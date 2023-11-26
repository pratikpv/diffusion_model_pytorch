import os
from datetime import datetime
import matplotlib.pyplot as plt
from config import *
from dataset import *
from models import *


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
           + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image), cmap='gray')


def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)


@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(epoch, batch_num, IMG_DIR):
    # Sample noise
    img = torch.randn((1, 1, IMG_SIZE, IMG_SIZE), device=device)
    fig = plt.figure(figsize=(15, 5))
    plt.axis('off')
    num_images = 10
    stepsize = int(T / num_images)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i / stepsize) + 1)

        show_tensor_image(img.detach().cpu())

    filename = f"{IMG_DIR}/epoch_{epoch}_batch_{batch_num}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


def save_and_plot_losses(epoch_losses, LOG_ROOT):
    fig = plt.figure(figsize=(15, 15))
    plt.plot(epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig(os.path.join(LOG_ROOT, LOSSES_PNG))
    plt.close(fig)

    np.savetxt(os.path.join(LOG_ROOT, LOSSES_DATA), np.array(epoch_losses))


#############
# main
#############
current_time_str = str(datetime.now().strftime("%d-%b-%Y_%H_%M_%S"))
LOG_ROOT = os.path.join(LOG_MASTER_DIR, current_time_str)
os.makedirs(LOG_ROOT)
IMG_DIR = os.path.join(LOG_ROOT, OUTPUT_IMG_ROOT)
os.makedirs(IMG_DIR)
MODEL_SAVE_PATH = os.path.join(LOG_ROOT, MODEL_NAME)
os.makedirs(MODEL_SAVE_PATH)

betas = linear_beta_schedule(timesteps=T)
# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = SimpleUnet()
print(f"LOG_ROOT {LOG_ROOT}")
print("Num params: ", sum(p.numel() for p in model.parameters()))

model.to(device)
optimizer = Adam(model.parameters(), lr=lr)
epoch_losses = []
for epoch in range(epochs):

    batch_losses = []
    for batch_num, batch_data in enumerate(dataloader):
        optimizer.zero_grad()

        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
        loss = get_loss(model, batch_data[0], t)
        batch_losses.append(loss.cpu().detach())
        loss.backward()
        optimizer.step()

        if batch_num == 0:
            print(f"Epoch {epoch} | step {batch_num:03d} Loss: {loss.item()} ")
            sample_plot_image(epoch, batch_num, IMG_DIR)

    # save model every 10 epoch
    if epoch % 10 == 0:
        MODEL_SAVE_EPOCH_PATH = os.path.join(MODEL_SAVE_PATH, f"{MODEL_NAME}_{epoch}")
        OPT_SAVE_EPOCH_PATH = os.path.join(MODEL_SAVE_PATH, f"{OPT_NAME}_{epoch}")

        torch.save(model.state_dict(), MODEL_SAVE_EPOCH_PATH)
        torch.save(optimizer.state_dict(), OPT_SAVE_EPOCH_PATH)

    # save losses every epoch
    epoch_losses.append(sum(batch_losses) / len(batch_losses))
    save_and_plot_losses(epoch_losses, LOG_ROOT)
