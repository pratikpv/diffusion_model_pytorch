import torch

LOG_MASTER_DIR = "logs"
OUTPUT_IMG_ROOT = "sample_diffusion_op"
MODEL_NAME = "mnist_diffusion_model"
OPT_NAME = "mnist_diffusion_opt"
LOSSES_DATA = "losses.txt"
LOSSES_PNG = 'losses.png'

use_cuda = True
device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"

# hyper-params
T = 300
epochs = 300
lr = 0.001
IMG_SIZE = 28
BATCH_SIZE = 512
