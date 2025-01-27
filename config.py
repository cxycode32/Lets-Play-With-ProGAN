import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_MODEL = True
LOAD_MODEL = False

Z_DIM = 256  #512
IN_CHANNELS = 256  #512
IMG_CHANNELS = 3

LEARNING_RATE = 1e-3
START_TRAIN_AT_IMG_SIZE = 128
BATCH_SIZES = [32, 32, 32, 16, 16, 16, 16, 8, 4]
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)
NUM_WORKERS = 4

LAMBDA_GP = 10
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)

MODEL_DIR = "models"
DATASET_DIR = "dataset"
ASSETS_DIR = "assets"
IMAGE_DIR = "generated_images"
LOG_DIR = "logs"
DIRECTORIES=[MODEL_DIR, ASSETS_DIR, IMAGE_DIR, LOG_DIR]