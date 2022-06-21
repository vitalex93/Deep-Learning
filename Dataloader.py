from Dataset import *
from torch import nn
from torch.utils.data import DataLoader

BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 0.1
LATENT_DIM = 2

ANNOTATIONS_FILE = "./data/features_30_sec.csv"
AUDIO_DIR = "./data/genres_original/"
SAMPLE_RATE = 22050
FRAME_SIZE = 512
HOP_LENGTH = 256
DURATION = 1
NUM_SAMPLES = DURATION*SAMPLE_RATE
INPUT_SHAPE = 256

md = MusicSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, FRAME_SIZE, HOP_LENGTH, SAMPLE_RATE, NUM_SAMPLES)
spec, label, max, min = md[0]

DIM_1 = spec.size(1)
DIM_2 = spec.size(2)




def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader





