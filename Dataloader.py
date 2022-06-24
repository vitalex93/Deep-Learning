from Dataset import *
from Autoencoder import Autoencoder
from VAE import VAE
from torch import nn
from torch.utils.data import DataLoader

BATCH_SIZE = 128
EPOCHS = 1
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "./data/features_30_sec.csv"
AUDIO_DIR = "./data/genres_original/"
SAMPLE_RATE = 22050
FRAME_SIZE = 512
HOP_LENGTH = 256
NUM_SAMPLES = 44100
INPUT_SHAPE = 256


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

md = MusicSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, FRAME_SIZE, HOP_LENGTH, SAMPLE_RATE, NUM_SAMPLES)
train_dataloader = create_data_loader(md, BATCH_SIZE)


