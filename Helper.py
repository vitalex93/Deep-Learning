from torch import nn
import math
from MusicSoundDataset import *

BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 0.1
LATENT_DIM = 50

ANNOTATIONS_FILE = "./data/features_30_sec.csv"
AUDIO_DIR = "./data/genres_original/"
SAMPLE_RATE = 22050
FRAME_SIZE = 512
HOP_LENGTH = 256
DURATION = 1
NUM_SAMPLES = DURATION * SAMPLE_RATE
INPUT_SHAPE = 256


# latent_dim=50, dim1=256, dim2=862


md = MusicSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, FRAME_SIZE, HOP_LENGTH, SAMPLE_RATE, NUM_SAMPLES)
spec, label, max, min = md[0]
DIM_1 = spec.size(1)
DIM_2 = spec.size(2)
print(DIM_1)
print(DIM_2)


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x[:, :, :self.dim1, :self.dim2]


def typos(DIM_1, DIM_2):
    total = 0
    # input
    # total += DIM_1 * DIM_2 * 1  #dimensions of spectogram * channel
    # First Conv2d
    dim1 = math.floor(((DIM_1 + 2*1 - 3)/1)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((DIM_2 + 2*1 - 3)/1)) + 1  # (input_dimension + 2* padding)/stride + 1
    # Second Conv2
    dim1 = math.floor(((dim1 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((dim2 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    # Third Conv2d
    dim1 = math.floor(((dim1 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((dim2 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    # Fourth Conv2d
    dim1 = math.floor(((dim1 + 2 * 1 - 3) / 1)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((dim2 + 2 * 1 - 3) / 1)) + 1  # (input_dimension + 2* padding)/stride + 1

    return dim1 * dim2 * 64  # returns Flatten output

def typos2(DIM_1, DIM_2):
    total = 0
    # input
    # total += DIM_1 * DIM_2 * 1  #dimensions of spectogram * channel
    # First Conv2d
    dim1 = math.floor(((DIM_1 + 2*1 - 3)/1)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((DIM_2 + 2*1 - 3)/1)) + 1  # (input_dimension + 2* padding)/stride + 1
    # Second Conv2
    dim1 = math.floor(((dim1 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((dim2 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    # Third Conv2d
    dim1 = math.floor(((dim1 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((dim2 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    # Fourth Conv2d
    dim1 = math.floor(((dim1 + 2 * 1 - 3) / 1)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((dim2 + 2 * 1 - 3) / 1)) + 1  # (input_dimension + 2* padding)/stride + 1

    return dim1  # returns Flatten output

def typos3(DIM_1, DIM_2):
    total = 0
    # input
    # total += DIM_1 * DIM_2 * 1  #dimensions of spectogram * channel
    # First Conv2d
    dim1 = math.floor(((DIM_1 + 2*1 - 3)/1)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((DIM_2 + 2*1 - 3)/1)) + 1  # (input_dimension + 2* padding)/stride + 1
    # Second Conv2
    dim1 = math.floor(((dim1 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((dim2 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    # Third Conv2d
    dim1 = math.floor(((dim1 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((dim2 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    # Fourth Conv2d
    dim1 = math.floor(((dim1 + 2 * 1 - 3) / 1)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((dim2 + 2 * 1 - 3) / 1)) + 1  # (input_dimension + 2* padding)/stride + 1

    return dim2