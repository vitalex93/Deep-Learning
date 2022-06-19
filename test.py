from VAE import *
from Dataset import *

BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "./data/features_30_sec.csv"
AUDIO_DIR = "./data/genres_original/"
SAMPLE_RATE = 22050
FRAME_SIZE = 512
HOP_LENGTH = 256
NUM_SAMPLES = 22050
INPUT_SHAPE = 256

md = MusicSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, FRAME_SIZE, HOP_LENGTH, SAMPLE_RATE, NUM_SAMPLES)
print(f"There are {len(md)} samples in the dataset.")
spec, label = md[0]



