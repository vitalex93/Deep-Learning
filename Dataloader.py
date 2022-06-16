from Dataset import *

ANNOTATIONS_FILE = "/home/vitalex93/Desktop/Data_Science/Deep_Learning/DLproject/Data/features_30_sec.csv"
AUDIO_DIR = "/home/vitalex93/Desktop/Data_Science/Deep_Learning/DLproject/Data/genres_original/"
SAMPLE_RATE = 22050
FRAME_SIZE = 512
HOP_LENGTH = 256
NUM_SAMPLES = 22050

dataset = MusicSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, FRAME_SIZE, HOP_LENGTH, SAMPLE_RATE, NUM_SAMPLES)
first_data = dataset[0]
spec, labels = first_data 

print(spec)

