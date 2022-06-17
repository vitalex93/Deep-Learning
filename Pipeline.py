from Dataset import *
from Spectrogram import LogSpectrogramExtractor, MinMaxNormaliser
from Saver import *



class Pipeline:

    def __init__(self, SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR, FILES_DIR,
    ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, FRAME_SIZE, HOP_LENGTH, NUM_SAMPLES):
        self.spectrograms_save_dir = SPECTROGRAMS_SAVE_DIR
        self.min_max_values_save_dir = MIN_MAX_VALUES_SAVE_DIR
        self.files_dir = FILES_DIR
        self.annotations_file =  ANNOTATIONS_FILE
        self.audio_dir = AUDIO_DIR
        self.sample_rate = SAMPLE_RATE
        self.frame_size = FRAME_SIZE
        self.hop_length = HOP_LENGTH
        self.num_samples = NUM_SAMPLES
    
    def load_process(self):
        








SPECTROGRAMS_SAVE_DIR = "/home/vitalex93/Desktop/Data_Science/Deep_Learning/DLproject/Data/log_spectrograms"
MIN_MAX_VALUES_SAVE_DIR = "/home/vitalex93/Desktop/Data_Science/Deep_Learning/DLproject/Data/min_max_values"
FILES_DIR = "/home/valerio/datasets/fsdd/audio/"
ANNOTATIONS_FILE = "/home/vitalex93/Desktop/Data_Science/Deep_Learning/DLproject/Data/features_30_sec.csv"
AUDIO_DIR = "/home/vitalex93/Desktop/Data_Science/Deep_Learning/DLproject/Data/genres_original/"
SAMPLE_RATE = 22050
FRAME_SIZE = 512
HOP_LENGTH = 256
NUM_SAMPLES = 22050