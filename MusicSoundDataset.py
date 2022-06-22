import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import numpy as np
from Spectrogram import LogSpectrogramExtractor, MinMaxNormaliser

#TODO REMOVE unused imports
import librosa
from sklearn.cluster import spectral_clustering


class MusicSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir, frame_size, hop_length,
                 target_sample_rate, num_samples):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        #TODO REMOVE label
        #label = self._get_audio_sample_label(index)
        label = torch.tensor([1])
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        log_spectrogram = LogSpectrogramExtractor(self.frame_size, self.hop_length).extract(signal[0].numpy())
        max, min = np.amax(log_spectrogram), np.amin(log_spectrogram)
        min_max = MinMaxNormaliser(0,1).normalise(log_spectrogram)
        spectrogram = min_max[np.newaxis,:,:]
        tensor = torch.from_numpy(spectrogram)
        return tensor, label, max, min

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        #singnal tensor with two dimensions (num_of_channels, samples)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        fold = self.annotations.iloc[index, 59]
        path = os.path.join(self.audio_dir, fold , self.annotations.iloc[
            index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 59]






#TODO REMOVE
'''
if __name__ == "__main__":
    ANNOTATIONS_FILE = "./data/features_30_sec.csv"
    AUDIO_DIR = "./data/genres_original/"
    SAMPLE_RATE = 22050
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    NUM_SAMPLES = 220500



    md = MusicSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, FRAME_SIZE, HOP_LENGTH, SAMPLE_RATE, NUM_SAMPLES)
    print(f"There are {len(md)} samples in the dataset.")
    spec, label, max, min = md[800]
    #print(spec.size())
    #print(md[0])

    M = []
    N = []
    for i in range(len(md)):
        spec, label, max, min = md[i]
        
        M.append(max)
        N.append(min)
    
    M.sort()
    N.sort()

    median_max = M[int(len(M)/2 - 0.5)]
    median_min = N[int(len(N)/2 - 0.5)]

    max_mean = sum(M)/len(M)
    min_mean = sum(N)/len(N)

    print(max_mean, min_mean)
    print(median_max, median_min)
 
'''


