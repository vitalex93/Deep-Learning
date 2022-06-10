import os

from torch.utils.data import Dataset
import pandas as pd
import torchaudio

class MusicSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, label

    def _get_audio_sample_path(self, index):
        fold = self.annotations.iloc[index, 59]
        path = os.path.join(self.audio_dir, fold , self.annotations.iloc[
            index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 59]


if __name__ == "__main__":
    ANNOTATIONS_FILE = "/home/vitalex93/Desktop/Data_Science/Deep_Learning/DLproject/Data/features_30_sec.csv"
    AUDIO_DIR = "/home/vitalex93/Desktop/Data_Science/Deep_Learning/DLproject/Data/genres_original/"
    md = MusicSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR)
    print(f"There are {len(md)} samples in the dataset.")
    signal, label = md[0]

