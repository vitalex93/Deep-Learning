from Dataset import *

class Noise(MusicSoundDataset):

    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples):
        super().__init__(annotations_file, audio_dir, transformation, target_sample_rate, num_samples)

    def _get_audio_sample_label(self, index):
        return super()._get_audio_sample_label(index)   

    def _get_audio_sample_path(self, index):
        return super()._get_audio_sample_path(index)

    def _cut_if_necessary(self, signal):
        return super()._cut_if_necessary(signal)

    def _right_pad_if_necessary(self, signal):
        return super()._right_pad_if_necessary(signal)
    
    def _resample_if_necessary(self, signal, sr):
        return super()._resample_if_necessary(signal, sr)
    
    def _mix_down_if_necessary(self, signal):
        return super()._mix_down_if_necessary(signal)
    
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal + (0.1**0.5)*torch.randn(1, list(signal.shape[1]))
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

if __name__ == "__main__":
    ANNOTATIONS_FILE = "/home/vitalex93/Desktop/Data_Science/Deep_Learning/DLproject/Data/features_30_sec.csv"
    AUDIO_DIR = "/home/vitalex93/Desktop/Data_Science/Deep_Learning/DLproject/Data/genres_original/"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    md = MusicSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram,
                            SAMPLE_RATE, NUM_SAMPLES)
    print(f"There are {len(md)} samples in the dataset.")
    signal, label = md[0]
    

