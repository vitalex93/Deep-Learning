import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from VAE import *
from Spectrogram import *

FILE = 'feedforwardnet.pth'
model = VAE()
model.load_state_dict(torch.load(FILE))
model.eval()

HOP_LENGTH = 256

with torch.no_grad():
    new_image = model.decoder(torch.tensor([-0.0, 0.03]).to('cpu'))
    print(new_image.size())
    new_image.squeeze_(0)
    new_image.squeeze_(0)
plt.imshow(new_image.to('cpu').numpy(), cmap='binary')
plt.show()


log_spectrogram = new_image[:, :, 0]
spec = librosa.db_to_amplitude(log_spectrogram)
min_max = MinMaxNormaliser(0,1).denormalise(log_spectrogram)
signal = librosa.istft(min_max, hop_length=HOP_LENGTH)

print(signal)



def convert_spectrograms_to_audio(spectrograms, min_max_values):
    signals = []
    #for spectrogram, min_max_value in zip(spectrograms, min_max_values):
        # reshape the log spectrogram
    log_spectrogram = spectrogram[:, :, 0]
    # apply denormalisation
    denorm_log_spec = self._min_max_normaliser.denormalise(
        log_spectrogram, min_max_value["min"], min_max_value["max"])
    # log spectrogram -> spectrogram
    spec = librosa.db_to_amplitude(denorm_log_spec)
    # apply Griffin-Lim
    signal = librosa.istft(spec, hop_length=self.hop_length)
    # append signal to "signals"
    signals.append(signal)
    return signals



