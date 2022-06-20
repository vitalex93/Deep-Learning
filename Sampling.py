

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from VAE import *
from Spectrogram import *
import soundfile as sf
from Dataset import *
from Dataloader import *



FILE = 'feedforwardnet.pth'
model = VAE()
model.load_state_dict(torch.load(FILE))
model.eval()

HOP_LENGTH = 256

with torch.no_grad():
    new_image = model.decoder(torch.tensor([2.2552, 0.6494]).to('cpu'))
    

    new_image.squeeze_(0)
    print(new_image.size())
    #new_image.squeeze_(0)
    #print(new_image)
#plt.imshow(new_image.to('cpu').numpy(), cmap='binary')
#plt.show()

log_spectrogram = new_image.numpy()
log_spec = log_spectrogram[0,:,:]
min_max = MinMaxNormaliser(0,1).denormalise(log_spec, -49, 30)
print(log_spec.shape)
spec = librosa.db_to_amplitude(min_max)

signal = librosa.istft(spec, hop_length=HOP_LENGTH)

print(signal.shape)


save_dir = './data/sampling/'
sample_rate = 22050
save_path = os.path.join(save_dir + "test.wav")
sf.write(save_path, signal, sample_rate)




for input, _, _, _ in train_dataloader:    
    with torch.no_grad():
        latent = model.encoding_fn(input)
        
        latent.squeeze_(0)
        latent.squeeze_(0)
        print(latent)
    plt.imshow(latent.to('cpu').numpy(), cmap='binary')
    plt.show()










