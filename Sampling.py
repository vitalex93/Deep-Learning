

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

L = []
for input, _, _, _ in train_dataloader:    
    with torch.no_grad():
        latent = model.encoding_fn(input)
        
        latent.squeeze_(0)
        latent.squeeze_(0)
        L.append(latent)

#print(L[0].tolist()[1])
with torch.no_grad():
    new_image = model.decoder(torch.tensor(L[0].tolist()[1]).to('cpu'))
    

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




    #plt.imshow(latent.to('cpu').numpy(), cmap='binary')
    #plt.show()


class Sampler:
    
    def __init__(self, model = VAE(), sample_rate = 22050):
        self.model = model
        self.sample_rate = sample_rate

    def load_model(self, file ='feedforwardnet.pth'):
        model = self.model
        model.load_state_dict(torch.load(file))
        model.eval()
        return model

    def audio_converter(self, vector, min = -49, max = 30, HOP_LENGTH = 256):
        with torch.no_grad():
            new_image = model.decoder(torch.tensor(vector).to('cpu'))
            new_image.squeeze_(0)
        log_spectrogram = new_image.numpy()
        log_spec = log_spectrogram[0,:,:]
        min_max = MinMaxNormaliser(0,1).denormalise(log_spec, min, max)
        spec = librosa.db_to_amplitude(min_max)
        signal = librosa.istft(spec, hop_length=HOP_LENGTH)
        return signal
    
    def saver(self, save_dir = './data/sampling/', name = 'test.wav'):
        save_path = os.path.join(save_dir + name)
        sf.write(save_path, signal, self.sample_rate)

    def sampling(self):
        model = Sampler(self).load_model()
        signal = 


            

        







