
import torch
from torch import nn



class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :128, :128]


class Autoencoder(nn.Module):
   

    def __init__(self):
        super().__init__()

        
        self.encoder = nn.Sequential( #784
                #input volume = 28*28*1
                nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                #output volume = [(28 + 2*1 - 3)/1] + 1 = 28  (28*28*32 = 25088)
                nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                #output volume = [(28 + 2*1 - 3)/2] + 1 = 14 (14*14*64 = 12544)
                nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                #output volume = [(14 + 2*1 - 3)/2] + 1 = 8 (8*8*64 = 4096)
                nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
                #output volume = [(8 + 2*1 - 3)/1] + 1 = 8 (8*8*64 = 4096)
                nn.Flatten(),
                nn.Linear(3136, 2)
                )

        #self.final_linear = nn.Linear(3136, 2)

        self.decoder = nn.Sequential(
            torch.nn.Linear(2, 3136),
            Reshape(-1, 64, 7, 7),
            nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),                
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=1),                
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=1), 
            Trim(),  # 1x29x29 -> 1x28x28
            nn.Softmax(dim=1)
            )

    def forward(self, x):
        x = self.encoder(x)
        #self.final_linear = nn.Linear(3136, 2)
        x = self.decoder(x)
        return x

    def reconstruct(self, images):
        latent_representations = self.encoder(images)
        reconstructed_images = self.decoder(latent_representations)
        return reconstructed_images, latent_representations
