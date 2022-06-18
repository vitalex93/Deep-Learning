
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
        return x[:, :, :256, :87]


class Autoencoder(nn.Module):
   

    def __init__(self):
        super().__init__()

        
        self.encoder = nn.Sequential( 
                #input volume = 256*87*1 = 22272
                nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                #dim1 = [(256 + 2*1 - 3)/1] + 1 = 256  
                #dim2 = [(87 + 2*1 - 3)/1] + 1 = 87   
                # output volume = 256*87 = 22272 
                nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                #dim1 = [(256 + 2*1 - 3)/2] + 1 = 129  
                #dim2 = [(87 + 2*1 - 3)/2] + 1 = 44   
                # output volume = 129*44 = 5676 
                nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                #dim1 = [(129 + 2*1 - 3)/2] + 1 = 65  
                #dim2 = [(44 + 2*1 - 3)/2] + 1 = 23   
                # output volume = 65*23 = 1495 
                nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
                #dim1 = [(65 + 2*1 - 3)/1] + 1 = 65  
                #dim2 = [(23 + 2*1 - 3)/1] + 1 = 23   
                # output volume = 65*23 = 1495 
                
                nn.Flatten(),
                nn.Linear(90112, 2)
                )
                #64*22*64 = 90112

        #self.final_linear = nn.Linear(3136, 2)

        self.decoder = nn.Sequential(
            torch.nn.Linear(2, 90112),
            Reshape(-1, 64, 64, 22),
            nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            #dim1 = 64
            #dim2 = 22
            nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=0),                
            nn.LeakyReLU(0.01),
            #dim1 = 127 
            #dim2 = 43
            nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),                
            nn.LeakyReLU(0.01),
            #dim1 = 255
            #dim2 = 87
            nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=1), 
            #dim1 = 257
            #dim1 = 89
            Trim(),  # 1x257x89 -> 1x256x287
            nn.Sigmoid()
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
