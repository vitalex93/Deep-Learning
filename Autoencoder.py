
from locale import DAY_1
import torch
from torch import nn
from typos import *
#from torchsummary import summary



class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, *args, d1, d2):
        super().__init__()
        self.d1 = d1
        self.d2 = d2
    def forward(self, x):
        return x[:, :, :self.d1, :self.d2]


class Autoencoder(nn.Module):
   

    def __init__(self, latent_dim = 50, dim1 = 256, dim2 = 862):
        super().__init__()

        self.latent_dim = latent_dim 
        self.dim1 = dim1
        self.dim2 = dim2
        self.encoder = nn.Sequential( 
                #input volume = 256*862*1 = 220672
                nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                #dim1 = [(256 + 2*1 - 3)/1] + 1 = 256  
                #dim2 = [(862 + 2*1 - 3)/1] + 1 = 862   
                # output volume = 256*87 = 22272 
                nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                #dim1 = [(256 + 2*1 - 3)/2] + 1 = 128  
                #dim2 = [(862 + 2*1 - 3)/2] + 1 = 431   
                # output volume = 128*431 = 55168
                nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                #dim1 = [(128 + 2*1 - 3)/2] + 1 = 64  
                #dim2 = [(431 + 2*1 - 3)/2] + 1 = 216   
                # output volume = 64*216 = 13824
                nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
                #dim1 = [(64 + 2*1 - 3)/1] + 1 = 64  
                #dim2 = [(216 + 2*1 - 3)/1] + 1 = 216   
                # output volume = 64*216 = 13824 
                
                nn.Flatten(),
                #nn.Linear(90112, self.latent_dim)
                )
                #64*216*64 = 884736

        
        self.final_linear = nn.Linear(typos(self.dim1,self.dim2), self.latent_dim)
        self.decoder = nn.Sequential(
            torch.nn.Linear(self.latent_dim, typos(self.dim1,self.dim2)),
            Reshape(-1, 64, 64, 216),
            nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            #dim1 = 1*(64-1) + 3 - 2*1 = 64
            #dim2 = 1*(216-1) + 3 - 2*1 = 216
            nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=0),                
            nn.LeakyReLU(0.01),
            #dim1 = 2*(64-1) + 3 - 2*0 = 129 
            #dim2 = 2*(215-1) + 3 - 2*0 = 433
            nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),                
            nn.LeakyReLU(0.01),
            #dim1 = 2*(129-1) + 3 - 2*0 = 259 
            #dim2 = 2*(433-1) + 3 - 2*0 = 867
            nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=1), 
            #dim1 = 1*(259-1) + 3 - 2*1 = 259 
            #dim2 = 1*(867-1) + 3 - 2*1 = 867
            Trim(d1=self.dim1, d2=self.dim2),  # 1x259x867 -> 1x256x862
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.final_linear = nn.Linear(typos(self.dim1,self.dim2), self.latent_dim)(x)
        x = self.decoder(x)
        return x

    def reconstruct(self, images):
        latent_representations = self.encoder(images)
        reconstructed_images = self.decoder(latent_representations)
        return reconstructed_images, latent_representations
