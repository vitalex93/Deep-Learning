
import torch
from torch import nn



class Autoencoder(nn.Module):
   

    def __init__(self, input):
        super().__init__()

        self.input = input
        self.encoder = nn.Sequential( #784
                nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.Flatten()
                )

        self.final_linear = nn.Linear(3136, 2)

        self.decoder = nn.Sequential(
            torch.nn.Linear(2, 3136),
            #Reshape(-1, 64, 7, 7),
            nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),                
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),                
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=0), 
            #Trim(),  # 1x29x29 -> 1x28x28
            nn.Softmax(dim=1)
            )

    def forward(self):
        x = self.encoder(self.input)
        self.final_linear = nn.Linear(3136, 2)
        x = self.decoder(x)
        return x
