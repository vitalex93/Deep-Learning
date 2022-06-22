import torch
#from torch import 
from Autoencoder import *


class VAE(Autoencoder):

    def __init__(self, latent_dim, dim1, dim2):

        super().__init__(latent_dim = latent_dim, dim1 = dim1, dim2 = dim2)
        self.z_mean = torch.nn.Linear(216*64*64, self.latent_dim)
        self.z_log_var = torch.nn.Linear(216*64*64, self.latent_dim)





    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded

        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to('cpu')
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
        
    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded
