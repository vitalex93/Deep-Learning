from torch import nn

#hyperparameter
latent_size= 2



def reparameterise(self, mu, logvar):
    if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
    return eps.mul(std).add_(mu)
    else:
    return mu

    def encode(self, x):
        mu_logvar = self.encoder(x.view(-1, 784)).view(-1, 2, latent_size)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        return mu, logvar

def decode(self, z):
    return self.decoder(z)

def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparameterise(mu, logvar)