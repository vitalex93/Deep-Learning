import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from VAE import *

FILE = 'feedforwardnet.pth'
model = VAE()
model.load_state_dict(torch.load(FILE))
model.eval()

for param in model.parameters():
    print(param)


#with torch.no_grad():
    #new_image = model.decoder(torch.tensor([-0.0, 0.03]).to('cpu'))
    #new_image.squeeze_(0)
    #new_image.squeeze_(0)
#plt.imshow(new_image.to('cpu').numpy(), cmap='binary')
#plt.show()