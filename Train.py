from Dataset import *
from Autoencoder import Autoencoder
from VAE import VAE
from torch import nn
from torch.utils.data import DataLoader

BATCH_SIZE = 128
EPOCHS = 2
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "/home/vitalex93/Desktop/Data_Science/Deep_Learning/DLproject/Data/features_30_sec.csv"
AUDIO_DIR = "/home/vitalex93/Desktop/Data_Science/Deep_Learning/DLproject/Data/genres_original/"
SAMPLE_RATE = 22050
FRAME_SIZE = 512
HOP_LENGTH = 256
NUM_SAMPLES = 22050
INPUT_SHAPE = 256

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device = 'cpu'):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, epochs, device = 'cpu' ):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

   
    md = MusicSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, FRAME_SIZE, HOP_LENGTH, SAMPLE_RATE, NUM_SAMPLES)
    train_dataloader = create_data_loader(md, BATCH_SIZE)

    # construct model and assign it to device
    
    autoencoder = Autoencoder().to(device)
    print(autoencoder)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(autoencoder.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(autoencoder, train_dataloader, loss_fn, optimiser, EPOCHS, device)

    # save model
    torch.save(autoencoder.state_dict(), "feedforwardnet.pth")
    print("Trained autoencoder saved at feedforwardnet.pth")