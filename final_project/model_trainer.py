from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
import matplotlib
import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils import data
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
from tqdm import tqdm, tqdm_notebook
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Tuple
from model import CVAE
from factor_data_loader import FactorDataLoader
from market_data_loader import MarketDataLoader
from alpaca_trade_api.rest import TimeFrame

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_mnist_data(batch_size: int, transform: transforms.Compose) -> Tuple[data.DataLoader, data.DataLoader]:
    
    dataset = MNIST('./mnist_data', transform=transform, download=True)
    train_data, test_data = data.random_split(dataset, (50000,10000))
    
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_dataset, val_dataset
    
def plot_loss(history):
    loss, val_loss = zip(*history)
    plt.figure(figsize=(15, 9))
    plt.plot(loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

def one_hot(x, max_x):
    return torch.eye(max_x + 1)[x]

def vae_loss_fn(x, recon_x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def evaluate(losses, cvae: CVAE, dataloader: data.DataLoader, device: torch.device, flatten=False):
    
    model = lambda x, y: cvae(x, y)[0]    
    loss_sum = []
    loss_fn = nn.MSELoss()
    
    for (_, batch) in enumerate(dataloader):
        
        inputs = batch['Data']
        labels = batch['Class']
        
        inputs = inputs.to(device)
        labels = one_hot(labels, 1).to(device)

        if flatten:
            inputs = inputs.view(inputs.size(0), 28*28)

        outputs = model(inputs.float(), labels)
        loss = loss_fn(inputs.float(), outputs)            
        loss_sum.append(loss)

    losses.append((sum(loss_sum)/len(loss_sum)).item())
    
def train_model(cvae: CVAE, dataloader: data.DataLoader, test_dataloader: data.DataLoader, flatten=False, epochs=20):
   
    validation_losses = []
    optim = torch.optim.Adam(cvae.parameters())
    log_template = "\nEpoch {ep:03d} val_loss {v_loss:0.4f}"
    
    with tqdm(desc="epoch", total=epochs) as pbar_outer:  
        for i in range(epochs):
            for (_, batch) in enumerate(dataloader):
               
                data_batch = batch['Data']
                class_batch = batch['Class']
                
                data_batch = data_batch.to(DEVICE)
                labels = one_hot(class_batch, 1).to(DEVICE)

                if flatten:
                    data_batch = data_batch.view(data_batch.size(0), 28*28)

                optim.zero_grad()
                x, mu, logvar = cvae(data_batch.float(), labels)
                loss = vae_loss_fn(data_batch.float(), x[:, :5], mu, logvar)
                loss.backward()
                optim.step()
            
            evaluate(validation_losses, cvae, test_dataloader, DEVICE)
            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=i+1, v_loss=validation_losses[i]))
    
    plt.show()
    return validation_losses


class MarketDataset(Dataset):
    def __init__(self, data, labels):
        self.labels = labels
        self.data = data
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        d = self.data[idx]
        sample = {"Data": d, "Class": label}
        return sample
    
def load_market_data(batch_size: int = 1) -> Tuple[data.DataLoader, data.DataLoader]:
    
    mdl = MarketDataLoader()
    mdl.load_data('SPY', '2015-01-01', '2020-01-01', TimeFrame.Day)
    
    eod_data = mdl.get_eod_price()
    normalized_eod_data = eod_data / np.linalg.norm(eod_data)
    
    weekly_batches = np.array([
        normalized_eod_data[i:i+5] 
        for i in range(0, len(normalized_eod_data), 5)
        if i+5 < len(normalized_eod_data)])
    
    avg_batch = np.mean(weekly_batches)
    labels = [1 if np.mean(batch) > avg_batch else 0 for batch in weekly_batches]
    md = MarketDataset(weekly_batches, labels)
    
    train_size = int(len(weekly_batches) * 0.8)
    train_data, test_data = data.random_split(md, (train_size, len(weekly_batches) - train_size))
    
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_dataset, val_dataset


def main():
    
    cvae = CVAE(5, 20, 2).to(DEVICE)
    #train_dataset, val_dataset = load_mnist_data(128, transform)
    train_dataset, val_dataset = load_market_data()
    history = train_model(cvae, train_dataset, val_dataset)

    val_loss = history
    plt.figure(figsize=(15, 9))
    plt.plot(val_loss, label="val_loss")
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

if __name__ == "__main__":
    main()

    
    