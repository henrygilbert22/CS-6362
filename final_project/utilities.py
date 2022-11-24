from typing import Tuple, List
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from matplotlib import pyplot as plt
import torch
from torch.utils import data
from torch import nn
from tqdm import tqdm

from model import CVAE


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

class ConditionedMarketDataset(Dataset):
    def __init__(self, data: Tuple[float, np.array]):
        # TODO: Add docstring
        self.factor_data = [t[0] for t in data]
        self.price_data = [t[1] for t in data]
    
    def __len__(self):
        return len(self.factor_data)
    
    def __getitem__(self, idx):
        factor = self.factor_data[idx]
        prices = self.price_data[idx]
        return {"price_data": prices, "factor_data": factor}
    
    
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

def rmse_loss_fn(x, recon_x):
    criterion = nn.MSELoss()
    return torch.sqrt(criterion(x, recon_x))

def evaluate(losses: List, cvae: CVAE, dataloader: data.DataLoader, device: torch.device):
    
    loss_sum = []
    for (_, batch) in enumerate(dataloader):
        
        price_batch = batch['price_data']
        factor_batch = batch['factor_data']
        price_batch = price_batch.to(device)
        
        outputs = cvae(price_batch.float(), factor_batch.float())       
        loss = rmse_loss_fn(price_batch.float(), outputs)            
        loss_sum.append(loss)

    losses.append((sum(loss_sum)/len(loss_sum)).item())


def train_model(cvae: CVAE, dataloader: data.DataLoader, test_dataloader: data.DataLoader, epochs:int=30):
   
    validation_losses = []
    optim = torch.optim.Adam(cvae.parameters())
    for i in range(epochs):
        for (_, batch) in enumerate(dataloader):
            
            price_batch = batch['price_data']
            factor_batch = batch['factor_data']

            price_batch = price_batch.to(DEVICE)
            optim.zero_grad()
            
            x = cvae(price_batch.float(), factor_batch.float())         
            loss = rmse_loss_fn(price_batch.float(), x)
            loss.backward()
            optim.step()
        
       # evaluate(validation_losses, cvae, test_dataloader, DEVICE)
    
    plt.show()
    return validation_losses

def save_loss(val_loss, name: str):
    plt.figure(figsize=(15, 9))
    plt.plot(val_loss, label="val_loss")
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(name)
    plt.clf()