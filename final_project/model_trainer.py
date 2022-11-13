from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils import data
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from typing import Dict, List, Tuple
import pandas as pd
import random

from factor_data_loader import FactorDataLoader, Factor
from market_data_loader import MarketDataLoader, GroupPeriod
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

def rmse_loss_fn(x, recon_x):
   
    criterion = nn.MSELoss()
    return torch.sqrt(criterion(x, recon_x))
   

def evaluate(losses, cvae: CVAE, dataloader: data.DataLoader, device: torch.device):
    
    loss_sum = []
    for (_, batch) in enumerate(dataloader):
        
        price_batch = batch['price_data']
        factor_batch = batch['factor_data']
        price_batch = price_batch.to(device)
        
        outputs = cvae(price_batch.float(), factor_batch.float())       
        loss = rmse_loss_fn(price_batch.float(), outputs)            
        loss_sum.append(loss)

    losses.append((sum(loss_sum)/len(loss_sum)).item())
    
def train_model(cvae: CVAE, dataloader: data.DataLoader, test_dataloader: data.DataLoader, epochs=30):
   
    validation_losses = []
    optim = torch.optim.Adam(cvae.parameters())
    log_template = "\nEpoch {ep:03d} val_loss {v_loss:0.4f}"
    
    with tqdm(desc="epoch", total=epochs) as pbar_outer:  
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
            
            evaluate(validation_losses, cvae, test_dataloader, DEVICE)
            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=i+1, v_loss=validation_losses[i]))
    
    plt.show()
    return validation_losses


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
    
def load_weekly_data_conditioned_on_factor() -> List[Tuple[np.array, np.array]]:
    
    mdl = MarketDataLoader()
    fdl = FactorDataLoader()
    
    start_ts = pd.Timestamp('2016-01-01')
    end_ts = pd.Timestamp('2021-02-01')
    
    monthly_eod_prices = mdl.get_eod_price_data_grouped('SPY', start_ts, end_ts, GroupPeriod.MONTHLY)
    monthly_factors = fdl.get_factor_data_by_month(Factor.INFLATION, start_ts, end_ts)
    assert set(monthly_eod_prices.keys()) == set(monthly_factors.keys()), "Price and Factor dates are misaligned"
    
    all_eod_prices = np.concatenate([prices for prices in list(monthly_eod_prices.values())])
    eod_norm = np.linalg.norm(all_eod_prices)
    normalized_eod_data = {k: v/eod_norm for k, v in monthly_eod_prices.items()}
    
    weekly_batched_eod_data = {
        k:  np.array([prices[i:i+5] for i in range(0, len(prices), 5)])
        for k, prices in normalized_eod_data.items()}
    
    weekly_data = [
        (np.array([monthly_factors[month]]), weekly_price) 
        for month, weekly_prices in weekly_batched_eod_data.items()
        for weekly_price in weekly_prices
        if len(weekly_price) == 5]
    
    return weekly_data, eod_norm

def load_weekly_data_conditoned_on_previous_week_and_factor() -> List[Tuple[np.array, np.array]]:
    
    mdl = MarketDataLoader()
    fdl = FactorDataLoader()
    
    start_ts = pd.Timestamp('2016-01-01')
    end_ts = pd.Timestamp('2021-02-01')
    
    monthly_eod_prices = mdl.get_eod_price_data_grouped('SPY', start_ts, end_ts, GroupPeriod.MONTHLY)
    monthly_factors = fdl.get_factor_data_by_month(Factor.INFLATION, start_ts, end_ts)
    assert set(monthly_eod_prices.keys()) == set(monthly_factors.keys()), "Price and Factor dates are misaligned"
    
    all_eod_prices = np.concatenate([prices for prices in list(monthly_eod_prices.values())])
    min_price = np.min(all_eod_prices)
    max_price = np.max(all_eod_prices)
    diff = max_price - min_price
    normalized_eod_data = {k: (v-min_price)/diff for k, v in monthly_eod_prices.items()}

    
    weekly_batched_eod_data = {
        k:  np.array([prices[i:i+5] for i in range(0, len(prices), 5)])
        for k, prices in normalized_eod_data.items()}
    
    weekly_data = [
        (np.concatenate((np.array([monthly_factors[month]]), weekly_prices[i-1])), weekly_prices[i]) 
        for month, weekly_prices in weekly_batched_eod_data.items()
        for i in range(1, len(weekly_prices))
        if len(weekly_prices[i]) == 5 and len(weekly_prices[i-1]) == 5]
    
    return weekly_data, diff, min_price
   

def create_train_val_dataloaders(weekly_data: list, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    
    md = ConditionedMarketDataset(weekly_data)
    train_size = int(len(weekly_data) * 0.8)
    train_data, test_data = data.random_split(md, (train_size, len(weekly_data) - train_size))
    
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_dataset, val_dataset
    
def save_loss(val_loss, name: str):
    plt.figure(figsize=(15, 9))
    plt.plot(val_loss, label="val_loss")
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(name)
    plt.clf()
   
def run_experiment():
    
    cvae = CVAE(5, 6).to(DEVICE)
    weekly_data, diff, min_price = load_weekly_data_conditoned_on_previous_week_and_factor()
    train_dataset, val_dataset = create_train_val_dataloaders(weekly_data)
    
    history = train_model(cvae, train_dataset, val_dataset)
    save_loss(history, "loss.png")
    evaluate_model(cvae, val_dataset, diff, min_price, "eval_after_training.png")
    
    
    
def evaluate_model(cvae, val_dataset, diff, min_price, fig_name):
    
    predicted_val_prices = []
    predicted_synthetic_prices = []
    actual_val_prices = []
    
    for batch in val_dataset:
        
        price_batch = batch['price_data']
        synthetic_price_batch = torch.FloatTensor(np.array([np.random.randn(len(b)) for b in batch['price_data']]))
        factor_batch = batch['factor_data']
        price_batch = price_batch.to(DEVICE)
        
        predicted_prices = cvae(price_batch.float(), factor_batch.float())
        predicted_synthetic_price = cvae(synthetic_price_batch.float(), factor_batch.float())
        
        # Really more interested in average of synthetic prices. Did we capture the correct disitrbution?
        # Should I take out the linear increase in SPY? Ie, normalie out the trend
        
        predicted_val_prices += (predicted_prices*diff+min_price).detach().numpy().tolist()
        predicted_synthetic_prices += (predicted_synthetic_price*diff+min_price).detach().numpy().tolist()
        actual_val_prices += (price_batch*diff+min_price).detach().numpy().tolist()

    plt.plot(np.array(predicted_val_prices).flatten(), label="predicted")
    plt.plot(np.array(predicted_synthetic_prices).flatten(), label="synthetic")
    plt.plot(np.array(actual_val_prices).flatten(), label="actual")
    plt.legend()
    plt.savefig(fig_name)
    plt.clf()
        
     
def main():
    run_experiment()

if __name__ == "__main__":
    main()

    
    