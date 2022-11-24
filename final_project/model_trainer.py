from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils import data
from typing import List, Tuple
import pandas as pd

from factor_data_loader import FactorDataLoader, Factor
from market_data_loader import MarketDataLoader, GroupPeriod
from model import CVAE
import utilities


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
    std_price = np.std(all_eod_prices)
    mean_price = np.mean(all_eod_prices)
    
    normalized_eod_data = {k: (v-mean_price)/std_price for k, v in monthly_eod_prices.items()}
    weekly_batched_eod_data = {
        k:  np.array([prices[i:i+5] for i in range(0, len(prices), 5)])
        for k, prices in normalized_eod_data.items()}
    
    weekly_data = [
        (np.concatenate((np.array([monthly_factors[month]]), weekly_prices[i-1])), weekly_prices[i]) 
        for month, weekly_prices in weekly_batched_eod_data.items()
        for i in range(1, len(weekly_prices))
        if len(weekly_prices[i]) == 5 and len(weekly_prices[i-1]) == 5]
    
    return weekly_data, std_price, mean_price
   

def create_train_val_dataloaders(weekly_data: list, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    
    md = utilities.ConditionedMarketDataset(weekly_data)
    train_size = int(len(weekly_data) * 0.8)
    train_data, test_data = data.random_split(md, (train_size, len(weekly_data) - train_size))
    
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_dataset, val_dataset
    

   
def run_experiment():
    
    cvae = CVAE(5, 6).to(utilities.DEVICE)
    weekly_data, std_price, mean_price = load_weekly_data_conditoned_on_previous_week_and_factor()
    train_dataset, val_dataset = create_train_val_dataloaders(weekly_data)
    
    history = utilities.train_model(cvae, train_dataset, val_dataset)
    utilities.save_loss(history, "loss.png")
    evaluate_model(cvae, val_dataset, std_price, mean_price, "eval_after_training.png")
    
    
def evaluate_model(cvae, val_dataset, std_price, mean_price, fig_name):
    
    predicted_val_prices = []
    predicted_synthetic_prices = []
    actual_val_prices = []
    
    for batch in val_dataset:
        
        price_batch = batch['price_data']
        synthetic_price_batches = [torch.FloatTensor(np.array([np.random.randn(len(b)) for b in batch['price_data']])) for _ in range(100)]
        
        factor_batch = batch['factor_data']
        price_batch = price_batch.to(utilities.DEVICE)
        
        predicted_prices = cvae(price_batch.float(), factor_batch.float())
        predicted_val_prices += (predicted_prices*std_price+mean_price).detach().numpy().tolist()
        actual_val_prices += (price_batch*std_price+mean_price).detach().numpy().tolist()
       
        sample_synthetic_prices = [cvae(synthetic_b.float(), factor_batch.float()).detach().numpy() for synthetic_b in synthetic_price_batches]
        mean_synthetic_prices = np.mean(sample_synthetic_prices, axis=0)
        predicted_synthetic_prices += (mean_synthetic_prices*std_price+mean_price).tolist()
                
        

    plt.plot(np.array(predicted_val_prices).flatten(), label="predicted")
    plt.plot(np.array(predicted_synthetic_prices).flatten(), label="synthetic mean over 100 samples")
    plt.plot(np.array(actual_val_prices).flatten(), label="actual")
    plt.legend()
    plt.savefig(fig_name)
    plt.clf()
        
     
def main():
    run_experiment()

if __name__ == "__main__":
    main()

    
    