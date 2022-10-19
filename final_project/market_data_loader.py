from datetime import tzinfo
import pandas as pd
from dotenv import load_dotenv
import os
from alpaca_trade_api.rest import REST, TimeFrame
import torch
import numpy as np

class MarketDataLoader():
    
    def __init__(self) -> None:
        
        self._load_api_keys()
    
    def _load_api_keys(self) -> None:
        
        if not os.path.exists('secrets.env'):
            raise ValueError('secrets.env not found - please create your own secrets.env " \
                " file with the following format: APCA_API_KEY_ID=YOUR_API_KEY APCA_API_SECRET_KEY=YOUR_SECRET_KEY')
        
        load_dotenv('secrets.env')
        
        
    def _download_data(self, ticker: str, start_date: str, end_date: str, timeframe: TimeFrame) -> pd.DataFrame:
        # TODO: Add docstring on date string format
        
        print(f"Downloading data for {ticker} from {start_date} to {end_date} with timeframe {timeframe}")
        api = REST()
        return api.get_bars(ticker, timeframe, start_date, end_date, adjustment='raw').df        
    
    def load_data(self, ticker: str, start_date: str, end_date: str, timeframe: TimeFrame) -> pd.DataFrame:
        # TODO: Add docstring on date string format
        
        if os.path.exists(f'market_data/{ticker}_{timeframe}.csv'):
            
            existing_df = pd.read_csv(f'market_data/{ticker}_{timeframe}.csv', index_col='timestamp', parse_dates=True)
            existing_start = existing_df.index[0]
            existing_end = existing_df.index[-1]
            
            if pd.to_datetime(start_date, utc=True) < existing_start:
                start, end = start_date, existing_start.strftime('%Y-%m-%d')
            
            if pd.to_datetime(end_date, utc=True) > existing_end:
                start, end = existing_end.strftime('%Y-%m-%d'), end_date
               
            requested_df = pd.concat([self._download_data(ticker, start, end, timeframe), existing_df])            
        
        else:
            requested_df = self._download_data(ticker, start_date, end_date, timeframe)
        
        requested_df.drop_duplicates(inplace=True)
        requested_df.to_csv(f'market_data/{ticker}_{timeframe}.csv', index=True)
        self.df = requested_df
        
    def get_eod_price_data_grouped_by_month(self):
        
        if self.df is None:
            raise ValueError('No data loaded')
        
        return {
            group[-1].index[0].to_period('M'): group[-1].close.to_numpy() 
            for group in self.df.groupby(pd.Grouper(freq='M'))}
        
    def get_eod_price(self):
        
        if self.df is None:
            raise ValueError('No data loaded')
        
        return self.df.close.to_numpy()
            
def main():
    
    mdl = MarketDataLoader()
    mdl.load_data('SPY', '2021-01-01', '2021-11-02', TimeFrame.Day)
    mdl.get_eod_price_data_grouped_by_month()

if __name__ == '__main__':
    main()