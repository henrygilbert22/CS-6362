import pandas as pd
from dotenv import load_dotenv
import os
from alpaca_trade_api.rest import REST, TimeFrame
from enum import Enum
from typing import Dict

class GroupPeriod(Enum):
    DAILY = 'D'
    MONTHLY = 'M'
    YEARLY = 'Y'
    
class MarketDataLoader():
    
    def __init__(self) -> None:
        self._load_api_keys()
    
    def _load_api_keys(self) -> None:
        
        if not os.path.exists('secrets.env'):
            raise ValueError('secrets.env not found - please create your own secrets.env " \
                " file with the following format: APCA_API_KEY_ID=YOUR_API_KEY APCA_API_SECRET_KEY=YOUR_SECRET_KEY')
        
        load_dotenv('secrets.env')
        
    def _download_data(self, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp, timeframe: TimeFrame) -> pd.DataFrame:
        # TODO: Add docstring on date string format
        
        api = REST()
        return api.get_bars(ticker, timeframe, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), adjustment='raw', limit=10_000).df
    
    def _get_data(self, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp, timeframe: TimeFrame) -> pd.DataFrame:
        # TODO: Add docstring and include dates are assumed US/Eastern
        
        localized_start_date = start_date.tz_localize('US/Eastern')
        localized_end_date = end_date.tz_localize('US/Eastern')
        
        if os.path.exists(f'data/market_data/{ticker}_{timeframe}.csv'):
            
            existing_df = pd.read_csv(f'data/market_data/{ticker}_{timeframe}.csv', index_col='timestamp', parse_dates=True)
            start = existing_start = existing_df.index[0]
            end = existing_end = existing_df.index[-1]
            
            if localized_start_date < existing_start:
                start, end = localized_start_date, existing_start
            
            if localized_end_date > existing_end:
                start, end = existing_end, localized_end_date
               
            price_df = pd.concat([self._download_data(ticker, start, end, timeframe), existing_df]) 
                       
        else:
            price_df = self._download_data(ticker, localized_start_date, localized_end_date, timeframe)
        
        
        price_df.drop_duplicates(inplace=True)
        price_df.to_csv(f'data/market_data/{ticker}_{timeframe}.csv', index=True)
        return price_df[price_df.index.to_series().between(localized_start_date,localized_end_date)]
        
    def get_eod_price_data_grouped(self, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp, group_by: GroupPeriod) -> Dict[pd.Timestamp, float]:
        """ Returns dictionary of dataframes with keys being the group_by period and values being the dataframes.
        Grouped values are averaged over the period. """
        
        data = self._get_data(ticker, start_date, end_date, TimeFrame.Day)
        return {
            group.index[0].to_period(group_by.value).to_timestamp(): group['close'].to_numpy() 
            for _, group in data.groupby(pd.Grouper(freq=group_by.value))}
