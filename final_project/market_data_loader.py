import pandas as pd
from dotenv import load_dotenv
import os
class MarketDataLoader():
    
    def __init__(self) -> None:
        pass
    
    def _load_keys(self) -> None:
        
        if not os.path.exists('secrets.env'):
            raise ValueError('secrets.env not found - please create your own secrets.env file')
        
        load_dotenv('secrets.env')
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.api_url = os.getenv('ALPHA_VANTAGE_API_URL')

def main():
    pass

if __name__ == '__main__':
    main()