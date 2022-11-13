import pandas as pd
from enum import Enum
from typing import Dict

class Factor(Enum):
    INFLATION = 'inflation'
    UNEMPLOYMENT = 'unemployment'
    CONSUMER_PRICE_INDEX = 'consumer_price_index'
    PRODUCER_PRICE_INDEX = 'producer_price_index'
    EXPORT_PRICE_INDEX = 'export_price_index'
    IMPORT_PRICE_INDEX = 'import_price_index'
    AVERAGE_HOURLY_EARNINGS = 'average_hourly_earnings'
    
factor_to_file_name = {
    Factor.INFLATION: 'SeriesReport-20221018154634_4f9f2e.xlsx',
    Factor.UNEMPLOYMENT: 'SeriesReport-20221018162036_4119b9.xlsx',
    Factor.CONSUMER_PRICE_INDEX: 'SeriesReport-20221018162729_d7c62b.xlsx',
    Factor.PRODUCER_PRICE_INDEX: 'SeriesReport-20221018162824_85d041.xlsx',
    Factor.EXPORT_PRICE_INDEX: 'SeriesReport-20221018162918_c24229.xlsx',
    Factor.IMPORT_PRICE_INDEX: 'SeriesReport-20221018163001_81b92c.xlsx',
    Factor.AVERAGE_HOURLY_EARNINGS: 'SeriesReport-20221018163151_ec4930.xlsx'}

class FactorDataLoader():
    
    def _create_df_from_xlsx(factor: Factor):
    
        if factor not in factor_to_file_name:
            raise ValueError('Factor not supported')
        
        file_name = factor_to_file_name[factor]
        df = pd.read_excel(f'factor_data/{file_name}').astype(str)
        
        # TODO: This is a bit dumb
        year_index = df.index[df.iloc[:, 0] == 'Year'].to_list()[0]
        df = pd.read_excel(f'factor_data/{file_name}', skiprows=year_index+1).astype(str)
    
        date_to_value = {'date': [], factor: []}
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for _, row in df.iterrows():
            for m in months:
                date_to_value['date'].append(pd.to_datetime(f"{row['Year']}-{m}-01", format='%Y-%b-%d')) 
                date_to_value[factor].append(float(row[m]))
    
        return pd.DataFrame(date_to_value).set_index('date')

    def get_factor_data_by_month(self, factor: Factor, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Dict[pd.Timestamp, float]:
        
        data = FactorDataLoader._create_df_from_xlsx(factor)
        
        if start_date < data.index[0]:
            raise ValueError('Start date is before data')
        
        if end_date > data.index[-1]:
            raise ValueError('End date is after data')
        
        return data[data.index.to_series().between(start_date,end_date)].to_dict()[factor]
