import pandas as pd

factor_to_file_name = {
    'inflation': 'SeriesReport-20221018154634_4f9f2e.xlsx',
    'unemployment': 'SeriesReport-20221018162036_4119b9.xlsx',
    'consumer_price_index': 'SeriesReport-20221018162729_d7c62b.xlsx',
    'producer_price_index': 'SeriesReport-20221018162824_85d041.xlsx',
    'export_price_index': 'SeriesReport-20221018162918_c24229.xlsx',
    'import_price_index': 'SeriesReport-20221018163001_81b92c.xlsx',
    'average_hourly_earnings': 'SeriesReport-20221018163151_ec4930.xlsx'}

class FactorDataLoader():
    
    def __init__(self):
        self.df = pd.DataFrame()
        for factor in factor_to_file_name:
            self.df = pd.concat([self.create_df_from_bls_xlsx(factor), self.df], axis=1)
    
    @staticmethod
    def create_df_from_bls_xlsx(factor: str):
    
        if factor not in factor_to_file_name:
            raise ValueError('Factor not supported')
        
        df = pd.read_excel(f'data/{factor_to_file_name[factor]}').astype(str)
        
        # TODO: This is a bit dumb
        year_index = df.index[df.iloc[:, 0] == 'Year'].to_list()[0]
        df = pd.read_excel(f'data/{factor_to_file_name[factor]}', skiprows=year_index+1).astype(str)
    
        date_to_value = {'date': [], factor: []}
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for _, row in df.iterrows():
            for m in months:
                date_to_value['date'].append(pd.to_datetime(f"{row['Year']}-{m}-01", format='%Y-%b-%d')) 
                date_to_value[factor].append(row[m])
    
        return pd.DataFrame(date_to_value).set_index('date')


    
def main():
    
    fdl = FactorDataLoader()
    print(fdl.df)
    
if __name__ == '__main__':
    main()