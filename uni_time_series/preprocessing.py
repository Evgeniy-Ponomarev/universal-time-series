import numpy as np
import pandas as pd
import yaml

with open("config/config.yaml", 'r') as stream:
    configuration_param = yaml.safe_load(stream)

class Preprocess_Financial():

    def __init__(self, start_date=configuration_param["preprocessing"]["start_date"], 
                 end_date=configuration_param["preprocessing"]["end_date"], 
                 sample_size=configuration_param["preprocessing"]["sample_size"]):
        self.start_date = start_date
        self.end_date = end_date
        self.sample_size = sample_size

    def get_returns(self, df):
        df['close2open']=np.log(df['Close']/df['Open'])
        df=df.reset_index()
        df=df.rename(columns={"Date": "timestamp"})
        df_2=pd.DataFrame(columns=['timestamp', 'value'])
        df_2['timestamp']=df['timestamp']
        df_2['value']=df['close2open']
        return df_2

    def get_diff(self, df_etf, df_stks):
        df_comb = pd.merge(df_etf, df_stks, on="timestamp")
        df_comb["value_diff"] = df_comb["value_y"] - df_comb["value_x"]
        df_comb.rename(columns={"value_x": "etf_r", "value_y": "stock_r"}, inplace=True)
        return df_comb
    
    def create_df_split(self, df):
        l_dfs=[]
        df_full=pd.DataFrame()
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)
        hist_end_date=end_date + pd.DateOffset(days=self.sample_size)
        df=df[(df['timestamp']>start_date)&(df['timestamp']<hist_end_date)]
        df.reset_index(inplace=True)
        num_data_points=df.shape[0]
        for i in range(num_data_points):
            df_tr=df.iloc[i:i+self.sample_size, :].copy(deep=True)
            df_tr['unique_id']=i
            l_dfs.append(df_tr)
        for df_item in l_dfs:
            df_full=pd.concat([df_full,df_item], axis=0)
        df_full.reset_index(inplace=True)
        df_full.drop(columns=['level_0'], inplace=True)
        df_full.drop(columns=['index'], inplace=True)
        df_full_2=df_full[['timestamp', 'value_diff', 'unique_id']]
        df_full_2=df_full_2.rename(columns={"value_diff": "value"})
        #df_full_2=df_full_2.groupby('unique_id').filter(lambda x: len(x)==self.sample_size)
        
        return df_full_2
    
    def create_df_split_temp(self, df):
        l_dfs=[]
        df_full=pd.DataFrame()
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)
        df=df[(df['timestamp']>start_date)&(df['timestamp']<end_date)]
        df.reset_index(inplace=True)
        num_data_points=df.shape[0]
        size_df=144
        for i in range(num_data_points):
            df_tr=df.iloc[i:i+size_df, :].copy(deep=True)
            df_tr['unique_id']=i
            l_dfs.append(df_tr)
        for df_item in l_dfs:
            df_full=pd.concat([df_full,df_item], axis=0)
        df_full.reset_index(inplace=True)
        df_full.drop(columns=['level_0'], inplace=True)
        df_full.drop(columns=['index'], inplace=True)
        df_full_2=df_full[['timestamp', 'value_diff', 'unique_id']]
        df_full_2=df_full_2.rename(columns={"value_diff": "value"})
        
        return df_full_2