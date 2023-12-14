import numpy as np
import pandas as pd
import yaml

with open("config/config.yaml", 'r') as stream:
    configuration_param = yaml.safe_load(stream)

class Postprocess_Financial():

    def __init__(self): 

        return None
        
    def get_merged(self, df_input, df_output):
        df_input['timestamp']=pd.to_datetime(df_input['timestamp'])
        df_output['timestamp']=pd.to_datetime(df_output['timestamp'])
        df_comb = pd.merge(df_input, df_output, on='timestamp', how='inner')
        stock_m_final=df_comb[['timestamp','etf_r', 'stock_r', 'value_diff','TimeGPT']]
        stock_m_final['sign'] = np.where(stock_m_final['TimeGPT']>0, 1, -1)
        stock_m_final['PnL']=stock_m_final.sign*stock_m_final.value_diff
        
        return stock_m_final
    
    def get_sr(self, result_dict):
        SR_l={}
        for k, otp in result_dict.items():
            sr=252*otp['PnL'].mean()/((((otp['PnL']-otp['PnL'].mean())**2).sum())**0.5)
            SR_l[k] = sr
        return SR_l