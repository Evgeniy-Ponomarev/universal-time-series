# import utils.preprocessing from outside of the current directory
import os
# add current working directory to path
# os.chdir("/home/azureuser/cloudfiles/code/Users/Jiahui.Kang/Universal_Forecast_AIsuite/")
from uni_time_series.preprocessing import *
from uni_time_series.forecasting import *
from uni_time_series.postprocessing import *
# from utils_ts.plot import *
from pandas_datareader import data as pdr 
import yfinance as yf 
yf.pdr_override()
from nixtlats import TimeGPT

tickers=['XLY','HD','NKE',
         'XLP','CL','EL','KO','PEP',
         'XLE','APA','OXY',
         'XLF','WFC','GS','BLK',
         'XLV','PFE','HUM',
         'XLI','FDX','GD',
         'XLB','ECL',
         'XLU','DTE']

stock_names=['HD','NKE',
        'CL','EL','KO','PEP',
        'APA','OXY',
        'WFC','GS','BLK',
        'PFE','HUM',
        'FDX','GD',
        'ECL','DTE']

match = {"XLY": ["HD", "NKE"],
         "XLP": ["CL", "EL", "KO", "PEP"],
         "XLE": ["APA", "OXY"],
         "XLF": ["WFC", "GS", "BLK"],
         "XLV": ["PFE", "HUM"],
         "XLI": ["FDX", "GD"],
         "XLB": ["ECL"],
         "XLU": ["DTE"]}


prep = Preprocess_Financial()
os.environ['TIMEGPT_TOKEN'] = 'get your token from https://api.nixtlats.ai/'
timegpt = TimeGPT(token=os.environ['TIMEGPT_TOKEN'])
forecst = Forecast(model=timegpt)
postp = Postprocess_Financial()

if __name__ == "__main__":
    
    data_dict = {key: pdr.get_data_yahoo(key, start="2000-01-01", end="2023-11-01") for key in tickers}
    result = {}
    for key, value in match.items():
        df_etf = data_dict[key]
        df_etf_rt = prep.get_returns(df_etf)
        for stk in value:
            df_stock = data_dict[stk]
            df_stock_rt = prep.get_returns(df_stock)
            df_comb = prep.get_diff(df_etf_rt, df_stock_rt)
            # df_input = prep.create_df_split(df_comb)
            stock_m_split=prep.create_df_split_temp(df_comb)
            df_input=stock_m_split[stock_m_split['unique_id']<252]
            df_pred = forecst.inference(df_input)
            # df_result = postp.get_merged(df_comb, df_pred)
            df_pred['timestamp']=pd.to_datetime(df_pred['timestamp'])
            df_pred=df_pred.rename(columns={"timestamp": "ts_tgpt"})
            df_pred['timestamp']=df_stock_rt[(df_stock_rt['timestamp']>'2014-12-31')&(df_stock_rt['timestamp']<='2015-12-31')].timestamp.to_list()
            stock_m_comb = pd.merge(df_pred, df_comb, on="timestamp")
            stock_m_final=stock_m_comb[['timestamp','value_diff','TimeGPT']]
            stock_m_final['sign'] = np.where(stock_m_final['TimeGPT']>0, 1, -1)
            stock_m_final['PnL']=stock_m_final.sign*stock_m_final.value_diff
            result[stk] = stock_m_final
            SR_l= postp.get_sr(result)
            # plot_sr(SR_l)
            # plot_pnl(result)
    all_stocks=pd.DataFrame()
    for key, value in result.items():
        all_stocks[key]=value['PnL']
    all_stocks['sum'] = all_stocks.sum(axis=1)
    sr_portfolio=252*all_stocks['sum'].mean()/((((all_stocks['sum']-all_stocks['sum'].mean())**2).sum())**0.5)
    print(f'Portfolio annual Sharpe ratio for 2015: {sr_portfolio:.2f}')