# import utils_ts from outside of the folder src
import yaml

with open("config/config.yaml", 'r') as stream:
    configuration_param = yaml.safe_load(stream)

class Forecast():
    def __init__(self, model, h=configuration_param["forecasting"]["h"], freq=configuration_param["forecasting"]["freq"], 
                 time_col=configuration_param["forecasting"]["time_col"], 
                 target_col=configuration_param["forecasting"]["target_col"]):
        self.model = model
        self.h = h
        self.freq = freq
        self.time_col = time_col
        self.target_col = target_col


    def inference(self, df):

        df_result = self.model.forecast(df,
                                h=1,
                                freq='B',
                                time_col='timestamp',
                                target_col='value')
        
        return df_result