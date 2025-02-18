import os
import sys
import pandas as pd
from src.exception.exception import customexception
from src.logger.logging import logging
from src.utils.utils import load_object


class PredictPipeline:

    def __init__(self):
        print("initialising the object")

    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            scaled_features=preprocessor.transform(features)
            pred=model.predict(scaled_features)

        except Exception as e:
            raise customexception(e,sys)
        
class CustomData:
    def __init__(self,
                 store: str,
                 holiday_flag: int,
                 temperature: float,
                 fuel_price: float,
                 cpi: float,
                 unemployment: float,
                 month: int,
                 season: str):

        self.store = store
        self.holiday_flag = holiday_flag
        self.temperature = temperature
        self.fuel_price = fuel_price
        self.cpi = cpi
        self.unemployment = unemployment
        self.month = month
        self.season = season

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'store': [self.store],
                'holiday_flag': [self.holiday_flag],
                'temperature': [self.temperature],
                'fuel_price': [self.fuel_price],
                'cpi': [self.cpi],
                'unemployment': [self.unemployment],
                'month': [self.month],
                'season': [self.season]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception occurred in prediction pipeline')
            raise customexception(e,sys)