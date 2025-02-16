import os
import sys
import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception
from dataclasses import dataclass
from pathlib import Path

from src.utils.utils import save_object,evaluate_model

from sklearn.linear_model import LinearRegression , Ridge 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from xgboost import XGBRegressor




@dataclass
class ModelTrainerConfig:
    pass
class ModelTrainer:
    def __init__(self):
        pass
    def initiate_model_training(self):
        try:
            pass
        except Exception as e:
            logging.info()
            raise customexception(e,sys)
