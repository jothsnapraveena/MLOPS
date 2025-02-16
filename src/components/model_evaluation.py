import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from src.logger.logging import logging
from src.exception.exception import customexception
from dataclasses import dataclass
from pathlib import Path
import pickle
from src.utils.utils import load_object



@dataclass
class ModelEvaluationConfig:
    pass
class ModelEvaluation:
    def __init__(self):
        pass
    def initiate_model_evaluation(self):
        try:
            pass
        except Exception as e:
            logging.info()
            raise customexception(e,sys)
