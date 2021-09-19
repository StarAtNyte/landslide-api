import pandas as pd
import numpy as np
import joblib
import requests

lr = joblib.load("model.pkl")
model_columns = joblib.load("model_cols.pkl")
print(lr)