import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv("final_data.csv")
X = dataset.drop(['output', 'DISTRICT', 'Unnamed: 0'], axis=1)
y = dataset['output']

import xgboost
xgb = xgboost.XGBRegressor(
    max_depth=2,
    gamma=2,
    eta=0.8,
    reg_alpha=0.5,
    reg_lambda=0.5
)
res=xgb.fit(X,y)
import joblib
joblib.dump(res, "model.pkl")
cols=X.columns
joblib.dump(cols, "model_cols.pkl")
