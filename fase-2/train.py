import argparse
import numpy as np
from lightgbm import LGBMRegressor
from loguru import logger
import os
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--data_file', required=True, type=str, help='a csv file with train data')
parser.add_argument('--model_file', required=True, type=str, help='where the trained model will be stored')
parser.add_argument('--overwrite_model', default=False, action='store_true', help='if sets overwrites the model file if it exists')

args = parser.parse_args()

model_file = args.model_file
data_file  = args.data_file
overwrite = args.overwrite_model

if os.path.isfile(model_file):
    if overwrite:
        logger.info(f"overwriting existing model file {model_file}")
    else:
        logger.info(f"model file {model_file} exists. exitting. use --overwrite_model option")
        exit(-1)

logger.info("loading train data")
z = pd.read_csv(data_file).values
X = z[:,:5]
y = z[:,-1]

X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=128)

logger.info("fitting model")
#----------------------------
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.2,
    'num_leaves': 128,
    'min_child_samples': 79,
    'max_depth': 9,
    'subsample_freq': 1,
    'subsample': 0.9,
    'bagging_seed': 11,
    'reg_alpha': 0.1,
    'reg_lambda': 0.3,
    'colsample_bytree': 1.0
}
#Entrenamiento del modelo de regresión con el algoritmo LGBMRegressor
model = LGBMRegressor(**LGB_PARAMS, n_estimators=1500, n_jobs = -1)
model.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='mae',
        verbose=True, early_stopping_rounds=200)

y_pred = model.predict(X_test)
np.log(mean_absolute_error(y_test, y_pred))


logger.info(f"saving model to {model_file}")
with open(model_file, "wb") as f:
    pickle.dump(model, f)