# -*- coding: utf-8 -*-
"""
Example of the use of the Estimated Minute Ventilation model

@author: oliver
"""
#install package joblib
#pip install joblib

#load package
from joblib import load
import pandas as pd
import numpy as np
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR


#load model
#Choose the right model for your data
#DI_model include ['HR', 'sex', 'age', 'height', 'weight'] 
#PAE_model include ['HR', 'PA_habit', 'METS', 'temp', 'hum'] 
#All_model include ['HR', 'sex', 'age', 'height', 'weight', 'PA_habit', 'METS', 'temp', 'hum'] 
VE_model = load('C:/xgb_model.joblib')

#load your data
#DI_model include ['HR', 'sex', 'age', 'height', 'weight'] 
#PAE_model include ['HR', 'PA_habit', 'METS', 'temp', 'hum'] 
#All_model include ['HR', 'sex', 'age', 'height', 'weight', 'PA_habit', 'METS', 'temp', 'hum'] 
#Column names are the same as above
your_data = pd.read_csv('your_data.csv')

#Predicting minute ventilation using models
VE_preds = VE_model.predict(your_data)

#Adding predictive data to the data frame
your_data['VE_pred'] = VE_preds

#Save data
your_data.to_csv('your_data.csv', index=False)

