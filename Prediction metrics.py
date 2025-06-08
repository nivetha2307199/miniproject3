#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pymysql
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import psutil
import os

# Database connection
conn = pymysql.connect(host='localhost', user='root', password='root@2024', database='power')
mycursor = conn.cursor()
mycursor.execute('select * from dataset')
result = mycursor.fetchall()

# Create DataFrame
df = pd.DataFrame(result, columns=('Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                                   'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'))

# Preprocess the data
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce', dayfirst=True)
df.drop(columns=['Date', 'Time'], inplace=True)
df = df.dropna(subset=['DateTime'])
for col in df.columns:
    if col != 'DateTime':
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()
df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day
df['WeekDay'] = df['DateTime'].dt.weekday
df['Month'] = df['DateTime'].dt.month
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
df['Rolling_Mean'] = df['Global_active_power'].rolling(window=60).mean().fillna(method='bfill')
df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')
df['Peak_Hours'] = df['Hour'].apply(lambda x: 1 if 17 <= x <= 20 else 0)

# Create the model dataset
df_model = df.drop(columns='DateTime')
X = df_model.drop('Global_active_power', axis=1)
Y = df_model['Global_active_power']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Define models
models = {
    'LinearR': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=50, n_jobs=2, random_state=42),  # Reduced trees and parallelism
    'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42),  # Reduced trees
    'NeuralNetworks': MLPRegressor(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
}

# Function to monitor memory usage
def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # Memory in MB

result={}
for name,model in models.items():
    model.fit(X_train, Y_train)
    preds = model.predict(X_test)
    result[name] = {
        'RMSE': np.sqrt(mean_squared_error(Y_test, preds)),
        'MAE': mean_absolute_error(Y_test, preds),
        'R2': r2_score(Y_test, preds)
    }
res_df=pd.DataFrame(result).T
res_df.head(10)


# In[ ]:




