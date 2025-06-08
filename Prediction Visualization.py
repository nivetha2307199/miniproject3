#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Database connection
conn = pymysql.connect(host='localhost', user='root', password='root@2024', database='power_house')
mycursor = conn.cursor()
mycursor.execute("SELECT DateTime, Global_active_power FROM dataset WHERE DateTime BETWEEN '2006-12-16 17:24:00' AND '2007-04-30 00:00:00'")
result = mycursor.fetchall()
df = pd.DataFrame(result, columns=('DateTime', 'Global_active_power'))

# Preprocessing
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'].replace('?', np.nan), errors='coerce')
df = df.dropna(subset=['Global_active_power'])

# Feature engineering
df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day
df['WeekDay'] = df['DateTime'].dt.weekday
df['Month'] = df['DateTime'].dt.month
df['Peak_Hours'] = df['Hour'].apply(lambda x: 1 if 17 <= x <= 20 else 0)

# Lag features
for lag in [1, 5, 15, 30, 60]:
    df[f'Lag_{lag}'] = df['Global_active_power'].shift(lag)
df['Rolling_Mean'] = df['Global_active_power'].rolling(window=60).mean()
df = df.dropna()

# Define features and target
target = 'Global_active_power'
features = [col for col in df.columns if col not in ['DateTime', target]]
X = df[features]
y = df[target]

# Train/test split (chronological)
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train RandomForest
model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=2)
model.fit(X_train_scaled, y_train)

# Forecasting next 3 months (minute-level)
last_timestamp = df['DateTime'].max()
future_dates = pd.date_range(start=last_timestamp + timedelta(minutes=1), periods=60 * 24 * 90, freq='min')

# Sliding window initialization
prediction_window = df.tail(60).copy()
forecast_rows = []

for dt in future_dates:
    new_row = {
        'DateTime': dt,
        'Hour': dt.hour,
        'Day': dt.day,
        'WeekDay': dt.weekday(),
        'Month': dt.month,
        'Peak_Hours': 1 if 17 <= dt.hour <= 20 else 0
    }
    for lag in [1, 5, 15, 30, 60]:
        new_row[f'Lag_{lag}'] = prediction_window['Global_active_power'].iloc[-lag]
    new_row['Rolling_Mean'] = prediction_window['Global_active_power'].iloc[-60:].mean()

    row_df = pd.DataFrame([new_row])
    row_scaled = scaler.transform(row_df[features])
    prediction = model.predict(row_scaled)[0]
    new_row['Global_active_power'] = prediction

    forecast_rows.append(new_row)
    prediction_window = pd.concat([prediction_window, pd.DataFrame([new_row])], ignore_index=True)

# Forecast DataFrame
forecast_df = pd.DataFrame(forecast_rows)

# Plot
plt.figure(figsize=(15, 6))
plt.plot(forecast_df['DateTime'], forecast_df['Global_active_power'], label='Forecast (Next 3 Months)', color='orange')
plt.plot(df['DateTime'].iloc[-len(forecast_df):], df['Global_active_power'].iloc[-len(forecast_df):], 
         label='Recent Historical Data', linestyle='--', alpha=0.6)
plt.title("Power Forecast for Next 3 Months (Random Forest)")
plt.xlabel("DateTime")
plt.ylabel("Global Active Power (kilowatts)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




