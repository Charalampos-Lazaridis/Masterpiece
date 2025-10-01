#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import requests


# In[2]:


poc_entities = { 
    "Chemistry Faculty": [
        "urn:ngsi-ld:DeviceMeasurement:UMU-POC-ChemistryFaculty-MachineryRoom-GlobalMeter-activeEnergyImport"
    ],
    "Veterinary Faculty": [
        "urn:ngsi-ld:DeviceMeasurement:UMU-POC-VetFaculty-MachineryRoom-GlobalMeter-activeEnergyImport"
    ],
    "Work Sciences Faculty": [
        "urn:ngsi-ld:DeviceMeasurement:UMU-POC-WorkSciencesFaculty-MachineryRoom-GlobalMeter-activeEnergyImport"
    ],
    "Psychology Faculty": [
        "urn:ngsi-ld:DeviceMeasurement:UMU-POC-PsicologyFaculty-MachineryRoom-GlobalMeter-activeEnergyImport"
    ],
    "Mathematics Faculty and General Lecturing Building": [
        "urn:ngsi-ld:DeviceMeasurement:UMU-POC-MathematicsFacultyandGeneralLecturingBuilding-MachineryRoom-GlobalMeter-1-activeEnergyImport",
        "urn:ngsi-ld:DeviceMeasurement:UMU-POC-MathematicsFacultyandGeneralLecturingBuilding-MachineryRoom-GlobalMeter-2-activeEnergyImport"
    ],
    "Giner Rios Lecturing Building": [
        "urn:ngsi-ld:DeviceMeasurement:UMU-POC-GinerRiosLecturingBuilding-MachineryRoom-GlobalMeter-activeEnergyImport"
    ]
}


# In[3]:


type_ = "DeviceMeasurement"
startTime_ = "2023-01-28T00:00:00Z"
endTime_ = "2025-06-23T23:59:59Z"
property_ = "numValue"
base_url = "https://masterpiece.odins.es:443/temporal/entities"

headers = {
    'fiware-service': 'masterpiece',
    'fiware-servicepath': '/',
    'x-auth-token': '{{AuthZToken}}'  # Replace with actual token
}


# In[4]:


entity_dataframes = {}

# Fetch data for each entity
for building, entity_list in poc_entities.items():
    entity_dataframes[building] = {}  # Initialize nested dictionary for the building

    for entityID_ in entity_list:
        url = f"{base_url}/{entityID_}/type/{type_}/time/{startTime_}/endTime/{endTime_}/attrs/{property_}"

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            r = response.json()
            data = [(item['observedAt'], item['value']) for item in r.get(property_, [])]

            if data:
                df = pd.DataFrame(data, columns=["Datetime", "Value"])
                df["Datetime"] = pd.to_datetime(df["Datetime"])
                df.set_index("Datetime", inplace=True)

                # Store DataFrame separately for each meter
                entity_dataframes[building][entityID_] = df
                
                print(f"*** {building}: OK ***\n")
        else:
            print(f"Failed to retrieve data for {entityID_} in {building}: {response.status_code}\n")


# In[5]:


from collections import defaultdict

# Dictionary to store the count of days with < 96 time steps per entity
missing_data_counts = defaultdict(lambda: defaultdict(int))

for building, meters in entity_dataframes.items():
    for entityID_, df in meters.items():
        daily_counts = df.resample('D').count()  # Count entries per day
        missing_days = (daily_counts["Value"] < 96).sum()  # Count days with < 96 records
        
        missing_data_counts[building][entityID_] = missing_days

# Print results
for building, meters in missing_data_counts.items():
    print(f"\n{building}:")
    for entityID_, count in meters.items():
        print(f"  {entityID_}: {count} days with less than 96 time steps")


# In[6]:


df_chemistry = entity_dataframes["Chemistry Faculty"]["urn:ngsi-ld:DeviceMeasurement:UMU-POC-ChemistryFaculty-MachineryRoom-GlobalMeter-activeEnergyImport"].copy()

df_veterinary = entity_dataframes["Veterinary Faculty"]["urn:ngsi-ld:DeviceMeasurement:UMU-POC-VetFaculty-MachineryRoom-GlobalMeter-activeEnergyImport"].copy()

df_work_sciences = entity_dataframes["Work Sciences Faculty"]["urn:ngsi-ld:DeviceMeasurement:UMU-POC-WorkSciencesFaculty-MachineryRoom-GlobalMeter-activeEnergyImport"].copy()

df_psychology = entity_dataframes["Psychology Faculty"]["urn:ngsi-ld:DeviceMeasurement:UMU-POC-PsicologyFaculty-MachineryRoom-GlobalMeter-activeEnergyImport"].copy()

df_mathematics_1 = entity_dataframes["Mathematics Faculty and General Lecturing Building"]["urn:ngsi-ld:DeviceMeasurement:UMU-POC-MathematicsFacultyandGeneralLecturingBuilding-MachineryRoom-GlobalMeter-1-activeEnergyImport"].copy()

df_mathematics_2 = entity_dataframes["Mathematics Faculty and General Lecturing Building"]["urn:ngsi-ld:DeviceMeasurement:UMU-POC-MathematicsFacultyandGeneralLecturingBuilding-MachineryRoom-GlobalMeter-2-activeEnergyImport"].copy()

df_giner_rios = entity_dataframes["Giner Rios Lecturing Building"]["urn:ngsi-ld:DeviceMeasurement:UMU-POC-GinerRiosLecturingBuilding-MachineryRoom-GlobalMeter-activeEnergyImport"].copy()


# In[7]:


dfs = {
    "Chemistry": df_chemistry,
    "Veterinary": df_veterinary,
    "Work Sciences": df_work_sciences,
    "Psychology": df_psychology,
    "Mathematics 1": df_mathematics_1,
    "Mathematics 2": df_mathematics_2,
    "Giner Rios": df_giner_rios
}

# Modify values in place
for name, df in dfs.items():
    if df is not None:
        df.loc[df["Value"] > 200, "Value"] = 40
        print(f"{name}: Capped values above 200 to 40.")
    else:
        print(f"{name}: No data available.")


# In[8]:


import matplotlib.pyplot as plt

# Dictionary to store filled DataFrames
filled_dfs = {}

# List of DataFrames to process
dfs = {
    "Chemistry": df_chemistry,
    "Veterinary": df_veterinary,
    "Work Sciences": df_work_sciences,
    "Psychology": df_psychology,
    "Mathematics 1": df_mathematics_1,
    "Mathematics 2": df_mathematics_2,
    "Giner Rios": df_giner_rios
}

# Process and store each filled DataFrame
for name, df in dfs.items():
    if df is not None:
        # Remove duplicate timestamps
        df = df[~df.index.duplicated(keep="first")]

        # Sort index to avoid errors in resampling
        df = df.sort_index()

        # Resample to ensure all 15-minute intervals exist
        df_filled = df.resample('15T').asfreq()

        # Identify missing values before filling
        missing_values = df_filled["Value"].isna()

        # Fill missing values using linear interpolation
        df_filled["Value"] = df_filled["Value"].interpolate(method="linear")

        # Store the processed DataFrame
        filled_dfs[name] = df_filled

        # Create the plot
        plt.figure(figsize=(15, 6))
        plt.plot(df_filled.index, df_filled["Value"], label="Original & Filled Data", color="blue", alpha=0.7)

        # Highlight injected missing values
        plt.scatter(df_filled.index[missing_values], df_filled["Value"][missing_values], 
                    color="red", label="Injected Values", s=4, zorder=3)  # Smaller dot size (s=4)

        plt.title(f"{name} Faculty Energy Data (Injected Values Highlighted)")
        plt.xlabel("Datetime")
        plt.ylabel("Energy Consumption")
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"{name}: Total injected values = {missing_values.sum()}")
    else:
        print(f"{name}: No data available.")


# In[9]:


def time_features_cyclical(df):
    df = df.copy()
    
    # Standard time features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)  # Binary flag
    
    # Apply cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23)
    
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 6)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 6)
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    
    # Drop original non-cyclical columns
    df.drop(columns=['hour', 'dayofweek', 'month', 'quarter'], inplace=True)
    
    return df

def create_sequences(data, lookback=24, forecast_horizon=24, step=24):
    """
    - `lookback`: The number of past time steps used as input.
    - `forecast_horizon`: The number of future time steps to predict.
    - `step`: The interval between starting points of consecutive sequences (now 24 instead of 1).
    """
    data = np.array(data)  # Ensure data is a NumPy array
    X, y = [], []
    
    for i in range(0, len(data) - lookback - forecast_horizon + 1, step):  # Shift by `step=24`
        X.append(data[i:i+lookback])  # Input: Last 24 hours
        y.append(data[i+lookback:i+lookback+forecast_horizon, 0])  # Output: Next 24 hours

    return np.array(X), np.array(y)


def quantile_loss(q, y_true, y_pred):
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))

def build_model_1(input_shape):

    model = Sequential([
        Conv1D(filters=256, kernel_size=4, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=1),
        Conv1D(filters=128, kernel_size=4, activation='relu'),
        MaxPooling1D(pool_size=3),
        
        Bidirectional(LSTM(units=150, return_sequences=True)),
        LSTM(units = 100, return_sequences=False),
        Flatten(),
        Dropout(0.2),
        Dense(units=forecast_horizon)
    ])
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: quantile_loss(0.8, y_true, y_pred), 
                  metrics=['mean_squared_error'])
    
    return model

def build_model_2(input_shape):

    model = Sequential([
        Conv1D(filters=256, kernel_size=4, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=1),
        Conv1D(filters=128, kernel_size=4, activation='relu'),
        MaxPooling1D(pool_size=3),
        
        Bidirectional(LSTM(units=150, return_sequences=True)),
        LSTM(units = 100, return_sequences=False),
        Flatten(),
        Dropout(0.2),
        Dense(units=forecast_horizon)
    ])
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: quantile_loss(0.7, y_true, y_pred), 
                  metrics=['mean_squared_error'])
    
    return model

def build_model_3(input_shape):

    model = Sequential([
        Conv1D(filters=256, kernel_size=4, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=1),
        Conv1D(filters=128, kernel_size=4, activation='relu'),
        MaxPooling1D(pool_size=3),
        
        Bidirectional(LSTM(units=150, return_sequences=True)),
        Bidirectional(LSTM(units=50, return_sequences=False)),
        Flatten(),
        Dropout(0.2),
        Dense(units=forecast_horizon)
    ])
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: quantile_loss(0.65, y_true, y_pred), 
                  metrics=['mean_squared_error'])
    
    return model

def build_model_4(input_shape):

    model = Sequential([
        Conv1D(filters=256, kernel_size=4, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=1),
        Conv1D(filters=128, kernel_size=4, activation='relu'),
        MaxPooling1D(pool_size=3),
        
        Bidirectional(LSTM(units=150, return_sequences=True)),
        Bidirectional(LSTM(units=100, return_sequences=False)),
        Flatten(),
        Dropout(0.2),
        Dense(units=forecast_horizon)
    ])
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: quantile_loss(0.7, y_true, y_pred), metrics=['mean_squared_error'])
    
    return model

def train_and_evaluate(model_func, X_train, y_train, X_test, model_name):
    tf.keras.backend.clear_session()
    model = model_func((X_train.shape[1], X_train.shape[2]))  # Build model
    history = model.fit(X_train, y_train, validation_split=0.1,
                        epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1)
    
    y_pred = model.predict(X_test)

    return model, history, y_pred

early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)


# In[10]:


def plot_training_validation_metrics(history, building, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))  # Create two side-by-side plots

    # Plot 1: Loss function (Quantile Loss or MSE)
    axes[0].plot(history.history['loss'], label='Training Loss', color='blue')
    axes[0].plot(history.history['val_loss'], label='Validation Loss', color='red', linestyle='dashed')
    axes[0].set_title(f"{model_name} - Training vs Validation Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Mean Squared Error (MSE)
    if 'mean_squared_error' in history.history:
        axes[1].plot(history.history['mean_squared_error'], label='Training MSE', color='blue')
        axes[1].plot(history.history['val_mean_squared_error'], label='Validation MSE', color='red', linestyle='dashed')
        axes[1].set_title(f"{model_name} - Training vs Validation MSE")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Mean Squared Error (MSE)")
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()  # Adjust layout for better spacing
    plt.savefig(f"visualizations/{building}_loss.png")
    plt.show()
    
def plot_predictions(y_test, y_pred_cnn_lstm, building):
    title=f"Predictions vs Actual Energy Consumption {building}"

    y_test_flat = y_test.reshape(-1)
    y_pred_cnn_lstm_flat = y_pred_cnn_lstm.reshape(-1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=y_test_flat,
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        y=y_pred_cnn_lstm_flat,
        mode='lines',
        name='CNN-LSTM Prediction',
        line=dict(color='orange')
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time Steps",
        yaxis_title="Energy Consumption",
        legend_title="Legend",
        template="plotly_white"
    )
    fig.show()
    fig.write_image(f"visualizations/{building}_data_and_training.png")


# In[ ]:





# In[11]:


test_size_days = 14
test_size = test_size_days * 24
lookback,forecast_horizon = 24,24
batch_size = 64
epochs = 500


# In[12]:


import tensorflow as tf


# In[13]:


import plotly.graph_objects as go


# In[14]:


buildings_left = ["Veterinary", "Work Sciences", "Psychology"]

for building in buildings_left:
    df_nn = filled_dfs[building].copy()
    df_nn.index = df_nn.index.tz_localize(None)
    df_nn = df_nn.resample('1H').sum().copy()

    for lag in [1, 6*24, 13*24]:  # Different lag intervals
        df_nn[f'lag_{lag}'] = df_nn['Value'].shift(lag)
    df_nn.dropna(inplace=True)
    
    df_nn = time_features_cyclical(df_nn)
   
    train_data = df_nn[:-test_size]
    test_data = df_nn[-test_size:]

    X_train, y_train = create_sequences(train_data, lookback, forecast_horizon)
    X_test, y_test = create_sequences(test_data, lookback, forecast_horizon)
    
    cnn_lstm_model, history_cnn_lstm, y_pred_cnn_lstm = train_and_evaluate(build_model_1, X_train, y_train, X_test, "CNN-LSTM")    
    
    plot_training_validation_metrics(history_cnn_lstm,building, "CNN-LSTM")    
    plot_predictions(y_test, y_pred_cnn_lstm, building)
#     cnn_lstm_model.save(f"models/{building}_model.h5")


# In[15]:


buildings_left = ["Chemistry"]

for building in buildings_left:
    df_nn = filled_dfs[building].copy()
    df_nn.index = df_nn.index.tz_localize(None)
    df_nn = df_nn.resample('1H').sum().copy()

    for lag in [1, 6*24, 13*24]:  # Different lag intervals
        df_nn[f'lag_{lag}'] = df_nn['Value'].shift(lag)
    df_nn.dropna(inplace=True)
    
    df_nn = time_features_cyclical(df_nn)
   
    train_data = df_nn[:-test_size]
    test_data = df_nn[-test_size:]

    X_train, y_train = create_sequences(train_data, lookback, forecast_horizon)
    X_test, y_test = create_sequences(test_data, lookback, forecast_horizon)
    
    cnn_lstm_model, history_cnn_lstm, y_pred_cnn_lstm = train_and_evaluate(build_model_2, X_train, y_train, X_test, "CNN-LSTM")    
    
    plot_training_validation_metrics(history_cnn_lstm, building, "CNN-LSTM")    
    plot_predictions(y_test, y_pred_cnn_lstm, building)
#     cnn_lstm_model.save(f"models/{building}_model.h5")


# In[16]:


batch_size = 32
buildings_left = ["Giner Rios","Mathematics 2","Mathematics 1"]

for building in buildings_left:
    df_nn = filled_dfs[building].copy()
    df_nn.index = df_nn.index.tz_localize(None)
    df_nn = df_nn.resample('1H').sum().copy()

    for lag in [1, 6*24, 13*24]:  # Different lag intervals
        df_nn[f'lag_{lag}'] = df_nn['Value'].shift(lag)
    df_nn.dropna(inplace=True)
    
    df_nn = time_features_cyclical(df_nn)
   
    train_data = df_nn[:-test_size]
    test_data = df_nn[-test_size:]

    X_train, y_train = create_sequences(train_data, lookback, forecast_horizon)
    X_test, y_test = create_sequences(test_data, lookback, forecast_horizon)
    
    cnn_lstm_model, history_cnn_lstm, y_pred_cnn_lstm = train_and_evaluate(build_model_3, X_train, y_train, X_test, "CNN-LSTM")    
    
    plot_training_validation_metrics(history_cnn_lstm, building, "CNN-LSTM")    
    plot_predictions(y_test, y_pred_cnn_lstm, building)
#     cnn_lstm_model.save(f"models/{building}_model.h5")


# In[22]:


def build_model_3(input_shape):

    model = Sequential([
        Conv1D(filters=256, kernel_size=4, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=1),
        Conv1D(filters=128, kernel_size=4, activation='relu'),
        MaxPooling1D(pool_size=3),
        
        Bidirectional(LSTM(units=150, return_sequences=True)),
        Bidirectional(LSTM(units=50, return_sequences=False)),
        Flatten(),
        Dropout(0.2),
        Dense(units=forecast_horizon)
    ])
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: quantile_loss(0.6, y_true, y_pred), metrics=['mean_squared_error'])
    
    return model


# In[ ]:





# In[17]:


batch_size = 64
buildings_left = ["Mathematics 1"]

for building in buildings_left:
    df_nn = filled_dfs[building].copy()
    df_nn.index = df_nn.index.tz_localize(None)
    df_nn = df_nn.resample('1H').sum().copy()

    for lag in [1, 6*24, 13*24]:  # Different lag intervals
        df_nn[f'lag_{lag}'] = df_nn['Value'].shift(lag)
    df_nn.dropna(inplace=True)
    
    df_nn = time_features_cyclical(df_nn)
   
    train_data = df_nn[:-test_size]
    test_data = df_nn[-test_size:]

    X_train, y_train = create_sequences(train_data, lookback, forecast_horizon)
    X_test, y_test = create_sequences(test_data, lookback, forecast_horizon)
    
    cnn_lstm_model, history_cnn_lstm, y_pred_cnn_lstm = train_and_evaluate(build_model_4, X_train, y_train, X_test, "CNN-LSTM")    
    
    plot_training_validation_metrics(history_cnn_lstm, building, "CNN-LSTM")    
    plot_predictions(y_test, y_pred_cnn_lstm, building)
#     cnn_lstm_model.save(f"models/{building}_model.h5")


# In[ ]:





# In[ ]:





# In[ ]:




