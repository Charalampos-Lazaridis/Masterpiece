import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import requests

# Visualization settings
sns.set(style="whitegrid")
plt.style.use('fivethirtyeight')

# Constants
BASE_URL = "https://masterpiece.odins.es:443/temporal/entities"
ENTITY_TEMPLATE = "urn:ngsi-ld:DeviceMeasurement:UEDAS-TR-{building}-Mainfloor-GlobalMeter-activeEnergyImport"
TYPE = "DeviceMeasurement"
START_TIME = "2024-01-01T00:00:00Z"
END_TIME = "2025-01-27T23:59:59Z"
PROPERTY = "numValue"
HEADERS = {
    'fiware-service': 'masterpiece',
    'fiware-servicepath': '/',
    'x-auth-token': '{{AuthZToken}}'
}
BUILDINGS = [f"B{i}" for i in range(1, 14)]  # B1 to B13
LOOKBACK, N_OUT = 24, 24  # Input and output time steps
EPOCHS, BATCH_SIZE = 100, 64

# Function to fetch data
def fetch_data(building):
    entity_id = ENTITY_TEMPLATE.format(building=building)
    url = f"{BASE_URL}/{entity_id}/type/{TYPE}/time/{START_TIME}/endTime/{END_TIME}/attrs/{PROPERTY}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data for {building}: {response.status_code}")
    r = response.json()
    data = [(item['observedAt'], item['value']) for item in r.get('numValue', [])]
    df = pd.DataFrame(data, columns=["Datetime", "Value"])
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df.set_index("Datetime")

# Data preprocessing steps
def time_features(df):
    """
    Create time series features based on datetime index
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    return df

def cyclic_encoding(df):
    """
    Apply cyclic encoding to time-based features
    """
    df = df.copy()
    for column in ['hour', 'dayofweek', 'month', 'quarter']:
        max_value = df[column].max()
        df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / max_value)
        df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / max_value)
    return df

def preprocess_data(df):
    """
    Preprocess the raw data by filling missing values, generating features, and scaling.
    """
    daily_counts = df.groupby(df.index.floor('D')).size()
    incomplete_days = daily_counts[daily_counts < 96].index
    df_cleaned = df[~df.index.floor('D').isin(incomplete_days)]
    
    df_hourly = df_cleaned.resample('1H').sum()
    df_hourly.rename(columns={'Value': 'consumption'}, inplace=True)
    df_hourly = time_features(df_hourly)
    df_hourly_c = cyclic_encoding(df_hourly)
    df_hourly_c.drop(columns=df_hourly.columns[1:], inplace=True)
    return df_hourly_c

def random_day_pairs(data, lookback, n_out):
    X, y = [], []
    total_steps = lookback + n_out

    for i in range(0, len(data) - total_steps + 1, total_steps):  # Skip overlapping windows
        X.append(data[i:i + lookback, :])
        y.append(data[i + lookback:i + lookback + n_out, 0])  # Assuming target is column 0

    X, y = shuffle(np.array(X), np.array(y), random_state=42)
    return X, y

# Model definition
def build_model(input_shape, output_units):
    print(input_shape)
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(units=150, return_sequences=True)),
        LSTM(units=100, return_sequences=False),
        Flatten(),
        Dropout(0.2),
        Dense(units=output_units)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mean_absolute_error'])
    return model

# Plot and save visualizations
def save_visualizations(df, history, building):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    df['consumption'].plot(ax=axes[0], title=f"{building} - Data Overview")
    axes[0].set_xlabel("Datetime")
    axes[0].set_ylabel("Consumption")
    
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title("Training vs Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"visualizations/{building}_data_and_training.png")


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    for building in BUILDINGS:
        print(f"Processing {building}...\n")
        try:
            # Fetch and preprocess data
            df = fetch_data(building)
            df_preprocessed = preprocess_data(df)
            
            # Prepare training data
#             scaler = MinMaxScaler()
#             scaled_data = scaler.fit_transform(df_preprocessed)
            X, y = random_day_pairs(df_preprocessed.values, LOOKBACK, N_OUT)
            
            # Build and train the model
            model = build_model((X.shape[1], X.shape[2]), y.shape[1])
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            history = model.fit(X, y, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping], verbose=1)
            
            # Save model and visualizations
            model.save(f"models/{building}_model.h5")
            save_visualizations(df_preprocessed, history, building)
            print(f"Completed {building}.\n")
        
        except Exception as e:
            print(f"Error processing {building}: {e}\n")

if __name__ == "__main__":
    main()
    plt.close(fig)
