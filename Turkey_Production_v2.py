import os, time
import pandas as pd
import numpy as np
import requests
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from pathlib import Path #<--

RUN_AT = os.getenv("RUN_AT", "23:20")
TIMEZONE = os.getenv("TIMEZONE", "Europe/Athens") 

BASE_DIR = Path(__file__).resolve().parent #<--
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://masterpiece.odins.es:443/temporal/entities"
ENTITY_TEMPLATE = "urn:ngsi-ld:DeviceMeasurement:UEDAS-TR-{building}-Mainfloor-GlobalMeter-activeEnergyImport"
PATCH_URL_TEMPLATE = "https://masterpiece.odins.es/ngsi-ld/v1/entities/urn:ngsi-ld:ExpectedThermalDemand:UEDAS-TR-{building}-Mainfloor-GlobalMeter-expectedThermalDemand/attrs/"
HEADERS = {
    "Content-Type": "application/json",
    "fiware-service": "masterpiece",
    "fiware-servicepath": "/",
    "x-auth-token": "{{AuthZToken}}"
}
BUILDINGS = [f"B{i}" for i in range(1, 14)]  # B1 to B13
LOOKBACK = 24
PREDICTION_HOURS = 24


def fetch_todays_data(building):
    """
    Fetch today's data for testing purposes or production.
    If there is an error, retry by stepping back one day at a time until successful.
    """
    attempt_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    while True:
        try:
            start_time = attempt_date.isoformat() + "Z"
            end_time = (attempt_date + timedelta(days=1)).isoformat() + "Z"

            entity_id = ENTITY_TEMPLATE.format(building=building)
            url = f"{BASE_URL}/{entity_id}/type/DeviceMeasurement/time/{start_time}/endTime/{end_time}/attrs/numValue"
            response = requests.get(url, headers=HEADERS)

            # Check if the request was successful
            if response.status_code == 200:
                r = response.json()
                data = [(item['observedAt'], item['value']) for item in r.get('numValue', [])]
                df = pd.DataFrame(data, columns=["Datetime", "Value"])
                df["Datetime"] = pd.to_datetime(df["Datetime"])
                df = df.set_index("Datetime").tz_localize(None)
                return df, attempt_date
            else:
                print(f"Failed to fetch data for {building} on {attempt_date.date()}. Trying previous day...")

        except Exception as e:
            print(f"Error fetching data for {building} on {attempt_date.date()}: {e}. Trying previous day...")

        # Move to the previous day
        attempt_date -= timedelta(days=1)


def fetch_yesterdays_data(building):
    """
    Fetch yesterday's data for imputation.
    """
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)
    start_time = yesterday.isoformat() + "Z"
    end_time = (yesterday + timedelta(days=1)).isoformat() + "Z"

    entity_id = ENTITY_TEMPLATE.format(building=building)
    url = f"{BASE_URL}/{entity_id}/type/DeviceMeasurement/time/{start_time}/endTime/{end_time}/attrs/numValue"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"Error {response.status_code} \nFailed to fetch data for {building}: {response.status_code}")
    r = response.json()
    data = [(item['observedAt'], item['value']) for item in r.get('numValue', [])]
    df = pd.DataFrame(data, columns=["Datetime", "Value"])
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime").tz_localize(None)
    return df


def time_features(df):
    """
    Create time series features based on datetime index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    return df


def cyclic_encoding(df):
    """
    Apply cyclic encoding to time-based features with fixed maximum values.
    """
    df = df.copy()
    # Define fixed maximum values for each time-based feature
    max_values = {
        'hour': 23,  # 24 hours in a day, indexed from 0 to 23
        'dayofweek': 6,  # 7 days in a week, indexed from 0 to 6
        'month': 12,  # 12 months in a year, indexed from 1 to 12
        'quarter': 4  # 4 quarters in a year, indexed from 1 to 4
    }

    for column, max_value in max_values.items():
        if column in df.columns:  # Ensure the column exists in the DataFrame
            df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / max_value)
            df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / max_value)

    return df


def heal_data(df, attempt_date, building):
    """
    Impute today's missing data with today's prediction.
    But first make sure that yesterday's data are ok
    """
    start_time = attempt_date
    end_time = attempt_date + timedelta(days=1)

    try:
        df_yest = fetch_yesterdays_data(building)
        full_index = pd.date_range(start=start_time - timedelta(days=1),
                                   end=end_time - timedelta(days=1) - timedelta(minutes=1), freq='15T')
        df_yest = df_yest.reindex(full_index)

        # Check for yesterday's missing data in 15-minute intervals
        df_yest['Value'] = df_yest['Value'].interpolate(method='linear').fillna(0)
        df_yest = df_yest.resample('1H').sum()

        df_yest = time_features(df_yest)
        df_yest_c = cyclic_encoding(df_yest)
        df_yest_c.drop(columns=df_yest.columns[1:], inplace=True)
        todays_predictions = predict_next_day(building, df_yest_c)

    except Exception as e:
        print(f"Error fetching yesterday's data for {building}: {e}")
        print("Skipping yesterday's data check and filling missing values with a basic method.")
        todays_predictions = None

    ##
    df_hourly = df.resample('1H').sum()
    full_index = pd.date_range(start=start_time, end=end_time - timedelta(minutes=1), freq='1H')
    df_hourly = df_hourly.reindex(full_index)

    if todays_predictions is not None:
        df_hourly['Value'] = [
            pred if np.isnan(val) else val
            for val, pred in zip(df_hourly['Value'], todays_predictions)
        ]

    else:
        df_hourly['Value'] = df_hourly['Value'].fillna(method="ffill").fillna(method="bfill")

    df_hourly = time_features(df_hourly)
    df_hourly_c = cyclic_encoding(df_hourly)
    df_hourly_c.drop(columns=df_hourly.columns[1:], inplace=True)

    print(f"{building} initial 15-minute data points: {len(df)}")
    print(f"{building} final hourly data points: {len(df_hourly)}\n")

    return df_hourly_c


def predict_next_day(building, df):
    """
    Predict the next day's energy consumption using the trained model.
    """
    clear_session()
    model_path = f"models/{building}_model.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {building} not found at {model_path}")

    model = load_model(model_path)

    # Prepare input for prediction
    data_values = df.values[-LOOKBACK:]
    data_values = data_values.reshape(1, LOOKBACK, data_values.shape[1])
    predictions = model.predict(data_values).flatten()
    predictions = np.where(predictions < 0, 0, predictions)
    clear_session()
    print(f"{building} predictions: {predictions}")
    # predictions = np.full(24,2)
    return predictions


def patch_prediction(building, predictions):
    """
    Patch the prediction data to the API.
    """
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)
    date_start = tomorrow.isoformat() + "Z"
    date_end = (tomorrow + timedelta(hours=23)).isoformat() + "Z"

    attributes = {
        "baselineLoad": predictions.tolist(),
        "dateStart": date_start,
        "dateEnd": date_end
    }
    payload = {prop: {"type": "Property", "value": val} for prop, val in attributes.items()}

    url = PATCH_URL_TEMPLATE.format(building=building)
    response = requests.patch(url, headers=HEADERS, json=payload)
    if response.status_code == 204:
        print(f"Successfully patched prediction for {building}.\n")
    else:
        print(f"Failed to patch prediction for {building}: {response.status_code}\n")


def save_predictions_to_csv(building, predictions, target_dir):
    date = datetime.utcnow() + timedelta(days=1)
    df_predictions = pd.DataFrame({
        'hour': [i for i in range(24)],
        'prediction': predictions
    })
    df_predictions['date'] = date.strftime('%Y-%m-%d')

    # always save next to the code file (absolute path)
    out_path = target_dir / f"predictions_{building}_{date.strftime('%Y-%m-%d')}.csv" #<--
    df_predictions.to_csv(out_path, index=False) #<--


def main():
    all_predictions = []
    next_day = (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"*********** RESULTS FOR {next_day} ***********\n")

    folder_name = f"predictions_for_{next_day}"
    folder_path = OUTPUT_DIR / folder_name #<--
    folder_path.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for building in BUILDINGS:
        print(f"Processing {building}...\n")

        try:
            
            df, valid_date = fetch_todays_data(building)            # 1. Fetch today's data
            df_healed = heal_data(df, valid_date, building)         # 2. Heal the data
            predictions = predict_next_day(building, df_healed)     # 3. Load the model and predict the next day
            patch_prediction(building, predictions)                 # 4. Patch the prediction

            save_predictions_to_csv(building, predictions, folder_path)

            with open(folder_path / f"log_{next_day}.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"{building} processed successfully on {valid_date}\n")

            all_predictions.append({
                'building': building,
                'predictions': predictions
            })

            print(f"{building} Ok! :)\n")

        except Exception as e:
            print(f"Error processing {building}: {e}\n")
            with open(folder_path / f"log_{next_day}.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"Error processing {building}: {e}\n")

    all_predictions_df = pd.DataFrame(all_predictions)
    all_predictions_df.to_csv(folder_path / f"all_predictions_{next_day}.csv", index=False)


# ======

def next_run_at(hhmm: str, tz: str) -> datetime:
    tzinfo = ZoneInfo(tz)
    now = datetime.now(tzinfo)
    hh, mm = map(int, hhmm.split(":"))
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target <= now:  # if time already passed today, schedule tomorrow
        target += timedelta(days=1)
    return target

if __name__ == "__main__":
    
    while True:
        run_at = next_run_at(RUN_AT, TIMEZONE)
        print(f"⏳ Waiting until {run_at.strftime('%Y-%m-%d %H:%M:%S %Z')} to run again...", flush=True)

        while True:
            now = datetime.now(ZoneInfo(TIMEZONE))
            remaining = (run_at - now).total_seconds()

            if remaining <= 0:
                break 

            sleep_time = min(remaining, 30)
            time.sleep(sleep_time)

        main()