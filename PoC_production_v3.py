import os, time
import pandas as pd
import numpy as np
import requests
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from pathlib import Path #<--


poc_entities = {
    "Pleiades Machinery Room": [
        "urn:ngsi-ld:DeviceMeasurement:UMU-POC-Pleiades-MachineryRoom-GlobalMeter-activeEnergyImport"
    ],
    "Chemistry Faculty": [
        "urn:ngsi-ld:DeviceMeasurement:UMU-POC-ChemistryFaculty-MachineryRoom-GlobalMeter-activeEnergyImport"
    ],
    "Veterinary Faculty": [
        "urn:ngsi-ld:DeviceMeasurement:UMU-POC-VetFaculty-MachineryRoom-GlobalMeter-activeEnergyImport"
    ],
    "Work Sciences Faculty": [
        "urn:ngsi-ld:DeviceMeasurement:UMU-POC-WorkSciencesFaculty-MachineryRoom-GlobalMeter-activeEnergyImport"
    ],
    "Computer Science Faculty": [
        "urn:ngsi-ld:DeviceMeasurement:UMU-POC-ComputerScienceFaculty-MachineryRoom-GlobalMeter-activeEnergyImport"
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
    ],
    "North Lecturing Building": [
        "urn:ngsi-ld:DeviceMeasurement:UMU-POC-NorthLecturingBuilding-MachineryRoom-GlobalMeter-LineaAlimentacionGeneral-activeEnergyImport"
    ]
}

RUN_AT = os.getenv("RUN_AT", "23:20")
TIMEZONE = os.getenv("TIMEZONE", "Europe/Athens") 

BASE_DIR = Path(__file__).resolve().parent #<--
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


BASE_URL = "https://masterpiece.odins.es:443/temporal/entities"
ENTITY_TEMPLATE = "urn:ngsi-ld:DeviceMeasurement:UMU-POC-{building}-activeEnergyImport"
PATCH_URL_TEMPLATE = "https://masterpiece.odins.es/ngsi-ld/v1/entities/urn:ngsi-ld:ExpectedThermalDemand:UMU-POC-{building}-expectedThermalDemand/attrs/"
HEADERS = {
    "Content-Type": "application/json",
    "fiware-service": "masterpiece",
    "fiware-servicepath": "/",
    "x-auth-token": "{{AuthZToken}}"
}
BUILDINGS = ["Pleiades-MachineryRoom-GlobalMeter",
            "ChemistryFaculty-MachineryRoom-GlobalMeter", 
            "VetFaculty-MachineryRoom-GlobalMeter",
            "WorkSciencesFaculty-MachineryRoom-GlobalMeter",
            "ComputerScienceFaculty-MachineryRoom-GlobalMeter",
            "PsicologyFaculty-MachineryRoom-GlobalMeter",
            "MathematicsFacultyandGeneralLecturingBuilding-MachineryRoom-GlobalMeter-1",
            "MathematicsFacultyandGeneralLecturingBuilding-MachineryRoom-GlobalMeter-2",
            "GinerRiosLecturingBuilding-MachineryRoom-GlobalMeter",
            "NorthLecturingBuilding-MachineryRoom-GlobalMeter-LineaAlimentacionGeneral"]

MATH = ["MathematicsFacultyandGeneralLecturingBuilding-MachineryRoom-GlobalMeter-1", "MathematicsFacultyandGeneralLecturingBuilding-MachineryRoom-GlobalMeter-2"]

LOOKBACK = 24
PREDICTION_HOURS = 24

my_models = {
    "Pleiades-MachineryRoom-GlobalMeter": "Machinery",
    "ChemistryFaculty-MachineryRoom-GlobalMeter": "Chemistry",
    "VetFaculty-MachineryRoom-GlobalMeter": "Veterinary",
    "WorkSciencesFaculty-MachineryRoom-GlobalMeter": "Work Sciences",
    "ComputerScienceFaculty-MachineryRoom-GlobalMeter": "Computer Sciences",
    "PsicologyFaculty-MachineryRoom-GlobalMeter": "Psychology",
    "MathematicsFacultyandGeneralLecturingBuilding-MachineryRoom-GlobalMeter-1": "Mathematics 1",
    "MathematicsFacultyandGeneralLecturingBuilding-MachineryRoom-GlobalMeter-2": "Mathematics 2",
    "GinerRiosLecturingBuilding-MachineryRoom-GlobalMeter": "Giner Rios",
    "NorthLecturingBuilding-MachineryRoom-GlobalMeter-LineaAlimentacionGeneral": "North Lecturing"
}


# In[4]:


def fetch_todays_data(building):
    """
    Fetch today's data for testing purposes or production.
    If there is an error, retry by stepping back one day at a time until successful.
    """
    attempt_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    while True:
        try:
            start_time = (attempt_date - timedelta(days=13)).isoformat() + "Z"
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


# In[5]:


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


# In[6]:


def time_features_cyclical(df):
    df = df.copy()

    for lag in [1, 6 * 24, 13 * 24]:
        df[f'lag_{lag}'] = df['Value'].shift(lag)
    df.dropna(inplace=True)

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


# In[7]:


def heal_data(df, attempt_date, building):
    """
    Impute today's missing data with today's prediction.
    But first make sure that yesterday's data are ok
    """
    start_time = attempt_date - timedelta(days=13)
    end_time = attempt_date + timedelta(days=1)

    #     try:
    #         df_yest = fetch_yesterdays_data(building)
    #         full_index = pd.date_range(start=start_time - timedelta(days=1), end=end_time - timedelta(days=1) - timedelta(minutes=1), freq='15T')
    #         df_yest = df_yest.reindex(full_index)

    #         df_yest['Value'] = df_yest['Value'].interpolate(method='linear').fillna(0)
    #         df_yest = df_yest.resample('1H').sum()

    #         df_yest_c = time_features_cyclical(df_yest)
    #         print(df_yest_c)
    #         todays_predictions = predict_next_day(building, df_yest_c)
    #         print(f"here 7")

    #     except Exception as e:
    #         print(f"Error fetching yesterday's data for {building}: {e}")
    #         print("Skipping yesterday's data check and filling missing values with a basic method.")
    #         todays_predictions = None

    ##
    todays_predictions = None
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

    df_hourly_c = time_features_cyclical(df_hourly)

    print(f"{building} initial 15-minute data points: {len(df)}")
    print(f"{building} final hourly data points: {len(df_hourly)}\n")

    return df_hourly_c


# In[8]:


def predict_next_day(building, df):
    """
    Predict the next day's energy consumption using the trained model.
    """
    clear_session()
    model_path = f"models/{my_models[building]}_model.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {my_models[building]} not found at {model_path}")

#     q_value = quantile_values[my_models[building]]

    #     custom_objects = {"quantile_loss": quantile_loss}  # Use registered global function
    #     model = load_model(model_path, custom_objects=custom_objects)
    #     model = load_model(model_path, custom_objects={ 'loss': quantile_loss(q_value) })
    model = load_model(model_path, compile=False)

    # Prepare input for prediction
    data_values = df.values[-LOOKBACK:]
    data_values = data_values.reshape(1, LOOKBACK, data_values.shape[1])
    predictions = model.predict(data_values).flatten()
    predictions = np.where(predictions < 0, 0, predictions)
    clear_session()
    print(f"{building} predictions: {predictions}")
    return predictions


# In[9]:


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


# In[10]:


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


# In[11]:


def main():
    all_predictions = []
    math = 0

    next_day = (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')

    folder_name = f"predictions_for_{next_day}"
    folder_path = OUTPUT_DIR / folder_name #<--
    folder_path.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for building in BUILDINGS:
        print(f"\n------ Processing {building} ------\n")

        try:
            df, valid_date = fetch_todays_data(building)
            df_healed = heal_data(df, valid_date, building)
            predictions = predict_next_day(building, df_healed)
            patch_prediction(building, predictions)

            save_predictions_to_csv(building, predictions, folder_path)

            with open(folder_path / f"log_{next_day}.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"{building} processed successfully on {valid_date}\n")

            all_predictions.append({
                'building': building,
                'predictions': predictions
            })
        
            if building in MATH:
                math += predictions
                if building == "MathematicsFacultyandGeneralLecturingBuilding-MachineryRoom-GlobalMeter-2":
                    patch_prediction("MathematicsFacultyandGeneralLecturingBuilding-MachineryRoom-GlobalMeter", math)
        
            if building == "NorthLecturingBuilding-MachineryRoom-GlobalMeter-LineaAlimentacionGeneral":
                patch_prediction("NorthLecturingBuilding-MachineryRoom-GlobalMeter", predictions)
            
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