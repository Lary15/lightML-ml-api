import schedule
import time
import os
import datetime
import requests
import pandas as pd

BANANA_API="http://192.168.0.43"
SENSORS_API="http://192.168.0.35"
LAMP_API="http://192.168.0.46"

CORRECT_PRED = 0
INCORRECT_PRED = 0

os.environ['TZ'] = 'Brazil/East'
time.tzset()

def get_min_max():
  df = pd.DataFrame(requests.get(f"{BANANA_API}:5000/peripherals", timeout=100).json())
  return df["temp"].min(), df["temp"].max(), df["hum"].min(), df["hum"].max(), df["mic"].min(), df["mic"].max()

def predict():
  global CORRECT_PRED, INCORRECT_PRED
  try:
    sensors_state = requests.get(f"{SENSORS_API}", timeout=10).json()
    sensors = sensors_state.copy()
    temp_min, temp_max, hum_min, hum_max, mic_min, mic_max = get_min_max()

    sensors["temp"] = (sensors["temp"]-min(temp_min, sensors["temp"]))/(max(sensors["temp"], temp_max)-min(temp_min, sensors["temp"]))
    sensors["hum"] = (sensors["hum"]-min(hum_min, sensors["hum"]))/(max(sensors["hum"], hum_max)-min(hum_min, sensors["hum"]))
    sensors["mic"] = (sensors["mic"]-min(mic_min, sensors["mic"]))/(max(sensors["mic"], mic_max)-min(mic_min, sensors["mic"]))

    sensors["timestamp"] = int(time.strftime('%H', time.localtime()))
    sensors["weekday"] = int(time.strftime('%w', time.localtime()))

    pred = requests.post(f"{BANANA_API}:3000/predict", json=[sensors]).json()

    print("[PREDICT] ", sensors, flush=True)
    print("[PREDICT] ", pred, flush=True)

    if pred[0][0] >= 0.5:
      requests.get(f"{LAMP_API}/cm?cmnd=Power%20ON", timeout=10)
      print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} [PREDICT] ', "LAMP ON", flush=True)
    else:
      requests.get(f"{LAMP_API}/cm?cmnd=Power%20OFF", timeout=10)
      print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} [PREDICT] ', "LAMP OFF", flush=True)
  
    # Check prediction
    time.sleep(5*60)
    res = requests.get(f"{LAMP_API}/cm?cmnd=Power", timeout=10).json()

    if res["POWER"] == "ON" and pred[0][0] >= 0.5 or \
      res["POWER"] == "OFF" and pred[0][0] < 0.5:
      CORRECT_PRED += 1
    else:
      # Save incorrect prediction
      INCORRECT_PRED += 1
      requests.post(f"{BANANA_API}:5000/peripherals", json=sensors_state, timeout=100)

    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} [PREDICT] ', f"CP: {CORRECT_PRED} IP: {INCORRECT_PRED}", flush=True)
  except Exception as e:
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} [PREDICT] ', e, flush=True)

def retrain():
  global CORRECT_PRED, INCORRECT_PRED
  try:
    start = int((datetime.datetime.today() - datetime.timedelta(days=3)).timestamp())
    end = int(datetime.datetime.today().timestamp())

    data = pd.DataFrame(requests.get(f"{BANANA_API}:5000/peripherals?start={start}&end={end}", timeout=100).json())
    
    if data.empty:
      print("[TRAIN] NO DATA", flush=True)
      return

    # Extract target
    target = data.pop("led_on")

    # Drop led color cols
    data.drop(columns=["led_color", "led_dimmer"], inplace=True)

    # Drop rows with negative values
    data = data[data["temp"]>=0]
    data = data[data["hum"]>=0]
    data = data[data["mic"]>=0]

    # Turn timestamp integer to category type based on hour
    data["weekday"] = data["timestamp"].apply(lambda x: int(time.strftime('%w', time.localtime(x))))
    data["timestamp"] = data["timestamp"].apply(lambda x: int(time.strftime('%H', time.localtime(x))))

    # Normalize data for necessary fields
    mins = {}
    maxs = {}
    mins["temp"], maxs["temp"], mins["hum"], maxs["hum"], mins["mic"], maxs["mic"] = get_min_max()

    for col in ["temp", "hum", "mic"]:
      data[col] = (data[col]-mins[col])/(maxs[col]-mins[col])

    res = requests.post(f"{BANANA_API}:3000/train", json={"x": data.to_dict('records'), "y": target.tolist(), "cp": CORRECT_PRED, "ip": INCORRECT_PRED})

    # Reset counters
    CORRECT_PRED = 0
    INCORRECT_PRED = 0

    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} [TRAIN] DONE - RES {res.status_code}', flush=True)
  except Exception as e:
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} [TRAIN] ', e, flush=True)

schedule.every(3).days.at("00:15").do(retrain)
schedule.every(30).minutes.do(predict)

while True:
  # Checks whether a scheduled task
  # is pending to run or not
  schedule.run_pending()
  time.sleep(10)