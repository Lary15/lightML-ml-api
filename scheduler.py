import schedule
import time
import datetime
import requests
import pandas as pd

BANANA_API="http://192.168.0.16"
SENSORS_API="http://192.168.0.35"
LAMP_API="http://192.168.0.46"

def get_min_max():
  df = pd.DataFrame(requests.get(f"{BANANA_API}:5000/peripherals", timeout=100).json())
  return df["temp"].min(), df["temp"].max(), df["hum"].min(), df["hum"].max(), df["mic"].min(), df["mic"].max()

def predict():
  try:
    sensors = requests.get(f"{SENSORS_API}", timeout=10).json()
    temp_min, temp_max, hum_min, hum_max, mic_min, mic_max = get_min_max()

    sensors["temp"] = (sensors["temp"]-temp_min)/(max(sensors["temp"], temp_max)-temp_min)
    sensors["hum"] = (sensors["hum"]-hum_min)/(max(sensors["hum"], hum_max)-hum_min)
    sensors["mic"] = (sensors["mic"]-mic_min)/(max(sensors["mic"], mic_max)-mic_min)

    sensors["timestamp"] = int(time.strftime('%H', time.localtime()))

    pred = requests.post(f"{BANANA_API}:3000/predict", json=[sensors]).json()

    print("[PREDICT] ", pred)

    if pred[0][0] >= 0.5:
      requests.get(f"{LAMP_API}/cm?cmnd=Power%20ON", timeout=10)
      print("[PREDICT] ", "LAMP ON")
    else:
      requests.get(f"{LAMP_API}/cm?cmnd=Power%20OFF", timeout=10)
      print("[PREDICT] ", "LAMP OFF")
  except Exception as e:
    print("[PREDICT] ", e)

def retrain():
  try:
    start = int((datetime.datetime.today() - datetime.timedelta(days=1)).timestamp())
    end = int(datetime.datetime.today().timestamp())

    data = pd.DataFrame(requests.get(f"{BANANA_API}:5000/peripherals?start={start}&end={end}", timeout=100).json())
    
    if data.empty:
      print("[TRAIN] NO DATA")
      return

    # Extract target
    target = data.pop("led_on")

    # Drop led color cols
    data.drop(columns=["led_color", "led_dimmer"], inplace=True)

    # Turn timestamp integer to category type based on hour
    data["timestamp"] = data["timestamp"].apply(lambda x: int(time.strftime('%H', time.localtime(x))))
    data["timestamp"] = data["timestamp"].astype('category')

    # Normalize data for necessary fields
    for col in ["temp", "hum", "mic"]:
      data[col] = (data[col]-data[col].min())/(data[col].max()-data[col].min())

    res = requests.post(f"{BANANA_API}:3000/train", json={"x": data.to_dict('records'), "y": target.tolist()})
    print(f"[TRAIN] DONE - RES {res.status_code}")
  except Exception as e:
    print('[TRAIN] ', e)


schedule.every().day.at("00:00").do(retrain)
schedule.every().hour.do(predict)

while True:
  # Checks whether a scheduled task
  # is pending to run or not
  schedule.run_pending()
  time.sleep(10)