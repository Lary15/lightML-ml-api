import schedule
import time
import os
import datetime
import requests
import pandas as pd
import threading

BANANA_API="http://192.168.0.43"
SENSORS_API="http://192.168.0.35"
LAMP_API="http://192.168.0.46"

CORRECT_PRED = 0
INCORRECT_PRED = 0

os.environ['TZ'] = 'Brazil/East'
time.tzset()

def retrain():
  global CORRECT_PRED, INCORRECT_PRED
  try:
    start = int((datetime.datetime.today() - datetime.timedelta(days=1)).timestamp())
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

    res = requests.post(f"{BANANA_API}:3000/train", json={"x": data.to_dict('records'), "y": target.tolist(), "cp": CORRECT_PRED, "ip": INCORRECT_PRED})

    # Reset counters
    CORRECT_PRED = 0
    INCORRECT_PRED = 0

    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} [TRAIN] DONE - RES {res.status_code}', flush=True)
  except Exception as e:
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} [TRAIN] ', e, flush=True)

schedule.every(1).days.at("00:15").do(retrain)

while True:
  # Checks whether a scheduled task
  # is pending to run or not
  schedule.run_pending()
  time.sleep(10)