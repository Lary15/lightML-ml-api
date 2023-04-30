import schedule
import time
import os
import requests
import pandas as pd

BANANA_API="http://192.168.0.43"
SENSORS_API="http://192.168.0.35"
LAMP_API="http://192.168.0.46"

os.environ['TZ'] = 'Brazil/East'
time.tzset()

def save_data():
  try:
    sensors = requests.get(f"{SENSORS_API}", timeout=10).json()
    requests.post(f"{BANANA_API}:5000/peripherals", json=sensors, timeout=100)
  except Exception as e:
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} [SAVE] ', e, flush=True)

schedule.every(3).minutes.do(save_data)

while True:
  # Checks whether a scheduled task
  # is pending to run or not
  schedule.run_pending()
  time.sleep(10)