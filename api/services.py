import time

import numpy as np

from model.ml_training_report import MLTrainingReport
from repository.tiny_db import TinyDBRepository
from tensorflow.keras.models import Model
from typing import List

def predict(model: Model, data: List[dict]):
    data_list = [list(dictionary.values()) for dictionary in data]
    return model.predict(np.array(data_list)).tolist()

def retrain(model: Model, data: List[dict]):
    start = time.time()

    data_list = [list(dictionary.values()) for dictionary in data["x"]]
    target_list = data["y"]
    
    history = model.fit(np.array(data_list), np.array(target_list), validation_split=0.15, epochs=200)
    model.save("execution_dnn")

    return time.time() - start, history.history

def save_report(repo: TinyDBRepository, number_of_records: int, training_time: float, history: dict):
    report = MLTrainingReport()

    report.number_of_records = number_of_records
    report.training_time = int(training_time)
    report.timestamp = int(time.time())

    # Save average of all metrics
    for key, value in history.items():
        setattr(report, key, float(np.average(value)))

    repo.add(report)
