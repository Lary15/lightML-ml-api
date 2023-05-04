import time
import os
import shutil

import numpy as np
import tensorflow as tf

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

    if os.path.exists("execution_dnn"):
        if os.path.exists("execution_dnn_old"):
            shutil.rmtree("execution_dnn_old")
        os.rename("execution_dnn", "execution_dnn_old")

    history = model.fit(np.array(data_list), np.array(target_list), validation_split=0.1, epochs=5)
    tf.saved_model.save(model, "execution_dnn")

    return time.time() - start, history.history

def save_report(repo: TinyDBRepository, correct_predictions: int, incorrect_predictions: int, number_of_records: int, training_time: float, history: dict):
    report = MLTrainingReport()

    report.correct_predictions = correct_predictions
    report.incorrect_predictions = incorrect_predictions
    report.number_of_records = number_of_records
    report.training_time = int(training_time)
    report.timestamp = int(time.time())

    # Save average of all metrics
    for key, value in history.items():
        setattr(report, key, float(np.average(value)))

    repo.add(report)
