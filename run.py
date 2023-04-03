import os

from api.endpoints import Prediction, Training, Reports
from flask import Flask
from flask_restful import Api
from tinydb import TinyDB
from repository.tiny_db import TinyDBRepository
import tensorflow as tf

#Setup Repository
db = TinyDB("db.json")
repo = TinyDBRepository(db)

# Setup model interpreter
if os.path.exists("execution_dnn"):
  model = tf.keras.models.load_model("execution_dnn")
else:
  model = tf.keras.models.load_model("dnn")

# Setup API
app = Flask(__name__)
api = Api(app)

api.add_resource(Prediction, "/predict", resource_class_kwargs={'repo': repo, 'model': model })
api.add_resource(Training, "/train", resource_class_kwargs={'repo': repo, 'model': model })
api.add_resource(Reports, "/reports", resource_class_kwargs={'repo': repo })

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=3000)