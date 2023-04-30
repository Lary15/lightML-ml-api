from flask_restful import Resource
from flask import Response, request
from repository.repository import AbstractRepository
from tensorflow.keras.models import Model
from api.services import predict, retrain, save_report

class Prediction(Resource):
  def __init__(self, repo: AbstractRepository, model: Model):
    self.repo = repo
    self.model = model

  def post(self) -> Response:
    return predict(self.model , request.get_json())


class Training(Resource):
  def __init__(self, repo: AbstractRepository, model: Model):
    self.repo = repo
    self.model = model

  def post(self) -> Response:
    params = request.get_json()
    training_time, history = retrain(self.model , params)
    save_report(self.repo, correct_predictions=params["cp"], incorrect_predictions=params["ip"], training_time=training_time, number_of_records=len(request.get_json()["x"]), history=history)
    return "OK"


class Reports(Resource):
  def __init__(self, repo: AbstractRepository):
    self.repo = repo

  def get(self) -> Response:
    return self.repo.list()
