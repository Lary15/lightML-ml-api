from repository.repository import AbstractRepository
from dataclasses import asdict
from tinydb import Query

class TinyDBRepository(AbstractRepository):
  def __init__(self, db):
    self.db = db

  def add(self, model):
    self.db.insert(asdict(model))

  def get(self, reference):
    Report = Query()
    return self.db.search(Report.timestamp == reference)

  def list(self):
    return self.db.all()

  def query_by_epoch(self, start: int, end: int) -> dict:
    Report = Query()

    is_in_interval = lambda epoch: start <= epoch <= end
    return self.db.search(Report.timestamp.test(is_in_interval))