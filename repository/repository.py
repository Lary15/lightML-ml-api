import abc

class AbstractRepository(abc.ABC):
  @abc.abstractmethod
  def add(self, model):
    raise NotImplementedError
  
  @abc.abstractmethod
  def get(self, reference) -> dict:
      raise NotImplementedError

  @abc.abstractmethod
  def list(self) -> dict:
      raise NotImplementedError

  @abc.abstractmethod
  def query_by_epoch(self, start: int, end: int) -> dict:
      raise NotImplementedError