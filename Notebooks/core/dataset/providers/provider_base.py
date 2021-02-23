from abc import ABC, abstractmethod
import database_interface as dbi


class DataProviderBase(ABC):

    @abstractmethod
    def _validate(self):
        pass

    @staticmethod
    def database_hook():
        return dbi.PgsqlInterface()

    @abstractmethod
    def __name__(self):
        return self.__class__.__name__
