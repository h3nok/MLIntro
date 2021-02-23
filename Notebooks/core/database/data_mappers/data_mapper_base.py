from abc import ABC, abstractmethod
import database_interface as dbi
from decorators import viAi


class DatabaseObjectMapper(ABC):
    """ DataMapper pattern - a layer of mappers that move data between objects and database
        while keeping them independent of each other and the mapper itself.
    """

    @abstractmethod
    def get_persisted_data(self):
        pass

    @abstractmethod
    def insert(self):
        pass

    @staticmethod
    def database_hook():
        server = dbi.PgsqlInterface()
        server.connect(autocommit=True)

        return server.connection


class MappedDatabaseObject(ABC):
    """ Python object representing a database table entry (usually)"""

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

