from logger import Logger as L


class viAi:
    """
    Decorators for code readability etc
    """

    @staticmethod
    def database_method(function):
        print(function.__name__)
        return function()

    @staticmethod
    def core_method(function):
        print(function.__name__)

    @staticmethod
    def cli_method(function):
        print(function.__name__)

    @staticmethod
    def database_object(cls):
        print(cls.__name__)

    @staticmethod
    def core_object(cls):
        print(cls.__name__)

    @staticmethod
    def cli_object(cls):
        print(cls.__name__)
