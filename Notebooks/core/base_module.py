import abc


class BaseModule(abc.ABC):
    _module_name = None

    def __init__(self, name):
        self._module_name = name

    @abc.abstractmethod
    def __name__(self):
        return self._module_name
