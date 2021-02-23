class Avian:
    __protected = False
    __deployed = False

    def __init__(self, common_name: str, scientific_name: str, protected: bool, deployed: bool, vinet_version=''):
        """
        Abstract datatype that defines viNet avian classifications
        @type vinet_version:the version of viNet that leaned this species
        @param common_name: The main name used with  viNet
        @param scientific_name: the scientific name of the bird
        @param protected: whether or not its protected
        @param deployed:  whether or not it has been deployed with the active viNet
        """
        self.__vinet_version = vinet_version
        self.__name = common_name
        self.__scientific_name = scientific_name
        self.__protected = protected
        self.__deployed = deployed

    @property
    def name(self):
        return self.__name

    @property
    def scientific_name(self):
        return self.__scientific_name

    @property
    def protected(self):
        return self.__protected

    @property
    def deployed(self):
        return self.__deployed


class InanimateObject(object):
    __deployed = False

    def __init__(self, name, description, deployed: bool):
        self.__name = name
        self.__description = description
        self.__deployed = deployed

    @property
    def name(self):
        return self.__name

    @property
    def description(self):
        return self.__description

    @property
    def deployed(self):
        return self.__deployed


class viNetClassifications(object):
    def __init__(self, classifications):
        """
        The main interface to get viNet active classifications
        @param classifications:
        """
        self.__classifications = classifications

    @property
    def protected(self):
        return [x for x in self.__classifications if self._is_protected(x)]

    @property
    def deployed_classifications(self):
        return [x for x in self.__classifications if self._is_deployed(x)]

    @property
    def all_classifications(self):
        return [x for x in self.__classifications]

    @staticmethod
    def _is_protected(category):
        if isinstance(category, InanimateObject):
            return False

        return category.protected

    @staticmethod
    def _is_deployed(category):
        assert isinstance(category, Avian) or isinstance(category, InanimateObject)
        return category.deployed
