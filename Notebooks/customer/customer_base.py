from abc import ABC, abstractmethod
from application.applications import Application
from core.datatype.geo import GeographicalLocation, WindFarm


class CustomerBase(ABC):
    """
    Abstract base class for all customers of this application.
    For new customer:
    1. create file inside this folder with the name of the file matching the customer name.
    2. create a class with name of the class matching the customer name - ex. class Vattenfall
    3. have that class inherit from this base class.
    4. implement all abstract methods ( enforced)
    5. add customized methods if needed
    """

    @property
    @abstractmethod
    def name(self) -> str:
        return ""

    @property
    @abstractmethod
    def windfarms(self) -> [WindFarm]:
        """
        Return a list of windfarms (WindFarm objects) that the customer is operating.
        Can be farms with active system deployed or potential for future deployment
        @return:
        """
        return []

    @property
    @abstractmethod
    def geo_location(self) -> GeographicalLocation:
        """
        Returns the main geographical location for this customer - not necessarily each site.
        Geo location of each site can be retrieved from @windfarms
        @return:
        """
        return GeographicalLocation()

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        return ''

    @property
    @abstractmethod
    def classification_group_name(self) -> str:
        return ''

    @property
    @abstractmethod
    def training_dataset_tag(self) -> str:
        return ''

    @property
    @abstractmethod
    def validation_dataset_tag(self) -> str:
        return ''

    @property
    def application(self) -> Application:
        """
        Returns the type of application the customer is using.
        viNet by default for most
        @return:
        """
        return Application.viNet

    @abstractmethod
    def get_avian(self, name):
        """
        Will get the Avain with the name `name`.

        :param name: capitalization is ignored, and so it using `_`, `-`, or ` ` for spacers.
        :return: Avain Type
        """
        return None

    @abstractmethod
    def lookup(self):
        """
        A database routine to fetch information about the customer
        @return:
        """
        pass

    @property
    @abstractmethod
    def colors(self):
        pass

    @abstractmethod
    def default_query(self):
        return None
