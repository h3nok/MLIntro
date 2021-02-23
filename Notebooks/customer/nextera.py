from typing import Optional

from customer.customer_base import CustomerBase
from core.datatype.geo import GeographicalLocation, WindFarm
from core.datatype.vinet_categories import Avian, viNetClassifications

import regex as re


class NextEra(CustomerBase):
    __geo_location = None
    __farms = None
    __vinet_classifications = []

    def __init__(self):
        """
        NextEra
        """
        self.__name = 'NextEra'
        self.__geo_location = GeographicalLocation(lat=37.745081, long=-121.657254)
        self.__farms = [WindFarm('Golden Hills', self.__geo_location)]
        classifications = [Avian('Golden-Eagle', '', False, True),
                           Avian('Bald-Eagle', '', False, True),
                           Avian('Raven', '', False, True),
                           Avian('Hawk', '', False, True),
                           Avian('Turkey-Vulture', '', False, True),
                           Avian('Young-Bald-Eagle', '', False, True),
                           Avian('Young-Golden-Eagle', '', False, True)]

        self.__vinet_classifications = viNetClassifications(classifications)

        self.__dataset_name = 'viNet_2.5.1_NA_v2'
        self.__dataset_tag = 'NA 5 Class v2.5 {} Final'
        self.__dataset_classification_group_name = 'NorthAmerica 5 Class'

    @property
    def windfarms(self) -> []:
        return self.__farms

    @property
    def name(self) -> str:
        return self.__name

    @property
    def colors(self):
        return 'coolwarm'

    @property
    def geo_location(self) -> GeographicalLocation:
        return self.__geo_location

    @property
    def protected_species(self) -> list:
        """
        An iterable list of protected species
        @return:
        """
        return self.__vinet_classifications.protected

    @property
    def deployed_network_classifications(self):
        """
        An iterable list of all species that are actively being protected at the sites.
        @return:
        """
        return self.__vinet_classifications.deployed_classifications

    @property
    def dataset_name(self) -> str:
        return self.__dataset_name

    @property
    def classification_group_name(self) -> str:
        return self.__dataset_classification_group_name

    @property
    def training_dataset_tag(self) -> str:
        return self.__dataset_tag.format('Training')

    @property
    def validation_dataset_tag(self) -> str:
        return self.__dataset_tag.format('Validation')

    def get_avian(self, name) -> Optional[Avian]:
        for classification in self.__vinet_classifications.all_classifications:
            if re.search(classification.name.replace('-', '.'), name, re.IGNORECASE):
                return classification

        return None

    def lookup(self):
        pass

