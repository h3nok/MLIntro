from typing import Optional

from customer.customer_base import CustomerBase
from core.datatype.geo import GeographicalLocation, WindFarm
from core.datatype.vinet_categories import Avian, viNetClassifications

import regex as re


class TOTW(CustomerBase):
    __geo_location = None
    __farms = None
    __vinet_classifications = []

    def __init__(self):
        """
        TOTW
        """
        self.__name = 'Duke'
        self.__geo_location = GeographicalLocation(lat=42.89779, long=-106.47371)
        self.__farms = [WindFarm('TOTW', self.__geo_location)]
        classifications = [Avian('Golden-Eagle', '', False, True),
                           Avian('Bald-Eagle', '', False, True),
                           Avian('Turbine-Blade', '', False, True),
                           Avian('Turbine-Antenna', '', False, True),
                           Avian('Turbine-Hub', '', False, True),
                           Avian('Raven', '', False, True),
                           Avian('Hawk', '', False, True),
                           Avian('Turbine-Other', '', False, True),
                           Avian('Calibration-Target', '', False, True),
                           Avian('Turkey-Vulture', '', False, True),
                           Avian('Young-Bald-Eagle', '', False, True),
                           Avian('Young-Golden-Eagle', '', False, True),
                           Avian('Other-Avian', '', False, True),
                           Avian('Turbine-General', '', False, True),
                           Avian('Turbine-Dark', '', False, True),
                           Avian('Turbine-Vent', '', False, True),
                           Avian('Condor', '', False, True)]

        self.__vinet_classifications = viNetClassifications(classifications)

        self.__dataset_name = 'viNet_TOTW_V_2.6_Training_v2_5_class_less_bald_aug'
        self.__dataset_tag = "NA 4 Class + Turbine v2.6.0 AllSites {}"
        self.__dataset_classification_group_name = 'NorthAmerica 4 Class + Turbines'

    @property
    def windfarms(self) -> []:
        return self.__farms

    @property
    def name(self) -> str:
        return self.__name

    @property
    def colors(self):
        return 'Blues'

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
