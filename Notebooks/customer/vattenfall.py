from typing import Optional

from customer.customer_base import CustomerBase
from core.datatype.geo import GeographicalLocation, WindFarm
from core.datatype.vinet_categories import Avian, viNetClassifications

import regex as re


class Vattenfall(CustomerBase):
    __name = None
    __geo_location = None
    __farms = None
    __vinet_classifications = []
    __dataset_name = None
    __dataset_tag = None
    __dataset_classification_group_name = None

    def __init__(self):
        """
        Vattenfall
        TODO - henok - set vinet_version of each classification
        """
        self.__name = 'Vattenfall'
        self.__geo_location = GeographicalLocation(lat=57.105306, long=18.210556)  # I think these are flipped
        self.__farms = [WindFarm('Gotland-Wind', self.__geo_location)]
        classifications = [Avian('Golden-Eagle', 'Aquila chrysaetos', False, True),
                           Avian('White-Tailed-Eagle', 'Haliaeetus albicilla', True, True),
                           Avian('Red-Or-Black-Kite', 'Milvus milvus or Milvus migrans', False, False),
                           Avian('Raven', 'Corvus corax', False, False),
                           Avian('Buzzard', 'Buteo buteo', False, True),
                           Avian('Other-Avian-Gotland', '', False, True),
                           Avian('Eagle-Or-Kite', '', False, True),
                           Avian('Gull', '', False, True)]

        self.__vinet_classifications = viNetClassifications(classifications)

        self.__dataset_name = 'viNet_2.7_Vattenfall_Proper_4_class_v2'
        self.__dataset_tag = 'Vattenfall-Gotland Training V2 (4 class) - {}'
        self.__dataset_classification_group_name = 'Vattenfall Proper 4 Class'

    @property
    def windfarms(self) -> []:
        return self.__farms

    @property
    def name(self) -> str:
        return self.__name

    @property
    def colors(self):
        return 'twilight'

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

    def default_query(self, limit=None):
        return """
        select
            tf.frame_id,
            grp_class.name,
            fd.boundingbox_image as frame_data

        from vinet.tagged_frames tf
            join "source".frame_data fd on fd.frame_id = tf.frame_id
            join viclassify.frame_results_latest frl on frl.frame_id = tf.frame_id
            join vinet.network_classification_group_entry grp ON grp.truth_classification_id = frl.classification_id
            join source.classifications grp_class ON grp.group_classification_id = grp_class.id
	        where tf.tag_id= (select tag_id from vinet.tags where name='Vattenfall V3 Training')
        and grp.classification_group_id = (SELECT classification_group_id FROM vinet.network_classification_group 
        WHERE name = 'Vattenfall Proper v3')
        order by tf.frame_id
        """
