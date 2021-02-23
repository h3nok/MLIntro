from typing import Optional

from customer.customer_base import CustomerBase
from core.datatype.geo import GeographicalLocation, WindFarm
from core.datatype.vinet_categories import Avian, viNetClassifications

import regex as re


class GWA(CustomerBase):

    __geo_location = None
    __farms = None
    __vinet_classifications = []

    def __init__(self):
        """
        GWA
        """
        self.__name = 'GWA'
        self.__geo_location = GeographicalLocation(lat=-42.157474, long=146.675036)
        self.__farms = [WindFarm('Cattle Hill Wind Farm', self.__geo_location)]
        classifications = [Avian('Wedge-Tailed-Eagle', 'Aquila audax', True, True),
                           Avian('Hawk-Falcon', '', False, True),
                           Avian('Raven', 'Corvus corax', False, True),
                           Avian('Other-Avian', '', False, False)]
        self.__vinet_classifications = viNetClassifications(classifications)

        self.__dataset_name = 'viNet_2.8_Goldwind_3_Class_Clean'
        self.__dataset_tag = "Goldwind-Cattle Hill {} v1 (WTE, Raven,Other-Avian, Hawk-Falcon)"
        # "Goldwind-Cattle Hill {} v2 (WTE, Raven,Hawk-Falcon)"
        self.__dataset_classification_group_name = 'Goldwind Proper V2'

    @property
    def windfarms(self) -> []:
        return self.__farms

    @property
    def name(self) -> str:
        return self.__name

    @property
    def colors(self):
        return 'Wistia'

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

    def default_query(self, benchmark=False):

        if benchmark:
            return """
                  select
                        COUNT(*) 
                  from vinet.tagged_frames tf
                        join "source".frame_data fd on fd.frame_id = tf.frame_id
                        join viclassify.frame_results_latest frl on frl.frame_id = tf.frame_id
                        join vinet.network_classification_group_entry grp ON 
                            grp.truth_classification_id = frl.classification_id
                        join source.classifications grp_class ON grp.group_classification_id = grp_class.id
                            where tf.tag_id= (select tag_id from vinet.tags where name='Goldwind-Cattle 
                        Hill Training v1 (WTE, Raven,Other-Avian, Hawk-Falcon)') and
                            grp.classification_group_id = (SELECT classification_group_id FROM 
                            vinet.network_classification_group
                  WHERE name = 'Goldwind Proper V2')
                  """

        return """
        select
            tf.frame_id as frame_id,
            grp_class.name as groundtruth,
            fd.boundingbox_image as frame_data
        from vinet.tagged_frames tf
            join "source".frame_data fd on fd.frame_id = tf.frame_id
            join viclassify.frame_results_latest frl on frl.frame_id = tf.frame_id
            join vinet.network_classification_group_entry grp ON grp.truth_classification_id = frl.classification_id
            join source.classifications grp_class ON grp.group_classification_id = grp_class.id
            where tf.tag_id= (select tag_id from vinet.tags where name='Goldwind-Cattle 
            Hill Training v1 (WTE, Raven,Other-Avian, Hawk-Falcon)') and
            grp.classification_group_id = (SELECT classification_group_id FROM vinet.network_classification_group
        WHERE name = 'Goldwind Proper V2')
        """
