from typing import Optional

from customer.customer_base import CustomerBase
from core.datatype.geo import GeographicalLocation, WindFarm
from core.datatype.vinet_categories import Avian, viNetClassifications

import regex as re


class E3(CustomerBase):
    __geo_location = None
    __farms = None
    __vinet_classifications = []

    def __init__(self):
        """
        E3
        TODO - populate available tags and configs corresponding to the customer from the db
        """
        self.__name = 'E3'
        self.__geo_location = GeographicalLocation(lat=53.344770, long=12.484574)
        self.__farms = [WindFarm('Butow', GeographicalLocation(lat=53.344770, long=12.484574)),
                        WindFarm('Gerbstedt', GeographicalLocation(lat=51.646800, long=11.648521)),
                        WindFarm('Lubesse', GeographicalLocation(lat=53.495585, long=11.451206)),
                        WindFarm('Zichow', GeographicalLocation(lat=53.173647, long=14.036385)),
                        WindFarm('Wilsickow', GeographicalLocation(lat=53.503770, long=13.840067)),
                        WindFarm('Buschmühlen', GeographicalLocation(lat=54.028314, long=11.633510))
                        ]

        # TODO: Did not include the REUBENKOGE windfarm as i could not find it on google maps
        classifications = [Avian('White-Tailed-Eagle', 'Haliaeetus albicilla', True, True),
                           Avian('Red-Or-Black-Kite', 'Milvus milvus or Milvus migrans', True, False),
                           Avian('Red-Kite', 'Milvus milvus', True, True),
                           Avian('Buzzard', 'Buteo buteo', False, True),
                           Avian('Other-Avian-DE', '', False, True),
                           Avian('Black-Kite', 'Milvus migrans', True, True),
                           Avian('Harrier', '', False, True)]

        self.__vinet_classifications = viNetClassifications(classifications)

        self.__dataset_name = "viNet_2.5.1_NA_v2"
        self.__latest_dataset_tag = 'E3_v2.2 {}'
        self.__latest_classification_group = 'E3 4 Class'

    @property
    def windfarms(self) -> []:
        return self.__farms

    @property
    def name(self) -> str:
        return self.__name

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
    def colors(self):
        return 'YlGn'

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
        return self.__latest_classification_group

    @property
    def training_dataset_tag(self) -> str:
        return self.__latest_dataset_tag.format('Training')

    @property
    def validation_dataset_tag(self) -> str:
        return self.__latest_dataset_tag.format('Validation')

    def get_avian(self, name) -> Optional[Avian]:
        for classification in self.__vinet_classifications.all_classifications:
            if re.search(classification.name.replace('-', '.'), name, re.IGNORECASE):
                return classification

        return None

    def default_query(self, benchmark=False, limit=None):

        if not limit:
            limit = ""
        else:
            limit = f"\nLIMIT {limit}"

        if benchmark:
            return """
            SELECT
                COUNT(*)
            FROM vinet.tagged_frames tag
                JOIN source.frame_data src ON tag.frame_id = src.frame_id
                JOIN vinet.network_classification_group_entry grp
       	            ON tag.truth_classification_id = grp.truth_classification_id
       	        JOIN source.classifications grp_class ON grp.group_classification_id = grp_class.id
                WHERE
       	            tag.tag_id = (SELECT tag_id FROM vinet.tags WHERE name = 'E3_v2.2 Training')
                AND
                    grp.classification_group_id = (
                    SELECT classification_group_id
      	            FROM vinet.network_classification_group
      	            WHERE name = 'E3 4 Class')
            """ + limit

        return """
            SELECT
                src.frame_id,
                grp_class.name as truth_classification,
                src.padded_image as image
            FROM vinet.tagged_frames tag
                JOIN source.frame_data src ON tag.frame_id = src.frame_id
                JOIN vinet.network_classification_group_entry grp
                    ON tag.truth_classification_id = grp.truth_classification_id
                JOIN source.classifications grp_class ON grp.group_classification_id = grp_class.id
                WHERE
                    tag.tag_id = (SELECT tag_id FROM vinet.tags WHERE name = 'E3_v2.2 Training')
                AND
                    grp.classification_group_id = (
                    SELECT classification_group_id
                    FROM vinet.network_classification_group
                    WHERE name = 'E3 4 Class')
            ORDER BY tag.frame_id
            """ + limit

    def lookup(self):
        pass
