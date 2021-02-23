import uuid
from enum import Enum
from time import sleep

from common.db.stored_procedures import StoredProcedures as sp
from data_mapper_base import DatabaseObjectMapper, MappedDatabaseObject
from logger import Logger


class TagType(Enum):
    Training = 2
    Validation = 1
    Verification = 3  # on-site usually, ah-hoc metrics for now
    Unknown = 0


def _decode_tag_type(tag_type: TagType):
    if tag_type == TagType.Training:
        return "Training"
    elif tag_type == TagType.Validation:
        return "Validation"
    elif tag_type == TagType.Verification:
        return "Verification"
    else:
        return "Unknown"


def _encode_tag_type(tag_type_str):
    tag_type_str = tag_type_str.lower()
    if 'train' in tag_type_str:
        return TagType.Training
    elif 'val' in tag_type_str:
        return TagType.Validation
    elif 'ver' in tag_type_str:
        return TagType.Verification
    else:
        return TagType.Unknown


class DatasetTag(MappedDatabaseObject):
    """ A dataset tag
    """

    def __init__(self, name: str,
                 comment: str = None,
                 tag_type: str = 'train', tag_id: uuid = None):
        self.name = name
        self.comment = comment
        self.type = _encode_tag_type(tag_type)
        self.tag_id = tag_id

    def __str__(self):
        return f""" 
        DatasetTag: 
                   Name: \'{self.name}\'
                   Type: \'{_decode_tag_type(self.type)}\'
                   Comment: \'{self.comment}\'
                   Type: {self.type}
                   Id: {self.tag_id}"""

    def __eq__(self, other):
        if self.tag_id is not None and other.tag is not None:
            return self.tag_id == other.tag_id
        elif self.name == other.name and self.type == other.type:
            return True
        else:
            return False

    def set_id(self, tag_id):
        self.tag_id = tag_id


class DatasetTagMapper(DatabaseObjectMapper):
    """
    ADT that maps viNet's tags table entry to python Tag object
    """

    def __init__(self, tag: DatasetTag = None):
        self.tag = tag
        self.tags = []
        self.tag_ids = []
        self.connection = self.database_hook()
        self.server = None
        self.get_persisted_data()
        self._logger = Logger(module=self.__class__.__name__, console=True).configure()
        self.tag_id = None

    def __enter__(self):
        return self

    def _exists(self):
        if self.tag.name in self.tags:
            self._logger.warning("Dataset tag \'{}\' "
                                 "already exists in database. "
                                 "\'{}\'".format(self.tag.name, self.tag))
            self.tag_id = self.tags.index(self.tag.name.strip())
            return True

        return False

    def get_persisted_data(self):
        self.tags, self.tag_ids = sp.dataset_tags(self.connection)

    def insert(self, tag: DatasetTag = None):
        """
        Insert the tag to database, return tag_id
        """
        if not tag:
            assert self.tag
            tag = self.tag
        else:
            self.tag = tag
            self.get_persisted_data()

        if self._exists():
            self.tag_id = self.tag_ids[self.tag_id]
            return self.tag_id

        sp.add_new_tag(tag.name, tag.comment, tag.type.value, self.connection or self.database_hook())
        sleep(5)
        while not self._exists():
            self.get_persisted_data()

        self._logger.debug("Successfully inserted tag to "
                           "database, tag:{}".format(self.tag))

        self.tag_id = self.tag_ids[self.tag_id]

        return self.tag_id

    def __exit__(self, exc_type, exc_value, exc_info):
        self.connection.close()
