from unittest import TestCase

from dataset_tag import TagType, DatasetTag, DatasetTagMapper


class TestTagMapper(TestCase):
    type = TagType.Training
    tag = DatasetTag("viAi-Unittest-TagMapper-2", comment="TagMapper test", tag_type=type.value)

    def test_insert(self):
        with DatasetTagMapper(self.tag) as tm:
            print(tm.insert())
