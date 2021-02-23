from unittest import TestCase
from data_mappers.classification_group import viNetClassGroup, viNetClassGroupMapper
from tabulate import tabulate
from vinet_frame_label import viNetTrainingFrameLabel


class TestClassGroupMapper(TestCase):
    class_group = viNetClassGroup('Python-Test-Classification-Group', comment="Test Case")
    cgm = viNetClassGroupMapper(class_group)

    def test__get(self):
        groups = self.cgm.get_persisted_data()
        if not groups.empty:
            print(tabulate(groups, headers='keys', tablefmt='psql'))
        else:
            self.fail()

    def test_insert(self):
        if not isinstance(self.cgm.insert(), bool):
            self.fail()

    def test_update(self):
        # 1. Create classification group
        class_group = viNetClassGroup(name="E3 LSE 5 class-Test",
                                      comment="WTE, Buzzard, "
                                              "Other-Avian,"
                                              "Red-Kite, LSE")

        with viNetClassGroupMapper(class_group) as cgm:
            # 2. Create a list of classifications to be grouped together
            classifications = [viNetTrainingFrameLabel("Buzzard-sp", False),
                               viNetTrainingFrameLabel("Eurasian-Buzzard", False),
                               viNetTrainingFrameLabel("Buzzard", False)]

            # 3. A group class name for the above list
            mapped_to = viNetTrainingFrameLabel('Buzzard', False)

            # 4. Map frame groundtruth to class group names
            cgm.map_classifications(classifications, mapped_to)

