from unittest import TestCase
from vinet_frame_label import viNetTrainingFrameLabel, viNetTrainingFrameLabelMapper


class TestviNetTrainingFrameLabelAndMapper(TestCase):
    label = viNetTrainingFrameLabel('Unknown', protected=False, class_id=1)
    lm = viNetTrainingFrameLabelMapper(label)

    def test__exists(self):
        if not self.lm._exists():
            self.fail()
        else:
            print(self.label, "exists")

    def test__str__(self):
        print(self.label)

    def test__eq__(self):
        assert self.label == self.label

    def test_get_persisted_labels(self):
        for label in self.lm.get_persisted_data():
            print(label)

    def test_insert(self):
        assert isinstance(self.lm.insert(), bool)
