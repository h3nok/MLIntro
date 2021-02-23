from unittest import TestCase
from deployment.inference import InferenceModel
from application.viNet import viNet


class TestviNet(TestCase):
    model_path = r"E:\viNet_RnD\Research\Candidates\GWA\B0\viNet_3.0_B0_done.ckpt_GWA.RnD.pb"
    class_map = r"E:\viNet_RnD\Research\Candidates\GWA\B0\viNet_3.0_B0_done.ckpt_GWA.RnD_ClassMap.txt"
    input_size = 224
    model = InferenceModel(model_path, class_map, input_size)
    candidate = viNet(model, 'GWA')

    def test_upload(self):
        self.fail()

    def test_deploy(self):
        self.fail()

    def test_evaluate(self):
        self.fail()

    def test_visualize(self):
        self.fail()

    def test_predict(self):
        self.fail()
