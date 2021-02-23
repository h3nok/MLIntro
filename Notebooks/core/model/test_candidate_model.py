from candidate_model import CandidateModel


class TestCandidateModel:
    def test_customer(self):
        cm = CandidateModel('test1', 'test2')
        assert cm.customer == 'test2'

    def test_config(self):
        cm = CandidateModel('test1', 'test2')
        assert cm.config == 'test1'

    def test_evaluate(self):
        cm = CandidateModel('viNet_2.7_Vattenfall_4_class_6.3m', 'Vattenfall-Gotland')
        cm.evaluate('Vattenfall-Gotland Training V2 (4 class) - Validation')

    def test_upload(self):
        assert False

    def test_plot(self):
        cm = CandidateModel(r'C:\ProgramData\viNet\Deployment\Inference Models\Inception\inception_v2_6_class.pb')
        cm._classified_frames_distribution(save_as="Test_plot.png", show=True)
