from model_factory import ModelArch
from training_config import NeuralNetConfig, Config
from nn.neuralnet import NeuralNet
from customer.customer_base import CustomerBase
from model.ocr.model.ocr_models import OpticalCharacterRecognition


class Trainer:
    """
        An interface to model training
    """
    def __init__(self, model: ModelArch,
                 customer: CustomerBase,
                 training_config: Config):
        """

        @param model:
        @param customer: customer of the neural network
        @param training_config: used to instantiate a model and kick-off training
        """

        assert model
        assert isinstance(model, ModelArch)
        assert issubclass(customer, CustomerBase)

        self._model_type = model
        self._customer = customer
        self._net = NeuralNet(training_config, self._customer)
        if self._model_type == ModelArch.OCR:
            self._net = OpticalCharacterRecognition()

    def run(self, training_config):
        if self._model_type == ModelArch.EfficientNet:
            self._train(training_config)

    def _train(self, training_config: NeuralNetConfig):
        assert self._model_type == ModelArch.EfficientNet
        net = NeuralNet(training_config, self._customer)
        return net.train()

    def _train_legacy(self):
        """ Backward compatibility """
        pass
