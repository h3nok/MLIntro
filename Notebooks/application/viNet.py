import os

from common.io.database import evaluate_candidate_formal, upload_model_to_db
from common.nn_utils import visualize_graph
from core import base_module
from logs import logger as L
from core.analytics.performance import ClassifierPerformanceReport
from core.dataset.providers.reportdata_provider import viNetResultDataProvider
from core.deployment.inference import InferenceModel


class viNet(base_module.BaseModule):
    _customer = None
    _model_path = None
    _description = None
    _module_name = 'viNet'
    _is_candidate = True

    def __init__(self, im, customer, comments=""):
        """
        An interface for the final candidate model.
        @param im:
        @param customer:
        @param comments:
        """
        super().__init__(name=self.__name__)
        assert im
        assert isinstance(im, InferenceModel), "Supplied model is not of type InferenceModel"
        self._inference_model = im
        self._customer = customer
        self._comments = comments
        self._model_path = im.pb
        self._input_size = im.input_size
        self._class_map = im.classmap
        self._config = os.path.basename(self._model_path).replace('.pb', '')
        self._logger = L.Logger(module=self.__name__, console=True).configure()
        self._logger.info("Successfully initialized viNet, args: %s, %s, "
                          "%s " % (self._customer, self._description, self._model_path))
        self._verification_tag = None

    @property
    def __name__(self):
        return self._module_name

    @property
    def config(self):
        return self._config

    def upload(self) -> None:
        """
        Upload model to database

        @return:
        """
        self._config = upload_model_to_db(self._model_path,
                                          self._class_map,
                                          self._input_size[0])

    def deploy(self):
        pass

    def verify(self, tag) -> bool:
        """
        Runs a candidate model on a formal validation tag

        @param tag:
        @return:
        """
        assert tag
        self._verification_tag = tag
        return evaluate_candidate_formal(self.config, self._verification_tag)

    def visualize(self) -> None:
        visualize_graph(self._model_path)

    def predict(self, frame):
        return self._inference_model.predict(input_frame=frame)

    @staticmethod
    def app() -> str:
        return 'viNet: Vision Inspector Neural Network'

    def rename(self, candidate_index=1) -> None:
        """

        @param candidate_index:
        @return:
        """
        new_path = os.path.splitext(self._model_path)[0] + f".Candidate.{candidate_index}.pb"
        if os.path.exists(new_path):
            os.remove(new_path)
        os.rename(self._model_path, new_path)

        self._model_path = new_path

    def generate_report(self, classification_group, output_dir, tag, limit=None):
        """

        @param classification_group:
        @param output_dir:
        @param tag:
        @param limit:
        @return:
        """
        reporter = ClassifierPerformanceReport
        assert os.path.exists(output_dir)
        filename = os.path.join(output_dir, f"{self._config}.pdf")
        if tag:
            self._verification_tag = tag
        with viNetResultDataProvider(self._verification_tag,
                                     self._config,
                                     classification_group=classification_group,
                                     include_images=True) as dp:
            if limit.lower() != 'all':
                dp.set_limit(int(limit))
            dp.fetch_results()
            dp.map_results()
            vinet_datum = dp.dataframe

            vinet_datum.vi.set_attributes(customer=self._customer)
            reporter.create_complete_report(vinet_datum,
                                            save_loc=filename,
                                            appendix=False,
                                            colors=self._customer.colors)

            print(f"\nReport saved to: {filename}")
