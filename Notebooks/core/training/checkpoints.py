import os
from customer.customers import CustomerObjectMap
from core.training.training_system import TRAINING_DIR, DEPLOYMENT_DIR
import shutil


class Checkpoints:
    def __init__(self, customer=None, vinet_version='2.9', dataset_version='4',
                 checkpoint_version_tf='v2'):
        assert customer is not None
        self._customer = customer
        self._net_version = vinet_version
        self._dataset_version = dataset_version
        training_output = None
        for output in os.listdir(TRAINING_DIR):
            if self._customer in output:
                training_output = output
        self._path = os.path.join(TRAINING_DIR, training_output)
        assert os.path.exists(self._path)
        self._iteration = None
        self._checkpoints = None
        self._copy_to_local()

    @property
    def path(self):
        return self._path

    @property
    def latest_checkpoint(self):
        return self._iteration

    @property
    def iteration(self):
        return self._iteration.split('-')[1]

    def _copy_to_local(self, output_dir=DEPLOYMENT_DIR):
        self._checkpoints = self._latest()
        output_dir = os.path.join(output_dir, self._customer)
        assert os.path.exists(output_dir)
        output_dir = os.path.join(output_dir, str(self._net_version),
                                  str(self._dataset_version),
                                  self._iteration.split('-')[1])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for ckpt in self._checkpoints:
            if os.path.isfile(ckpt):
                shutil.copy(ckpt, output_dir)

        assert len(self._checkpoints) == len(os.listdir(output_dir)), "\'{}\'".format(output_dir)
        self._path = output_dir

    def _latest(self):
        training_outputs = os.listdir(self._path)
        assert len(training_outputs) > 0
        training_outputs = [os.path.join(self._path, d) for d in os.listdir(self._path)]
        latest_checkpoints = max(training_outputs, key=os.path.getmtime)
        checkpoints = [os.path.join(latest_checkpoints, d) for d in
                       os.listdir(latest_checkpoints) if 'events' not in d]
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)

        self._iteration = os.path.splitext(os.path.basename(latest_checkpoint))[0]
        # self._iteration = "".join()

        return [ckpt for ckpt in checkpoints if self._iteration in ckpt]

