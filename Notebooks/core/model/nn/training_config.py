import configparser
import os

from graph_functions.classification_model_functions import loss_function, metrics, optimizer, distributed_strategy


class Config:
    _file_path = None

    __config = None
    __nets = dict()
    __dataset = dict()

    # Default values
    _train_dataset_dir = 'datatype'
    _validation_dataset_dir = 'None'

    def __init__(self, file_path):
        """
        Creates viNetConfig instance from the data in file_path.
        Note, file_path does not have to exist. Any data missing from the file will be filled in with defaults.

        Use `dump_data` to update the `file_path` with the missing data with the defaults.

        :param file_path:
        """
        self._file_path = file_path
        self.__config = configparser.ConfigParser()

        if os.path.exists(self._file_path):
            self.__config.read(self._file_path)

            for entry_name in self.__config.sections():
                if entry_name == 'Dataset':
                    dataset_config_data = self.__config[entry_name]
                    self.__dataset = dict()

                    for elem in list(dataset_config_data):
                        self.__dataset[elem] = dataset_config_data[elem]
                else:
                    net_config_data = self.__config[entry_name]
                    net_data = dict()

                    for elem in list(net_config_data):
                        net_data[elem] = net_config_data[elem]

                    self.__nets[entry_name] = net_data

        if 'train_dataset_dir' in self.__dataset:
            assert os.path.exists(self.__dataset['train_dataset_dir']), f"{self.__dataset['train_dataset_dir']} does " \
                                                                        f"not exist "
            self._train_dataset_dir = self.__dataset['train_dataset_dir']

        if 'validation_dataset_dir' in self.__dataset:
            self._validation_dataset_dir = self.__dataset['validation_dataset_dir']

    def get_net_data(self, net_name):
        if net_name in self.__nets:
            return self.__nets[net_name]
        else:
            return dict()

    @property
    def file_path(self):
        return self._file_path

    @property
    def validation_dataset_dir(self):
        if os.path.exists(self._validation_dataset_dir):
            return self._validation_dataset_dir
        else:
            return self._train_dataset_dir

    def set_validation_dataset_dir(self, dataset_dir):
        self._validation_dataset_dir = dataset_dir

    def set_file_path(self, file_path):
        """
        If you want to save the file somewhere else you can override the file path before calling `dump_data`.

        :param file_path:
        :return:
        """
        self._file_path = file_path

    def dump_data(self):
        """
        saves all data to the file that isn't already there. Will create the file to save to if the file doesn't exist.
        Will keep comments and order of existing data.

        :return:
        """
        config_writer = configparser.RawConfigParser()
        config_writer.read(self._file_path)
        section_name = 'Dataset'
        if section_name not in config_writer.sections():
            config_writer.add_section(section_name)

        config_writer.set(section_name, 'train_dataset_dir', self._train_dataset_dir)
        config_writer.set(section_name, 'validation_dataset_dir', self._validation_dataset_dir)

        with open(self._file_path, 'w') as f:
            config_writer.write(f)


class NeuralNetConfig(Config):
    __net_data = None

    # Default values
    __net = None
    __input_shape = None
    model_dir = 'datatype'
    _checkpoint_dir = 'None'
    num_classes = 4
    _optimizer = 'adam'
    momentum = 0.9
    learning_rate = 0.001
    _loss_fn = 'sparse_categorical_cross_entropy'
    _metrics = 'accuracy'
    _evaluate_metrics = 'accuracy,categorical_accuracy'
    _distributed_strategy = 'default'
    _cached_distributed_strategy = None
    batch_size = 25
    log_dir = 'datatype'
    log_every_n_steps = 10

    _max_steps_per_epoch = None
    num_validation_images = 500
    __epochs = 5

    bench_timings = False
    bench_path = r'.\data.csv'

    def __init__(self, file_path):
        """
        Creates NeuralNetConfig instance from the data in file_path.
        Note, file_path does not have to exist. Any data missing from the file will be filled in with defaults.
        Note, any data that is invalid will cause a exception to be raised.

        Use `dump_data` to update the `file_path` with the missing data with the defaults.

        :param file_path:
        """
        assert os.path.exists(file_path), "Config file doesn't exist, filepath: {}".format(file_path)
        self._file_path = file_path
        super(NeuralNetConfig, self).__init__(self._file_path)
        # object.__init__(self._file_path)
        self.__net_data = self.get_net_data('CNN')

        assert os.path.exists(self._train_dataset_dir), f'train_dataset_dir \'{self._train_dataset_dir}\' does not exist'

        if 'version' in self.__net_data.keys():
            # if this is wrong, then it will fail when initializing neuralnet
            self.__net = self.__net_data['version']

        if 'model_dir' in self.__net_data:
            self.model_dir = self.__net_data['model_dir']

        if 'checkpoint_dir' in self.__net_data:
            self._checkpoint_dir = self.__net_data['checkpoint_dir']

        if 'num_classes' in self.__net_data:
            self.num_classes = int(self.__net_data['num_classes'])

        if 'optimizer' in self.__net_data:
            assert self.__net_data['optimizer'].lower() in optimizer,\
                "Supplied optimizer, \'{}\', doesn\'t exist, available options are {}".format(
                    self.__net_data['optimizer'], list(optimizer))
            self._optimizer = self.__net_data['optimizer'].lower()

        if 'learning_rate' in self.__net_data:
            self.learning_rate = float(self.__net_data['learning_rate'])

        if 'momentum' in self.__net_data:
            self.momentum = float(self.__net_data['momentum'])

        if 'loss' in self.__net_data:
            assert self.__net_data['loss'].lower() in loss_function,\
                "Supplied loss fn, \'{}\', doesn\'t exist, available options are {}".format(
                    self.__net_data['loss'], list(loss_function))
            self._loss_fn = self.__net_data['loss'].lower()

        if 'metrics' in self.__net_data:
            if self.__net_data['metrics'].lower() != 'none':
                for m in self.__net_data['metrics'].lower().split(','):
                    assert m.strip() in metrics, "Supplied metric, \'{}\', doesn\'t exist, available options are {}".format(
                        m, list(metrics))
            self._metrics = self.__net_data['metrics'].lower()

        if 'evaluate_metrics' in self.__net_data:
            for m in self.__net_data['evaluate_metrics'].lower().split(','):
                assert m in metrics, "Supplied evaluate_metrics, \'{}\', doesn\'t exist, available options are {}"\
                    .format(m, list(metrics))
            self._evaluate_metrics = self.__net_data['evaluate_metrics'].lower()

        if 'distributed_strategy' in self.__net_data:
            assert self.__net_data['distributed_strategy'].lower() in distributed_strategy, \
                "Supplied distributed_strategy, \'{}\', doesn\'t exist, available options are {}".format(
                    self.__net_data['distributed_strategy'], list(distributed_strategy))
            self._distributed_strategy = self.__net_data['distributed_strategy'].lower()

        if 'batch_size' in self.__net_data:
            self.batch_size = int(self.__net_data['batch_size'])

        if 'log_dir' in self.__net_data:
            self.log_dir = self.__net_data['log_dir']

        if 'log_every_n_steps' in self.__net_data:
            self.log_every_n_steps = int(self.__net_data['log_every_n_steps'])

        if 'max_steps_per_epoch' in self.__net_data:
            self._max_steps_per_epoch = self.__net_data['max_steps_per_epoch']

        if 'num_validation_images' in self.__net_data:
            self.num_validation_images = int(self.__net_data['num_validation_images'])

        if 'epochs' in self.__net_data:
            self.__epochs = int(self.__net_data['epochs'])

        if 'bench_timings' in self.__net_data:
            assert bool('False')  # I have this to remind me of what python does here
            self.bench_timings = self.__net_data['bench_timings'].lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']

        if 'bench_file' in self.__net_data:
            self.bench_path = self.__net_data['bench_file']

        if 'input_shape' in self.__net_data:
            self.__input_shape = self.__net_data['input_shape']

    @property
    def optimizer(self):
        if self._optimizer == 'adam':
            return optimizer['adam'](learning_rate=self.learning_rate)
        elif self._optimizer == 'sgd':
            return optimizer['sgd'](learning_rate=self.learning_rate, momentum=self.momentum)
        elif self._optimizer == 'rmsprop':
            return optimizer['rmsprop'](learning_rate=self.learning_rate, momentum=self.momentum)
        else:
            return optimizer[self._optimizer]()

    @property
    def epochs(self):
        return self.__epochs

    @property
    def optimizer_str(self):
        return self._optimizer

    @property
    def loss_fn(self):
        return loss_function[self._loss_fn]

    @property
    def metrics(self):
        if self._metrics.lower() == 'none':
            return None
        else:
            return list(map(lambda m: metrics[m.strip()], self._metrics.split(',')))

    @property
    def input_shape(self):
        return self.__input_shape

    @property
    def evaluate_metrics(self):
        # TODO - make sure the same strategy used to compile model is used to compile the metrics
        return list(map(lambda m: metrics[m], self._evaluate_metrics.split(',')))

    @property
    def distributed_strategy(self):
        if self._cached_distributed_strategy is None:
            self._cached_distributed_strategy = distributed_strategy[self._distributed_strategy]
        return self._cached_distributed_strategy  # return the same class each time (although is might be copied)

    @property
    def distributed_strategy_str(self):
        return self._distributed_strategy

    @property
    def max_steps_per_epoch(self):
        res = None
        try:
            res = int(self.__net_data['max_steps_per_epoch'])
        except ValueError:
            res = None
        finally:
            return res

    @property
    def net(self):
        return self.__net

    @property
    def checkpoint_dir(self):
        if self._checkpoint_dir.lower() == 'none' or self._checkpoint_dir.lower() == 'null':
            return None
        else:
            return self._checkpoint_dir

    def set_file_path(self, file_path):
        self._file_path = file_path
        super(NeuralNetConfig, self).set_file_path(file_path)

    def dump_data(self):
        super(NeuralNetConfig, self).dump_data()

        config_writer = configparser.RawConfigParser()
        config_writer.read(self._file_path)

        section_name = 'CNN'
        if section_name not in config_writer.sections():
            config_writer.add_section(section_name)

        config_writer.set(section_name, 'version', self.__net)
        config_writer.set(section_name, 'model_dir', self.model_dir)
        config_writer.set(section_name, 'checkpoint_dir', self._checkpoint_dir)
        config_writer.set(section_name, 'num_classes', str(self.num_classes))
        config_writer.set(section_name, 'optimizer', self._optimizer)
        config_writer.set(section_name, 'learning_rate', str(self.learning_rate))
        config_writer.set(section_name, 'momentum', str(self.momentum))
        config_writer.set(section_name, 'loss', self._loss_fn)
        config_writer.set(section_name, 'metrics', self._metrics)
        config_writer.set(section_name, 'evaluate_metrics', self._evaluate_metrics)
        config_writer.set(section_name, 'distributed_strategy', self._distributed_strategy)
        config_writer.set(section_name, 'batch_size', str(self.batch_size))
        config_writer.set(section_name, 'log_dir', self.log_dir)
        config_writer.set(section_name, 'log_every_n_steps', str(self.log_every_n_steps))

        config_writer.set(section_name, 'max_steps_per_epoch', self._max_steps_per_epoch)
        config_writer.set(section_name, 'num_validation_images', str(self.num_validation_images))
        config_writer.set(section_name, 'epochs', str(self.__epochs))

        with open(self._file_path, 'w') as f:
            config_writer.write(f)

    def set_net(self, neuralnet):
        assert neuralnet
        self.__net = neuralnet

    def set_input_shape(self, input_shape: tuple):
        assert input_shape
        self.__input_shape = input_shape

    def set_epochs(self, epochs):
        assert epochs
        self.__epochs = epochs
