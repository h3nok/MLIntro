import datetime
import itertools
import math
import os
import time
from typing import Optional

import cv2
import numpy as np
import regex as re
import tensorflow as tf
from tensorflow.keras.utils import plot_model

from core.analytics.reporting.csv_generator import CSVGenerator
from customer.customer_base import CustomerBase
from logs import logger as L
from nn_factory import ClassificationNet as CNN
from training_config import NeuralNetConfig
from tfrecord_explorer import DatasetExplorer


# disable gpu for tensorflow
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def tf_version_at_or_above_220():
    """
    Some code needs tf-2.2.0 or higher to run

    :return:
    """
    min_tf_ver = [2, 2, 0]
    return [
               int(v) >= min_tf_ver[i]
               for i, v in enumerate(tf.__version__.split('.')[:3])
           ].count(True) == 3


class NeuralNetLoggerCallback(tf.keras.callbacks.Callback):
    """
    Class used to log the training progress. The default verbose
    setting didn't give enough control.

    Just pass an instance of this class into the `model.fit` method.

    Will print:
        - on_epoch_begin
        - on_epoch_end
        - on_batch_end
        - on_train_begin
        - on_train_end
    """

    _epoch = 0
    _step = 0  # not used
    _time = None

    def __init__(self, settings: NeuralNetConfig,
                 num_batches, global_batch_size,
                 print_fn=print, print_logs=False,
                 **kwargs):
        """
        Set up the callback, I sort of followed this:
        https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/callbacks.py#L2371-L2456

        :param settings:
        :param num_batches:
        :param print_fn:
        :param print_logs: whether we parse the logs or just print it.
        :param kwargs:
        """
        super(NeuralNetLoggerCallback, self).__init__()
        self.__dict__.update(kwargs)
        self.__settings = settings
        self.__print_fn = print_fn
        self.__print_logs = print_logs

        self.on_epoch_begin = lambda epoch, logs: self.__on_epoch_begin(epoch + 1, logs)  # starts at epoch 1 not 0
        self.on_epoch_end = lambda epoch, logs: self.__print_epoch_end_data(epoch + 1, logs)
        self.on_batch_begin = lambda batch, logs: self.__on_batch_begin(batch, logs)
        self.on_batch_end = lambda batch, logs: self.__print_batch_end_data(batch, logs)
        self.on_train_begin = lambda logs: self.__print_fn(
            'Trainer: Training with {} records, {} batches per epoch, and {} epochs.'.format(
                num_batches * global_batch_size, num_batches, self.__settings.epochs))
        self.on_train_end = lambda logs: self.__print_fn('Trainer: End of training, logs: {}.'.format(logs))

    def __on_batch_begin(self, batch, _logs):
        """
        Starts the timer

        :param batch:
        :param _logs:
        :return:
        """
        if batch % self.__settings.log_every_n_steps == 1:
            self._time = time.time()

    def __on_epoch_begin(self, epoch, _logs):
        self.__print_fn('Trainer: Start of epoch {}.'.format(epoch))
        self._epoch = epoch

    def __print_batch_end_data(self, batch, logs):
        if batch % self.__settings.log_every_n_steps == 0:
            if self.__print_logs or 'loss' not in logs:
                self.__print_fn('Trainer: Epoch {}, Batch {}, logs {}.'.format(self._epoch, batch, logs))
            else:
                # note: loss will always be calculated
                build_str = 'Trainer: Epoch {}, Batch {}, loss {:.4f}'.format(self._epoch, batch, logs['loss'])
                if 'accuracy' in logs:  # accuracy must be a metric
                    build_str += ', train_accuracy: {:.4f}'.format(logs['accuracy'])
                if batch != 0:
                    time_elapsed = time.time() - self._time
                    build_str += ' ({:.4f} sec/step)'.format(time_elapsed / self.__settings.log_every_n_steps)
                build_str += '.'
                self.__print_fn(build_str)
        self._step += 1

    def __print_epoch_end_data(self, epoch, logs):
        if self.__print_logs:
            self.__print_fn('Trainer: End of epoch {}, logs {}.'.format(epoch, logs))
        else:
            if 'val_loss' in logs:
                build_str = 'Trainer: End of epoch {}, val_loss {:.4f}'.format(epoch, logs['val_loss'])
            else:
                build_str = 'Trainer: End of epoch {}, train_loss {:.4f}'.format(epoch, logs['loss'])

            if 'val_accuracy' in logs:  # accuracy must be a metric
                build_str += ', val_accuracy: {:.4f}'.format(logs['val_accuracy'])
            elif 'accuracy' in logs:  # accuracy must be a metric
                build_str += ', train_accuracy: {:.4f}'.format(logs['accuracy'])

            build_str += '.'
            self.__print_fn(build_str)


class BenchmarkerCallback(tf.keras.callbacks.Callback):
    """
    Class used to benchmark the training progress. The is also the profiler through tensorboard that works well.

    Just pass an instance of this class into the `model.fit` method.

    Will output to CSV:
        - time per batch
        - frames per batch
        - time per frame
        - stuff from settings
        - distributed_strategy info
        - tensorboard command to run (note, this assumes that a tensorboard callback is being used when training)
        - save dir
    """
    _time = None
    _ave_time_per_batch = 0
    _frames_per_batch = None

    _log_file = r'C:\viNet_RnD\Bench\data.csv'
    _bench_name = None

    _csv_gen = None

    def __init__(self, settings: NeuralNetConfig, bench_name, save_path, log_path, **kwargs):
        """
        Set up the callback, I sort of followed this:
        https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/callbacks.py#L2371-L2456

        :param settings:
        :param bench_name: maybe use the config file name/path
        :param save_path:  model save dir
        :param log_path: tensorboard log_path.
        :param kwargs:
        """

        super(BenchmarkerCallback, self).__init__()
        self.__dict__.update(kwargs)
        self.__settings = settings

        self._log_file = self.__settings.bench_path
        self._bench_name = bench_name
        self._save_path = save_path
        self._log_path = log_path

        self._csv_gen = CSVGenerator(self._log_file, entries_str='configs')
        assert self._csv_gen is not None

        self.on_epoch_end = lambda epoch, logs: self.__log_epoch_end_data(epoch + 1, logs)
        self.on_batch_begin = lambda batch, logs: self.__on_batch_begin(batch, logs)
        self.on_batch_end = lambda batch, logs: self.__on_batch_end(batch, logs)
        self.on_train_end = lambda logs: self._csv_gen.close()

    def __on_batch_begin(self, batch, _logs):
        """
        Starts the timer

        :param batch:
        :param _logs:
        :return:
        """
        if batch % self.__settings.log_every_n_steps == 1:
            self._time = time.time()

    def __on_batch_end(self, batch, logs):
        self._frames_per_batch = logs['size']
        if batch % self.__settings.log_every_n_steps == 0 and batch != 0:
            time_elapsed = time.time() - self._time
            ave_time = time_elapsed / self.__settings.log_every_n_steps

            num_logs = (batch // self.__settings.log_every_n_steps) - 1

            self._ave_time_per_batch = (self._ave_time_per_batch * num_logs + ave_time) / (num_logs + 1)

    def __log_epoch_end_data(self, epoch, logs):
        entry = self._bench_name
        entry_data = dict()
        entry_data['version'] = self.__settings.net
        entry_data['num_classes'] = self.__settings.num_classes
        entry_data['optimizer'] = self.__settings.optimizer_str
        entry_data['epoch'] = epoch
        entry_data['ave_time_per_batch'] = self._ave_time_per_batch
        entry_data['batch_size'] = self._frames_per_batch
        entry_data['ave_time_per_frame'] = self._ave_time_per_batch / self._frames_per_batch
        entry_data['distributed_strategy'] = self.__settings.distributed_strategy_str
        entry_data['num_replicas_in_sync'] = str(self.__settings.distributed_strategy.num_replicas_in_sync)
        entry_data['tensorboard'] = 'tensorboard --logdir {}'.format(self._log_path)
        entry_data['save dir'] = self._save_path

        self._csv_gen.write_line(entry, entry_data)

        self._ave_time_per_batch = 0


class NeuralNet:
    """
    NeuralNet class: handles building various networks (and loading a weights file), saving to network, training the
    network, prepossessing of input images, and evaluating the network.
    """

    def __init__(self, net_config: NeuralNetConfig, customer: CustomerBase):
        """
        An interface  Neural Network Models

        :param net_config: Efficient net settings read from ini file
        """
        assert net_config
        assert customer
        assert issubclass(customer, CustomerBase)

        self.__net_config = net_config
        self.__customer = customer()
        self.__settings_file_path = self.__net_config.file_path  # we need to back this up because it changes
        self._logger = L.Logger(self.__class__.__name__).configure()
        self.__version__ = self.__net_config.net
        assert self.__version__ in CNN.keys(), \
            "Supplied version, \'{}\', " \
            "doesn't exist, available options are {}".format(self.__version__, list(CNN))

        self.__input_shape = None
        self.__include_top = True

        # Override default input shape, tuple(w, h, ch)
        # the top layer must be excluded when using custom input shape
        if self.__net_config.input_shape:
            if isinstance(self.__net_config.input_shape, str):
                self.__input_shape = tuple(map(int, self.__net_config.input_shape.split(',')))
            else:
                self.__input_shape = self.__net_config.input_shape
            self.__include_top = False

        # build the graph
        with self.__net_config.distributed_strategy.scope():
            self.__model = CNN[self.__version__](classes=self.__net_config.num_classes,
                                                 weights=None,
                                                 input_shape=self.__input_shape,
                                                 include_top=self.__include_top)
        self._input_size = self.__model.input_shape[1]

        self._logger.info("Successfully constructed neural net, version: {}".format(self.__version__))

        self._time_string = "{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S'))

        # output directory
        save_dir = os.path.join(self.__net_config.model_dir,
                                self.__customer.name,
                                self.__version__,
                                self._time_string)

        self._logger.debug(f'Model dir: {save_dir}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.__net_config.set_file_path(os.path.join(save_dir,
                                                     f'{self.__customer.name}_{self.__version__}_train_config.ini'))
        self.__net_config.dump_data()

    @property
    def customer(self):
        return self.__customer.name

    @property
    def base_model(self):
        """

        @return: model
        """
        return self.__model

    @property
    def output_nodes(self):
        """

        @return:
        """
        return [node.op.name for node in self.__model.outputs]

    @property
    def input_nodes(self):
        """

        @return:
        """
        return [node.op.name for node in self.__model.inputs]

    def load_weights(self, weight_file):
        """
        Loads weights from a weight file.
        Will get rid of existing weight data.

        Can be use when you want to continue from where a save file left off. However, the `checkpoint_dir` setting in
        the settings file might be more useful.

        :param weight_file: `tf.train.latest_checkpoint` might be useful
        :return:
        """
        self.__model.load_weights(weight_file)

        self._logger.info("Successfully loaded weights, checkpoint_path: {}".format(weight_file))

    @property
    def summary(self) -> str:
        """
        Just calls `model.summary()`.

        :return: str
        """

        ret_str_arr = []
        self.__model.summary(print_fn=lambda x: ret_str_arr.append(x + '\n'),
                             line_length=150,
                             positions=[0.4, 0.65, 0.77, 1.0])
        return ''.join(ret_str_arr)

    @property
    def input_shape(self):
        """
        Note, the real input is (None, width, height, channels)

        :return: (width, height, channels)
        """
        return self.__model.input_shape[1:]

    def resize_input_image(self, image, crop=False):
        """
        Used to take a RGB image and preprocess it for the network.

        if crop:
            Right now this requires the network to take a square input.
            Will crop out the largest possible centered square in the image.

        Then will resize to (self._input_size, self._input_size, None).
        Then will normalize to range [-1, 1] (not 0-255) as a `np.float32` array or `tf.float32`.

        :param image: input image can be `np.ndarray` or `tf.Tensor` (but will always use CPU)
        :param crop: if false it will stretch
        :return: Will be np.ndarray` or `tf.Tensor` depending on the input
        """

        if crop:
            h, w = image.shape[:2]

            min_size = min(h, w)

            # step 1: make square
            offset_height = (h - min_size) // 2
            offset_width = (w - min_size) // 2

            image_crop = image[
                         offset_height: min_size + offset_height,
                         offset_width: min_size + offset_width,
                         ]
        else:
            image_crop = image

        # step 2: resize
        mean = 127.5  # image is [0, 255]
        if isinstance(image, tf.Tensor):
            # I don't really know how much this affects anything
            with tf.device('/cpu:0'):  # By default cpu:0 represents all cores available to the process
                # The reason we force tensorflow to use the CPU here is so we don't have to take GPU memory away from
                # the training.

                image_crop = tf.cast(tf.constant(image_crop), tf.float32)
                image_crop -= mean
                image_crop /= mean  # we want to have a image with range [-1, 1]
                return tf.image.resize(image_crop, (self._input_size, self._input_size))
        else:
            image_crop = np.asarray(image_crop).astype(np.float32)
            image_crop -= mean
            image_crop /= mean  # we want to have a image with range [-1, 1]
            return cv2.resize(image_crop, (self._input_size, self._input_size))

    def plot(self, output_path: str = None):
        """
        Creates an image of the network.

        :param output_path: optional
        :return:
        """
        if not output_path:
            output_path = os.path.join(self.__net_config.model_dir, self.__version__, self._time_string)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        plot_model(self.__model,
                   to_file=os.path.join(output_path,
                                        "{}_{}.png".format(self.__customer.name, self.__version__)),
                   show_shapes=True, show_layer_names=True)

    def save(self, output_dir=None, **kwargs):
        """
        Saves the model to a `.h5` file.
        This is need when running `to_operational_tfv2`.

        :return: path to saved model file.
        """

        filename = f'{self.__version__}_{self.__customer.name}.h5'

        if kwargs:
            filename = f"{self.__version__}_{kwargs['epochs']}_{kwargs['customer']}.h5"

        if not output_dir:
            save_dir = os.path.join(self.__net_config.model_dir, self.__version__, self._time_string)
        else:
            save_dir = output_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path_m = os.path.join(save_dir, filename)
        self.__model.save(save_path_m)

        self._logger.info('Successfully saved model file, filepath: {}'.format(save_path_m))

        return save_path_m

    def __unzip_tfrecords(self, image_label_iter, count, with_ids=False):
        """
        Takes in a iterator of (image, label, [id]) and creates two (or 3)
        numpy arrays of the data of the size `count`.Will only grab `count`
        samples from the iterator.

        note, [x] means x is optional

        @param image_label_iter: iter over type (image, label, [id])
        @param count:
        @param with_ids: indicates that the input has the id field
        @return: the tuple (image_ndarray, label_array, [id_array], count)
        """
        array_0 = np.zeros((count, self._input_size, self._input_size, 3), dtype=np.float32)
        array_1 = np.zeros(count, dtype=np.uint8)
        array_2 = []  # make into numpy array?
        i = 0
        for i, v in enumerate(itertools.islice(image_label_iter, count)):
            array_0[i, :, :, :] = v[0]
            array_1[i] = v[1]
            if with_ids:
                array_2.append(v[2])

        if with_ids:
            return array_0[0:i + 1, :, :, :], array_1[:i + 1], array_2[:i + 1], i + 1
        else:
            return array_0[0:i + 1, :, :, :], array_1[:i + 1], i + 1

    def __batched_dataset(self, datasetexpl: DatasetExplorer,
                          max_batches: Optional[int], batch_size: int):
        """
        takes the (image, label) iter in the form of a `DatasetExplorer` and
        creates a new iter that batches the image label data.

        Once the `datasetexpl.get_records()` iter is exhausted it will restart
        the iter. So this iter is infinite.

        :param datasetexpl:
        :param max_batches: Number of batches before restarting the `datasetexpl`
        iter
        :param batch_size: size of each batch
        :return: infinite iter over type (image_batch, label_batch)
        """

        while True:
            dataset = map(
                lambda e: (self.resize_input_image(tf.image.decode_image(e[0], channels=3)).numpy(), e[2]),
                datasetexpl.get_records())

            for i in itertools.count():
                if max_batches is not None and i >= max_batches:
                    break

                inputs, targets, size = self.__unzip_tfrecords(dataset, batch_size)
                if size < batch_size:
                    if i == 0:
                        return
                    break

                yield inputs, targets

    def __batched_dataset_with_ids(self, datasetexpl: DatasetExplorer,
                                   max_batches: Optional[int], batch_size: int):
        """
        takes the (image, label, id) iter in the form of a `DatasetExplorer` and creates a new iter that batches the
        image, label, id data.

        Once the `datasetexpl.get_records()` iter is exhausted it will restart the iter. So this iter is infinite.

        :param datasetexpl:
        :param max_batches: Number of batches before restarting the `datasetexpl` iter
        :param batch_size: size of each batch
        :return: infinite iter over type (image_batch, label_batch, id_batch)
        """

        while True:
            dataset = map(
                lambda e: (self.resize_input_image(tf.image.decode_image(e[0], channels=3)).numpy(), e[2], e[4]),
                datasetexpl.get_records())

            for i in itertools.count():
                if max_batches is not None and i >= max_batches:
                    break

                inputs, targets, ids, size = self.__unzip_tfrecords(dataset, batch_size, True)
                if size < batch_size:
                    if i == 0:
                        return
                    break

                yield inputs, targets, ids

    def __try_load_last_save(self, epoch_offset_override: int = None):
        """
        Will use checkpoint_dir from the config file to try and load weights from the latest checkpoint. If it fails to
        load the correct weights, then you would have to use `load_weights` manually before calling train and/or you
        might have to supply `epoch_offset_override`.

        :param epoch_offset_override: It is an override for the default case. If `epoch_offset_override` is None then
            the default case applies. If you don't specify `checkpoint_dir` in the config then the default is 0. If you
            specify `checkpoint_dir` then the latest checkpoint file will be used and the name will be parsed to attempt
            to get the epoch number. That is if the format is the following: `cp.ckpt-e{epoch_num}` the `epoch_num` will
            be use. Or if the training finished and `done.ckpt` is the latest checkpoint then epoch number will be read
            from the config.ini file in the same directory as the checkpoint. If neither of these can be used to get the
            epoch number then 0 will be used (this is when you should specify `epoch_offset_override`).
        :return: epoch_offset, can be None.
        """

        if self.__net_config.checkpoint_dir is not None:
            checkpoint_path = tf.train.latest_checkpoint(self.__net_config.checkpoint_dir)

            self._logger.info('Continuing from latest checkpoint file: {}'.format(checkpoint_path))

            res = re.match(r'cp.ckpt-e([0-9]{2}[0-9]*)', os.path.basename(checkpoint_path))
            res2 = re.match(r'done\.ckpt', os.path.basename(checkpoint_path))
            if epoch_offset_override is None:
                if res:  # if training stopped in the middle
                    epoch_offset_override = int(res.group(1))
                    self._logger.info("Latest checkpoint file, {}: Found epoch offset, will start at epoch {}."
                                      .format(os.path.basename(checkpoint_path), epoch_offset_override + 1))
                elif res2:  # if more epochs are needed but past training finished correctly
                    tmp_settings = NeuralNetConfig(os.path.join(os.path.dirname(checkpoint_path), 'config.ini'))
                    epoch_offset_override = tmp_settings.epochs
                    self._logger.info("Latest checkpoint file, {}: Found epoch offset, will start at epoch {}."
                                      .format(os.path.basename(checkpoint_path), epoch_offset_override + 1))
                else:
                    self._logger.info("Latest checkpoint file, {}: Can't find epoch number, will start at epoch 1."
                                      .format(os.path.basename(checkpoint_path)))

            self.load_weights(checkpoint_path)

        return epoch_offset_override

    def __get_global_batch_size(self):
        """
        Only different from `batch_size` in the settings if the `distributed_strategy` (from settings) is
        a mirror_strategy. This is because we want to give each clone of the network batch_size different frames.

        :return: `batch_size` from settings or `batch_size * num_gpus` if mirror_strategy.
        """
        global_batch_size = self.__net_config.batch_size
        # TODO: do we need this if (right now it doesn't account for other strategies that might clone the network)
        # if isinstance(self.__settings.distributed_strategy, tf.distribute.MirroredStrategy):
        global_batch_size *= self.__net_config.distributed_strategy.num_replicas_in_sync
        return global_batch_size

    def __setup_callbacks_and_file_system(self, num_batches, global_batch_size):
        """
        Does 2 things.
            - Sets up the save_dir and log_dir
            - Sets up the callbacks

        :param num_batches:
        :param global_batch_size:
        :return: (callbacks, save_dir, log_path)
        """
        # 4: Set up file system
        self._logger.debug("Setting up callbacks and filesystem.")

        save_dir = os.path.join(self.__net_config.model_dir,
                                self.__customer.name,
                                self.__version__,
                                self._time_string,
                                'checkpoints')

        self._logger.debug(f'Save dir:{save_dir}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        log_path = os.path.join(self.__net_config.log_dir,
                                self.__customer.name,
                                self.__version__,
                                self._time_string,
                                'events')

        self._logger.debug('To open tensorboard run `tensorboard --logdir {}`'.format(log_path))

        if not os.path.exists(log_path):
            os.makedirs(log_path)

        if self.__net_config.bench_timings:
            self._logger.debug('Bench data at: {}'.format(self.__net_config.bench_path))
            if not os.path.exists(os.path.dirname(self.__net_config.bench_path)):
                os.makedirs(os.path.dirname(self.__net_config.bench_path))

        # 5: Set up callbacks for saving checkpoints, logging, and tensorboard
        epoch_checkpoint_path = os.path.join(save_dir, 'cp.ckpt-e{epoch:02d}')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=epoch_checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1, save_freq='epoch')

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1,
                                                              update_freq=self.__net_config.log_every_n_steps,
                                                              profile_batch='100,200'
                                                              if tf_version_at_or_above_220()
                                                              else 0)

        logger_callback = NeuralNetLoggerCallback(self.__net_config, num_batches,
                                                  global_batch_size,
                                                  print_fn=self._logger.info)

        callbacks = [cp_callback, tensorboard_callback, logger_callback]

        if self.__net_config.bench_timings:
            bench_callback = BenchmarkerCallback(self.__net_config,
                                                 self.__settings_file_path,
                                                 save_dir, log_path)
            callbacks.append(bench_callback)

        return callbacks, save_dir, log_path

    @staticmethod
    def __save_classification_code_map_to_file(save_dir, datasetexpl: DatasetExplorer):
        """
        Saves the labels.txt file from datasetexpl to save_dir in the format that viNetClassifyTool expects

        :param save_dir:
        :param datasetexpl:
        :return:
        """
        path = os.path.join(save_dir, 'labels.txt')
        with open(path, 'w') as f:
            for i in range(len(list(datasetexpl.class_maps))):
                f.write(datasetexpl.class_maps[i] + '\n')

    def train(self, epoch_offset_override: int = None):
        """
        Trains the Neural Net model using Keras API 
        Only one call to model.fit

        If you want to continue from where a checkpoint left off, you must specify `checkpoint_dir` in the config.

        :param epoch_offset_override: It is an override for the default case. If `epoch_offset_override` is None then
            the default case applies. If you don't specify `checkpoint_dir` in the config then the default is 0. If you
            specify `checkpoint_dir` then the latest checkpoint file will be used and the name will be parsed to attempt
            to get the epoch number. That is if the format is the following: `cp.ckpt-e{epoch_num}` the `epoch_num` will
            be use. Or if the training finished and `done.ckpt` is the latest checkpoint then epoch number will be read
            from the config.ini file in the same directory as the checkpoint. If neither of these can be used to get the
            epoch number then 0 will be used (this is when you should specify `epoch_offset_override`).
        :return: save_dir
        """

        # TODO: save weights more then once every epoch, delete old saves if there are a lot (use a callback)
        # TODO: Update to tensorflow==2.2.0, this is because when you fit with validation data you can't specify the
        # TODO:     batch size meaning that it can overflow the GPU memory. You can specify it in 2.2.0
        # TODO:     I haven't been able to test it.

        epoch_offset = self.__try_load_last_save(epoch_offset_override)

        # 1: Set up the dataset iterators
        global_batch_size = self.__get_global_batch_size()

        datasetexpl = DatasetExplorer(self.__net_config._train_dataset_dir)

        real_len = math.floor(datasetexpl.num_records() / global_batch_size)
        if self.__net_config.max_steps_per_epoch is None:
            num_batches = real_len
        else:
            num_batches = min(real_len, self.__net_config.max_steps_per_epoch)

        # note, __batched_dataset will also "calculate" num_batches internally
        train_dataset = tf.data.Dataset.from_generator(
            lambda: self.__batched_dataset(datasetexpl, self.__net_config.max_steps_per_epoch, global_batch_size),
            (tf.float32, tf.int64),
            output_shapes=(tf.TensorShape([None, None, None, 3]), tf.TensorShape([None])))

        test_dataset = map(
            lambda e: (self.resize_input_image(tf.image.decode_image(e[0], channels=3)).numpy(), e[2]),
            DatasetExplorer(self.__net_config.validation_dataset_dir).get_records())

        # 2: Compile the model
        with self.__net_config.distributed_strategy.scope():
            self.__model.compile(optimizer=self.__net_config.optimizer,
                                 loss=self.__net_config.loss_fn,
                                 metrics=self.__net_config.metrics)

        # 3: Get validation frames
        if self.__net_config.num_validation_images == 0:
            test_data = None
        else:
            test_images, test_labels, test_size = self.__unzip_tfrecords(test_dataset,
                                                                         self.__net_config.num_validation_images)
            assert test_size == self.__net_config.num_validation_images, 'Validation dataset is to small'
            test_data = (test_images, test_labels)

        # 4: Set up file system
        # 5: Set up callbacks for saving checkpoints, logging, and tensorboard
        callbacks, save_dir, _log_path = self.__setup_callbacks_and_file_system(num_batches, global_batch_size)

        self.__save_classification_code_map_to_file(save_dir, datasetexpl)

        # 6: Call model.fit once
        if epoch_offset is None:
            epoch_offset = 0
        vsteps = int(
            math.ceil(self.__net_config.num_validation_images / global_batch_size)) if test_data is not None else None

        history = self.__model.fit(train_dataset, epochs=self.__net_config.epochs, steps_per_epoch=num_batches,
                                   validation_data=test_data, initial_epoch=epoch_offset,
                                   validation_steps=vsteps, shuffle=True, callbacks=callbacks,
                                   verbose=0)  # , validation_batch_size=self.__settings.batch_size)#only for >=tf-2.2.0

        # 7: Save final weights
        save_path_w = os.path.join(save_dir, 'done.ckpt')
        self.__model.save_weights(save_path_w)
        self._logger.info('Finished training model. Saved weight file: {}'.format(save_path_w))

        return save_dir

    def __metrics_to_dict(self, metrics):
        # tf-2.2.0 can return a dict but tf-2.1.0 can not
        metrics_names = self.__model.metrics_names
        if not isinstance(metrics, list):
            metrics = [metrics]
        assert len(metrics_names) == len(metrics)
        ret = dict()
        for i in range(len(metrics_names)):
            ret[metrics_names[i]] = metrics[i]

        return ret

    def predict_validation(self, hard_labels=True):
        """
        Evaluates the whole validation dataset by returning the truth and predicted values.
        Note, Uses `num_validation_images` internally.
        You should only use this if you have a validation_dataset_dir set up in settings.

        :param hard_labels: if the predicted values should be soft or hard labels.
        :return: list of ids, truth, and predicted
        """
        global_batch_size = self.__get_global_batch_size()

        datasetexpl = DatasetExplorer(self.__net_config.validation_dataset_dir)

        num_groups = math.floor(datasetexpl.num_records() / self.__net_config.num_validation_images)

        validation_dataset_gen = self.__batched_dataset_with_ids(datasetexpl, None,
                                                                 self.__net_config.num_validation_images)

        with self.__net_config.distributed_strategy.scope():
            self.__model.compile(optimizer=self.__net_config.optimizer,
                                 loss=self.__net_config.loss_fn,
                                 metrics=self.__net_config.evaluate_metrics)

        pred = []
        truth = []
        ids = []

        for i, group in enumerate(validation_dataset_gen):
            if i >= num_groups:
                break

            print("group", i + 1, "out of", num_groups)

            images, labels, this_ids = group

            res = self.__model.predict(images)

            if hard_labels:
                res = [list(this_pred).index(max(this_pred)) for this_pred in res]

            pred.extend(res)
            truth.extend(labels)
            ids.extend(this_ids)

        return ids, truth, pred

    def evaluate(self):
        """
        Evaluates the whole validation dataset. Ignores `num_validation_images` in settings.
        You should only use this if you have a validation_dataset_dir set up in settings.

        :return: dict of metrics
        """
        global_batch_size = self.__get_global_batch_size()

        datasetexpl = DatasetExplorer(self.__net_config.validation_dataset_dir)

        num_batches = math.floor(datasetexpl.num_records() / global_batch_size)

        # note, __batched_dataset will also "calculate" num_batches
        validation_dataset = tf.data.Dataset.from_generator(
            lambda: self.__batched_dataset(datasetexpl, None, global_batch_size),
            (tf.float32, tf.int64),
            output_shapes=(tf.TensorShape([None, None, None, 3]), tf.TensorShape([None])))

        with self.__net_config.distributed_strategy.scope():
            self.__model.compile(optimizer=self.__net_config.optimizer,
                                 loss=self.__net_config.loss_fn,
                                 metrics=self.__net_config.evaluate_metrics)

        # evaluate the model
        metrics = self.__model.verify(validation_dataset, verbose=1,
                                      steps=num_batches)
        # return_dict=True) only in tf-2.2.0

        return self.__metrics_to_dict(metrics)

    def evaluate_num_validation_images(self, checkpoint_path=None):
        """
        Evaluates the first `num_validation_images` (from settings) frames in the validation dataset.

        :return: dict of metrics
        """
        # Set up the dataset iterators
        dataset = DatasetExplorer(self.__net_config.validation_dataset_dir).get_records()
        dataset = map(
            lambda e: (self.resize_input_image(tf.image.decode_image(e[0], channels=3)).numpy(),
                       e[2]), dataset)
        validation_images, validation_labels, num = self.__unzip_tfrecords(dataset,
                                                                           self.__net_config.num_validation_images)

        # Compile the model
        with self.__net_config.distributed_strategy.scope():
            self.__model.compile(optimizer=self.__net_config.optimizer,
                                 loss=self.__net_config.loss_fn,
                                 metrics=self.__net_config.evaluate_metrics)

        # evaluate the model
        metrics = self.__model.verify(validation_images,
                                      validation_labels,
                                      verbose=1,
                                      batch_size=self.__net_config.batch_size)  # , return_dict=True)

        return self.__metrics_to_dict(metrics)

# if __name__ == '__main__':
#     efn_settings = NeuralNetConfig('vinet.settings.test.ini')
#     # efn_settings.dump_data() # call this function to update the ini file anytime you
#     # add a new param to settings
#     from common.nn_utils import visualize_graph
#
#     net = NeuralNetModel(efn_settings, CustomerObjectLookup['GWA'])
#     net.load_weights(r"E:\viNet_RnD\Research\EfficientNet\EfficientNet_20200817-111013\cp.ckpt-e11")
#     visualize_graph(net.save(output_dir=r"E:\viNet_RnD\Research\Candidates",
#                              epochs=11, customer='Goldwind'))
#     print(net.summary)
#     # net.save_model()
#     print(net.evaluate_num_validation_images())
