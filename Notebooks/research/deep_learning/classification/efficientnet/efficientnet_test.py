import os
import itertools

import tensorflow as tf
import numpy as np

import tensorflow.keras as keras

import neuralnet.tfkeras as efn
import matplotlib.pyplot as plt

from dataset_utils import read_label_file

import cv2

import time


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """
    Will crop out the largest square possible, and centered, in the size of the image.
    Then will resize to (size, size).
    Then will normalize to range 0-1 (not 0-255) as a `np.float32` array.

    :param image: input image
    :param size: size of returned image
    :return:
    """

    h, w = image.shape[:2]

    min_size = min(h, w)

    # step 1: make square
    offset_height = (h - min_size) // 2
    offset_width = (w - min_size) // 2

    image_crop = image[
                 offset_height: min_size + offset_height,
                 offset_width: min_size + offset_width,
                 ]

    # image_crop = tf.cast(tf.constant(image_crop), tf.float32) / 255.0
    # return tf.image.resize(image_crop, (size, size))

    image_crop = np.asarray(image_crop).astype(np.float32) / 255.0
    return cv2.resize(image_crop, (size, size))


def resize_image_tf(image: tf.Tensor, size: int) -> tf.Tensor:
    """
    Will crop out the largest square possible, and centered, in the size of the image.
    Then will resize to (size, size).
    Then will normalize to range 0-1 (not 0-255) as a `np.float32` array.

    :param image: input image
    :param size: size of returned image
    :return:
    """

    h, w = image.shape[:2]

    min_size = min(h, w)

    # step 1: make square
    offset_height = (h - min_size) // 2
    offset_width = (w - min_size) // 2

    image_crop = image[
                 offset_height: min_size + offset_height,
                 offset_width: min_size + offset_width,
                 ]

    image_crop = tf.cast(tf.constant(image_crop), tf.float32) / 255.0
    return tf.image.resize(image_crop, (size, size))


def load_efficientnetb0_model(save_path, from_weights_file=False, len_class_names=4, w=False):
    """
    Loads an efficientnetb0 model from a file in `this_dir`.

    :param this_dir: The dir/file where the save file/weights are
    :param from_weights_file: If we read from a .ckpt or a .h5 file
    :param len_class_names: only needed if `from_weights_file` is True
    :param w: for testing: loads imagenet weights
    :return: the model
    """
    if w:
        model = efn.EfficientNetB0(weights='imagenet')
    else:
        if os.path.isdir(save_path):
            data_file = tf.train.latest_checkpoint(save_path)
        else:
            data_file = save_path

        if from_weights_file:
            model = efn.EfficientNetB0()  # 0-7
            image_size = model.input_shape[1]

            top_model = keras.models.Sequential()
            top_model.add(keras.layers.Dense(len_class_names))

            model = keras.Model(inputs=model.input, outputs=top_model(model.output))

            model.load_weights(data_file)
        else:
            model = tf.keras.models.load_model(data_file)
    return model


class EfficientNetTrainer:
    __dataset_dir = None
    __save_dir = None

    _iterations = None
    _batch_size = 13000  # so we don't overfill the system memory

    _num_epochs = 10
    _train_batch_size = 10  # so we don't overfill the GPU memory

    _image_size = None
    _class_names = None

    __model = None

    def __create(self, v=True):
        self.__model = efn.EfficientNetB0()  # 0-7
        self._image_size = self.__model.input_shape[1]

        top_model = keras.models.Sequential()
        top_model.add(keras.layers.Dense(len(self._class_names)))

        self.__model = keras.Model(inputs=self.__model.input, outputs=top_model(self.__model.output))

        if v is True:
            self.__model.summary()

    def __init__(self, dataset_dir, save_dir, v=False):
        """
        Creates a new EfficientNetB0.


        :param dataset_dir:
        :param save_dir:
        :param v: verbose
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.__dataset_dir = dataset_dir
        self.__save_dir = save_dir

        self._class_names = read_label_file(self.__dataset_dir)

        if self.__model is None:
            self.__create(v)

    @classmethod
    def from_save_file(cls, dataset_dir, save_dir, from_weights_file=False):
        """
        Creates a EfficientNetB0 from a save file in `save_dir`.

        :param dataset_dir:
        :param save_dir:
        :param from_weights_file: If we read from a .ckpt or a .h5 file
        """
        trainer = cls(dataset_dir, save_dir)
        trainer.__model = load_efficientnetb0_model(save_dir, from_weights_file, len(trainer._class_names))
        return trainer

    @staticmethod
    def __map_tfrecord(raw_example):
        parsed = tf.train.Example.FromString(raw_example.numpy())

        feature = parsed.features.feature
        raw_img = feature['image/encoded'].bytes_list.value[0]
        img = tf.image.decode_image(raw_img).numpy()
        img = img[:, :, 0:3]

        lable = feature['image/class/label'].int64_list.value[0]

        shape = (feature['image/height'].int64_list.value[0], feature['image/width'].int64_list.value[0], 3)

        assert shape == img.shape

        return img, lable, shape

    def __read_tfrecords(self):
        file_paths = [
            os.path.join(self.__dataset_dir, fn)
            for fn in next(os.walk(self.__dataset_dir))[2]
            if os.path.splitext(fn)[1] == '.tfrecord'
        ]

        tfrecord_dataset = tf.data.TFRecordDataset(file_paths)

        return map(self.__map_tfrecord, iter(tfrecord_dataset))

    def train(self):
        """
        Will start training from the given `dataset_dir`

        :return:
        """
        test_acc = None
        test_loss = None

        parsed_dataset = self.__read_tfrecords()

        self.__model.compile(optimizer='adam',  # TODO: make good settings
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                             metrics=['accuracy'])

        # parsed_dataset = map(lambda x: (center_crop_and_resize(x[0], image_size), x[1]), parsed_dataset)
        parsed_dataset = map(lambda x: (resize_image(x[0], self._image_size), x[1]), parsed_dataset)

        if not os.path.exists(self.__save_dir):
            os.mkdir(self.__save_dir)

        for i in itertools.count():
            if self._iterations is not None and i > self._iterations:
                break

            print("Reading", self._batch_size, "frames from tfrecords")

            checkpoint_path = os.path.join(self.__save_dir, 'cp.ckpt-' + str(i))

            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)

            # TODO: fix this. It blows up the memory
            train_images = []
            train_labels = []
            for parsed_example in itertools.islice(parsed_dataset, self._batch_size):
                train_images.append(parsed_example[0])
                train_labels.append(parsed_example[1])
            train_images = np.asarray(train_images)
            train_labels = np.asarray(train_labels, dtype=np.uint32)

            # train_images, train_labels = list(zip(*itertools.islice(parsed_dataset, self._batch_size)))
            # train_images = np.asarray(list(train_images))
            # train_labels = np.asarray(list(train_labels), dtype=np.uint32)

            test_num = len(train_images) // 10
            train_num = len(train_images) - test_num

            if len(train_images) == 0:
                break

            test_images = train_images[train_num:train_num + test_num]
            test_labels = train_labels[train_num:train_num + test_num]

            history = self.__model.fit(train_images[0:train_num], train_labels[0:train_num], epochs=self._num_epochs,
                                validation_data=(test_images, test_labels), batch_size=self._train_batch_size,
                                callbacks=[cp_callback])

            test_loss, test_acc = self.__model.evaluate(test_images, test_labels, verbose=2)

        # TODO: use all test images
        # test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

        print('test acc:', test_acc, 'test loss:', test_loss)

        save_path_w = os.path.join(self.__save_dir, 'done.ckpt')
        save_path_m = os.path.join(self.__save_dir, 'done.h5')
        self.__model.save(save_path_m)
        self.__model.save_weights(save_path_w)

        return self.__save_dir

    def check_preprocessing(self, num):
        """
        Shows the training images before and after the resize_image call.

        :param num: repeat
        :return:
        """
        parsed_dataset = self.__read_tfrecords()

        i = 0
        while i < num:
            data = next(parsed_dataset)
            image_cropped = resize_image(data[0], self._image_size)
            # image_cropped = center_crop_and_resize(data[0], image_size)

            plt.imshow(data[0].astype(np.uint8))
            plt.show()

            plt.imshow(image_cropped.astype(np.uint8))
            plt.show()

            i += 1

    def eval(self):
        """
         Evals the network

        :return: test_loss, test_acc, total_time, image_time, num
        """
        self.__model.compile(optimizer='adam',
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                             metrics=['accuracy'])

        parsed_dataset = self.__read_tfrecords()
        parsed_dataset = list(itertools.islice(parsed_dataset, self._batch_size))

        t0 = time.time()

        parsed_dataset = map(lambda x: (resize_image(x[0], self._image_size), x[1]), parsed_dataset)

        images = []
        labels = []
        for parsed_example in itertools.islice(parsed_dataset, self._batch_size):
            images.append(parsed_example[0])
            labels.append(parsed_example[1])
        images = np.asarray(images)
        labels = np.asarray(labels, dtype=np.uint32)
        # images, labels = list(zip(*itertools.islice(parsed_dataset, self._batch_size)))
        # images = np.asarray(list(images))
        # labels = np.asarray(list(labels), dtype=np.uint32)

        t1 = time.time()

        test_loss, test_acc = self.__model.evaluate(images, labels, verbose=2)

        return test_loss, test_acc, time.time() - t0, t1-t0, self._batch_size

    def save_as_h5(self):
        save_path_m = os.path.join(self.__save_dir, 'done.h5')
        self.__model.save(save_path_m)


if __name__ == '__main__':
    SAVE_DIR = r'C:\EfficientNet_saves\viNet_2.7_Vattenfall_Proper_5_class_v2_jpg95'
    FILE_ROOT = r'C:\viNet_RnD\Datasets\Vattenfall-Gotland\TFRecords\viNet_2.7_Vattenfall_Proper_5_class_v2_jpg95\train'

    # folder = 'save_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M%S")
    # save_dir = os.path.join(SAVE_DIR, folder)
    # efn_t = EfficientNetTrainer(FILE_ROOT, save_dir)
    # efn_t.train()

    efn_t = EfficientNetTrainer.from_save_file(FILE_ROOT, SAVE_DIR, from_weights_file=True)
    test_loss, test_acc, total_time, image_time, num = efn_t.eval()
    # efn_t.save_as_h5()
    print("acc:", test_acc)
    print("total time per image:", total_time/num, "time to resize per image:", image_time/num)
