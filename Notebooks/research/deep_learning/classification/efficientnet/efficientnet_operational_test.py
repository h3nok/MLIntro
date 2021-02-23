import itertools
import os

import tensorflow as tf

import cv2

import efficientnet_test as efn

import dataset.tfrecord_explorer as tfrec_exp

from dataset_utils import read_label_file

from deployment.wrap_frozen_graph import wrap_frozen_graph


SAVE_MODEL_FILE = r'C:\EfficientNet_saves\done.h5'
# SAVE_MODEL_FILE = r'C:\viNet_RnD\Deployment\Frozen Models\Efficientnet_test.pb'
TFRECORD_DIR = r'C:\viNet_RnD\Datasets\Vattenfall-Gotland\TFRecords\viNet_2.7_Vattenfall_Proper_4_class_v2_jpg100\train'
IMAGE_SAVE_DIR = r'C:\Users\zjorquera\Pictures\train'


def use_model(model, input, input_size):
    input = efn.resize_image_tf(input, input_size)
    input = tf.expand_dims(input, axis=0)
    res = model(input)
    if isinstance(res, list):
        res = res[0]
    i = max(range(len(res[0])), key=res[0].__getitem__)
    return i, res


def tfrecord_save_to_file(tfrec_dir, save_dir, num_saves=100, class_names: dict = None):
    exp = tfrec_exp.DatasetExplorer(tfrec_dir)
    examples = itertools.islice(exp.get_records(), num_saves)

    for e in examples:
        #test = tf.image.decode_image(tf.constant(e[0]))
        file_name = e[4].decode('utf-8') + '.' + e[1].decode('utf-8')
        with open(os.path.join(save_dir, file_name), 'wb') as f:
            f.write(e[0])

        with open(os.path.join(save_dir, 'labels.txt'), 'a') as f:
            if class_names is not None:
                #label = '{}:{}'.format(e[2], class_names[e[2]].rstrip('\r'))
                label = '{}'.format(e[2])
            else:
                label = str(e[2])
            f.write('{}:{}\n'.format(file_name, label))


def load_image(image_path) -> tf.Tensor:
    with open(image_path, 'rb') as f:
        raw_image_data = f.read()

    if os.path.splitext(image_path)[1] == '.jpg':
        img = cv2.imread(image_path)
        # img = tf.image.decode_image(tf.constant(raw_image_data), channels=3)
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        return tf.cast(tf.constant(img), tf.float32)
    else:
        img = tf.image.decode_image(tf.constant(raw_image_data), channels=3)
        img = img[:, :, :3]
        return tf.cast(img, tf.float32)


def load_model(file_path):
    model = None
    if os.path.splitext(file_path)[1] == '.h5':
        model = efn.load_efficientnetb0_model(SAVE_MODEL_FILE, False)
    elif os.path.splitext(file_path)[1] == '.pb':
        with tf.io.gfile.GFile(file_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        model = inception_func = wrap_frozen_graph(graph_def, inputs='input_1:0', outputs=["Identity:0"])
    return model


if __name__ == '__main__':
    class_names = read_label_file(TFRECORD_DIR)

    # load net
    model = load_model(SAVE_MODEL_FILE)

    # load/save images
    if not os.path.exists(IMAGE_SAVE_DIR):
        os.makedirs(IMAGE_SAVE_DIR)
    files_in_save_path = [
        os.path.join(IMAGE_SAVE_DIR, fn)
        for fn in next(os.walk(IMAGE_SAVE_DIR))[2]
    ]
    if len(files_in_save_path) > 0:
        print(IMAGE_SAVE_DIR, ' has files in it')
    else:
        tfrecord_save_to_file(TFRECORD_DIR, IMAGE_SAVE_DIR, 1000, class_names)

    files_in_save_path = [
        os.path.join(IMAGE_SAVE_DIR, fn)
        for fn in next(os.walk(IMAGE_SAVE_DIR))[2]
        if os.path.splitext(fn)[1] == '.jpg' or
            os.path.splitext(fn)[1] == '.png'
    ]

    for file in files_in_save_path:
        image = load_image(file)
        file_name = os.path.basename(file)

        ret, res = use_model(model, image, 224)

        print('{}: {} {}'.format(file_name, ret, class_names[ret]))

