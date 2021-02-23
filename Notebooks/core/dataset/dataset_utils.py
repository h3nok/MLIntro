# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import, division, print_function

import glob
import imghdr
import os
import shutil
import sys
import tarfile
import random
from os import walk

import tensorflow.compat.v2 as tf
from six.moves import urllib
from tqdm import tqdm
from csv_generator import  CSVGenerator
LABELS_FILENAME = 'labels.txt'


def get_files(dir):
    if not os.path.exists(dir):
        raise ValueError("Directory \'{}\' not found".format(dir))

    if not os.path.isdir(dir):
        raise ValueError(
            "Supplied path is not directory, dir: \'{}\'.".format(dir))

    files = []

    for (dirpath, dirnames, filenames) in walk(dir):
        files.extend(filenames)
        break
    return files, len(files)


def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
    """Writes a file with the list of class names.

    Args:
      labels_to_class_names: A map of (integer) labels to class names.
      dataset_dir: The directory in which the labels file should be written.
      filename: The filename where the class names are written.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
    """Specifies whether or not the dataset directory contains a label map file.

    Args:
      dataset_dir: The directory in which the labels file is found.
      filename: The filename where the class names are written.

    Returns:
      `True` if the labels file exists and `False` otherwise.
    """
    return tf.io.gfile.exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    """Reads the labels file and returns a mapping from ID to class name.

    Args:
      dataset_dir: The directory in which the labels file is found.
      filename: The filename where the class names are written.

    Returns:
      A map from a label (integer) to class name.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.io.gfile.GFile(labels_filename, 'rb') as f:
        lines = f.read().decode()
    lines = lines.splitlines()
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index + 1:]
    return labels_to_class_names


def check_image_type(image_file):
    """[summary]

    Arguments:
        image_file {[type]} -- [description]
    """
    if not os.path.exists(image_file):
        raise ValueError("File not found {}".format(image_file))

    img_type = imghdr.what(image_file)

    # print("File: {}, Type: {}".format(os.path.basename(image_file), img_type))

    return img_type


def subsample(dataset_dir, output_dir, category, amount=1000):
    """
    Subsample some number of samples from dataset_dir 

    Arguments:
        dataset_dir {path} -- directory containing sample set_value
        output_dir {path} -- output directory 
        predicted_class {species} -- subsample for this species

    Keyword Arguments:
        amount {int} -- [number of examples to subsample] (get_as_tfexample: {1000})
    """
    output_dir = os.path.join(output_dir, category)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = sorted(glob.glob(os.path.join(dataset_dir, "*")))
    total_samples = len(files)
    print("TOTAL {}".format(total_samples))

    random_files = random.sample(range(0, total_samples), amount)

    print("Sampling {} files ".format(len(random_files)))

    for item in random_files:
        filepath = files[item]
        filename = os.path.basename(filepath)
        shutil.move(filepath, os.path.join(output_dir, filename))


def csv_to_dataframe(feature_description, output_dir='.'):
    assert os.path.exists(feature_description)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv = CSVGenerator(feature_description)
    return csv.dataframe

