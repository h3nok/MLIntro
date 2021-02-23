import math
import os
import random
import shutil
import sys
import ctypes
import itertools
import string
from typing import Union

import regex as re

from enum import Enum

from preprocessing.image_augmenter import ImageAugmenter
from dataset_utils import image_to_tfexample as im2tf, write_label_file
from customer.customer_base import CustomerBase
from customer.vattenfall import Vattenfall
from customer.e3 import E3
from customer.gwa import GWA
from customer.nextera import NextEra
from customer.totw import TOTW
from customer.avangrid import Avangrid
from customer.customers import CustomerObjectMap
import database.database_interface as dbi

import tensorflow as tf

from logs import logger

import image.image_compression as ic

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
DEFAULT_NUM_SHARDS = 5

_logger = logger.Logger(__name__, console=True).configure()

DEFAULT_OUTPUT_DIR_F = r"{}:\viNet_RnD\Datasets\{}\TFRecords"  # format string
DEFAULT_BATCH_SIZE = 200000


# None of these are needed, but I will keep them in for reference
# DEFAULT_NORTHAMERICA_DATASET_NAME = "viNet_2.5.1_NA_v2"
# DEFAULT_TOTW_DATASET_NAME = "viNet_TOTW_V_2.6_Training_v2_5_class_less_bald_aug"
# DEFAULT_VATTENFALL_DATASET_NAME = "viNet_2.7_Vattenfall_Proper_4_class_v2"
# DEFAULT_E3_DATASET_NAME = "viNet_2.5.1_NA_v2"
# DEFAULT_GOLDWIND_DATASET_NAME = "viNet_2.8_Goldwind_3_Class_Clean"
#
# DEFAULT_NORTHAMERICA_DATASET_TAG = "NA 5 Class v2.5 {} Final"
# DEFAULT_TOTW_DATASET_TAG = "NA 4 Class + Turbine v2.6.0 AllSites {}"
# DEFAULT_VATTENFALL_DATASET_TAG = "Vattenfall-Gotland Training V2 (4 class) - {}"
# DEFAULT_E3_DATASET_TAG = "E3_v2.2 {}"
# DEFAULT_GOlDWIND_DATASET_TAG = "Goldwind-Cattle Hill {} v1 (WTE, Raven,Other-Avian, Hawk-Falcon)"  # "Goldwind-Cattle Hill {} v2 (WTE, Raven,Hawk-Falcon)"
#
# DEFAULT_NORTHAMERICA_CLASSIFICATION_GROUP_NAME = 'NorthAmerica 5 Class'
# DEFAULT_TOTW_CLASSIFICATION_GROUP_NAME = 'NorthAmerica 4 Class + Turbines'
# DEFAULT_VATTENFALL_CLASSIFICATION_GROUP_NAME = 'Vattenfall Proper 4 Class'
# DEFAULT_E3_CLASSIFICATION_GROUP_NAME = 'E3 4 Class'
# DEFAULT_GOLDWIND_CLASSIFICATION_GROUP_NAME = 'Goldwind Proper V2'
#
# DEFAULT_TRAINING_STR = 'Training'
# DEFAULT_VALIDATION_STR = 'Validation'


class AugmenterLevels(Enum):
    NONE = 0
    X1 = 1
    X2 = 2
    X5 = 5
    X6 = 6
    X9 = 9


class ImageType(Enum):
    BoundingBox = 'boundingbox_image'
    Padded = 'padded_image'


class SplitName(Enum):
    Train = 'train'
    Validation = 'validation'


class DatasetGenerator:
    """
    Training/Validation set generator

    Generates a dataset based off of the customer. Supported customers are TOTW, NA, Vatterfall, E3.

    use `from_defaults` to use deafult setting for each customer like dataset_tag, dataset_name, and so on.

    Training set:
        will grab padded images (by default) from the database and augment them saving the original and augmented to
        the tfrecords.

    Validation set:
        will grab boundingbox images (by default) from the database and no augmentations will happen saving only what in
        the database to the tfrecords.
    """
    __customer = None

    __output_dir = None
    __batch_size = None
    __num_shards = None
    __dataset_name = None
    __split_name = None
    __dataset_tag = None
    __classification_group_name = None
    __image_type = None

    _ini_file_path = None

    _max_batches = None

    #### TESTING ####
    _use_jpg = False
    _jpg_quality = None

    #################

    def __init__(self, customer: CustomerBase, batch_size: int, output_dir: str, dataset_name: str,
                 split_name: str, num_shards: int, dataset_tag: str, classification_group_name: str,
                 ini_path: str = 'database.ini', image_type=ImageType.Padded):
        """
        init, all the settings needed except for the `use_jpg` setting. Use the `set_jpg` function if you want to the
        dataset to be converted to JPGs.

        :param customer:
        :param batch_size:
        :param output_dir:
        :param dataset_name:
        :param dataset_tag:
        :param split_name:
        :param num_shards:
        :param ini_path: optional
        :param image_type:
        """
        assert isinstance(customer, CustomerBase)
        self.__customer = customer
        self.__output_dir = output_dir
        self.__batch_size = batch_size
        self.__dataset_name = dataset_name
        self.__split_name = split_name
        self.__num_shards = num_shards
        self._ini_file_path = ini_path
        self.__dataset_tag = dataset_tag
        self.__image_type = image_type
        self.__classification_group_name = classification_group_name

    @classmethod
    def from_defaults(cls, customer: CustomerBase, ini_path: str = 'database.ini',
                      split_name: Union[str, SplitName] = 'train', is_test=False):
        """
        Will create the default values for customer, batch_size, output_dir, dataset_name, dataset_tag based on the
        customer type and split_name.

        :param split_name: if 'train' or 'validation'. Note, image_type is Padded if split name is train and BoundingBox
            if split name is validation. You can override it with the generate function
        :param customer:
        :param ini_path:
        :param is_test: If you are running test code. So the tests don't override the real datasets.
        :return:
        """
        assert isinstance(customer, CustomerBase)
        if isinstance(split_name, str):
            assert split_name in ['train', 'validation'], "Unknown dataset, please " \
                                                          "supply one from ['train', 'validation']"
        elif isinstance(split_name, SplitName):
            split_name = split_name.value

        # tag_type = DEFAULT_TRAINING_STR if split_name == 'train' else DEFAULT_VALIDATION_STR

        image_type = ImageType.Padded if split_name == 'train' else ImageType.BoundingBox

        drive = 'C'
        # This is a piece of code that I found on stack overflow that finds all drives on the system.
        drive_bitmask = ctypes.cdll.kernel32.GetLogicalDrives()
        drives = list(itertools.compress(string.ascii_uppercase,
                                         map(lambda x: ord(x) - ord('0'), bin(drive_bitmask)[:1:-1])))
        # This is mainly for testing. My laptop does not have an E drive
        if 'E' in drives:
            drive = 'E'

        if is_test:
            output_dir = DEFAULT_OUTPUT_DIR_F.format(drive,
                                                     "test_{}".format(customer.name.replace(" ", '').replace(",", "_")))
        else:
            output_dir = DEFAULT_OUTPUT_DIR_F.format(drive, customer.name.replace(" ", '').replace(",", "_"))

        dataset_name = customer.dataset_name
        dataset_tag = customer.training_dataset_tag if split_name == 'train' else customer.validation_dataset_tag
        classification_group_name = customer.classification_group_name

        return cls(customer=customer, batch_size=DEFAULT_BATCH_SIZE, output_dir=output_dir, dataset_name=dataset_name,
                   split_name=split_name, num_shards=DEFAULT_NUM_SHARDS, dataset_tag=dataset_tag, ini_path=ini_path,
                   image_type=image_type, classification_group_name=classification_group_name)

    def set_jpg(self, use_jpg, jpg_quality: int = None):
        assert (use_jpg and jpg_quality is not None) or not use_jpg
        self._use_jpg = use_jpg
        self._jpg_quality = jpg_quality
        if use_jpg:
            self.__dataset_name += '_jpg' + str(jpg_quality)

    def set_batches(self, batch_size, max_batches: int = None, num_shards=1):
        self.__batch_size = batch_size
        self._max_batches = max_batches
        self.__num_shards = num_shards

    def set_tag(self, tag):
        self.__dataset_tag = tag

    @staticmethod
    def __write_batch_frames(frames, truth, counts=None, output_dir='.', batch=0):
        with open(os.path.join(output_dir, 'frames_batch_%d.txt' % batch), 'w+') as f:
            f.write("Category\tCount\n")
            f.write("--------------------------------------------------\n\n")
            if counts:
                for key, value in counts.items():
                    f.write("{}\t{}\n\n".format(key, len(value)))
            f.write("--------------------------------------------------\n\n")
            for i in range(len(frames)):
                f.write("{}\t{}\n".format(frames[i], truth[i]))

    @staticmethod
    def count_batch_frames_in_file(file_path):
        # We can't just read the tfrecords because they contain augmented image. While we can filter them out
        # that just makes this function more inefficient.
        assert re.search(r'frames_batch_([0-9]+)\.txt', file_path, 0) is not None

        with open(file_path, 'r') as f:
            d = f.readline()
            assert d == "Category\tCount\n"
            d = f.readline()
            assert d == "--------------------------------------------------\n"

            while f.readline() != "--------------------------------------------------\n":
                pass
            d = f.readline()
            assert d == "\n"

            # Now we are at the frames
            count = 0

            while f.readline():
                count += 1

            return count

    @staticmethod
    def __get_tfrecord_filename(dataset_dir: str, split_name: str, shard_id: int, num_of_shards: int):
        """
        Generates file path for tfrecord with split_name and shard_id.

        :param dataset_dir: base dir
        :param split_name: 'train' or 'validation'
        :param shard_id:
        :param num_of_shards:
        :return: generated file path in format "dataset_dir\vinet_<split_name>_xxxxx_of_xxxxx.tfrecord"
        """
        output_filename = 'vinet_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, num_of_shards)

        return os.path.join(dataset_dir, output_filename)

    @staticmethod
    def __read_image_dims(image_data: bytes):
        """
        Will get the shape of the image that is encoded in `image_data`

        :param image_data: byte array of encoded image (png or jpg)
        :return: height, width
        """
        assert isinstance(image_data, bytes)
        assert tf.executing_eagerly()  # this is true as long as you don't use `disable_v2_behavior()` or a graph

        image_data = tf.constant(image_data)  # I'm not sure if this is needed
        image = tf.io.decode_image(image_data, channels=3)  # works for both png and jpg
        is_jpg = tf.io.is_jpeg(image_data)
        if is_jpg:
            image_format = b'jpg'
        else:
            image_format = b'png'

        assert len(image.shape) == 3
        assert image.shape[0] is not None
        assert image.shape[2] == 3

        return image.shape[0], image.shape[1], image_format

    def __convert(self, split_name: str, training_frames, truth_labels, class_names_to_ids, dataset_dir: str,
                  number_of_shards: int, shard_offset: int, augment_list=None):
        """
        Converts the given batch of training_frames to a TFRecord dataset.


        :param split_name: The name of the dataset, either 'train' or 'validation'.
        :param training_frames: A list of `(frame_id, groundtruth, frame_data)`.
        :param truth_labels:
        :param class_names_to_ids: A dictionary from class names (strings) to ids (integers).
        :param dataset_dir: The directory where the converted data-sets are stored.
        :param number_of_shards: number of tfrecords to split the batch into with no overlap.
        :param shard_offset: should be increased by `number_of_shards` before each call. Only used for file names.
        :param augment_list: dict of class name as key and augment type at value. types: x1, x2, x5, x6, x9
        :return:
        """

        augment = False
        if augment_list is not None and len(augment_list) > 0:
            augment = True
            augment_list = {x.lower(): augment_list[x] for x in list(augment_list)}

        frame_id = None
        class_name = None

        assert split_name in ['train', 'validation'], "Unknown dataset, please " \
                                                      "supply one from ['train', 'validation']"

        with open(os.path.join(dataset_dir, "converted_frames.txt"), 'a+') as f, open(
                os.path.join(dataset_dir, "bad_frames.txt"), 'a+') as f2:

            needs_rep = False
            for i in range(len(training_frames)):
                t = type(training_frames[i])
                if t is None:
                    _logger.warn("{} \t {} has no frame data.".format(truth_labels[i], t))
                    needs_rep = True
            if needs_rep:
                # We are basically iterating through the list twice, but we kind of have to log missing frames
                training_frames = list(filter(lambda elem: type(elem) is not None, training_frames))

            num_per_shard = int(math.ceil(len(training_frames) / float(number_of_shards)))
            # print(num_per_shard)

            assert shard_offset >= 0
            for shard_id in range(0 + shard_offset, number_of_shards + shard_offset):
                output_filename = self.__get_tfrecord_filename(dataset_dir, split_name, shard_id, number_of_shards)
                with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = (shard_id - shard_offset) * num_per_shard
                    end_ndx = min(((shard_id - shard_offset + 1) * num_per_shard), len(training_frames))

                    for i in range(start_ndx, end_ndx):
                        try:
                            sys.stdout.write(
                                '\r>> Converting image %d/%d shard %d' % (i + 1, len(training_frames), shard_id))
                            sys.stdout.flush()

                            image_data = bytes(training_frames[i][2])

                            ######## TESTING ########
                            # Used to test the effects of training with jpegs which out having to update the database
                            # TODO: Once testing is over, this block should be removed.
                            if self._use_jpg:
                                c = ic.Compressor.from_image_codex(image_data)
                                save_image_data = c.encode_to_jpg(self._jpg_quality).tobytes()
                            else:
                                save_image_data = image_data
                            #########################

                            frame_id = training_frames[i][0]
                            class_name = training_frames[i][1]
                            class_id = class_names_to_ids[class_name]
                            height, width, image_format = self.__read_image_dims(save_image_data)
                            example = im2tf(save_image_data, image_format, height, width, class_id, frame_id.encode())
                            tfrecord_writer.write(example.SerializeToString())
                            f.write(f"{frame_id}\t{class_name}\n")

                            if augment and class_name.lower() in augment_list and split_name == 'train':
                                aug = ImageAugmenter(image_data, False, ret_type=bytes, frame_id=frame_id)

                                if augment_list[class_name.lower()] == AugmenterLevels.X1:
                                    augmented_frames = aug.augment_x1()
                                elif augment_list[class_name.lower()] == AugmenterLevels.X2:
                                    augmented_frames = aug.augment_x2()
                                elif augment_list[class_name.lower()] == AugmenterLevels.X5:
                                    augmented_frames = aug.augment_x5()
                                elif augment_list[class_name.lower()] == AugmenterLevels.X6:
                                    augmented_frames = aug.augment_x6()
                                elif augment_list[class_name.lower()] == AugmenterLevels.X9:
                                    augmented_frames = aug.augment_x9()
                                else:
                                    augmented_frames = {}

                                for frame_id, image_data in augmented_frames.items():
                                    ######## TESTING ########
                                    # Used to test the effects of training with jpegs which out having to update the database
                                    # TODO: Once testing is over, this block should be removed.
                                    if self._use_jpg:
                                        c = ic.Compressor.from_image_codex(image_data)
                                        save_image_data = c.encode_to_jpg(self._jpg_quality).tobytes()
                                    else:
                                        save_image_data = image_data
                                    #########################
                                    # _logger.debug("Processing augmented frame, frame_id: {}".format(frame_id))
                                    height, width, image_format = self.__read_image_dims(save_image_data)

                                    example = im2tf(save_image_data, image_format, height, width, class_id,
                                                    frame_id.encode())
                                    tfrecord_writer.write(example.SerializeToString())
                                    f.write(f"{frame_id}\t{class_name}\n")

                        except BaseException as e:
                            f2.write(f"{frame_id}\t{class_name}\n")
                            _logger.error(
                                'Unable to process frame data, frame_id: {}, Error: {}'.format(frame_id, e))

        sys.stdout.write('\n')
        sys.stdout.flush()

    def __customer_specific_query(self, offset=0, exclude=None) -> str:
        """
        Gets a query that will grab frame_id, truth_classification, and padded_image from the database
        using self.__customer.

        :param exclude: object type to exclude.
        :return:
        """
        exclude_str = ""
        if exclude is not None:
            exclude_str = "AND grp_class.name <> '{}'".format(exclude)

        assert isinstance(self.__image_type, ImageType)

        if isinstance(self.__customer, Vattenfall):
            query = """
                    select
                        src.frame_id,
                        grp_class.name as truth_classification,
                        src.{} as image
                    from vinet.tagged_frames tag
                        join source.frame_data src on tag.frame_id = src.frame_id
                        join source.classifications grp_class ON tag.truth_classification_id = grp_class.id
                    where
                        tag.tag_id = (select tag_id from vinet.tags where name = '{}')
                    {}
                    order by frame_id
                    OFFSET {}
                    """.format(self.__image_type.value, self.__dataset_tag, exclude_str, offset)
        elif isinstance(self.__customer, GWA):
            query = """
                    select 
                        tf.frame_id as frame_id, 
                        grp_class.name as truth_classification, 
                        fd.{} as image
                    from vinet.tagged_frames tf
                        join "source".frame_data fd on fd.frame_id = tf.frame_id 
                        join viclassify.frame_results_latest frl on frl.frame_id = tf.frame_id 
                        join vinet.network_classification_group_entry grp ON grp.truth_classification_id = frl.classification_id 
                        join source.classifications grp_class ON grp.group_classification_id = grp_class.id
                    where tf.tag_id= (select tag_id from vinet.tags where name='{}') 
                    and    
                        grp.classification_group_id = (SELECT classification_group_id FROM vinet.network_classification_group
                            WHERE name = '{}') 
                    {}
                    ORDER BY
                        tf.frame_id
                    OFFSET {}
                    """.format(self.__image_type.value, self.__dataset_tag, self.__classification_group_name,
                               exclude_str, offset)
        else:
            query = """
                    SELECT
                        src.frame_id,
                        grp_class.name as truth_classification,
                        src.{} as image
                    FROM vinet.tagged_frames tag
                    JOIN source.frame_data src ON tag.frame_id = src.frame_id
                    JOIN vinet.network_classification_group_entry grp
                        ON tag.truth_classification_id = grp.truth_classification_id
                    JOIN source.classifications grp_class ON grp.group_classification_id = grp_class.id
                    WHERE
                        tag.tag_id = (SELECT tag_id FROM vinet.tags WHERE name = '{}')
                    AND
                    grp.classification_group_id = (
                        SELECT classification_group_id
                        FROM vinet.network_classification_group
                    WHERE name = '{}'
                    )
                    {}
                    --  Uncomment 2 lines below to limit data to data starting with given frame id
                    --  AND
                    --  tag.frame_id >= '076e1756-177c-4717-b811-6b0be0710021'

                    ORDER BY
                        tag.frame_id
                    OFFSET {}
                """.format(self.__image_type.value, self.__dataset_tag, self.__classification_group_name, exclude_str,
                           offset)

        # else:
        #     raise BaseException(
        #         "Function 'DatasetGenerator.__customer_specific_query' is not implemented for {0}".format(self.__customer))

        return query

    def __get_records(self, batch_size: int, offset=0, exclude=None):
        """
        A generator function that preforms a query and yields the result with limit `batch_size`.

        :param batch_size: size limit for yielded records
        :param exclude:
        :return: Iterator of lists of records
        """
        ini_path = 'database/database.ini'
        if self._ini_file_path is not None:
            ini_path = self._ini_file_path
        pgserver = dbi.PgsqlInterface()
        pgserver.connect(ini=ini_path, name="batch_convert")

        query = self.__customer_specific_query(offset, exclude)

        batch_counter = 0

        test = pgserver.cursor
        pgserver.cursor.itersize = batch_size
        pgserver.execute(query)

        while True:
            _logger.debug("Fetching {} frames, batch: {} ... ".format(pgserver.cursor.itersize, batch_counter + 1))

            records = pgserver.fetch_many(batch_size)
            if records is None:
                _logger.info("Trying again: Fetching {} frames, batch: {} ... ".format(
                    pgserver.cursor.itersize, batch_counter + 1))
                records = pgserver.fetch_many(batch_size)

            batch_counter += 1

            if records is not None and len(records) > 0:
                yield records
                if len(records) < batch_size:
                    break
            else:
                break

    def __get_offset(self, output_dir) -> (int, int, int):
        """
        get basic size information about the existing data like num frames, num batches, and num shards
        :param output_dir: training set dir
        :return: count_offset, batch_offset, shard_offset
        """
        file_paths = [
            os.path.join(output_dir, fn)
            for fn in next(os.walk(output_dir))[2]
            if re.match(r'frames_batch_[0-9]+\.txt', fn, 0)
        ]
        tfrecord_shard_nums = [
            int(re.search(r'vinet_([a-z]+)_([0-9]{5})-of-([0-9]{5})\.tfrecord', fn, 0).group(2))
            for fn in next(os.walk(output_dir))[2]
            if re.match(r'vinet_([a-z]+)_([0-9]{5})-of-([0-9]{5})\.tfrecord', fn, 0)
        ]
        count_offset = 0
        batch_offset = 0
        shard_offset = max(tfrecord_shard_nums) + 1 if len(tfrecord_shard_nums) > 0 else 0
        for fp in file_paths:
            search_obj = re.search(r'frames_batch_([0-9]+)\.txt', fp, 0)
            batch_num = int(search_obj.group(1))
            if batch_num + 1 > batch_offset:
                batch_offset = batch_num + 1
            count_offset += self.count_batch_frames_in_file(fp)

        return count_offset, batch_offset, shard_offset

    def __convert_with_named_cursor(self, set_type='train', augment_list=None, exclude=None, force_new=False):
        """
        Queries database and converts result to tfrecords.

        :param set_type: The name of the dataset, either 'train' or 'validation'.
        :param augment_list: dict of class name as key and augment type at value. augment types: x1, x2, x5, x6, x9
        :param exclude:
        :param force_new: By default if the training set dir has data in it then we will try to pick up where
        it left off by adding an offset to the query. This can turn that off.
        :return: output_dir
        """
        assert set_type in ['train', 'validation'], "Unknown dataset, please " \
                                                    "supply one from ['train', 'validation']"

        output_dir = os.path.join(self.__output_dir, self.__dataset_name, set_type)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            count_offset, batch_offset, shard_offset = (0, 0, 0)
        else:
            if force_new:
                _logger.info("Found existing tfrecords at {}, removing existing data.".format(output_dir))
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
                count_offset, batch_offset, shard_offset = (0, 0, 0)
            else:
                _logger.info("Found existing tfrecords at {}, attempting to continue where last generate left off."
                             .format(output_dir))
                count_offset, batch_offset, shard_offset = self.__get_offset(output_dir)
                if count_offset > 0:
                    _logger.info("Found {} existing samples starting with offset {}, batch offset {}, "
                                 "and shard offset {}.".format(count_offset, count_offset, batch_offset, shard_offset))

        batch_counter = 0

        for records in self.__get_records(self.__batch_size, count_offset, exclude):
            training_frames = dict()
            class_names = set()
            truth = []
            frame_ids = []
            frame_counter = 0

            for record in records:
                frame_id = record[0]
                groundtruth = record[1]
                frame_data = record[2]
                if exclude:
                    if exclude in groundtruth.lower():
                        _logger.info("Skipping frame, frame_id: {}, category: {}".format(frame_id, groundtruth))
                        continue

                if self.__customer.get_avian('GOLDEN_EAGLE') is not None and self.__customer.get_avian('YOUNG_GOLDEN') \
                        is not None and groundtruth.lower() == self.__customer.get_avian('YOUNG_GOLDEN').name.lower():
                    groundtruth = self.__customer.get_avian('GOLDEN_EAGLE').name
                if self.__customer.get_avian('BALD_EAGLE') is not None and self.__customer.get_avian('YOUNG_BALD') and \
                        groundtruth.lower() == self.__customer.get_avian('YOUNG_BALD').name.lower():
                    groundtruth = self.__customer.get_avian('BALD_EAGLE').name

                class_names.add(groundtruth)
                frame_ids.append(frame_id)
                truth.append(groundtruth)
                training_frames[frame_counter] = (frame_id, groundtruth, frame_data)
                frame_counter += 1

            assert len(training_frames) > 0
            class_names = sorted(class_names)
            number_of_samples = len(training_frames)
            class_names_to_ids = dict(zip(class_names, range(len(class_names))))

            random.seed(_RANDOM_SEED)
            random.shuffle(training_frames)

            self.__write_batch_frames(frame_ids, truth, output_dir=output_dir, batch=batch_counter + batch_offset)
            # First, convert the training and validation sets.
            self.__convert(set_type, training_frames, truth, class_names_to_ids, output_dir, self.__num_shards,
                           shard_offset=batch_counter * self.__num_shards + shard_offset,
                           augment_list=augment_list)

            # Finally, write the labels file:
            labels_to_class_names = dict(zip(range(len(class_names)), class_names))
            write_label_file(labels_to_class_names, output_dir)
            _logger.info('Finished converting batch: {}, frames: {}, '
                         'classes: {}.'.format(batch_counter + batch_offset + 1, number_of_samples, class_names))
            batch_counter += 1
            if self._max_batches is not None and batch_counter >= self._max_batches:
                break

        return output_dir

    def generate(self, force_new=False, image_type: ImageType = None):
        """
        Will start querying the database for records and then will convert them into tfrecords.
        Will end when query returns no records.

        :param force_new: By default if the training set dir has data in it then generate will try to pick up where
            it left off by adding an offset to the query. This can turn that off. Also, any existing data will be
            removed from the file system.
        :param image_type: override for the default
        :return: output_dir
        """
        # TODO: maybe move all the augmenter setting to a function call so that the caller can change them
        exclude = None

        # I would like to move the augment list to the customers, but I can't find a good way to do it.
        if isinstance(self.__customer, Avangrid) or isinstance(self.__customer, NextEra):
            # Note: This is weird and I don't really like it: `get_avian` will get the string name of the class from the
            # name I gave it (using regex.search). The reason I do this is to sort to assert that the name exists.
            # It doesn't cause any problems, but I don't think it is the best way to do it (best I could come up with).
            augment_these = {
                self.__customer.get_avian('bald_eagle').name: AugmenterLevels.X1,
                self.__customer.get_avian('GOLDEN_EAGLE').name: AugmenterLevels.X1,
                self.__customer.get_avian('TURKEY_VULTURE').name: AugmenterLevels.X1,
                self.__customer.get_avian('HAWK').name: AugmenterLevels.X1,
                self.__customer.get_avian('RAVEN').name: AugmenterLevels.X1
            }

        elif isinstance(self.__customer, TOTW):
            exclude = self.__customer.get_avian('TURBINE_DARK')
            augment_these = {
                self.__customer.get_avian('BALD_EAGLE').name: AugmenterLevels.X5,
                self.__customer.get_avian('RAVEN').name: AugmenterLevels.X2,
                self.__customer.get_avian('TURKEY_VULTURE').name: AugmenterLevels.X2,
                self.__customer.get_avian('HAWK').name: AugmenterLevels.X2,
                self.__customer.get_avian('TURBINE_GENERAL').name: AugmenterLevels.X1
            }

        elif isinstance(self.__customer, Vattenfall):
            augment_these = {
                self.__customer.get_avian('Gull').name: AugmenterLevels.X9,
                self.__customer.get_avian('Eagle_Or_Kite').name: AugmenterLevels.X5,
                self.__customer.get_avian('BUZZARD').name: AugmenterLevels.X9
            }

        elif isinstance(self.__customer, GWA):
            augment_these = {
                self.__customer.get_avian('RAVEN').name: AugmenterLevels.X9,
                self.__customer.get_avian('HAWK_FALCON').name: AugmenterLevels.X9,
                self.__customer.get_avian('WEDGE_TAILED_EAGLE').name: AugmenterLevels.NONE
            }

        elif isinstance(self.__customer, E3):
            augment_these = {}

        else:
            raise BaseException(
                "Function 'generate' is not implemented for {0}".format(self.__customer))

        augment_these = {x: augment_these[x] for x in augment_these.keys() if x is not None}

        if image_type is not None:
            self.__image_type = image_type

        return self.__convert_with_named_cursor(augment_list=augment_these, set_type=self.__split_name, exclude=exclude,
                                                force_new=force_new)

        # print_prof_data()


if __name__ == '__main__':
    # Note, I have only tested Vattenfall and Goldwind. The others should work just fine.
    customer = CustomerObjectMap['GWA']()
    g = DatasetGenerator.from_defaults(customer, 'database/database.ini', split_name=SplitName.Train)
    g.set_batches(200000, None, 1)
    g.set_jpg(False)
    g.generate()

    customer = CustomerObjectMap['GWA']()
    g = DatasetGenerator.from_defaults(customer, 'database/database.ini', split_name=SplitName.Validation)
    g.set_batches(200000, None, 1)
    g.set_jpg(False)
    g.generate()
