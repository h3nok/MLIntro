import os
import time

import matplotlib.pyplot as plt
import numpy as np
import regex as re
import tensorflow as tf

from dataset_utils import read_label_file


class DatasetExplorer:
    __dataset_dir = None
    __tfrecord_files = []

    __class_names = None

    def __init__(self, dataset_dir):
        """
        DatasetExplorer init function

        :param dataset_dir: the directory that contains the tfrecords.
        """
        self.__dataset_dir = dataset_dir

        self.__tfrecord_files = [
            os.path.join(self.__dataset_dir, fn)
            for fn in next(os.walk(self.__dataset_dir))[2]
            if os.path.splitext(fn)[1] == '.tfrecord'
        ]

        self.__class_names = read_label_file(self.__dataset_dir)


    @property
    def class_maps(self):
        return self.__class_names

    @staticmethod
    def __map_tfrecord(raw_example):
        parsed = tf.train.Example.FromString(raw_example.numpy())
        feature = parsed.features.feature

        raw_img = feature['image/encoded'].bytes_list.value[0]
        img_format = feature['image/format'].bytes_list.value[0]
        label = feature['image/class/label'].int64_list.value[0]
        shape = (feature['image/height'].int64_list.value[0], feature['image/width'].int64_list.value[0], 3)
        frame_id = feature['image/frame-id'].bytes_list.value[0]

        return raw_img, img_format, label, shape, frame_id

    def __to_classname_dict_entry(self, num, unique=False):
        if unique:
            return 'num_of_unique_' + self.__class_names[num].rstrip('\r')
        else:
            return 'num_of_' + self.__class_names[num].rstrip('\r')

    def __analyze_shard(self, file_path) -> (dict, int):

        def get_shard_id_from_filename(file_name) -> int:
            # Note, the format is: "vinet_<split_name>_xxxxx_of_xxxxx.tfrecord"
            # we want the first "xxxxx"
            return int(re.search(r'vinet_([a-z]+)_([0-9]{5})-of-([0-9]{5})\.tfrecord', file_name, 0).group(2))
            # return int(re.search(r'vinet_train_(\d+)-of-(\d+)\.tfrecord', file_name, 0).group(2))
            # return int(re.search(r'vinet_(\d+\w+)_\d+.tfrecord', file_name, 0).group(2))

        # make the base line structs
        # I know I don't need to do this but I think it help for knowing what is outputted.
        per_shard_struct = dict()
        for class_num in self.__class_names:
            per_shard_struct[self.__to_classname_dict_entry(class_num)] = 0
            per_shard_struct[self.__to_classname_dict_entry(class_num, True)] = 0
        per_shard_struct['total_frames'] = 0
        per_shard_struct['total_unique_frames'] = 0
        per_shard_struct['timestamp'] = time.time()  # this wont be kept as time.time()
        per_shard_struct['image_format'] = 'none'
        per_shard_struct['file_path'] = 'none'
        # Note, this shape is not the height vs width, it is max side vs min side. This is the case because we care
        # about the 'shape' and not doing this would give us a square for the average shape.
        per_shard_struct['average_image_shape'] = (0.0, 0.0)

        shard_id = get_shard_id_from_filename(file_path)

        tfrecord_dataset = tf.data.TFRecordDataset([file_path])
        record_iter = map(self.__map_tfrecord, iter(tfrecord_dataset))

        this_struct = per_shard_struct  # shallow copy

        # This just gets the timestamp of when the dataset was created.
        # it would be interesting to have the timestamp of when the images were taken.
        # This would require the DatasetGenerator to get timestamps from the database and
        # then put them in the tfrecords.
        this_struct['timestamp'] = os.path.getmtime(file_path)
        this_struct['file_path'] = file_path

        img_format = None

        for record in record_iter:
            if img_format is not None:
                assert img_format == record[1].decode("utf-8")
            img_format = record[1].decode("utf-8")

            image_shape = record[3]
            if this_struct['average_image_shape'] == (0.0, 0.0):
                this_struct['average_image_shape'] = (max(image_shape[0:2]), min(image_shape[0:2]))
            else:
                ais = this_struct['average_image_shape']
                totf = this_struct['total_frames']
                this_struct['average_image_shape'] = (
                    (ais[0] * totf + max(image_shape[0:2])) / (totf + 1),
                    (ais[1] * totf + min(image_shape[0:2])) / (totf + 1)
                )

            class_num = record[2]

            this_struct[self.__to_classname_dict_entry(class_num)] += 1
            this_struct['total_frames'] += 1

            if len(record[4]) == 36:
                # We can do this because record[4], or frame_id, is just a UUID if it wasn't augmented.
                this_struct[self.__to_classname_dict_entry(class_num, True)] += 1
                this_struct['total_unique_frames'] += 1

        this_struct['image_format'] = img_format

        return this_struct, shard_id

    def analyze(self) -> dict:
        """
        Reads the tfrecords and collects information on them.

        :return: A dict with all the information in it.
        """

        # make the base line structs
        # I know I don't need to do this but I think it help for knowing what is outputted.
        main_struct = dict()
        main_struct['shard_data'] = dict()
        for class_num in self.__class_names:
            main_struct[self.__to_classname_dict_entry(class_num)] = 0
            main_struct[self.__to_classname_dict_entry(class_num, True)] = 0
        main_struct['total_frames'] = 0
        main_struct['total_unique_frames'] = 0
        main_struct['average_image_shape'] = (0.0, 0.0)

        shard_data = [dict() for _ in range(len(self.__tfrecord_files))]

        # collect the data
        for file_path in self.__tfrecord_files:
            shard_struct, shard_id = self.__analyze_shard(file_path)
            shard_data[shard_id] = shard_struct

        # collect the data
        for shard_id in range(len(shard_data)):
            main_struct['shard_data'][shard_id] = shard_data[shard_id]

            for class_num in self.__class_names:
                main_struct[self.__to_classname_dict_entry(class_num)] += shard_data[shard_id][
                    self.__to_classname_dict_entry(class_num)]
                main_struct[self.__to_classname_dict_entry(class_num, True)] += shard_data[shard_id][
                    self.__to_classname_dict_entry(class_num, True)]

            main_struct['total_frames'] += shard_data[shard_id]['total_frames']
            main_struct['total_unique_frames'] += shard_data[shard_id]['total_unique_frames']
            # Right now we dont care about how many frames are in each shard for this average.
            main_struct['average_image_shape'] = (
                main_struct['average_image_shape'][0] + shard_data[shard_id]['average_image_shape'][0],
                main_struct['average_image_shape'][1] + shard_data[shard_id]['average_image_shape'][1]
            )
        main_struct['average_image_shape'] = (main_struct['average_image_shape'][0] / len(shard_data),
                                              main_struct['average_image_shape'][1] / len(shard_data))

        return main_struct

    @staticmethod
    def plot_class_distribution(analyze_data, show_plot=False, save_dir=None):
        """
        Takes the output of the `analyze` method and creates a bar graph plot of the distribution of classes in the\
        training set.

        @param analyze_data:  the dict to plot with (can be the whole dict or a 'shard_data' entry)
        @param show_plot:
        @param save_dir:
        :return: (labels, total_vals, unique_vals) if not show_plot

        """
        objects_tot = [entry for entry in analyze_data if 'num_of' in entry and 'unique' not in entry]
        objects_unique = [entry for entry in analyze_data if 'num_of_unique' in entry]

        labels = []

        total_vals = []
        unique_vals = []

        for obj in objects_tot:
            re_res = re.match('num_of_(.*)', obj)
            name = re_res.groups()[0]
            if 'num_of_unique_{}'.format(name) in objects_unique:
                labels.append(name)
                total_vals.append(analyze_data[obj])
                unique_vals.append(analyze_data['num_of_unique_{}'.format(name)])

        if not show_plot:
            return labels, total_vals, unique_vals

        # code from:
        # https://matplotlib.org/3.3.0/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, total_vals, width, label='Total')
        rects2 = ax.bar(x + width / 2, unique_vals, width, label='Unique')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('number of')
        ax.set_title('Numbers of classifications with and without augmentations')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()
        if save_dir:
            fig_file = os.path.join(save_dir, 'dist.png')
            print("Saving fig to {}".format(fig_file))
            plt.savefig(fig_file, dpi=300)

        if show_plot:
            plt.show()

    def get_records(self):
        """
        Just returns an iterator of the example data in a tuple.

        :return: iter of (raw_img, img_format, label, shape, frame_id)
        """
        tfrecord_dataset = tf.data.TFRecordDataset(self.__tfrecord_files)
        # return tfrecord_dataset.map(self.__map_tfrecord)
        return map(self.__map_tfrecord, iter(tfrecord_dataset))

    def num_records(self):
        """
        Get the number of records in the dataset

        :return:
        """
        converted_frames_file = os.path.join(self.__dataset_dir, 'converted_frames.txt')
        assert os.path.exists(converted_frames_file)

        count = 0

        with open(converted_frames_file, 'r') as f:
            while f.readline():
                count += 1
        return count


if __name__ == '__main__':
    this_exp = DatasetExplorer(
        r'C:\viNet_RnD\Datasets\Goldwind-CattleHill\TFRecords\viNet_2.8_Goldwind_3_Class_Clean\train')
    res = this_exp.analyze()
    DatasetExplorer.plot_class_distribution(res)
