from datatype.frame_datum import FrameDatum
import os
from common.audit import path_exists
import tensorflow as tf
from datatype.tfrecord import TFRecordDecoder
from logs import logger
from common.profiler import Timer


class TrainingSetGenerator:
    def __init__(self, output_dir, customer):
        """

        @param output_dir:
        @param customer:
        """

        self.__labels = set()
        self.__labels_encoding = None
        self.__output_dir = output_dir
        if not path_exists(self.__output_dir):
            os.makedirs(self.__output_dir)
        self.__customer = customer
        self.__filename = None
        self.__writer = None
        self.__frames = []
        self._logger = logger.Logger(self.__class__.__name__, console=True).configure()
        self.__converted_frames = open(os.path.join(output_dir, 'converted_frames.txt'), 'a+')
        self.__bad_frames = open(os.path.join(output_dir, 'corrupt_frames.txt'), 'a+')

    def __enter__(self, output_dir=None, customer=None):
        """

        @param output_dir:
        @param customer:
        @return:
        """
        return self

    def _write_labels(self):
        output_file = os.path.join(self.__output_dir, 'labels.txt')
        file_handle = open(output_file, 'w+')
        file_handle.writelines([f"{value}:{key}\n" for key, value in self.__labels_encoding.items()])
        file_handle.close()

    @Timer.function_timer
    def convert(self, frames, batch_index, total_batch=100, augment_cmd=None) -> str:
        """
        Converts a batch of FrameDatum to TFRecord

        @param augment_cmd: dict of groundtruth and  augment cmd. {'Turkey-Vulture': 'x9'}
        @param frames:
        @param batch_index:
        @param total_batch:
        @return:
        """
        assert len(frames) > 0
        self.__filename = os.path.join(self.__output_dir,
                                       "vinet_train_%05d-of-%05d.tfrecord" % (batch_index, total_batch))
        self.__writer = tf.io.TFRecordWriter(self.__filename)
        augmented_frames = []
        for sample in frames:
            frame = FrameDatum(frame_data=sample[2], frame_id=sample[0], groundtruth=sample[1])
            self.__frames.append(frame)
            self.__labels.add(frame.groundtruth)

            # Handle augmentation
            if augment_cmd:
                assert isinstance(augment_cmd, dict)
                if frame.groundtruth in augment_cmd.keys():
                    augmented_frames = frame.augment(augment_cmd=augment_cmd[frame.groundtruth])

        self.__frames += augmented_frames
        self.__labels_encoding = dict(zip(self.__labels, range(len(sorted(self.__labels)))))

        assert len(self.__frames) == len(frames) + len(augmented_frames)

        for frame in self.__frames:
            label_id = self.__labels_encoding[frame.groundtruth]
            tfex = frame.to_tfexample(label_code=label_id)
            if not tfex:
                self.__bad_frames.write(f"{frame.frame_id} - {frame.groundtruth}\n")
                continue
            self.__writer.write(tfex.SerializeToString())
            self.__converted_frames.write(f"{frame.frame_id} - {frame.groundtruth}\n")

        self._write_labels()
        self._verify()
        self.__frames.clear()

        return self.__filename

    def _verify(self):
        """
        Decode each shard ensure
        @return:
        """
        decode = TFRecordDecoder(self.__filename)
        dataset = decode.decode_v2()
        count = 0
        for _ in dataset:
            count += 1
        if len(self.__frames) != count:
            self._logger.warn("Not all frames were converted or some frames were augmented."
                              "Check output directory for detail ")

    def __exit__(self, output_dir=None, customer=None, batch=None, batch_index=None):
        """

        @param output_dir:
        @param customer:
        @param batch:
        @param batch_index:
        @return:
        """
        self.__converted_frames.flush()
        self.__converted_frames.close()
        self.__frames.clear()
        self.__labels.clear()

        del self.__labels
        del self.__frames
        del self.__converted_frames
        del self.__writer
