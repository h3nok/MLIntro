import os
import shutil

from unittest import TestCase
import dataset.trainingset as ts
import dataset.tfrecord_explorer as tfrexplr
from customers import CustomerObjectLookup
from logs import logger


class TestDatasetGenerator(TestCase):
    def test_small_batch(self):
        customer = CustomerObjectLookup['Vattenfall']()
        g = ts.DatasetGenerator.from_defaults(customer, 'database/database.ini', is_test=True)
        g.set_batches(20, 1, 5)
        out_dir = g.generate(force_new=True)

        # check to make sure it creates the file.
        assert os.path.exists(out_dir)
        assert os.path.exists(os.path.join(out_dir, 'frames_batch_0.txt'))
        assert os.path.exists(os.path.join(out_dir, 'vinet_train_00000-of-00005.tfrecord'))
        # we will assume that 1-3 exist too
        assert os.path.exists(os.path.join(out_dir, 'vinet_train_00004-of-00005.tfrecord'))
        assert os.path.getsize(os.path.join(out_dir, 'vinet_train_00000-of-00005.tfrecord')) > 0
        assert os.path.getsize(os.path.join(out_dir, 'vinet_train_00004-of-00005.tfrecord')) > 0

        assert ts.DatasetGenerator.count_batch_frames_in_file(os.path.join(out_dir, 'frames_batch_0.txt')) == 20

        # clean up
        shutil.rmtree(os.path.join(out_dir, '../'))

    def test_multiple_small_batches(self):
        customer = CustomerObjectLookup['Vattenfall']()
        g = ts.DatasetGenerator.from_defaults(customer, 'database/database.ini', is_test=True)
        g.set_batches(20, 3, 5)
        out_dir = g.generate(force_new=True)

        # check to make sure it creates the file.
        assert os.path.exists(out_dir)
        assert os.path.exists(os.path.join(out_dir, 'frames_batch_0.txt'))
        assert os.path.exists(os.path.join(out_dir, 'vinet_train_00000-of-00005.tfrecord'))
        # we will assume that 1-3 exist too
        assert os.path.exists(os.path.join(out_dir, 'vinet_train_00004-of-00005.tfrecord'))
        assert os.path.getsize(os.path.join(out_dir, 'vinet_train_00000-of-00005.tfrecord')) > 0
        assert os.path.getsize(os.path.join(out_dir, 'vinet_train_00004-of-00005.tfrecord')) > 0

        assert os.path.exists(os.path.join(out_dir, 'frames_batch_1.txt'))
        assert os.path.exists(os.path.join(out_dir, 'vinet_train_00005-of-00005.tfrecord'))
        # we will assume that 1-3 exist too
        assert os.path.exists(os.path.join(out_dir, 'vinet_train_00009-of-00005.tfrecord'))
        assert os.path.getsize(os.path.join(out_dir, 'vinet_train_00000-of-00005.tfrecord')) > 0
        assert os.path.getsize(os.path.join(out_dir, 'vinet_train_00009-of-00005.tfrecord')) > 0

        assert os.path.exists(os.path.join(out_dir, 'frames_batch_2.txt'))
        assert os.path.exists(os.path.join(out_dir, 'vinet_train_00010-of-00005.tfrecord'))
        # we will assume that 1-3 exist too
        assert os.path.exists(os.path.join(out_dir, 'vinet_train_00014-of-00005.tfrecord'))
        assert os.path.getsize(os.path.join(out_dir, 'vinet_train_00000-of-00005.tfrecord')) > 0
        assert os.path.getsize(os.path.join(out_dir, 'vinet_train_00014-of-00005.tfrecord')) > 0

        assert ts.DatasetGenerator.count_batch_frames_in_file(os.path.join(out_dir, 'frames_batch_0.txt')) == 20
        assert ts.DatasetGenerator.count_batch_frames_in_file(os.path.join(out_dir, 'frames_batch_1.txt')) == 20
        assert ts.DatasetGenerator.count_batch_frames_in_file(os.path.join(out_dir, 'frames_batch_2.txt')) == 20

        # clean up
        shutil.rmtree(os.path.join(out_dir, '../'))

    def test_small_batch_validation(self):
        customer = CustomerObjectLookup['Vattenfall']()
        g = ts.DatasetGenerator.from_defaults(customer, 'database/database.ini',
                                              split_name=ts.SplitName.Validation, is_test=True)
        g.set_batches(20, 1, 5)
        out_dir = g.generate(force_new=True)  # you only have to say Validation once

        # check to make sure it creates the file.
        assert os.path.exists(out_dir)
        assert os.path.exists(os.path.join(out_dir, 'frames_batch_0.txt'))
        assert os.path.exists(os.path.join(out_dir, 'vinet_validation_00000-of-00005.tfrecord'))
        # we will assume that 1-3 exist too
        assert os.path.exists(os.path.join(out_dir, 'vinet_validation_00004-of-00005.tfrecord'))
        assert os.path.getsize(os.path.join(out_dir, 'vinet_validation_00000-of-00005.tfrecord')) > 0
        assert os.path.getsize(os.path.join(out_dir, 'vinet_validation_00004-of-00005.tfrecord')) > 0

        assert ts.DatasetGenerator.count_batch_frames_in_file(os.path.join(out_dir, 'frames_batch_0.txt')) == 20
        assert tfrexplr.DatasetExplorer(out_dir).num_records() == 20  # we don't want augmentations

        # clean up
        shutil.rmtree(os.path.join(out_dir, '../'))

    def test_can_resume(self):
        customer = CustomerObjectLookup['Vattenfall']()
        g = ts.DatasetGenerator.from_defaults(customer, 'database/database.ini', is_test=True)
        g.set_batches(20, 1, 5)
        out_dir = g.generate(force_new=True)

        # we are generating another batch
        g = ts.DatasetGenerator.from_defaults(customer, 'database/database.ini', is_test=True)
        g.set_batches(20, 1, 5)  # get 20 more frames
        out_dir = g.generate(force_new=False)  # by default it is false

        # check to make sure it creates the files.
        assert os.path.exists(out_dir)
        assert os.path.exists(os.path.join(out_dir, 'frames_batch_1.txt'))
        assert os.path.exists(os.path.join(out_dir, 'vinet_train_00005-of-00005.tfrecord'))
        # we will assume that 6-8 exist too
        assert os.path.exists(os.path.join(out_dir, 'vinet_train_00009-of-00005.tfrecord'))
        assert os.path.getsize(os.path.join(out_dir, 'vinet_train_00005-of-00005.tfrecord')) > 0
        assert os.path.getsize(os.path.join(out_dir, 'vinet_train_00009-of-00005.tfrecord')) > 0

        assert ts.DatasetGenerator.count_batch_frames_in_file(os.path.join(out_dir, 'frames_batch_1.txt')) == 20

        # clean up
        shutil.rmtree(os.path.join(out_dir, '../'))
