import os
import shutil

from unittest import TestCase
import dataset.tfrecord_explorer as tfexp
import dataset.trainingset as ts
from dataset.customer import viNetCustomer

import matplotlib.pyplot as plt


class TestDatasetExplorer(TestCase):
    def test_analyze(self):
        batch_size = 20
        num_batches = 1
        shards = 5

        customer = viNetCustomer.Vattenfall
        g = ts.DatasetGenerator.from_defaults(customer, 'database/database.ini', is_test=True)
        g.set_batches(batch_size, num_batches, shards)
        out_dir = g.generate()

        assert os.path.exists(out_dir)

        this_exp = tfexp.DatasetExplorer(out_dir)
        res = this_exp.analyze()

        assert len(res['shard_data']) == shards * num_batches
        assert res['shard_data'][0]['total_unique_frames'] == batch_size // shards
        assert res['shard_data'][shards * num_batches - 1]['total_unique_frames'] == batch_size // shards
        assert res['total_unique_frames'] == batch_size * num_batches
        assert res['shard_data'][0]['image_format'] == 'png'
        assert res['shard_data'][shards * num_batches - 1]['image_format'] == 'png'
        assert os.path.exists(res['shard_data'][0]['file_path'])
        assert os.path.exists(res['shard_data'][shards * num_batches - 1]['file_path'])

        # clean up
        shutil.rmtree(os.path.join(out_dir, '../'))

    def test(self):

        customer = viNetCustomer.Vattenfall
        g = ts.DatasetGenerator.from_defaults(customer, 'database/database.ini', is_test=True)
        g.set_batches(20, 1, 5)
        out_dir = g.generate()

        assert os.path.exists(out_dir)

        this_exp = tfexp.DatasetExplorer(out_dir)
        res = this_exp.analyze()

        this_exp.plot_class_distribution(res, show_plot=False)
        this_exp.plot_class_distribution(res['shard_data'][3], show_plot=False)

        print(res)

        # clean up
        shutil.rmtree(os.path.join(out_dir, '../'))
