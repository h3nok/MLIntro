"""
This is used when you have a folder of vinet2 settings files. This script will run each network
with bench marks.
"""

import os

from training_config import NeuralNetConfig
from core.model.nn.neuralnet import NeuralNet

SETTINGS_FOLDER = r'./bench_settings'

if __name__ == '__main__':
    assert os.path.exists(SETTINGS_FOLDER)

    settings_files = [
        os.path.join(SETTINGS_FOLDER, fn)
        for fn in next(os.walk(SETTINGS_FOLDER))[2]
        if os.path.splitext(fn)[1] == '.ini'
    ]

    effnet_settings = [NeuralNetConfig(fp) for fp in settings_files]

    common_train_dataset_dir = None
    common_bench_path = None
    for settings_data in effnet_settings:
        if common_train_dataset_dir is None:
            common_train_dataset_dir = settings_data._train_dataset_dir
            common_bench_path = settings_data.bench_path
        else:
            assert common_train_dataset_dir == settings_data._train_dataset_dir, "All train_dataset_dir must be the same"
            assert common_bench_path == settings_data.bench_path, "All bench_path must be the same"

        assert settings_data.max_steps_per_epoch is None or \
               settings_data.max_steps_per_epoch > settings_data.log_every_n_steps
        assert settings_data.bench_timings

    for settings_data in effnet_settings:
        this_effnet = NeuralNet(settings_data)
        this_effnet.train()

