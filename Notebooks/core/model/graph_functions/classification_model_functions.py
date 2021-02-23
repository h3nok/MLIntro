import tensorflow as tf
import tensorflow.keras as tfk

distributed_strategy = {
    'default': tf.distribute.get_strategy(),
    'mirror_strategy_hierarchical_copy': tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(
            num_packs=len(tf.config.experimental.list_physical_devices('GPU')))),
    'mirror_strategy_nccl': tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce(  # you need a nccl driver which right now only exists for linux
            num_packs=len(tf.config.experimental.list_physical_devices('GPU')))),
    'mirror_strategy_reduction_to_one_device': tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.ReductionToOneDevice()),
    # note, 'one_device_cpu' is the same as 'default' when the computer does not have a GPU.
    'one_device_cpu': tf.distribute.OneDeviceStrategy(device='/CPU:0'),
    # CPU:0 represents all cores available to the process
    # for 'one_device_GPU', use 'default' when the computer has at least one GPU
}

optimizer = {
    'adam': tfk.optimizers.Adam,
    'sgd': tfk.optimizers.SGD,
    'rmsprop': tfk.optimizers.RMSprop
}

loss_function = {
    'categorical_cross_entropy': tfk.losses.CategoricalCrossentropy(),
    'sparse_categorical_cross_entropy': tfk.losses.SparseCategoricalCrossentropy(from_logits=True),
    'poisson': tfk.losses.Poisson(),
    'kl_divergence': tfk.losses.KLDivergence()
}

metrics = {
    'accuracy': 'accuracy',  # Note, the resulting tfk.metrics class is dependent on loss function and the input shape.
    'auc': tf.metrics.AUC(),
    'fp': tf.metrics.FalsePositives(),
    'fn': tf.metrics.FalseNegatives(),
    'precision': tf.metrics.Precision(),
    'recall': tf.metrics.Recall(),
    'categorical_accuracy': tfk.metrics.CategoricalAccuracy(),
    'root_mean_squared_error': tfk.metrics.RootMeanSquaredError()  # just for tests
}
