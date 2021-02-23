import tensorflow as tf
import tensorflow_datasets as tfds

print('tf version:', tf.__version__)

# when ever we do a operation this will print what device was used
tf.debugging.set_log_device_placement(True)

BUFFER_SIZE = 50000

BATCH_SIZE_PER_REPLICA = 512


def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255

  return image, label


def do_train(mnist_train, mnist_test, info, strategy):

    batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    num_train_examples = info.splits['train'].num_examples
    num_test_examples = info.splits['test'].num_examples

    train_dataset = mnist_train.map_classifications(scale).cache().shuffle(BUFFER_SIZE).batch(batch_size)
    eval_dataset = mnist_test.map_classifications(scale).batch(batch_size)

    # Note: making the model, compiling the model, and maybe loading weights (idk) need to be in the same strategy scope

    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

    # some stuff not in the scope if needed

    with strategy.scope():
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])

    model.fit(train_dataset, epochs=12)

    return model.evaluate(eval_dataset)


if __name__ == '__main__':
    """
    Trains a simple NN with the MirroredStrategy. This will use multiple GPUs
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')

    datasets, info = tfds._load(name='mnist', with_info=True, as_supervised=True)

    mnist_train, mnist_test = datasets['train'], datasets['test']

    # This will use all 4 GPUs
    cross_device_ops = tf.distribute.HierarchicalCopyAllReduce(num_packs=len(gpus))
    # we have to specify this cross device op because the default is nccl which is not supported on windows.
    strategy1 = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)
    print('strategy1:', strategy1)
    val1 = do_train(mnist_train, mnist_test, info, strategy1)

    # This will use only One GPU
    strategy2 = tf.distribute.get_strategy()
    print('strategy2:', strategy2)
    val2 = do_train(mnist_train, mnist_test, info, strategy2)

    print(val1, val2)
