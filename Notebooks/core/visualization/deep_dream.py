import tensorflow as tf
import numpy as np
# from IPython.display import clear_output
from matplotlib import pyplot as plt

from training_config import NeuralNetConfig
from core.model.nn.neuralnet import NeuralNet

from deployment.wrap_frozen_graph import wrap_frozen_graph

IMAGE1_PATH = r'..\image\test_images\test_image5.png'
IMAGE2_PATH = r'..\image\test_images\test_image5.png'
IMAGE3_PATH = r'..\image\test_images\test_image3.png'

MODEL1_PATH = r'\\Hgh-d-22\Deployment\E3\Candidates\viNet_2.2_E3_24M.pb'
MODEL3_PATH = r'\\qa\tmp\1zjorquera\efficientnet\saves\take1\done.ckpt'

global_graph_def: any = None  # I was getting an error about a variable not being local so I moved this here.


# I am not sure if this was the issue.
# Most of this code is from: https://www.tensorflow.org/tutorials/generative/deepdream


def _calc_loss(model, img):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return tf.reduce_sum(losses)


def _random_roll(img, maxroll):
    # Randomly shift the image to avoid tiled boundaries.
    shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
    shift_down, shift_right = shift[0], shift[1]
    img_rolled = tf.roll(tf.roll(img, shift_right, axis=1), shift_down, axis=0)
    return shift_down, shift_right, img_rolled


def _gradient_ascent_with_roll(model, img, step_size, tile_size=128):
    shift_down, shift_right, img_rolled = _random_roll(img, tile_size)

    # Initialize the image gradients to zero.
    gradients = tf.zeros_like(img_rolled)

    # Skip the last tile, unless there's only one tile.
    xs = tf.range(0, img_rolled.shape[0], tile_size)[:-1]
    if not tf.cast(len(xs), bool):
        xs = tf.constant([0])
    ys = tf.range(0, img_rolled.shape[1], tile_size)[:-1]
    if not tf.cast(len(ys), bool):
        ys = tf.constant([0])

    for x in xs:
        for y in ys:
            # Calculate the gradients for this tile.
            with tf.GradientTape() as tape:
                # This needs gradients relative to `img_rolled`.
                # `GradientTape` only watches `tf.Variable`s by default.
                tape.watch(img_rolled)

                # Extract a tile out of the image.
                img_tile = img_rolled[x:x + tile_size, y:y + tile_size]
                loss = _calc_loss(model, img_tile)

            # Update the image gradients for this tile.
            gradients = gradients + tape.gradient(loss, img_rolled)

    # Undo the random shift applied to the image and its gradients.
    gradients = tf.roll(tf.roll(gradients, -shift_right, axis=1), -shift_down, axis=0)

    # Normalize the gradients.
    gradients /= tf.math.reduce_std(gradients) + 1e-8

    # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
    # You can update the image by directly adding the gradients (because they're the same shape!)
    img = img + gradients * step_size
    img = tf.clip_by_value(img, -1, 1)

    return loss, img


# I was getting errors when this was a member function
def _gradient_ascent(model, img, step_size):
    with tf.GradientTape() as tape:
        # This needs gradients relative to `img`
        # `GradientTape` only watches `tf.Variable`s by default
        tape.watch(img)
        loss = _calc_loss(model, img)

    # Calculate the gradient of the loss with respect to the pixels of the input image.
    gradients = tape.gradient(loss, img)

    # Normalize the gradients.
    gradients /= tf.math.reduce_std(gradients) + 1e-8

    # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
    # You can update the image by directly adding the gradients (because they're the same shape!)
    img = img + gradients * step_size
    img = tf.clip_by_value(img, -1, 1)
    return loss, img


class DeepDreamVisualizer:
    __dream_model = None
    __type = None
    __net = None

    def __init__(self, dream_model, net_type, net=None):
        """
        Use a from_x classmethod not this

        :param dream_model:
        :param net_type:
        """
        self.__dream_model = dream_model
        self.__type = net_type
        self.__net = net

    @classmethod
    def from_vinet_tfv1(cls, model_file_path):
        """
        loads a .pb file containing the graph_def info. Must be InceptionV2.

        :param model_file_path:
        :return:
        """
        # import tensorflow.compat.v1 as tf
        # tf.disable_v2_behavior()

        with tf.io.gfile.GFile(model_file_path, 'rb') as f:
            global_graph_def = tf.compat.v1.GraphDef()
            global_graph_def.ParseFromString(f.read())

        # graph = tf.Graph()
        # sess = tf.InteractiveSession(graph=graph)
        # t_input = tf.placeholder(np.float32, name='input')
        # imagenet_mean = 117.0
        # t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
        # tf.import_graph_def(global_graph_def, {'input': t_input}, )
        #
        # layers = [op.name for op in graph.get_operations() if op.type ==
        #           'Conv2D' and 'import/' in op.name]
        # print(layers)
        # exit(0)

        inception_func = wrap_frozen_graph(
            global_graph_def, inputs='input:0',
            outputs=['InceptionV2/InceptionV2/Mixed_3c/Branch_1/Conv2d_0b_3x3/Conv2D:0',  # I sort of picked these at
                     'InceptionV2/InceptionV2/Mixed_5c/Branch_1/Conv2d_0b_3x3/Conv2D:0'])  # random. TODO: try more

        return cls(inception_func, 'vinet_tfv1')

    @classmethod
    def from_effnet(cls, weight_file_path):
        """

        :param weight_file_path:
        :return:
        """
        effnet = NeuralNet(NeuralNetConfig(r'../model/nn/vinet.settings.gpu.efn.ini'))
        effnet.load_weights(weight_file_path)
        base_model = effnet.base_model
        print(effnet.summary)
        # Maximize the activations of these layers
        layer_names = ['top_activation', 'block6d_activation']
        layers = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

        return cls(dream_model, 'effnet', effnet)

    @classmethod
    def from_inceptionv3(cls):
        """
        Used for testing. Loads an inceptionv3 network with imagenet weights

        :return:
        """
        base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        base_model.summary()
        # Maximize the activations of these layers
        layer_names = ['mixed3', 'mixed5']
        layers = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

        return cls(dream_model, 'inceptionv3')

    def to_input_image(self, img_path):
        """
        Converts an image to what the network takes.
        Uses the network type with creating DeepDreamVisualizer class.

        :param img_path:
        :return:
        """
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=None)
        img = tf.constant(np.array(img))

        if self.__type == 'inceptionv3':
            img = tf.cast(img, tf.float32)
            img = tf.keras.applications.inception_v3.preprocess_input(img)
        elif self.__type == 'effnet':
            img = self.__net.resize_input_image(img)
        elif self.__type == 'vinet_tfv1':
            img = tf.cast(img, tf.float32)
            img = tf.keras.applications.inception_v3.preprocess_input(img)
        else:
            raise Exception

        return img

    def show_image(self, img):
        """
        shows the image

        :param img:
        :return:
        """
        if self.__type == 'inceptionv3' or self.__type == 'vinet_tfv1':
            img = 255 * (img + 1.0) / 2.0
        elif self.__type == 'effnet':
            img = 255 * img
        else:
            raise Exception

        img = tf.cast(img, tf.uint8)

        plt.figure(figsize=(12, 12))
        plt.grid(False)
        plt.axis('off')
        plt.imshow(img)
        plt.show()

    def visualize(self, input_image, use_tiles=False, v=False):
        """
        Runs the basic deep dream algorithm

        :param input_image:
        :param use_tiles: Enables tile mode
        :param v:
        :return:
        """
        img = input_image

        if v is True:
            show_image = (img + 1.0) / 2.0
            self.show_image(show_image)

        step_size = 0.001
        steps = 1000

        for step in range(steps):
            if use_tiles:
                loss, img = _gradient_ascent_with_roll(self.__dream_model, img, step_size)
            else:
                loss, img = _gradient_ascent(self.__dream_model, img, step_size)
            if (step + 1) % 100 == 0 and v is True:
                show_image = (img + 1.0) / 2.0
                self.show_image(show_image)

        if v is True:
            show_image = (img + 1.0) / 2.0
            self.show_image(show_image)
        return img

    def visualize_with_octaves(self, img, steps_per_octave=1000, step_size=0.0001,
                               octaves=range(-2, 3), octave_scale=1.3, with_tiles=False, v=False):
        """
        Runs the deep dream algorithm with octaves

        :param img:
        :param steps_per_octave:
        :param step_size:
        :param octaves:
        :param octave_scale:
        :param with_tiles:
        :param v:
        :return:
        """
        assert self.__type != 'effnet'

        base_shape = tf.shape(img)

        initial_shape = img.shape[:-1]
        img = tf.image.resize(img, initial_shape)
        for octave in octaves:
            # Scale the image based on the octave
            new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32) * (octave_scale ** octave)
            img = tf.image.resize(img, tf.cast(new_size, tf.int32))

            img = self.visualize(img, use_tiles=with_tiles, v=v)

        return img


if __name__ == '__main__':
    # v = DeepDreamVisualizer.from_inceptionv3()
    # v = DeepDreamVisualizer.from_vinet_tfv1(MODEL1_PATH)
    v = DeepDreamVisualizer.from_effnet(MODEL3_PATH)
    # img = v.visualize_with_octaves(v.to_input_image(IMAGE3_PATH), with_tiles=False, v=False, octave_scale=1.1)
    img = v.visualize(v.to_input_image(IMAGE2_PATH), use_tiles=False, v=False)
    show_image = (img + 1.0) / 2.0
    v.show_image(show_image)
