import tensorflow as tf
import numpy as np

print('tf version:', tf.__version__)

# when ever we do a operation this will print what device was used
tf.debugging.set_log_device_placement(True)

if __name__ == '__main__':
    """
    We just want to test to see that we can use all 4 GPUs for 4 different tasks.
    I don't think any of this is concurrent. 
    """
    gpu_names = [gpu.name.replace('physical_device:', '') for gpu in tf.config.experimental.list_physical_devices('GPU')]  # I dont know why I have to do this
    print("GPUs Available: ", gpu_names)  # tf.config.experimental.list_physical_devices('GPU'))
    assert len(gpu_names) >= 4

    c = []
    # Create some tensors
    with tf.device(gpu_names[0]):  # with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c.append(tf.matmul(a, b))

    with tf.device(gpu_names[1]):  # with tf.device('/GPU:1'):
        a1 = tf.constant([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        b1 = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        c.append(tf.matmul(a1, b1))

    with tf.device(gpu_names[2]):  # with tf.device('/GPU:2'):
        ret = tf.add_n(c)

    with tf.device(gpu_names[3]):  # with tf.device('/GPU:3'):
        rett = tf.transpose(ret)

    print(c)
    print(ret)
    print(rett)

    assert np.allclose(rett.numpy(), np.array([[72.0, 163.0], [88.0, 204.0]]))

    # It should have printed that it used each GPU:
    """
    Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0
    Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:1
    Executing op AddN in device /job:localhost/replica:0/task:0/device:GPU:2
    Executing op Transpose in device /job:localhost/replica:0/task:0/device:GPU:3
    """
