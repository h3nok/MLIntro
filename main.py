import tensorflow as tf
# Build the graph
A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
x = tf.constant([[0, 10], [0, 0.5]])
b = tf.constant([[1, -1]], dtype=tf.float32)


y = tf.add(tf.matmul(A, x), b, name="result")
# y = Ax + b
writer = tf.summary.create_file_writer("log/matmul")

writer.close()
