import tensorflow as tf

# Simple hello world using TensorFlow

# Create a Constant op
# The op is added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
hello = tf.constant('Hello, TensorFlow!')

# Start tf session
sess = tf.Session()

a = tf.constant(2)
b = tf.constant(5)
c = a + b

print(c)

print(sess.run(c))

print(hello)

# Run the op
print (sess.run(hello))