import tensorflow as tf
import math

def sigmoid(x):
    return 1 / 1 + math.exp(-x);

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1, x2) # x1 * x2, arrage work

sess = tf.Session() 
print(sess.run(result)) # action, start the work

with tf.Session() as sess:
    output = sess.run(result)
    print(output)