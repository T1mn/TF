import tensorflow as tf 
import numpy as np

def add_layers(inputs,in_size,out_size,activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.zeros([1,out_size]) + 0.1
    Wx_plus_b = tf.matmul(Weights,inputs) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(outputs)
    return outputs



# 理解 [:,np.newaxis] 所用

# sess = tf.Session()
# print(np.linspace(-1,1,3)[:,np.newaxis])
# print(np.linspace(-1,1,3))
# print(sess.run(tf.random_normal([3,1])))
# print(sess.run(tf.matmul(tf.random_normal([1,10]),tf.random_normal([10,1]))))