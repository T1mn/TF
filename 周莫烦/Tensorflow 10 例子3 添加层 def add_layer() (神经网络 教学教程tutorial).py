import tensorflow as tf 

def add_layers(inputs,in_size,out_size,activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.zeros([1,out_size]) + 0.1
    Wx_plus_b = tf.matmul(Weights,inputs) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(outputs)
    return outputs