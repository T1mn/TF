import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
# 1~10
mnist = input_data.read_data_sets('MNIST',one_hot=True)
# Define layer
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.3)
    Wx_plus_biases = tf.add(tf.matmul(inputs,Weights),biases)
    if activation_function is None:
        outputs = Wx_plus_biases
    else:
        outputs = activation_function(Wx_plus_biases)
    return outputs
# Define inputs
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
# Build layer
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)
# Train preparation
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction)
                                ,reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# Sess preparation
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# Compute accuracy
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    accuracy_equal = tf.equal(tf.argmax(v_ys,1),tf.argmax(y_pre,1))
    accuracy_cast = tf.reduce_mean(tf.cast(accuracy_equal,tf.float32))
    result = sess.run(accuracy_cast,feed_dict={xs:v_xs,ys:v_ys})
    return result

# Train
for _ in range(1000):
    batch_x,batch_y = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_x,ys:batch_y})
    if _ % 50 == 0 :
        print(compute_accuracy(mnist.test.images,mnist.test.labels))