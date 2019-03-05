import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function = None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name = 'W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1,name = 'b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs,Weights) , biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

x_data = np.linspace(-10,10,100)[:,np.newaxis]
noises = np.random.normal(0,6,x_data.shape)
y_data = np.power(x_data,3) + noises + 0.5

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name = 'x_input')
    ys = tf.placeholder(tf.float32,[None,1],name = 'y_input')

l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,activation_function=None)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

sess = tf.Session()
writer = tf.summary.FileWriter("logs/",sess.graph)
sess.run(tf.global_variables_initializer())

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)

plt.ion()
plt.show()

for _ in range(100000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if _ % 500 == 0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data,ys:y_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=3)
        plt.pause(0.01)
plt.pause(0)
# 运行后，在终端输入 tensorboard -logdir=logs
# 在浏览器中打开: localhost:6006