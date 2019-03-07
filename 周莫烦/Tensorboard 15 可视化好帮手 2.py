import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('W'):
            Weight = tf.Variable(tf.random_normal([in_size,out_size]))
            tf.summary.histogram('weights',Weight)
        with tf.name_scope('b'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs,Weight),biases)
        with tf.name_scope('Ac_Function'):
            if activation_function is None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b)
            return outputs

# Define inputs_data
x_data = np.linspace(-7,3,300)[:,np.newaxis]
noises = np.random.normal(0,0.05,x_data.shape)
y_data = np.power(x_data+3,2) + noises

with tf.name_scope('input'):
    with tf.name_scope('input_x'):
        xs = tf.placeholder(tf.float32,[None,1])
    with tf.name_scope('input_y'):
        ys = tf.placeholder(tf.float32,[None,1])

# Define layer
with tf.name_scope('Layer1'):
    l1 = add_layer(xs,1,15,activation_function= tf.nn.relu)
with tf.name_scope('Layer2'):
    prediction = add_layer(l1,15,1,activation_function=None)

# Define loss&train
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - ys),reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.008).minimize(loss)

sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/",sess.graph)
sess.run(tf.global_variables_initializer())

# Graph
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data,1)

plt.ion()
plt.show()

# Run
for _ in range(10000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if _ % 500 == 0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,_)
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data,ys:y_data})
        lines = ax.plot(x_data,prediction_value,'r--',lw=1)
        plt.pause(0.001)
print('Done')
plt.pause(0)