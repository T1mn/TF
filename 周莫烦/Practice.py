import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
# AddLay
def add_layer (inputs,in_size,out_size,activation_function=None):
    with tf.name_scope('Layer'):
        with tf.name_scope('W'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]))
        with tf.name_scope('b'):
            baises = tf.Variable(tf.zeros([1,out_size])+0.1)
        Wx_plus_baises = tf.add(tf.matmul(inputs,Weights),baises)
        if activation_function is None:
            outputs = Wx_plus_baises
        else:
            outputs = activation_function(Wx_plus_baises)
        return outputs

# Data
x_data = np.linspace(-3,3,600)[:,np.newaxis]
noises = np.random.normal(0,0.1,x_data.shape)
y_data = np.square(x_data)+2*x_data+0.3+noises

with tf.name_scope('INPUTS'):
    with tf.name_scope('input_x'):
        xs = tf.placeholder(tf.float32,[None,1])
    with tf.name_scope('input_y'):
        ys = tf.placeholder(tf.float32,[None,1])

# Layer
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
l2 = add_layer(l1,10,1,activation_function=None)

# Train
with tf.name_scope('LOSS'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-l2),reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.03).minimize(loss)

# Run
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data,1)
plt.ion()
plt.show()

sess = tf.Session()

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/",sess.graph)# Tensorboard
sess.run(tf.global_variables_initializer())




for _ in range (1111):
    sess.run(train,feed_dict={xs:x_data,ys:y_data})
    if _ % 10 == 0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        prediction = sess.run(l2,feed_dict={xs:x_data,ys:y_data})
        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,_)
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        ax.lines = ax.plot(x_data,prediction,'r-',lw =1)
        plt.pause(0.00001)