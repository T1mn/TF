import tensorflow as tf 
import numpy as np 

def add_layer(inputs,in_size,out_size,activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_biases = tf.matmul(inputs,Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_biases
    else:
        outputs = activation_function(Wx_plus_biases)
    return outputs

x_data = np.linspace(-1,1,50)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

l1 = add_layer(xs,1,15,activation_function=tf.nn.relu)
prediction = add_layer(l1,15,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for _ in range(2000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if _ % 200 == 0:
        print(sess.run(loss,feed_dict= {xs:x_data,ys:y_data}))


# 理解loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))所用
# tf.reduce_sum( ,reduction_indices=[1]) 横着相加，并转置

# print('y_data=',y_data)
# print('prediction=',sess.run(prediction,feed_dict={xs:x_data,ys:y_data}))
# print('y_data-prediction=',sess.run(y_data-prediction,feed_dict={xs:x_data,ys:y_data}))
# print('tf.square(ys - prediction)=',sess.run(tf.square(ys - prediction),feed_dict={xs:x_data,ys:y_data}))
# print('tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1])=',sess.run(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]),feed_dict={xs:x_data,ys:y_data}))