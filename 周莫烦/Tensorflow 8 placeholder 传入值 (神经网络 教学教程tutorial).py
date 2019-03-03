import tensorflow as tf 

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

z = tf.multiply(x,y)

with tf.Session() as sess:
    print(sess.run(z,feed_dict ={x:[11.],y:[11.]}))