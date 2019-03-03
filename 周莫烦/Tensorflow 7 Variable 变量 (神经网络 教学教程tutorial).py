import tensorflow as tf 

state = tf.Variable(0,name = 'v')
one = tf.constant(1,name = 'one')

add = tf.add(state,one)
new_value = tf.assign(state,add)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for _ in range(100):
        print(sess.run(new_value))
