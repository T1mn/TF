import tensorflow as tf
v = tf.Variable(1,name='++')
state = tf.Variable(0,name = 'state')

value = tf.add(v,state)
update = tf.assign(state,value)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for _ in range(100):
        print()