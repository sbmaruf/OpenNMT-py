import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
W = tf.get_variable(shape=[10000, 10000], name='w')
b = tf.get_variable(shape=[10000], name='b')
x = tf.get_variable(shape=[10000, 10000], name='x')
init = tf.global_variables_initializer()
with tf.Session() as (sess):
    sess.run(init)
    while True:
        p = tf.matmul(W, x) + b
        out = sess.run(p)
        print(out)

