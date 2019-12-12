import tensorflow as tf
import numpy as np

g = tf.Graph()
with g.as_default():
   x = tf.placeholder(dtype=tf.float32,
            shape=(None, 4, 2),
            name='input_x')

   x2 = tf.reshape(x, shape=(-1, 8),
           name='x2')

   ## calculate the sum of each column, Summe der Spalten berechnen
   xsum = tf.reduce_mean(x2, axis=0, name='col_sum')

   ## calculate the mean of each column, Mittelwerte der Spalten berechnen
   xmean = tf.reduce_mean(x2, axis=0, name='col_mean')

with tf.Session(graph=g) as sess:
   x_array = np.arange(32).reshape(4, 4, 2)

   print('input shape: ', x_array.shape)
   print('Reshaped:\n',
      sess.run(x2, feed_dict={x:x_array}))
   print('Column Sums:\n',
      sess.run(xsum, feed_dict={x:x_array}))
   print('Column Means:\n',
      sess.run(xmean, feed_dict={x:x_array}))
