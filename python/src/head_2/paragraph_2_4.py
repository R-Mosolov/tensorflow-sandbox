import tensorflow as tf

# Listing 2.6
x = tf.constant([[1., 2.]])
neg_op = tf.negative(x)

with tf.compat.v1.Session() as sess:
    result = sess.run(neg_op)
print(result)

# Listing 2.7
sess = tf.compat.v1.InteractiveSession()

x = tf.constant([[1., 2.]])
neg_op = tf.negative(x)

result = neg_op.numpy()
print(result)

sess.close()