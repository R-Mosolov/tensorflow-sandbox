import tensorflow as tf
sess = tf.compat.v1.InteractiveSession()

# Listing 2.9
raw_data = [1., 2., 8., -1., 0., 5.5, 6., 13]
spike = tf.Variable(False)
spike.initializer.run()

for i in range(1, len(raw_data)):
    if raw_data[i] - raw_data[i - 1] > 5:
        updater = tf.compat.v1.assign(spike, True)
        updater.eval()
    else:
        tf.compat.v1.assign(spike, False).eval()
    print('Spike', spike.eval())

sess.close()
