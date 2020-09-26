const tf = require('@tensorflow/tfjs');
// tf.compat.v1.InteractiveSession();

const rawData = [1., 2., 8., -1., 0., 5.5, 6., 13];
const spike = tf.Variable(false);
spike.initializer.run();

for (let i in rawData.length) {
  if (rawData[i] - rawData[i - 1] > 5) {
    updater = spike.assign(true);
    updater.eval();
  } else {
    spike.assign(false);
    console.log('Spike', spike.eval());
  }
}

sess.close();