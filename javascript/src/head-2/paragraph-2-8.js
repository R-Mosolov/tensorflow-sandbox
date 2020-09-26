const tf = require('@tensorflow/tfjs');

// // Define a model which simply adds two inputs.
// const model1 = tf.sequential();
// model1.add(tf.layers.dense({inputShape: [4], units: 3, activation: 'relu'}));
// model1.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
// model1.summary();
// model1.predict(tf.zeros([1, 4])).print();

// // Construct another model, reusing the second layer of `model1` while
// // not using the first layer of `model1`. Note that you cannot add the second
// // layer of `model` directly as the first layer of the new sequential model,
// // because doing so will lead to an error related to the fact that the layer
// // is not an input layer. Instead, you need to create an `inputLayer` and add
// // it to the new sequential model before adding the reused layer.
// const model2 = tf.sequential();
// // Use an inputShape that matches the input shape of `model1`'s second
// // layer.
// model2.add(tf.layers.inputLayer({inputShape: [3]}));
// model2.add(model1.layers[1]);
// model2.summary();
// model2.predict(tf.zeros([1, 3])).print();

tf.scalar(3.14).print();