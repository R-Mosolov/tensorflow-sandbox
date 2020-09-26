import tensorflow as tf
import numpy as np

# Listing 2.3
m1 = [
  [1.0, 2.0],
  [3.0, 4.0]
]

m2 = np.array(
  [
    [1.0, 2.0],
    [3.0, 4.0]
  ],
  dtype=np.float32
)

m3 = tf.constant(
  [
    [1.0, 2.0],
    [3.0, 4.0]
  ]
)

print(type(m1))
print(type(m2))
print(type(m3))

t1 = tf.convert_to_tensor(m1, dtype=tf.float32)
t2 = tf.convert_to_tensor(m2, dtype=tf.float32)
t3 = tf.convert_to_tensor(m3, dtype=tf.float32)

print(type(t1))
print(type(t2))
print(type(t3))

# Listing 2.4
m1 = tf.constant([[1., 2.]])
m2 = tf.constant([[1], [2]])
m3 = tf.constant([ [[1, 2],
                  [3, 4],
                  [5, 6]],
                  [[7, 8],
                  [9, 10],
                  [11, 12]] ])

print(m1)
print(m2)
print(m3)

# Exercise 2.1
print('# Exercise 2.1')
result = tf.ones(shape=(500, 500)) * 0.5
print(result)