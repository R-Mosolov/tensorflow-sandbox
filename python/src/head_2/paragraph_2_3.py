import tensorflow as tf
import json
from math import pi

# Listing 2.5
number = tf.constant([1, 2])
negativeMatrix = tf.negative(number)
print(negativeMatrix)

# Custom 1
number1 = tf.constant([1, 2])
number2 = tf.constant([3, 4])
sum = tf.add(number1, number2)
print(sum)

multiply = tf.multiply(number1, number2)
print(multiply)

number3 = tf.constant([1, 2, 3])
number4 = tf.constant([4, 5, 6])
sum2 = tf.add(number3, number4)
print(sum2)


# Exercise 2.2
def calc_normal_distribution(arr):
    mean = 0.0
    sigma = 1.0

    result = tf.exp(tf.negative(tf.pow(x - mean, 2.0) /

    print(result)


with open('../../data/salaries.json') as json_file:
    arr = json.load(json_file)

calc_normal_distribution(arr)
