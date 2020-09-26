import numpy as np
import tensorflow as tf

# Listing 1
revenue = 0
prices = [1, 2, 3]
amounts = [10, 20, 33]

for price, amount in zip(prices, amounts):
  revenue += price * amount
  print(revenue)

# Listing 2
revenue = np.dot(prices, amounts)
