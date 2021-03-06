"""
This module based on the series of following 3 articles
1) The keys of Deep Learning in 100 lines of code:
https://towardsdatascience.com/the-keys-of-deep-learning-in-100-lines-of-code-907398c76504
2) Coding a 2 layer neural network from scratch in Python:
https://towardsdatascience.com/coding-a-2-layer-neural-network-from-scratch-in-python-4dd022d19fd2
3) Predict malignancy in cancer tumors with your own neural network:
https://towardsdatascience.com/predict-malignancy-in-breast-cancer-tumors-with-your-own-neural-network-and-the-wisconsin-dataset-76271a05e941
"""

import numpy as np  # Linear algebra
import pandas as pd  # Preparing data
import matplotlib.pyplot as plt  # Result visualisation

# Normalising data
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools

# """
# Setups and initializes our network
# """
#
#
# class dlnet:
#     def __init__(self, x, y):
#         self.X = x  # Holds our input layer, the data we give to the network
#         self.Y = y  # Holds our desired output, which we will use to train the network
#         self.Yh = np.zeros((1, self.Y.shape[1]))  # Holds the output that our network produces
#
#         self.L = 2  # Holds the number of layers of our network, 2.
#         self.dims = [9, 15, 1]  # Next, we define the number of neurons or units in each of our layers
#
#         self.param = {}  # A Python dictionary that will hold the W and b parameters of each of the layers of network
#         self.ch = {}  # A cache variable, a Python dictionary that will hold some intermediate calculations
#         self.grad = {}
#
#         self.loss = []  # An array where we will store the loss value of the network every x iterations
#         self.lr = 0.003  # Our learning rate
#         self.sam = self.Y.shape[1]  # The number of training samples we have
#
#
# """
# Initializes with random values the parameters of our network
# """
#
#
# def nInit(self):
#     np.random.seed(1)
#     self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0])
#     self.param['b1'] = np.zeros((self.dims[1], 1))
#     self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1])
#     self.param['b2'] = np.zeros((self.dims[2], 1))
#     return
#
#
# """
# Relu and Sigmoid functions that will compute the non-linear activation functions at the output of each layer
# """
#
#
# def Sigmoid(Z):
#     return 1 / (1 + np.exp(-Z))
#
#
# def Relu(Z):
#     return np.max(0, Z)
#
#
# def forward(self):
#     Z1 = self.param['W1'].dot(self.X) + self.param['b1']
#     A1 = Relu(Z1)
#     self.ch['Z1'], self.ch['A1'] = Z1, A1
#
#     Z2 = self.param['W2'].dot(A1) + self.param['b2']
#     A2 = Sigmoid(Z2)
#     self.ch['Z2'], self.ch['A2'] = Z2, A2
#
#     self.Yh = A2
#     loss = self.nloss(A2)
#     return self.Yh, loss
#
#
# # squared_errors = (self.Yh - self.Y) ** 2
# # self.Loss = np.sum(squared_errors)
#
#
# """
# Classifies problems
# """
#
#
# def nloss(self, Yh):
#     loss = (1. / self.sam) * (-np.dot(self.Y, np.log(Yh).T)) - np.dot(1 - self.Y, np.log(1 - Yh).T)
#     return loss
#
#
# # dLoss_Yh = -(np.divide(self.Y, self.Yh ) - np.divide(1 - self.Y, 1 - self.Yh))
# # dLoss_Z2 = dLoss_Yh * dSigmoid(self.ch['Z2'])
# # dLoss_A1 = np.dot(self.param['W2'].T,dLoss_Z2)
# # dLoss_Z1 = dLoss_A1 * dRelu(self.ch['Z1'])
# # dLoss_W1 = 1./self.X.shape[1] * np.dot(dLoss_Z1,self.X.T)
#
#
# def dRelu(x):
#     x[x <= 0] = 0
#     x[x > 0] = 1
#     return x
#
#
# def dSigmoid(Z):
#     s = 1 / (1 + np.exp(-Z))
#     dZ = s * (1 - s)
#     return dZ
#
#
# def backward(self):
#     dLoss_Yh = -(np.divide(self.Y, self.Yh) - np.devide((1 - self.Y) - (1 - self.Yh)))
#
#     dLoss_Z2 = dLoss_Yh * dSigmoid(self.ch('Z2'))
#     dLoss_A1 = np.dot(self.param['W2'].T, dLoss_Z2)
#     dLoss_W2 = 1. / self.ch['A1'].shape[1] * np.dot(dLoss_Z2, self.ch['A1'].T)
#     dLoss_b2 = 1. / self.ch['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1], 1]))
#
#     dLoss_Z1 = dLoss_A1 * dRelu(self.ch['Z1'])
#     dLoss_A0 = np.dot(self.param["W1"].T, dLoss_Z1)
#     dLoss_W1 = 1. / self.X.shape[1] * np.dot(dLoss_Z1, self.X.T)
#     dLoss_b1 = 1. / self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1], 1]))
#
#     self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
#     self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
#     self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
#     self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2
#
#
# nn = dlnet(x, y)
# nn.gd(x, y, iter=15000)
#
#
# def gd(self, X, Y, iter=3000):
#     np.random.seed(1)
#
#     self.nInit()
#
#     for i in range(0, iter):
#         Yh, loss = self.forward()
#         self.backward()
#
#         if i % 500 == 0:
#             print("Cost after iteration %i: %f" % (i, loss))
#             self.loss.append(loss)
#
#     return


"""
Test the connection to the dataset
"""


df = pd.read_csv('wisconsin-cancer-dataset.csv', header=None)
print(df.head(5))


"""
Handle the dataset values to prepare them to analyse
"""