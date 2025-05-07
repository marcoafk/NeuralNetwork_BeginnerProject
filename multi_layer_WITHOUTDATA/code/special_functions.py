import numpy as np


def sigmoid(Z):
    Z = np.clip(Z, -1000, 1000) #numerical instability
    return 1/(1 + np.exp(-Z))

def sigmoid_derivative(Z):
    s = sigmoid(Z)
    return s * (1 - s)

#------------------------------------------

def ReLu(Z):
    return np.maximum(Z, 0)

def ReLu_derivative(Z):
    return Z>0

#---------------------------------------------------------------------------------------------------------

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0)) #subtracting the max value to reduce numerical instability
    return expZ / expZ.sum(axis=0)

#----------------------------------------------------------------------------------------------------------

def convert_to_one_hot(Y, num_categories=10):
    num_samples = Y.shape[0] #num. of columns is the num. of examples
    one_hot = np.zeros((num_categories, num_samples))
    one_hot[Y, np.arange(0,num_samples)] = 1
    return one_hot


