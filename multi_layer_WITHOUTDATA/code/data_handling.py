import pandas as pd
import numpy as np


#--------------------Data setting------------------------------------------------
# data from: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

def load_data(train_path='../dataset/mnist_train.csv', test_path='../dataset/mnist_test.csv'):
    data_train = pd.read_csv(train_path)
    data_test = pd.read_csv(test_path)


    data_train = np.array(data_train)
    #m_train, n_train = data_train.shape
    
    data_test = np.array(data_test)
    #m_test, n_test = data_test.shape

    data_train = data_train.T
    data_test = data_test.T
    
    Y_train = data_train[0]
    X_train = data_train[1:] / 255.0
    
    Y_test = data_test[0]
    X_test = data_test[1:]/ 255.0
    
    
    return X_train, Y_train, X_test, Y_test

#-------------------------------------------------------------------------------