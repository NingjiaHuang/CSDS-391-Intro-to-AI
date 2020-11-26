import pandas as pd
import numpy as np

def read_data():
    df = pd.read_csv('irisdata.csv')
    return df

# neural network in list, data vector in list
def calc_h_xj(neural_network, data_vector,pattern_classes):
    h_xj = neural_network[0]
    for i in range(data_vector):
        h_xj = h_xj + data_vector[i] * neural_network[i+1]
    return h_xj

# data vectors in dataframe, neural network in list, pattern classes in list
def mean_square_error(data_vectors, neural_network, pattern_classes):
    n = data_vectors.shape[0]
    data_vectors_list = data_vectors.values.tolist()
    temp_mse = 0
    for i in range(n):
        temp_mse = temp_mse + np.square(np.subtract(data_vectors_list[i], calc_h_xj(neural_network, data_vectors_list[i])))
    mse = temp_mse/n
    return mse

def iris_customize():
    df = read_data()
    # X = feature values, all columns except the last column
    x = df.iloc[:,:-1]
    # y = target values, last column 
    y = df.iloc[:, -1]
    y.replace(['setosa','versicolor','virginica'], [1,2,3], inplace=True)

