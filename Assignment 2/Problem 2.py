import pandas as pd
import numpy as np
import math
from sklearn import linear_model

def read_data():
    df = pd.read_csv('irisdata.csv')
    return df

def one_layer_network(w0, w1, w2, petal_length, petal_width):
    h_xj = w0 + w1 * petal_length + w2 * petal_width
    logistic = 1 / (1 + math.exp(-h_xj)) 
    return logistic

# data vectors in dataframe, pattern classes in list
def mean_square_error(data_vectors, w0, w1, w2, pattern_classes):
    n = data_vectors.shape[0]
    print("# of rows: ", n)
    data_vectors_list = data_vectors.values.tolist()
    pattern_classes_list = pattern_classes.tolist()
    temp_mse = 0
    for i in range(n):
        temp_mse = temp_mse + np.square(pattern_classes_list[i] - one_layer_network(w0, w1, w2, data_vectors_list[i][0], data_vectors_list[i][1]))
    mse = temp_mse/n
    return mse

def iris_customize():
    df = read_data()
    df = df[~df['species'].isin(['setosa'])]
    x = df.iloc[: , 2:4]
    # y = target values, last column 
    y = df.iloc[:, -1]
    y.replace(['versicolor','virginica'], [0,1], inplace=True)
    return x, y

def main():
    x, y = iris_customize()
    mse = mean_square_error(x, -2.8, 0.25, 1, y)
    print(mse)

if __name__ == "__main__":
    main()
