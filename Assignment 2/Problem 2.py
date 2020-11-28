import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import math

def read_data():
    df = pd.read_csv('irisdata.csv')
    return df

def one_layer_network(w0, w1, w2, petal_length, petal_width):
    h_xj = w0 + w1 * petal_length + w2 * petal_width
    sigmoid = 1 / (1 + math.exp(-h_xj))
    if (sigmoid < 0.5):
	    return 0
    else:
	    return 1

# data vectors in dataframe, pattern classes in dataframe
def mean_square_error(data_vectors, w0, w1, w2, pattern_classes):
    n = data_vectors.shape[0]
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

def plot(x, y):
    df = read_data()
    versicolor = df[df['species']=='versicolor']
    virginica = df[df['species']=='virginica']
    plt.plot(versicolor["petal_length"], versicolor["petal_width"], '*',label="versicolor", color = 'orange')
    plt.plot(virginica["petal_length"], virginica["petal_width"], '+',label="virginica", color = 'blue')
    plt.xlabel("petal length (cm)")
    plt.ylabel("petal width (cm)")
    plt.plot(x,y)
    plt.legend()
    plt.show()

def main():
    x, y = iris_customize()
    w0 = -1
    w1 = 0.7
    w2 = 2
    mse = mean_square_error(x, w0, w1, w2, y)
    plot([0,-w0/w1],[-w0/w2,0])

if __name__ == "__main__":
    main()
