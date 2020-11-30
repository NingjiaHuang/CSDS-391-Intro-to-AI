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
    return sigmoid

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

def summed_gradient(x, w0, w1, w2, y): 
    sigmoid_list = []
    n = len(x)
    error_list = []
    coefficient_list = []
    x = x.values.tolist()
    y = y.tolist()
    for i in range(len(x)): 
        sigmoid_list.append(one_layer_network(w0, w1, w2, x[i][0], x[i][1]))
        error_list.append(sigmoid_list[i] - y[i])
        coefficient_list.append(error_list[i] * sigmoid_list[i] * (1 - sigmoid_list[i]))
    # number of rows = number of rows in data vectors, number of columns = number of columns in data vector + 1 since 1 for bias coefficient
    temp_matrix = np.ones((len(x), len(x[0]) + 1))
    temp_matrix[:, 1:] = x
    sum_term = np.zeros((len(x), len(x[0]) + 1))
    for i in range(len(coefficient_list)):
        sum_term[i] = (2/n) * temp_matrix[i] * coefficient_list[i]
    return np.sum(sum_term, axis = 0)

def illustrate_summed_gradient(x, w0, w1, w2, y):
    num_of_iter = []
    for i in range(10000):
        num_of_iter.append(i+1)
        temp1, temp2, temp3 = summed_gradient(x, w0, w1, w2, y)
        w0 = w0 - 0.01 * temp1
        w1 = w1 - 0.01 * temp2
        w2 = w2 - 0.01 * temp3
    plot([0,-w0/w1],[-w0/w2,0])

def calc_slope_b(w0, w1, w2):
    slope = -(w0/w2)/(w0/w1)  
    b = -w0/w2
    return slope, b

def main():
    x, y = iris_customize()
    # w0 = -5.00080742
    # w1 = 0.49665344
    # w2 = 1.9989812
    # old_weight = []
    w0 = -5
    w1 = 0.5
    w2 = 2
    old_weight = []
    old_weight.append(w0)
    old_weight.append(w1)
    old_weight.append(w2)
    print("Old weight: ", old_weight)
    new_weight = old_weight - 0.01 * summed_gradient(x, w0, w1, w2, y)
    print("New weight: ", new_weight)
    # # illustrate_summed_gradient(x, w0, w1, w2, y)
    # # plot([0,-w0/w1],[-w0/w2,0])
    mse1 = mean_square_error(x, -1, 0.7, 2, y)
    print("Old MSE: ", mse1)
    print("Decision Boundary: ", calc_slope_b(-1, 0.7, 2))
    mse2 = mean_square_error(x, -1.00011645, 0.69953903, 1.99985973, y)
    print("Decision Boundary: ", calc_slope_b(-1.00011645, 0.69953903, 1.99985973))
    print("New MSE: ", mse2)
    illustrate_summed_gradient(x, w0, w1, w2, y)


    # df = read_data()
    # versicolor = df[df['species']=='versicolor']
    # virginica = df[df['species']=='virginica']
    # plt.plot(versicolor["petal_length"], versicolor["petal_width"], '*',label="versicolor", color = 'orange')
    # plt.plot(virginica["petal_length"], virginica["petal_width"], '+',label="virginica", color = 'blue')
    # plt.xlabel("petal length (cm)")
    # plt.ylabel("petal width (cm)")
    # plt.plot([0,-w0/w1],[-w0/w2,0], label='unadjusted decision boundary', color='purple')
    # plt.plot([0, 5.00080742/0.49665344], [5.00080742/1.9989812, 0], label='adjusted decision boundary', color='olive')
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()
