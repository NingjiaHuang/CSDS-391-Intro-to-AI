import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import math
import numpy.linalg as la

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
        sum_term[i] = temp_matrix[i] * coefficient_list[i]
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

def square_error(data_vectors, w0, w1, w2, pattern_classes): 
    pattern_classes_list = pattern_classes.tolist()
    data_vectors_list = data_vectors.values.tolist()
    error = 0
    n = len(pattern_classes_list)
    for i in range(n):
        error = error + np.square(pattern_classes_list[i] - one_layer_network(w0, w1, w2, data_vectors_list[i][0], data_vectors_list[i][1]))
    return error * 0.5

def armijo_updating(a, b, x, y, w0, w1, w2):
    step_size = a
    gradient = summed_gradient(x, w0, w1, w2, y)
    while square_error(x, w0 - (step_size * gradient[0]), w1 - (step_size * gradient[1]), w2 - (step_size * gradient[2]), y) > square_error(x, w0, w1, w2, y) - (0.5 * step_size * la.norm(gradient) ** 2):
        step_size = step_size * b
    return step_size

def gradient_descent(a, b, max_iter, prec, x, w0, w1, w2, y): 
    precision = prec # mse we would like to reach
    max_iters = max_iter # max number of iterations
    iteration_counter = 0
    step_size = 0 # using armijo to update
    return_w = []
    current_w = [w0,w1,w2]
    current_mse = mean_square_error(x, w0, w1, w2, y)
    current_sum_g = summed_gradient(x, w0, w1, w2, y)
    mse_list = []
    mse_list.append(mean_square_error(x, w0, w1, w2, y))
    improv_checker = 1 # check whether performed better
    # if current mse > the precision we defined and the number of iteration does not exceed the max iteration
    # execute the gradient descent
    while mean_square_error(x, w0, w1, w2, y) > precision and iteration_counter < max_iters:
        if improv_checker > 0:
            return_w = current_w
        iteration_counter += 1
        temp0, temp1, temp2 = summed_gradient(x, w0, w1, w2, y)
        step_size = armijo_updating(a, b, x, y, w0, w1, w2)
        w0 = w0 -  step_size * temp0
        w1 = w1 -  step_size * temp1
        w2 = w2 - step_size * temp2
        current_w = [w0,w1,w2]
        next_mse = mean_square_error(x, w0, w1, w2, y)
        mse_list.append(next_mse)
        improv_checker = current_mse - next_mse
        current_mse = next_mse
        current_sum_g = summed_gradient(x, w0, w1, w2, y)
        if improv_checker > 0:
            return_w = current_w
    return return_w, mse_list

def plot_gradient_descent(a, b, x, w0, w1, w2, y):
    df = read_data()
    versicolor = df[df['species']=='versicolor']
    virginica = df[df['species']=='virginica']
    plt.plot(versicolor["petal_length"], versicolor["petal_width"], '*',label="versicolor", color = 'orange')
    plt.plot(virginica["petal_length"], virginica["petal_width"], '+',label="virginica", color = 'blue')
    # plot the decision boundary
    plt.xlabel("petal length (cm)")
    plt.ylabel("petal width (cm)")
    plt.plot([0, -w0/w1],[-w0/w2, 0], label="initial boundary", color='purple')
    mid_w, mse_list = gradient_descent(1, 0.5, 10000, 0.001, x, w0, w1, w2, y)
    mid_w0, mid_w1, mid_w2 = mid_w[0], mid_w[1], mid_w[2]
    plt.plot([0, -mid_w0/mid_w1],[-mid_w0/mid_w2, 0], label="middle boundary", color='skyblue')
    fin_w, mse_list = gradient_descent(1, 0.5, 20000, 0.001, x, w0, w1, w2, y)
    fin_w0, fin_w1, fin_w2 = fin_w[0], fin_w[1], fin_w[2]
    plt.plot([0, -fin_w0/fin_w1],[-fin_w0/fin_w2, 0], label="final boundary", color='olive')
    # plot the change of objective function
    # plt.plot(mse_list)
    # plt.xlabel("Number of Iterations")
    # plt.ylabel("Objective Function(MSE)")    
    plt.legend()
    plt.show()

def main():
    x, y = iris_customize()
    w0 = -8
    w1 = 0.6
    w2 = 1
    # gradient_descent(x, w0, w1, w2, y)
    # old_weight = []
    # w0 = -5
    # w1 = 0.5
    # w2 = 2
    # old_weight = []
    # old_weight.append(w0)
    # old_weight.append(w1)
    # old_weight.append(w2)
    # print("Old weight: ", old_weight)
    # new_weight = old_weight - 0.01 * summed_gradient(x, w0, w1, w2, y)
    # print("New weight: ", new_weight)
    # # # illustrate_summed_gradient(x, w0, w1, w2, y)
    # plot([0, 30.8786102/3.9227226],[30.8786102/7.15813436,0])
    # armijo_updating(1, 0.5, x, y, -3.9, 0.46, 0.95)
    plot_gradient_descent(1, 0.5, x, w0, w1, w2, y)
    # mse1 = mean_square_error(x, -1, 0.7, 2, y)
    # print("Old MSE: ", mse1)
    # print("Decision Boundary: ", calc_slope_b(-1, 0.7, 2))
    # mse2 = mean_square_error(x, -1.00011645, 0.69953903, 1.99985973, y)
    # print("Decision Boundary: ", calc_slope_b(-1.00011645, 0.69953903, 1.99985973))
    # print("New MSE: ", mse2)
    # illustrate_summed_gradient(x, w0, w1, w2, y)


    # df = read_data()
    # versicolor = df[df['species']=='versicolor']
    # virginica = df[df['species']=='virginica']
    # plt.plot(versicolor["petal_length"], versicolor["petal_width"], '*',label="versicolor", color = 'orange')
    # plt.plot(virginica["petal_length"], virginica["petal_width"], '+',label="virginica", color = 'blue')
    # plt.xlabel("petal length (cm)")
    # plt.ylabel("petal width (cm)")
    # plt.plot([0,-w0/w1],[-w0/w2,0], label='unadjusted decision boundary', color='purple')
    # plot([0, -w0/w1], [-w0/w2, 0])
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()
