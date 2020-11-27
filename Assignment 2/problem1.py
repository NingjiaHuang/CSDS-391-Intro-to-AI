import math
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d 
import seaborn as sns
import numpy as np
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression 
from mpl_toolkits.mplot3d import *

#load iris dataset
data = pd.read_csv("/Users/tianxi/Desktop/irisdata.csv")
#print(data.head())

setosa = data[data['species']=='setosa']
versicolor = data[data['species']=='versicolor']
virginica = data[data['species']=='virginica']

#plt.plot(setosa["petal_length"], setosa["petal_width"], 'o',label="setosa", mfc='none', color = 'red')
#plt.plot(versicolor["petal_length"], versicolor["petal_width"], '*',label="versicolor", color = 'orange')
#plt.plot(virginica["petal_length"], virginica["petal_width"], '+',label="virginica", color = 'blue')
#plt.xlabel("petal length (cm)")
#plt.ylabel("petal width (cm)")
#plt.legend()
#plt.show()


#Exercise 1. a. plot the 2nd and 3rd iris classes
def plotclasses23() -> None:
    plt.plot(versicolor["petal_length"], versicolor["petal_width"], '*',label="versicolor", color = 'orange')
    plt.plot(virginica["petal_length"], virginica["petal_width"], '+',label="virginica", color = 'blue')
    plt.xlabel("petal length (cm)")
    plt.ylabel("petal width (cm)")
    plt.legend()
    plt.show()

#plotclasses23()

#function that computes the output of simple one-layer neural network
def output(species):
    n = 1
    y = []
    for var in species:
        if var =='versicolor':
            y.append(0)
        elif var =='virginica':
            y.append(1)
        else:
            print("error\n")
    return y


#Exercise 1. b. computes the output of simple one-layer neural network
def logistic(length:float, width: float) -> float: 
	w = [-3.9, 0.46, 0.95]
	z = (w[0] + (w[1]*length) + (w[2]*width))
	sigmoid = 1 / (1 + math.exp(-z))
	if (sigmoid < 0.5):
		return 0
	else:
		return 1

#Exercise 1. d. plot the output of neural network
def plotNeuralNetwork(dataset: list) -> None: 
	ax = plt.gca(projection='3d')
	
	x = np.arange(3.0, 7.0, 0.01) 
	y = np.arange(1.0, 2.5, 0.01) 
	X, Y = np.meshgrid(x, y)
	z = np.array([logistic(a,b) for a,b in zip(np.ravel(X), np.ravel(Y))])
	Z = z.reshape(X.shape)

	z2 = output(versicolor['species'])
	z3 = output(virginica['species'])

	ax.plot_surface(X, Y, Z)
	ax.scatter3D(versicolor["petal_length"], versicolor['petal_width'], z2, marker = '*', color = 'orange') 
	ax.scatter3D(virginica["petal_length"], virginica['petal_width'], z3, marker = '+', color = 'blue') 
	plt.title("Output of neural network")
	ax.set_xlabel('Petal length (cm)')
	ax.set_ylabel('Petal width (cm)')
	ax.set_zlabel('output')
	plt.show()


def main() -> None:
	class23 = data[data['species'] != 'setosa']
	plotclasses23()
	plotNeuralNetwork(class23)

main()
