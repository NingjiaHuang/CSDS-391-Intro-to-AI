import math
import pandas as pd
import math
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d 
import seaborn as sns
import numpy as np
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression 
from mpl_toolkits.mplot3d import *
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

#load iris dataset
data = pd.read_csv("/Users/tianxi/Desktop/irisdata.csv")

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
	plt.title("The 2nd and 3rd Iris Classes")
	plt.legend()
	plt.show()

#Output of 0 for the 2nd iris class, and 1 for the 3rd.
def zValue(species:list) -> float:
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

#sigmoid function
def sigmoid(length:float, width:float) -> float:
	w = [-3.18, 0.30, 0.98]
	z = w[0] + w[1] * length + w[2] * width
	sigmoid = 1 / (1 + math.exp(-z))
	return sigmoid

#Exercise 1. b. computes the output of simple one-layer neural network
def output(length:float, width:float) -> float: 
	sig = sigmoid(length, width)
	if (sig < 0.5):
		return 0
	else:
		return 1

#Exercise 1. c. plot decision boundary
def decisionBoundary(dataset:list) -> None:
	w = [-3.18, 0.30, 0.98]
	x = dataset['petal_length']
	y = [-(w[1] * x_value + w[0]/ w[2]) for x_value in x] 

	plt.plot(versicolor["petal_length"], versicolor["petal_width"], '*',label="versicolor", color = 'orange')
	plt.plot(virginica["petal_length"], virginica["petal_width"], '+',label="virginica", color = 'blue')
    
	plt.plot(x, y, 'g-')

	plt.title("Decision Boundary")
	plt.xlabel('Petal Length (cm)')
	plt.ylabel('Petal Width (cm)')
	plt.legend(loc = 2)
	plt.show()

#Exercise 1. d. plot the output of neural network
def plotNeuralNetwork(dataset:list) -> None: 
	ax = plt.gca(projection='3d')
	
	x = np.arange(3.0, 7.0, 0.01) 
	y = np.arange(1.0, 2.5, 0.01) 
	X, Y = np.meshgrid(x, y)
	z = np.array([output(a,b) for a,b in zip(np.ravel(X), np.ravel(Y))])
	Z = z.reshape(X.shape)

	z2 = zValue(versicolor['species'])
	z3 = zValue(virginica['species'])

	ax.plot_surface(X, Y, Z, cstride=1, rstride=1, linewidth=0, antialiased=True, color = 'grey')
	ax.scatter3D(versicolor["petal_length"], versicolor['petal_width'], z2, marker = '*', color = 'orange') 
	ax.scatter3D(virginica["petal_length"], virginica['petal_width'], z3, marker = '+', color = 'blue') 
	plt.title("Output of Neural Network")
	ax.set_xlabel('Petal length (cm)')
	ax.set_ylabel('Petal width (cm)')
	ax.set_zlabel('output')
	plt.show()

#Exercise 1. e. show the output of your simple classifier using examples
def examples(dataset:list) -> None:
	# 4.8 1.8 versicolor
	# 5.0 1.7 versicolor
	# 3.5 1.0 versicolor
	# 3.8 1.1 versicolor
	versicolorLength = [4.8, 5.0, 3.5, 3.8]
	versicolorWidth = [1.8, 1.7, 1.0, 1.1]
	l1 = []
	w1 = []
	l2 = []
	w2 = []
	z = []
	for var1, var2 in zip(versicolorLength, versicolorWidth):
		z.append(output(var1, var2))
	for var1, var2, var3 in zip(versicolorLength, versicolorWidth, z):
		if (var3 == 0):
			l1.append(var1)
			w1.append(var2)
		else:
			l2.append(var1)
			w2.append(var2)
	plt.plot(l1, w1, '*',label="versicolor, classified as versicolor", color = 'orange')
	plt.plot(l2, w2, '*',label="versicolor, classified as virginica", color = 'blue')
		
	# 5.1 1.8 virginica
	# 5.0 1.5 virginica
	# 4.9 1.8 virginica
	# 5.9 2.3 virginica
	virginicaLength = [5.1, 5.0, 4.9, 5.9]
	virginicaWidth = [1.8, 1.5, 1.8, 2.3]
	l3 = []
	w3 = []
	l4 = []
	w4 = []
	z = []
	for var1, var2 in zip(virginicaLength, virginicaWidth):
		z.append(output(var1, var2))
	for var1, var2, var3 in zip(virginicaLength, virginicaWidth, z):
		if (var3 == 0):
			l3.append(var1)
			w3.append(var2)
		else:
			l4.append(var1)
			w4.append(var2)
	plt.plot(l3, w3, '+',label="virginica, classified as versicolor", color = 'orange')
	plt.plot(l4, w4, '+',label="virginica, classified as virginica", color = 'blue')

	w = [-3.18, 0.30, 0.98]
	x = dataset['petal_length']
	y = [-(w[1] * x_value + w[0]/ w[2]) for x_value in x] 
	plt.plot(x, y, 'g-')

	plt.title("Examples")
	plt.xlabel('Petal Length (cm)')
	plt.ylabel('Petal Width (cm)')
	plt.legend(loc = 2, prop={'size': 9})
	plt.show()

def main() -> None:
	class23 = data[data['species'] != 'setosa']
	plotclasses23()
	decisionBoundary(class23)
	#plotNeuralNetwork(class23)
	examples(class23)

main()
