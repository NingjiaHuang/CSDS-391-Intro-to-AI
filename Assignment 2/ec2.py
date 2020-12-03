import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("irisdata.csv",header=None)
attributes = df.iloc[1:, :4].astype(float)
classes = df.iloc[1: , 4]
enc = OneHotEncoder()
classes = enc.fit_transform(classes[:, np.newaxis]).toarray()
classes = classes.tolist()
# plt.scatter(attributes.iloc[0:50,0], attributes.iloc[0:50,1],c='olive',label='Versicolor')
# plt.scatter(attributes.iloc[50:100,0], attributes.iloc[50:100,1],c='skyblue',label='Virginica')
# plt.xlabel('Pental length')
# plt.ylabel('Pental width')
# plt.legend(loc='upper left')
# X_train, X_test, y_train, y_test = train_test_split(attributes, classes,test_size=0.25,shuffle=True)

# def modelNN():
# 	model = Sequential()
# 	model.add(Dense(1,input_dim=2,activation='sigmoid'))
# 	model.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=['accuracy'])
# 	return model

# model = modelNN()
# model.fit(x=inputTrain,y=outputTrain,epochs=2000, validation_data=(inputVal,outputVal))
###################
# #4b)
# flowerData = dataSet[1:, :4].astype(float)
# verboseTypes = dataSet[1:, 4]
# flowerTypes = []
# # #need to make output a vector now because it's not binary
# for i in classes:
# 	if i == "setosa":
# 		flowerTypes.append([1, 0, 0])
# 	elif i == "versicolor":
# 		flowerTypes.append([0, 1, 0])
# 	else:
# 		flowerTypes.append([0, 0, 1])
print(classes)
# #now that data is multidimensional needs to be numpy array to cooperate
# flowerTypes = numpy.array(flowerTypes)

# #split data
# inputTrain,inputVal,outputTrain,outputVal = train_test_split(flowerData,flowerTypes,test_size=0.25,shuffle=True)
# def flowerNN():
# 	model = Sequential()
# 	#add 2 more dimensions to input for more data 
# 	#add 2 more dimensions to output for more classes
# 	model.add(Dense(3,input_dim=4,activation='sigmoid'))
# 	model.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=['accuracy'])
# 	return model

# irisNN = flowerNN()
# irisNN.fit(x=inputTrain,y=outputTrain,epochs=2000, validation_data=(inputVal,outputVal), verbose=0)
# print("Training loss/accuracy " + str(irisNN.evaluate(inputTrain, outputTrain)))
# print("Validation loss/accuracy " + str(irisNN.evaluate(inputVal, outputVal)))
# #plot training data and validation data
# train1 = []
# train2 = []
# train3 = []
# val1 = []
# val2 = []
# val3 = []

# for i in range(len(outputTrain)):
# 	c = outputTrain[i]
# 	if c[0] == 1:
# 		train1.append(inputTrain[i])
# 	elif c[1] == 1:
# 		train2.append(inputTrain[i])
# 	else:
# 		train3.append(inputTrain[i])

# for i in range(len(outputVal)):
# 	c = outputVal[i]
# 	if c[0] == 1:
# 		val1.append(inputVal[i])
# 	elif c[1] == 1:
# 		val2.append(inputVal[i])
# 	else:
# 		val3.append(inputVal[i])

# train1 = numpy.array(train1)
# train2 = numpy.array(train2)
# train3 = numpy.array(train3)
# val1 = numpy.array(val1)
# val2 = numpy.array(val2)
# val3 = numpy.array(val3)
# plt.figure()
# plt.subplot(211)
# plt.scatter(train1[:,0], train1[:,1], color='red', marker="2")
# plt.scatter(train2[:,0], train2[:,1], color='blue', marker="2")
# plt.scatter(train3[:,0], train3[:,1], color='green', marker="2")
# plt.scatter(val1[:,0], val1[:,1], color='red', marker="+")
# plt.scatter(val2[:,0], val2[:,1], color='blue', marker="+")
# plt.scatter(val3[:,0], val3[:,1], color='green', marker="+")
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
# plt.title("Sepal Data Training(tri) vs Validation(+)")

# plt.subplot(212)
# plt.scatter(train1[:,2], train1[:,3], color='red', marker="2")
# plt.scatter(train2[:,2], train2[:,3], color='blue', marker="2")
# plt.scatter(train3[:,2], train3[:,3], color='green', marker="2")
# plt.scatter(val1[:,2], val1[:,3], color='red', marker="+")
# plt.scatter(val2[:,2], val2[:,3], color='blue', marker="+")
# plt.scatter(val3[:,2], val3[:,3], color='green', marker="+")
# plt.xlabel('Pental length')
# plt.ylabel('Pental width')
# plt.title("Petal Data Training(tri) vs Validation(+)")
# plt.tight_layout()
# ###################
# #show plots
# plt.show()