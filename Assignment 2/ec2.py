import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def model():
	model = Sequential()
    # output = activation(dot(input, kernel) + bias)
	model.add(Dense(3, input_dim=4, activation='sigmoid'))
	model.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=['accuracy'])
	return model

def config_data(): 
    df = pd.read_csv("irisdata.csv",header=None)
    attributes = df.iloc[1:, :4].astype(float)
    classes = df.iloc[1: , 4]
    enc = OneHotEncoder()
    classes = enc.fit_transform(classes[:, np.newaxis]).toarray()
    X_train, X_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.25, shuffle=True)
    return X_train, X_test, y_train, y_test

def neural_network(): 
    neural_network = model()
    X_train, X_test, y_train, y_test = config_data()
    neural_network.fit(x=X_train, y=y_train, epochs=500, validation_data=(X_test, y_test), verbose=0)
    print("Training: " + str(neural_network.evaluate(X_train, y_train)))
    print("Validation: " + str(neural_network.evaluate(X_test, y_test)))

neural_network()