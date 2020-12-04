from sklearn import datasets
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

iris = datasets.load_iris()
X = iris.data
y = iris.target
# To check the target data (3 points)

X = iris.data[:,:2]
y = iris.target
plt.scatter(X[y==0,0],X[y==0,1],color = 'r',marker='o')
plt.scatter(X[y==1,0],X[y==1,1],color = 'b',marker='*')
plt.scatter(X[y==2,0],X[y==2,1],color = 'g',marker='+')
plt.title('the relationship between sepal and target classes')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

# sepal 花萼
X = iris.data[:,2:]
y = iris.target
plt.scatter(X[y==0,0],X[y==0,1],color = 'r',marker='o')
plt.scatter(X[y==1,0],X[y==1,1],color = 'b',marker='*')
plt.scatter(X[y==2,0],X[y==2,1],color = 'g',marker='+')
plt.title('the relationship between Petal and target classes')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()


X_train, X_test, y_train, y_test = train_test_split(iris.data[:,:2], iris.target, test_size=0.3, random_state=0)
lin_svc = svm.SVC(kernel='linear').fit(X_train, y_train)
rbf_svc = svm.SVC(kernel='rbf').fit(X_train, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3).fit(X_train, y_train)

# SVM分类器Iris Sepal features
# the step of the grid
h = .02
# to create the grid , so that we can plot the images on it
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# the title of the graph
titles = ['LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


for i, clf in enumerate((lin_svc, rbf_svc, poly_svc)):
    # to plot the edge of different classes
    # to create a 2*2 grid , and set the i image as current image
    plt.subplot(2, 2, i + 1)
    # set the margin between different sub-plot
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # SVM input :xx and yy output: an array
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # to plot the result
    Z = Z.reshape(xx.shape) #(220, 280)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()

lin_svc_pre = lin_svc.predict(X_test)
acc_lin_svc = sum(lin_svc_pre==y_test)/len(y_test)
rbf_svc_pre = rbf_svc.predict(X_test)
acc_rbf_svc = sum(rbf_svc_pre==y_test)/len(y_test)
poly_svc_pre = poly_svc.predict(X_test)
acc_poly_svc = sum(poly_svc_pre==y_test)/len(y_test)

print(lin_svc_pre)
print(rbf_svc_pre)
print(poly_svc_pre)

print(acc_lin_svc)
print(acc_rbf_svc)
print(acc_poly_svc)

X_train, X_test, y_train, y_test = train_test_split(iris.data[:,2:], iris.target, test_size=0.3, random_state=0)  #[:,2:]
# svc = svm.SVC(kernel='linear').fit(X_train, y_train)
lin_svc = svm.SVC(kernel='linear').fit(X_train, y_train)
rbf_svc = svm.SVC(kernel='rbf').fit(X_train, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3).fit(X_train, y_train)
# print(X_train)

#
h = .02
# to create the grid , so that we can plot the images on it
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# the title of the graph
titles = ['LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


for i, clf in enumerate((lin_svc, rbf_svc, poly_svc)):
    # to plot the edge of different classes
    # to create a 2*2 grid , and set the i image as current image
    plt.subplot(2, 2, i + 1)
    # set the margin between different sub-plot
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # SVM input :xx and yy output: an array
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # to plot the result
    Z = Z.reshape(xx.shape) #(220, 280)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()

# a = np.array([[1.3,0.4], [4.4,1.3], [6.0, 2.2]])
# print(a)
lin_svc_pre = lin_svc.predict(X_test)
acc_lin_svc = sum(lin_svc_pre==y_test)/len(y_test)
rbf_svc_pre = rbf_svc.predict(X_test)
acc_rbf_svc = sum(rbf_svc_pre==y_test)/len(y_test)
poly_svc_pre = poly_svc.predict(X_test)
acc_poly_svc = sum(poly_svc_pre==y_test)/len(y_test)

print(lin_svc_pre)
print(rbf_svc_pre)
print(poly_svc_pre)

print(acc_lin_svc)
print(acc_rbf_svc)
print(acc_poly_svc)
