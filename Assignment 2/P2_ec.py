from sklearn import datasets
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC

data = pd.read_csv('irisdata.csv')

features = pd.DataFrame({"A": data['sepal_length'], "B": data['sepal_width'], 'C': data['petal_length'], 'D': data['petal_width']}).to_numpy()

# function to assign target
def assignT(species:list) -> float:
    n = 1
    y = []
    for var in species:
        if var =='versicolor':
            y.append(1)
        elif var =='virginica':
            y.append(2)
        else:
            y.append(0)
    return y

t = assignT(data['species'])
target = np.array(t)

#sepal
X = features[:,:2]
y = target
plt.scatter(X[y==0,0],X[y==0,1],color = 'r',marker='o')
plt.scatter(X[y==1,0],X[y==1,1],color = 'orange',marker='*')
plt.scatter(X[y==2,0],X[y==2,1],color = 'b',marker='+')
plt.title('the relationship between sepal and target classes')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

#petal
X = features[:,2:]
y = target
plt.scatter(X[y==0,0],X[y==0,1],color = 'r',marker='o')
plt.scatter(X[y==1,0],X[y==1,1],color = 'orange',marker='*')
plt.scatter(X[y==2,0],X[y==2,1],color = 'b',marker='+')
plt.title('the relationship between Petal and target classes')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(features[:,:2], target, test_size=0.3, random_state=0)

sig_svc = svm.SVC(kernel='sigmoid').fit(X_train, y_train)
rbf_svc = svm.SVC(kernel='rbf').fit(X_train, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3).fit(X_train, y_train)

#SVM sepal features
h = .02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

titles = ['Sigmoid Kernel','RBF Kernel','Polynomial (degree=3) Kernel']

for i, clf in enumerate((sig_svc, rbf_svc, poly_svc)):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape) 
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

sig_svc_pre = sig_svc.predict(X_test)
acc_sig_svc = sum(sig_svc_pre==y_test)/len(y_test)
rbf_svc_pre = rbf_svc.predict(X_test)
acc_rbf_svc = sum(rbf_svc_pre==y_test)/len(y_test)
poly_svc_pre = poly_svc.predict(X_test)
acc_poly_svc = sum(poly_svc_pre==y_test)/len(y_test)

print("Sigmoid Result(Sepal): ", sig_svc_pre)
print("RBF Result(Sepal): ", rbf_svc_pre)
print("Polynomial Result(Sepal)): ", poly_svc_pre)

print("Accuracy of Sigmoid(Sepal): ",acc_sig_svc)
print("Accuracy of Sigmoid(Sepal): ",acc_rbf_svc)
print("Accuracy of Sigmoid(Sepal): ",acc_poly_svc)

#petal
X_train, X_test, y_train, y_test = train_test_split(features[:,2:], target, test_size=0.3, random_state=0)  #[:,2:]

#SVM petal features
sig_svc = svm.SVC(kernel='sigmoid').fit(X_train, y_train)
rbf_svc = svm.SVC(kernel='rbf').fit(X_train, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3).fit(X_train, y_train)

h = .02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

titles = ['Sigmoid Kernel', 'RBF Kernel', 'Polynomial (degree=3) Kernel']

for i, clf in enumerate((sig_svc, rbf_svc, poly_svc)):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape) 
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

sig_svc_pre = sig_svc.predict(X_test)
acc_lin_svc = sum(sig_svc_pre==y_test)/len(y_test)
rbf_svc_pre = rbf_svc.predict(X_test)
acc_rbf_svc = sum(rbf_svc_pre==y_test)/len(y_test)
poly_svc_pre = poly_svc.predict(X_test)
acc_poly_svc = sum(poly_svc_pre==y_test)/len(y_test)

print("Sigmoid Result(Petal): ", sig_svc_pre)
print("RBF Result(Petal): ", rbf_svc_pre)
print("Polynomial Result(Petal): ", poly_svc_pre)

print("Accuracy of Sigmoid(Petal): ", acc_sig_svc)
print("Accuracy of RBF(Petal): ", acc_rbf_svc)
print("Accuracy of Polynomial(Petal): ", acc_poly_svc)