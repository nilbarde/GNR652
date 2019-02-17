import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

global DataFilePath
global X_Train, Y_Train, X_Test, Y_Test

import cvxopt
DataFilePath = "./data/data.csv"

filepath ="data/data.csv"
myFile = np.genfromtxt(filepath, delimiter=',')
# print("MyData_shape =",myFile.shape)

# Adding biases i.e y=W'XItrain + b (accounting for b by adding ones in XItrain)
num_examples ,num_features  = myFile.shape
myFile1=np.ones((num_examples,num_features+1))
myFile1[:,1:] = myFile
myFile = myFile1
X_Train=myFile[1:800,:-1]
Y_Train=myFile[1:800,30]
X_Test=myFile[800:1000,:-1]
Y_Test=myFile[800:1000,30]
TotalTrainData, TotalParams = X_Train.shape

print("read")
X, y = X_Train, Y_Train

n_samples, n_features = X.shape

K = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        K[i,j] = np.dot(X[i], X[j])
print("poiuytre")
P = cvxopt.matrix(np.outer(y,y) * K)
q = cvxopt.matrix(np.ones(n_samples) * -1)
A = cvxopt.matrix(y, (1,n_samples))
b = cvxopt.matrix(0.0)

G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
h = cvxopt.matrix(np.zeros(n_samples))
print("for solution")
solution = cvxopt.solvers.qp(P, q, G, h, A, b)
print("got solution")
# Lagrange multipliers
alphas = np.ravel(solution['x'])
print(alphas)
# Support vectors are non zero lagrange multipliers
sv = alphas > 1e-12
print(alphas.shape,sv)
ind = np.arange(len(alphas))[sv]
sv_a = alphas[sv]
sv_x = X[sv]
sv_y = y[sv]

# Intercept
b = 0
for i in range(len(sv_a)):
    b += sv_y[i]
    b -= np.sum(sv_a * sv_y * K[ind[i],sv])
b /= len(sv_a)

# Weight vector
w = np.zeros(n_features)
for n in range(len(sv_a)):
    w += sv_a[n] * sv_y[n] * sv_x[n]

y_pred_for_train = np.sign(X_Train.dot(w) + b)
correct = y_pred_for_train==Y_Train
print(Y_Test)
print(np.sum(correct),len(correct))
y_pred_for_train = np.sign(X_Test.dot(w) + b)
correct = y_pred_for_train==Y_Test
print(np.sum(correct),len(correct))
