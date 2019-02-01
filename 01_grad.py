import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

global DataFilePath
global X_Train, Y_Train, X_Test, Y_Test

DataFilePath = "./data/data.csv"

global LearningRate, Weights, NumIter, Lambda
LearningRate = 0.0002
NumIter = 4000000
Lambda = 0.01

def GetData():
	global DataFilePath
	global TotalData, TotalParams, TotalTrainData, TotalTestData
	global X_Train, Y_Train, X_Test, Y_Test
	global Weights, MLEWeights

	TrafficData = pd.read_csv(DataFilePath,delimiter=";",decimal=",")

	TotalData = TrafficData.shape[0]
	TotalParams = 1 + (TrafficData.shape[1] - 1)

	TotalTrainData = int(TotalData*0.8)
	TotalTestData = TotalData - TotalTrainData

	Weights = np.random.random_sample((TotalParams))
	MLEWeights = np.random.random_sample((TotalParams))

	X       = np.ones((TotalData,TotalParams),dtype=float) 
	Y       = np.ones((TotalData),dtype=float)
	permutation = np.random.permutation(Y.shape[0])
	X,Y     = X[permutation],Y[permutation]
	
	X_Train = np.ones((TotalTrainData,TotalParams),dtype=float)
	Y_Train = np.ones((TotalTrainData),dtype=float)
	X_Test  = np.ones((TotalTestData,TotalParams),dtype=float)
	Y_Test  = np.ones((TotalTestData),dtype=float)

	X[:,1:] = TrafficData.iloc[:,:-1]
	Y[:]    = TrafficData.iloc[:,-1]

	X_Train, X_Test = X[:TotalTrainData], X[TotalTrainData:]
	Y_Train, Y_Test = Y[:TotalTrainData], Y[TotalTrainData:]

def CalcMSE(X,Y,W):
	Error = (np.dot(X,W)) - Y
	Error = np.average(Error*Error)
	return Error

def FindMLELoss(X, Y, Weights, Sigma, Alpha):
	LossRoot = (np.dot(X,Weights)) - Y
	ExpTerm = np.exp(-((LossRoot)**2)/(2*(Sigma**2)))
	Prod = np.product(ExpTerm)
	Prod /= ((2*np.pi)**(0.5))*(Sigma)
	Loss = np.sum((Prod*LossRoot)/(Sigma**2))
	NewWeights = Weights - (float(8*Alpha)/X.shape[0])*(np.dot(LossRoot,X))*Prod
	return Loss, NewWeights

def FindLoss(X, Y, Weights, Alpha):
	LossRoot = (np.dot(X,Weights)) - Y
	Loss = 0.5*np.sum(LossRoot*LossRoot) + 0.5*Lambda*(np.sum(Weights*Weights))# - Weights[0]**2)
	Loss /= X.shape[0]
	NewWeights = Weights - (float(Alpha)/X.shape[0])*((np.dot(LossRoot,X) + 2*Lambda*Weights))
	return Loss, NewWeights

GetData()

for i in range(1,NumIter+1):
	Loss, Weights = FindLoss(X_Train,Y_Train,Weights,LearningRate)
	Test_Loss, _ = FindLoss(X_Test,Y_Test,Weights,LearningRate)
	if(i%20000==0):
		Error = CalcMSE(X_Test,Y_Test,Weights)
		Error2 = CalcMSE(X_Train,Y_Train,Weights)
		# print(Weights)
		print(i/20000, Loss,Test_Loss,Error,Error2)

CFS_Weights = np.dot((np.linalg.inv(np.dot(X_Train.transpose(),X_Train))),np.dot(X_Train.transpose(),Y_Train))
CFSError = CalcMSE(X_Test, Y_Test, CFS_Weights)

# for i in range(1,NumIter+1):
# 	MLELoss, MLEWeights = FindMLELoss(X_Train,Y_Train,MLEWeights,10000,LearningRate)
# 	Test_Loss, _ = FindMLELoss(X_Test,Y_Test,MLEWeights,10000,LearningRate)
# 	if(i%20000==0):
#		Error = CalcMSE(X_Test,Y_Test,MLEWeights)
# 		print(MLEWeights)
# 		print(i/20000, MLELoss,Test_Loss,Error)

print("------")
print(Weights, Error)
print(CFS_Weights, CFSError)

# print(MLEWeights)
