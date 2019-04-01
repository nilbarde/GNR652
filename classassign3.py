import scipy.io
import numpy as np
import matplotlib.pyplot as plt

class SoftmaxClass():
	def __init__(self):
		self.Normalization = True
		self.ShuffleData = True
		self.LoadData()
		self.GetInfo()
		self.Train()

	def LoadData(self):
		self.DataFilePath = './data/Indian_pines_corrected.mat'
		self.GTFilePath = './data/Indian_pines_gt.mat'
		self.ReadFiles()
		self.PreProcessData()

	def ReadFiles(self):
		self.RawData = scipy.io.loadmat(self.DataFilePath)
		self.RawGT = scipy.io.loadmat(self.GTFilePath)

		self.RawX = self.RawData["indian_pines_corrected"]
		self.RawY = self.RawGT["indian_pines_gt"]

		print("Data files reading completed")

	def PreProcessData(self):
		self.TotalClasses = self.GetClassCount(self.RawY)

		self.XTrainPre, self.XTestPre = [], []
		self.YTrainPre, self.YTestPre = [], []
		self.ClassWiseCount = [0 for i in range(self.TotalClasses)]

		for i in range(self.RawY.shape[0]):
			for j in range(self.RawY.shape[1]):
				y = self.RawY[i,j]
				if y:
					if not(self.ClassWiseCount[y-1]%2==0):
						self.XTrainPre.append(self.RawX[i,j])
						self.YTrainPre.append(self.RawY[i,j])
					else:
						self.XTestPre.append(self.RawX[i,j])
						self.YTestPre.append(self.RawY[i,j])
					self.ClassWiseCount[y-1] += 1
		self.XTrain, self.XTest = self.GetData(self.XTrainPre), self.GetData(self.XTestPre)
		self.YTrain, self.YTest = self.GetLabels(self.YTrainPre), self.GetLabels(self.YTestPre)

		print("preprocessing done")

	def GetData(self,DataX):
		DataXNumpy = np.array(DataX).astype(float)
		if self.Normalization:
			MinValue = np.amin(DataXNumpy)
			MaxValue = np.amax(DataXNumpy)
			DataXNumpy = DataXNumpy - MinValue
			if(MinValue!=MaxValue):
				DataXNumpy = DataXNumpy/(MaxValue - MinValue)
			else:
				DataXNumpy /= MaxValue
		return DataXNumpy

	def GetLabels(self,RawLabels):
		NpLabels = np.array(RawLabels)
		OneHotEncoded = np.zeros((len(RawLabels),self.TotalClasses),dtype="uint8")
		for i in range(self.TotalClasses):
			layer = ((NpLabels==i+1)).astype("uint8")
			OneHotEncoded[:,i] = layer
		return OneHotEncoded

	def GetClassCount(self,DataY):
		ClassCount = {}
		for i in range(DataY.shape[0]):
			for j in range(DataY.shape[1]):
				y = DataY[i,j]
				if(y):
					if not y in ClassCount:
						ClassCount[y] = 0
					ClassCount[y] += 1
		return len(ClassCount)

	def GetInfo(self):
		self.TotalFeatures = (self.RawX.shape)[-1]
		self.Weights = np.random.randn(self.TotalFeatures,self.TotalClasses)
		self.Epochs = 50000
		self.LearningRate = 0.00012
		self.TrainLoss = []
		self.TestLoss = []
		self.TrainAccuracy = []
		self.TestAccuracy = []

	def Train(self):
		for i in range(self.Epochs):
			print("Running Epoch " + str(i+1) + " / " + str(self.Epochs))
			TrainAcc, TrainLoss, self.Weights = self.GetSoftmax(self.XTrain,self.YTrain,self.Weights)
			print("Training Accuracy is " + str(TrainAcc) + "%, Training Loss is " + str(TrainLoss))

			TestAcc, TestLoss, DummyWeights = self.GetSoftmax(self.XTest,self.YTest,self.Weights)
			print("Testing  Accuracy is " + str(TestAcc)  + "%, Testing  Loss is " + str(TestLoss) )

			if((i+1)%100==0):
				self.TrainAccuracy.append(TrainAcc)
				self.TrainLoss.append(TrainLoss)
				self.TestAccuracy.append(TestAcc)
				self.TestLoss.append(TestLoss)

				plt.plot(range(len(self.TrainAccuracy)),self.TrainAccuracy,color='blue')
				plt.plot(range(len(self.TestAccuracy)),self.TestAccuracy,color='red')
				plt.xlabel('epoch number')
				plt.legend(('Train Accuracy','Test Accuracy'), loc = 'lower right')
				plt.ylabel('Accuracy')
				plt.savefig('./Plots/Accuracy.png')

				plt.clf()

				plt.plot(range(len(self.TrainLoss)),self.TrainLoss,color='blue')
				plt.plot(range(len(self.TestLoss)),self.TestLoss,color='red')
				plt.xlabel('epoch number')
				plt.legend(('Train Loss','Test Loss'), loc = 'upper right')
				plt.ylabel('Loss')
				plt.savefig('./Plots/Loss.png')

				plt.clf()

	def GetSoftmax(self,Data,Labels,Weights):
		ClasswiseScore = np.dot(Data,Weights)
		ExpClass = np.exp(ClasswiseScore)
		ExpClassNorm =  np.transpose(np.transpose(ExpClass)/np.sum(np.transpose(ExpClass),axis=0))
		# ExpClassNorm = ExpClass
		# for i in range(len(ExpClass)):
		# 	ExpClassNorm[i] /= np.sum(ExpClass[i])

		Accuracy = np.sum((ExpClassNorm*Labels),axis=1)
		AvgAccuracy = np.sum(Accuracy)/len(Accuracy)

		Loss = -np.sum(np.log(Accuracy))/len(Accuracy)
		NewWeights = Weights + self.LearningRate*(np.dot(Data.T,(Labels - ExpClassNorm)))

		return AvgAccuracy, Loss, NewWeights

mySoft = SoftmaxClass()
