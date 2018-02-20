import numpy
import pandas
from sklearn import preprocessing
import matplotlib
#This is used so that Python is not installed as a framework
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#Method is directly from Linear Regression from Mingon Kang python code.  There is a small change
def SolverLinearRegression(X, y):
    return numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(X.transpose(), X)), X.transpose()), y)

#Method to classify the test data and determine the accuracy of this algorithm
def classifyTestDataAndAccuracy(X, Y, bOpt, threshold):
    Y_Pred = numpy.dot(X, bOpt)
    classify = numpy.array(Y_Pred > threshold)
    return float(sum(classify == Y)) / float(len(Y))

#Normalize the data to help solve for gradient linear regression problem
def min_max_normalization(data):
    return data / 255

#Gradient Linear Regression Formula
def GD_LR(X, y, b):
    return -numpy.dot(X.transpose(), y) + numpy.dot(numpy.dot(X.transpose(), X), b)

# cost function using OLS
def cost(X, y, b):
    return numpy.sum((numpy.dot(X, b) - numpy.array(y))**2)
#------------------------------->TASK 1<-----------------------------------------

#Get training data and test data from CSV files
trainData = numpy.genfromtxt('MNIST_training.csv', delimiter=',', skip_header=1)
testData = numpy.genfromtxt('MNIST_test.csv', delimiter=',', skip_header=1)

#Assign labels from first row of training and test data
yTrainingLabel = numpy.array(trainData[:, 0])
yTestLabel = numpy.array(testData[:, 0])

#Make array of data other than label row
xTraining = numpy.array(trainData[:, 1:])
xTest = numpy.array(testData[:, 1:])

#Solve for Linear Regression
bOpt = SolverLinearRegression(xTraining, yTrainingLabel)
threshold =  0.5
testBOptAccuracy = classifyTestDataAndAccuracy(xTest, yTestLabel, bOpt, threshold)

#Display the accuracy
print('Accuracy of the algorithm is: ')
print'{percent:.2%}'.format(percent=testBOptAccuracy)

#------------------------------->TASK 2<-----------------------------------------

#Normalize data based on Mingon Kang Linear Regression Python code
X_nor = min_max_normalization(xTraining)
y_nor = min_max_normalization(yTrainingLabel)

#initial coefficients
r,c = X_nor.shape
bEst = numpy.zeros(c)
bs = [bEst]
costs = [cost(X_nor, y_nor, bEst)]
#learning rate
learning_rate = 1e-4
for i in range(0, 100):
    #Gradient Descent Algorithm
    bEst = bEst - learning_rate * GD_LR(X_nor, y_nor, bEst)
    #test for convergence
    b_cost = cost(X_nor, y_nor, bEst)
    bs.append(bEst)
    costs.append(b_cost)

testBEstAccuracy = classifyTestDataAndAccuracy(xTest, yTestLabel, bEst, threshold)

#Display the accuracy
print('\nAccuracy of the algorithm is: ')
print'{percent:.2%}'.format(percent=testBEstAccuracy)

#Display the total differences between b_opt and b_est
testTotalAccuracy = sum(abs(bOpt - bEst))
print('\nAccuracy difference between these algorithms: ')
print'{percent:.2%}'.format(percent=testTotalAccuracy)

#Show a plot graph of cost function using OLS
plt.plot(costs)
plt.show()