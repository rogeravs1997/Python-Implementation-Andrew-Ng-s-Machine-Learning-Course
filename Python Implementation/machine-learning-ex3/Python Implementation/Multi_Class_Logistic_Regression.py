import numpy as np
from scipy.io import loadmat
import scipy.optimize as opt
import matplotlib.pyplot as plt
from PIL import Image
#loading the data

##########################we also could use the loadmath method from scipy#######################
# data=loadmat('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex3\ex3\ex3data1.mat')
# X=data['X']
# y = data['y']
#################################################################################################
#loading the features
data_X=open('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex3\ex3\ex3data1.txt','r').readlines()

X=[]
for i in data_X:
    X.append(i.split(","))

for i in X:
    for e in range(len(i)):
        i[e]=float(i[e])

X=np.array(X)

#loading the labels
data_y=open('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex3\ex3\ex3data1_2.txt','r').readlines()

y=[]
for i in data_y:
    y.append([float(i)])

y=np.array(y)

#Display 100 random training examples

def display(X):
    _, axarr = plt.subplots(10,10,figsize=(10,10))
    for i in range(10):
        for j in range(10):
           axarr[i,j].imshow(X[np.random.randint(X.shape[0])].\
    reshape((20,20), order = 'F'))          
           axarr[i,j].axis('off')  


#add an extra column of 1's to X
m = len(y)
ones = np.ones((m,1))
X = np.hstack((ones, X)) #add the intercept
(m,n) = X.shape

#Defining cost function and gradient regularized
def sigmoid(z):
    return ((1)/(1+np.exp(-z)))    

def costFunctionReg(theta, X, y, lmbda):
    m = len(y)
    temp1 = np.multiply(y, np.log(sigmoid(np.dot(X, theta))))
    temp2 = np.multiply(1-y, np.log(1-sigmoid(np.dot(X, theta))))
    return np.sum(temp1 + temp2) / (-m) + np.sum(theta[1:]**2) * lmbda / (2*m)

def gradRegularization(theta, X, y, lmbda):
    m = len(y)
    temp = sigmoid(np.dot(X, theta)) - y
    temp = np.dot(temp.T, X).T / m + theta * lmbda / m
    temp[0] = temp[0] - theta[0] * lmbda / m
    return temp


#setting initial values
(m,n) = X.shape
lam=1
k=10 #number of labels
max_iter=500
initial_theta = np.zeros((k,n)) #inital parameters

#Optimizing parameters for the 10 models, we will find 10 diferents optimal thetas to clasify if a training example is part or not from a class, ONE VS ALL
def optimizeTheta(lam,X,y,k,max_iter,initial_theta):
    for i in range(k):
        digit_class = i if i else 10
        initial_theta[i] = opt.fmin_cg(f = costFunctionReg, x0 = initial_theta[i],  fprime = gradRegularization, args = (X, (y == digit_class).flatten(), lam), maxiter = 500)
    theta_found=initial_theta.copy()
    return theta_found

theta_found=optimizeTheta(lam,X,y,k,max_iter,initial_theta)

#defininf function to predict new values
def predict(newX,theta_found):
    pred=sigmoid(newX.dot(theta_found.T))
    result = pred.argmax(axis=1)

    result = [e if e else 10 for e in result]
    result=np.array(result)
    result.shape=(len(newX),1)
    return result
    
#Defining function to compute the acurracy of the model
def acuraccy(X,y,theta_found):
    result=predict(X,theta_found)
    counter=0
    for i in range(len(y)):
        if y[i][0]==result[i][0]:
            counter+=1
    acc=(counter/len(y)*100)
    print("Acurracy of the model: {}%".format(acc))
    return acc

acuraccy(X,y,theta_found)

