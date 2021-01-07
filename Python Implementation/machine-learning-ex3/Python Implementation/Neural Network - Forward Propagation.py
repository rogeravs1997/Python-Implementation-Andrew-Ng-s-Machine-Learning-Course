import numpy as np
from scipy.io import loadmat
import scipy.optimize as opt
import matplotlib.pyplot as plt
from PIL import Image  

#loading DATA
data=loadmat('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex3\ex3\ex3data1.mat')
X=data['X']
y = data['y']


#loading Weights
weights=loadmat('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex3\ex3\weights_1000its.mat')
Theta1=weights['Theta1']
Theta2= weights['Theta2']

#useful function
def sigmoid(z):
    return ((1)/(1+np.exp(-z)))   

#defining function to predict values

def predict(Theta1,Theta2,X):
    #adding a column of 1's to X before computing with Theta1
    X=np.concatenate((np.ones((len(X),1)),X),axis=1)
    a2=sigmoid(X.dot(Theta1.T))
    #adding a column of 1's to a2 before computing with Theta2
    a2=np.concatenate((np.ones((len(a2),1)),a2),axis=1)
    a3=sigmoid(a2.dot(Theta2.T))
    a3=a3.argmax(axis=1)+1
    a3=np.array(a3)
    return a3

prediction=predict(Theta1, Theta2, X)

def acurracy(prediction,y):
    counter=0
    for i in range(len(y)):
        if y[i][0]==prediction[i]:
            counter+=1
    acc=(counter/len(y)*100)
    print("Acurracy of the model: {}%".format(acc))
    return acc

acurracy(prediction, y)
