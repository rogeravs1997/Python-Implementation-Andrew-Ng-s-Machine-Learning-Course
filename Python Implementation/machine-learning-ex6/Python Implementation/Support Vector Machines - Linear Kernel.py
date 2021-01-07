# USEFUL LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.optimize #fmin_cg to train the linear regression
from sklearn.svm import SVC #SVM software
import math
# Loading data
data= scipy.io.loadmat('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex6\ex6\ex6data1.mat')

X=data['X']
y=data['y'].reshape(-1) #necessary for SVM that labels array has 0 dimensions


# defining function to plot training set
def plotData(X,y):
    posIndex=[i for i,x in enumerate(y) if x == 1]
    negIndex=[i for i,x in enumerate(y) if x == 0]
    pos=[]
    for i in posIndex:
        pos.append(X[i])
    pos=np.array(pos)
    neg=[]
    for i in negIndex:
        neg.append(X[i])
    neg=np.array(neg)
    positive=plt.plot(pos[:,0],pos[:,1], 'k+', label='Pos. examples')
    negative=plt.plot(neg[:,0],neg[:,1], 'yo', label='Neg. examples')
    plt.xlabel('Column 1 Variable')
    plt.ylabel('Column 2 Variable')
    plt.legend(loc='upper right')
    plt.grid(True)   

# plotData(X,y)


# Training the SVM ussing specialized library
linear_svm  = SVC(C=1, kernel='linear')
linear_svm .fit(X, y)

#Function to draw the SVM boundary
def plotBoundary(X,y,linear_svm):

    xs = np.linspace(min(X[:, 0]), max(X[:, 0]))
    
    # Calculate the decision boundary
    b = linear_svm.intercept_[0]
    w_0 = linear_svm.coef_[0, 0]
    w_1 = linear_svm.coef_[0, 1]
    a = - w_0 / w_1
    db_1 = a * xs - b / w_1
    
    # Store support vectors
    svs = linear_svm.support_vectors_
    # Calculate margins
    c = svs[0]
    margin_low = a * (xs - c[0]) + c[1] # line of slope "a" passing through point "(c[0], c[1])"
    c = svs[-2]
    margin_high = a * (xs - c[0]) + c[1]

    plt.figure()
    plotData(X,y)
    plt.plot(xs, db_1, 'b-', lw=1, label='Decision boundary')
    plt.plot(xs, margin_low, 'b--', lw=0.5, label='Margin')
    plt.plot(xs, margin_high, 'b--', lw=0.5)
    plt.plot(svs.T[0], svs.T[1], marker='o', ls='none', ms=15, mfc='none', mec='b', mew=0.5, label='Support vectors')
    plt.xlim(math.floor(min(X[:, 0]))-0.5, math.ceil(max(X[:, 0])))
    plt.ylim(math.floor(min(X[:, 1])), math.ceil(max(X[:, 1]))+2)
    plt.legend(numpoints = 1, loc = 1)
    plt.show()

# plotBoundary(X,y,linear_svm)

# To show acurracy of the model uncomment the next line.
# print(linear_svm.score(X,y))

# To predict values in some new data set, uncomment the next line
# linear_svm.predict(newX)