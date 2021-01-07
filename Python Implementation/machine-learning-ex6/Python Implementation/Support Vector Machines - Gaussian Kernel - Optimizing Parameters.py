# USEFUL LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.optimize #fmin_cg to train the linear regression
from sklearn.svm import SVC #SVM software
# Loading data
data= scipy.io.loadmat('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex6\ex6\ex6data3.mat')

X=data['X']
y=data['y'].reshape(-1) #necessary for SVM that labels array has 0 dimensions

Xval= data['Xval']
yval = data['yval']
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
    plt.grid(True)   


# FINDING THE OPTIMUS PARAMETERS C AND SIGMA BY ITERATION

def optimizeParameters(X,y,Xval,yval):
    vals = [i for i in np.arange(0.1,10,0.1)]
    C_pts_mesh, sigma_pts_mesh = np.meshgrid(vals, vals)
    scores = np.vstack((C_pts_mesh.ravel(), sigma_pts_mesh.ravel(), np.zeros(len(vals)**2))).T
    for i, (C, sigma) in enumerate(scores[:, :2]):
        gamma = 1 / 2 / sigma**2
        gaussian_svm = SVC(C=C, gamma=gamma, kernel='rbf')
        gaussian_svm.fit(X, y)
        scores[i, 2] = gaussian_svm.score(Xval, yval)
    
    highest_acc_index = scores[:, 2].argmax()
    C, sigma = scores[highest_acc_index][:2]
    return C, sigma

# Training the SVM ussing specialized library
C , sigma = optimizeParameters(X,y,Xval,yval)
gamma = 1 / 2 / sigma**2
gaussian_svm = SVC(C=C, gamma=gamma, kernel='rbf')
gaussian_svm.fit(X, y)


#Function to draw the SVM boundary
def plotBoundary(X,y,gaussian_svm):

    x_0_pts, x_1_pts = np.linspace(min(X[:, 0]), max(X[:, 0]), 500), np.linspace(min(X[:, 1]), max(X[:, 1]), 500)
    x_0_pts_mesh, x_1_pts_mesh = np.meshgrid(x_0_pts, x_1_pts)
    z = gaussian_svm.predict(np.vstack((x_0_pts_mesh.ravel(), x_1_pts_mesh.ravel())).T).reshape(x_0_pts_mesh.shape)
    plotData(X,y)
    plt.contour(x_0_pts, x_1_pts, z, [0])
    plt.xlim(min(X[:, 0]), max(X[:, 0]))
    plt.ylim(min(X[:, 1]), max(X[:, 1]))
    plt.legend(numpoints = 1, loc = 1)
    plt.show()

# plotBoundary(X,y,gaussian_svm)
    
# To show acurracy of the model uncomment the next line.
# print(gaussian_svm.score(Xval,yval))

# To predict values in some new data set, uncomment the next line
# gaussian_svm.predict(newX)