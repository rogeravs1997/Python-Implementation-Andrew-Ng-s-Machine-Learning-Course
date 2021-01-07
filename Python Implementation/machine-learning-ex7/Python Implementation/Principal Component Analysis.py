# USEFUL LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import scipy.io #Used to load the OCTAVE *.mat files

# Loading data
data= scipy.io.loadmat('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex7\ex7\ex7data1.mat')

X=data['X']

# DEFINING FUNCTION TO PLOT THE DATA (JUST FOR 2D DATA)
def plotData(X):
    plt.figure()
    plt.plot(X[:,0],X[:,1],'bo')
    plt.xlabel('X 1')
    plt.ylabel('X 2')
    plt.title('Data Set in 2D')
# plotData(X)

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X_norm=(X-mu)/sigma
    return X_norm , mu , sigma

#dumping the results of scaling the features in 3 variables
X_norm , mu , sigma = featureNormalize(X)

# DEFINING FUNCTION TO COMPUTE THE EIGENVECTORS TO PROJECT DATA AND PARAMETER S TO COMPUTE VARIANCE RETAINED
def pca(X):
    m , n = X.shape
    U = np.zeros((n,n))
    S = np.zeros((n,n))
    
    sigma=(1/m)*(X.T.dot(X))
    U , S , V = np.linalg.svd(sigma)
    
    return U , S

U , S = pca(X_norm)

# DEFINING FUNCTION TO PROJECT DATA INTO EIGENVECTORS
def projectData(X,U,K):
    m, n = X.shape
    Z=np.zeros((m,K))
    
    U_reduce=U[:,0:K]
    Z=X.dot(U_reduce)
    
    return Z

K=1 #NUMBER OF FEATURES WHE WANT TO GET
Z=projectData(X_norm,U,K)

# DEFINING FUNCTION TO RECOVER O RECONSTRUCT DATA FROM PROJECTIONS
def recoverData(Z,U,K):
    U_reduce=U[:,0:K]
    X_rec=Z.dot(U_reduce.T)
    
    return X_rec

X_rec=recoverData(Z,U,K)

# FUNCTION TO PLOT THE ORIGINAL DATA VS RECOVERED DATA (JUST FOR 2D DATA)

def plotProjections(X,X_rec):
    plt.figure()
    plt.plot(X[:,0],X[:,1],'bo',label='Original Data')
    plt.plot(X_rec[:,0],X_rec[:,1],'ro',label='Recovered Data')
    plt.xlabel('X 1')
    plt.ylabel('X 2')
    plt.legend()
    plt.title('Data Set in 2D VS Projected Data in 1D')
plotProjections(X_norm,X_rec)