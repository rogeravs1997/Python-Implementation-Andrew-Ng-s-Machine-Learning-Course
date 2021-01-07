# IMPORT USEFUL LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import scipy.io #Used to load the OCTAVE *.mat files
from scipy.stats import multivariate_normal

# LOADING DATA FROM MAT FILE AND DUMPING INTO VARIABLES
data=scipy.io.loadmat('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex8\ex8\ex8data2')

X=data['X']
Xval=data['Xval']
yval=data['yval']

# COMPUTING MEAN AND VARIANCE FOR THE TRAINING DATASET, USEFUL TO COMPUTE GAUSSIAN FORMULA
mu=np.mean(X,0)
sigma2=np.var(X,0)

# COMPUTING DENSITY PROBABILITY FUNCTION FOR MULTIVARIATE GAUSSIAN DISTRIBUTION in TRAINING AND CV DATA SET
def multivariateGaussian(X,mu,sigma2):
    return multivariate_normal.pdf(X,mu,sigma2)

p=multivariateGaussian(X, mu, sigma2)
pval=multivariateGaussian(Xval, mu, sigma2)

# FINDING OPTIMAL THRESHOLD (EPSILON) GIVEN A CROSS VALIDATION LABELED DATA SET
def selectThreshold(yval,pval):
    bestEpsilon=0
    bestF1=0
    n=0
    partitions=1000000
    for epsilon in np.arange(np.min(pval),np.max(pval),(np.max(pval)-np.min(pval))/partitions):
        cvPredictions=(pval<epsilon)
        cvPredictions.shape = (cvPredictions.size,1)
        tp=sum(np.logical_and((cvPredictions==1), (yval==1)))
        fp=sum(np.logical_and((cvPredictions==1), (yval==0)))
        fn=sum(np.logical_and((cvPredictions==0), (yval==1)))
        precission=tp/(tp+fp)
        recall=tp/(tp+fn)
        F1=(2*precission*recall)/(precission+recall)
        
        if F1>bestF1:
            bestF1=F1
            bestEpsilon=epsilon
        n+=1
        print('Completed: {}/{}'.format(n,partitions))
            
    return bestEpsilon, bestF1

bestEpsilon , bestF1 = selectThreshold(yval, pval)
print('F1 Score: {}'.format(bestF1))
print('Optimal Epsilon Value: {}'.format(bestEpsilon))

# Creating a Variable with the odd values
ind=np.where(p<bestEpsilon)
X_outer=X[ind]
# IF WE WOULD TRY A NEW VALUE TO DECIDE IF ITS ANOMALOUS OR NOT, WE WOULD HAVE TO COMPARE IT WITH THE EPSILON FOUND

# newX = load('')
# newp = multivariateGaussian(newX, mu, sigma2)
# if newp < bestEpsilon >>>> anomaly

######### IM GOING TO USE PCA TO VISUALIZE THE DATA IN 2D AND HIGHLIGHT THE OUTLIER VALUES #######
def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X_norm=(X-mu)/sigma
    return X_norm , mu , sigma

#dumping the results of scaling the features in 3 variables
X_norm , muNorm , sigmaNorm = featureNormalize(X)
X_outer_norm, mu_outer_norm , sigma_outer_norm = featureNormalize(X_outer)
# DEFINING FUNCTION TO COMPUTE THE EIGENVECTORS TO PROJECT DATA AND PARAMETER S TO COMPUTE VARIANCE RETAINED
def pca(X):
    m , n = X.shape
    U = np.zeros((n,n))
    S = np.zeros((n,n))
    
    sigma=(1/m)*(X.T.dot(X))
    U , S , V = np.linalg.svd(sigma)
    
    return U , S

U , S = pca(X_norm)
U_outer , S_outer = pca(X_outer_norm)
# DEFINING FUNCTION TO PROJECT DATA INTO EIGENVECTORS
def projectData(X,U,K):
    m, n = X.shape
    Z=np.zeros((m,K))
    
    U_reduce=U[:,0:K]
    Z=X.dot(U_reduce)
    
    return Z

K=2 #NUMBER OF FEATURES WHE WANT TO GET
Z=projectData(X_norm,U,K)
Z_outer=projectData(X_outer_norm,U_outer,K)


# FUNCTION TO PLOT THE ORIGINAL DATA VS RECOVERED DATA (JUST FOR 2D DATA)

def plotProjections(Z):
    plt.figure()
    plt.plot(Z[:,0],Z[:,1],'bo',label='Non-Anomalies')
    plt.plot(Z_outer[:,0],Z_outer[:,1],'ro',label='Anomalies')
    plt.legend()
    plt.title('Data Set in 2D')
plotProjections(Z)


