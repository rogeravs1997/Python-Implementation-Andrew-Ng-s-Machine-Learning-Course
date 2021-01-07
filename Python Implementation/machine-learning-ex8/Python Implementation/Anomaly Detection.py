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


# IF WE WOULD TRY A NEW VALUE TO DECIDE IF ITS ANOMALOUS OR NOT, WE WOULD HAVE TO COMPARE IT WITH THE EPSILON FOUND

# newX = load('')
# newp = multivariateGaussian(newX, mu, sigma2)
# if newp < bestEpsilon >>>> anomaly



