import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt



# LOADING THE DATA FROM A .MAT FILE (MATLAB/OCTAVE FILE)
data=loadmat('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex5\ex5\ex5data1.mat')
X=data['X']
y = data['y']

m=np.size(X,0)


# defining cross validation data set
Xval=data['Xval']
yval=data['yval']

# defining test data set
Xtest=data['Xtest']
ytest=data['ytest']

# VISUALIZING THE DATASET (TRAIN DATA SET)
def plotTrain(X,y):
    fig = plt.figure()
    plt.plot(X,y,'rx')
    fig.suptitle('Training Data', fontsize=15)
    plt.xlabel('Change in water level (X)', fontsize=8)
    plt.ylabel('Water flowing out of the dam (y)', fontsize=8)

plotTrain(X,y)

# Defining regularized linear regression cost function
def costFunction(theta,X, y, Lambda):
    m=np.size(X,0)
    X=np.concatenate((np.ones((m,1)),X),axis=1)

    predictions=X.dot(theta)


    sqrError=np.square(predictions-y)

    J=(1/(2*m))*np.sum(sqrError)
    
    # Adding regularization
    regtheta=theta.copy()
    regtheta[0]=0
    reg=(Lambda/(2*m))*np.sum(np.square(regtheta))

    J=J+reg
    return J
    
theta=np.ones((2,1))
Lambda=0
cost=costFunction(theta,X, y, Lambda)

# DEFINING FUNCTION TO COMPUTE Regularized Linear Regression Gradient
def Gradient(theta,X, y, Lambda):
    m=np.size(X,0)
    X=np.concatenate((np.ones((m,1)),X),axis=1)


    predictions=X.dot(theta)
    regtheta=theta.copy()
    regtheta[0]=0

    grad=np.zeros(theta.shape)
    grad=((1/m)*X.T.dot(predictions-y))

    # Addding regularization
    reg=((Lambda/m)*regtheta)

    grad=grad+reg
    return grad
    
grad=Gradient(theta,X, y, Lambda)
    
# DEFINING FUNCTION TO FIND OPTIMAL VALUES OF THETA WITH FMNICG ALGORITHM FROM SCIPY
def optimizeTheta(X,y,Lambda=0):
    initial_theta=np.zeros((np.size(X,1)+1,1))
    optimal_theta = opt.fmin_cg(f = costFunction, x0 = initial_theta,  fprime = Gradient, args = (X, y.flatten(), Lambda),maxiter=10000)
    optimal_theta.shape=(initial_theta.shape)
    return optimal_theta


optimal_theta=optimizeTheta(X, y)

# DEFINING FUNCTION TO FOUND PREDICTED VALUES WITH OPTIMAL THETA
def predict(X,optimal_theta):
    m=np.size(X,0)
    X=np.concatenate((np.ones((m,1)),X),axis=1)
    predictions=X.dot(optimal_theta)
    predictions.shape=(m,1)
    return predictions

predictions=predict(X,optimal_theta)

# FUNCTION TO PLOT THE FITTING LINE FOUNDED WITH OPTIMAL THETA
def plotFit(X,y,predictions):
    fig = plt.figure()
    plt.plot(X,y,'rx')
    fig.suptitle('Training Data', fontsize=15)
    plt.xlabel('Chnage in water level (X)', fontsize=8)
    plt.ylabel('Water flowing out of the dam (y)', fontsize=8)
    plt.plot(X,predictions,'-',label='Linear fit')
    plt.legend()
    

plotFit(X,y,predictions)

# DEFINING FUNCTION TO COMPUTE LEARNING CURVES / VECTORS WITH ERROR IN TRAINING AND CROSS VAL DATA SETS FOR DIFERENT SIZES OF DATA SETS (m)
def learningCurves(X,y,Xval,yval,Lambda):
    m=np.size(X,0)
    error_train=np.zeros((m,1))
    error_val=np.zeros((m,1))
    
    for i in range(1,m+1):
        Xtrain=X[0:i,:]
        ytrain=y[0:i,:]
        optimal_theta=optimizeTheta(Xtrain,ytrain,Lambda)
        error_train[i-1]=costFunction(optimal_theta,Xtrain,ytrain,0)
        error_val[i-1]=costFunction(optimal_theta,Xval,yval,0)
    # plotting the learning curve
    fig2 = plt.figure()
    plt.plot(range(1,m+1),error_train,'-',label='Train error')
    plt.plot(range(1,m+1),error_val,'-',label='Cross Val. error')
    plt.xlabel('Data set Size', fontsize=10)
    plt.ylabel('cost(J): error', fontsize=10)
    plt.legend()
    return error_train, error_val

error_train , error_val = learningCurves(X,y,Xval,yval,0)

# DEFINING FUNCTION TO ADD POLINOMIAL FEATURES TO A DATA SET 'X' WHEN 'X' HAS JUST 1 FEATURE
def polyFeatures(X,p):
    m=np.size(X,0)
    X_poly=np.zeros((m,p))
    for i in range(1,p+1):
        X_poly[:,i-1]=(X**i)[:,0]

    return X_poly

polinomial_grade=8
X_poly=polyFeatures(X,polinomial_grade)
    
# DEFINING FUNCTION TO NORMALIZE THE FEATURE VAUES OF X
def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X_norm=(X-mu)/sigma
    return X_norm , mu , sigma
    
X_poly_norm,mu,sigma=featureNormalize(X_poly)
mu.shape=(1,polinomial_grade)
sigma.shape=(1,polinomial_grade)

# MAP XVAL TO POLINOMIAL FEATURES AND NORMALIZE WITH MU AND SIGMA FROM TRAINING SET
Xval_poly=polyFeatures(Xval, polinomial_grade)
Xval_poly_norm=(Xval_poly-mu)/sigma
# MAP XTEST TO POLINOMIAL FEATURES AND NORMALIZE WITH MU AND SIGMA FROM TRAINING SET
Xtest_poly=polyFeatures(Xtest, polinomial_grade)
Xtest_poly_norm=(Xtest_poly-mu)/sigma


# NOW WE ABLE TO FIND A NEW SET OF VALUES FOR THETA THAT MINIMIZE THE COST FUNCTION ERROR
Lambda=0
optimal_theta=optimizeTheta(X_poly_norm, y,Lambda)
predictions=predict(Xval_poly_norm, optimal_theta)

# and compute and visualize the error values in training and cross val data sets
error_train , error_val =learningCurves(X_poly_norm, y, Xval_poly_norm, yval, Lambda)

# FUNCTION TO VISUALIZE THE POLYNOMIAL FIT 
def plotFit2(polyFeatures, min_x, max_x, mu, sigma, theta, p):
    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.arange(min_x - 15, max_x + 10, 0.05).reshape(-1, 1)

    # Map the X values
    X_poly = polyFeatures(x, p)
    X_poly -= mu
    X_poly /= sigma

    # Add ones
    X_poly = np.concatenate([np.ones((x.shape[0], 1)), X_poly], axis=1)

    # Plot
    fig3=plt.figure()
    plt.plot(x, np.dot(X_poly, theta), '--', lw=2)
    plt.plot(X, y, 'ro', ms=10, mew=1.5, mec='k')
    plt.xlabel('Chnage in water level (X)', fontsize=8)
    plt.ylabel('Water flowing out of the dam (y)', fontsize=8)
    plt.title('Polynomial Regression Fit (lambda = %f)' % Lambda)
    plt.grid()

plotFit2(polyFeatures, np.min(X), np.max(X), mu, sigma, optimal_theta, polinomial_grade)


# NOW WE COMPUTE A FUNCTION TO FIND AN OPTIMAL VALUE FOR LAMBDA WITH ITERATION OPTIMIZATION
def optimizeLambda(X_poly_norm,y,Xval_poly_norm,yval):
    lambda_values=[i for i in np.arange(0,10,0.001)]
    train_errorDict={}
    crossval_errorDict={}
    for l in lambda_values:
        optimal_theta=optimizeTheta(X_poly_norm, y,l)
        train_errorDict[l]=costFunction(optimal_theta, X_poly_norm, y, 0)
        crossval_errorDict[l]=costFunction(optimal_theta, Xval_poly_norm, yval, 0)

    optimal_Lambda=min(crossval_errorDict, key=crossval_errorDict.get)

    return optimal_Lambda


optimal_Lambda=optimizeLambda(X_poly_norm,y,Xval_poly_norm,yval)

# optimal_Lambda=2.972
