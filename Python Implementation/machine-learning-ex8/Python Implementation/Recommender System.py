# IMPORT USEFUL LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.optimize as opt
import re 
# LOADING DATA FROM MAT FILE AND DUMPING INTO VARIABLES
data=scipy.io.loadmat('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex8\ex8\ex8_movies.mat')

Y=data['Y'] # Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users.
R=data['R'] # R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i.

# LOADING PRE-TRAINED WEIGHTS FROM MAT FILE AND DUMPING INTO VARIABLES
data2=scipy.io.loadmat('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex8\ex8\ex8_movieParams.mat')

X=data2['X'] # X is a 1682x10 matrix, containing 10 features of 1682 movies.
Theta=data2['Theta'] # Theta is a 943x10 matrix, containing pre-trained weights of 943 users for 10 features.

# LOADING THE LIST OF MOVIES
movie_list=np.loadtxt('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex8\ex8\movie_ids.txt', delimiter='/n', dtype=str)

# DEFINING COST FUNCTION

def cofiCostFunc(params,Y,R,num_users, num_movies, num_features,Lambda):
    
    X = params[:num_movies * num_features].reshape(num_movies, num_features)
    Theta = params[num_movies * num_features:].reshape(num_users, num_features)
    
    J=1/2*sum(sum((R*(np.dot(X,Theta.T)-Y))**2))
    reg=(Lambda/2)*(sum(sum(Theta**2))+sum(sum(X**2)))
    return J+reg

# DEFINING GRADIENT FUNCTION

def gradFunction(params,Y,R,num_users, num_movies, num_features,Lambda):
    X = params[:num_movies * num_features].reshape(num_movies, num_features)
    Theta = params[num_movies * num_features:].reshape(num_users, num_features)
    
    X_grad=(R*(np.dot(X,Theta.T)-Y)).dot(Theta)
    Theta_grad=(R*(np.dot(X,Theta.T)-Y)).T.dot(X)
    
    X_reg=Lambda*X
    Theta_reg=Lambda*Theta
    
    X_grad=X_grad+X_reg
    Theta_grad=Theta_grad+Theta_reg
    
    grad=np.concatenate((X_grad.ravel(), Theta_grad.ravel()))
    
    return grad

# ADDING MY RATINGS
num_movies, num_users = Y.shape
num_features=10
my_ratings = np.zeros(num_movies)

my_ratings[1] = 4
my_ratings[98] = 2
my_ratings[7] = 3
my_ratings[12]= 5
my_ratings[54] = 4
my_ratings[64]= 5
my_ratings[66]= 3
my_ratings[69] = 5
my_ratings[183] = 4
my_ratings[226] = 5
my_ratings[355] = 5

Y = np.hstack((my_ratings.reshape(-1, 1), Y))
R = np.hstack(((my_ratings != 0).reshape(-1, 1), R))

num_movies, num_users = Y.shape



# SETTING INITIAL PARAMETERS
Lambda = 10
initial_X = np.random.normal(size=(num_movies, num_features))
initial_Theta = np.random.normal(size=(num_users, num_features))
params = np.hstack((initial_X.ravel(), initial_Theta.ravel()))



res = opt.fmin_cg(f = cofiCostFunc, x0 = params,  fprime = gradFunction, args = (Y, R ,num_users, num_movies, num_features,Lambda), maxiter=1000)
X = res[:num_movies * num_features].reshape(num_movies, num_features)
Theta = res[num_movies * num_features:].reshape(num_users, num_features)
my_predictions = X.dot(Theta.T)[:, 0]

top15=my_predictions.argsort()[::-1][:15]

print('15 recommended movies for you:')
for i in top15:
    print('- {} with an estimated rating of: {:.1f}.'.format(movie_list[i],my_predictions[i]))