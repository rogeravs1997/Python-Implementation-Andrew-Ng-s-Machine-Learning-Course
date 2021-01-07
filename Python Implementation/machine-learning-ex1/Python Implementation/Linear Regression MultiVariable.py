import numpy as np
from matplotlib import pyplot as plt 

#load data
data=open('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex1\ex1\ex1data2.txt','r').readlines()

#loading variables in X
X=[]
for i in data:
    X.append((i.split(',')[:-1]))

for m in X:
    for x in range(len(m)):
        m[x]=float(m[x])

#add an extra column of 1's to X
for m in X:
    m.insert(0,1.0)    
X=np.array(X,dtype=np.float64)

#loading labels in y
y=[]
for i in data:
    y.append([float(i.split(',')[-1])])
y=np.array(y,dtype=np.float64)

#defining function to compute the Cost of J(theta)
def computeCost(X,y,theta):
    m=len(y)
    pred=X.dot(theta)
    sqrError=np.square(pred-y)
    J=(1/(2*m))*np.sum(sqrError)
    return J

def gradientDescent(X, y, theta, alpha=0.01, num_iters=100):
    m=len(y)
    J=[]
    for i in range(num_iters):
        error=X.dot(theta)-y
        delta=(1/m)*error.T.dot(X)
        theta=theta-(alpha*delta.T)
        J.append([computeCost(X,y,theta)])
    J=np.array(J)
    return theta , J

#scaling the features
def featureScaling(X):
    mu=np.mean(X[:,1:],0)
    sigma=np.std(X[:,1:],0)
    X_norm=(X[:,1:]-np.mean(X[:,1:],0))/np.std(X[:,1:],0)
    X_norm=np.concatenate((np.ones((len(X),1)),X_norm),axis=1)
    
    return X_norm , mu , sigma

#dumping the results of scaling the features in 3 variables
X,mu,sigma=featureScaling(X)



#setting an initial values for theta , alpha and # of iters
initial_theta=np.zeros((len(X[1]),1))
alpha=0.0033
num_iters=4000

#dumping values of theta in theta_found and the history o costJ in J_history 
theta_found,J_history=gradientDescent(X,y,initial_theta,alpha,num_iters)

print(theta_found)


#plotting cost of J in function of number of iterations
def plotCost(J_history):
    fig = plt.figure()
    plt.plot(J_history)
    #plt.axis([0, num_iters+1, 4, 6.5]) #we modify this values for the ones of our firts Step learning, so we can see the behaviour o J(theta) in the time
    fig.suptitle('Cost(J) in the time', fontsize=15)
    plt.xlabel('Number of Iterations', fontsize=8)
    plt.ylabel('Cost(J)', fontsize=8)
    
plotCost(J_history)
    