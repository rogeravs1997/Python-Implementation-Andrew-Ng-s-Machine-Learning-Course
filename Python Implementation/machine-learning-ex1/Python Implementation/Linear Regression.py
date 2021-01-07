import numpy as np
from matplotlib import pyplot as plt 

#load data
data=open('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex1\ex1\ex1data1.txt','r').readlines()


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


#setting an initial values for theta , alpha and # of iters
initial_theta=np.zeros((len(X[1]),1))
alpha=0.01
num_iters=4000

#dumping values of theta in theta_found and the history o costJ in J_history 
theta_found,J_history=gradientDescent(X,y,initial_theta,alpha,num_iters)

print(theta_found)
#ploting DATA: y in function of X (just for one-variable problems)
def plotData(X,y):
    fig = plt.figure()
    plt.plot(X[:,1],y,'bx')
    fig.suptitle('Training Data: Profit x Population', fontsize=15)
    plt.xlabel('Population of City in 10,000s', fontsize=8)
    plt.ylabel('Profit in $10,000s', fontsize=8)


#plotting cost of J in function of number of iterations
def plotCost(J_history):
    fig = plt.figure()
    plt.plot(J_history)
    #plt.axis([0, num_iters+1, 4, 6.5]) #we modify this values for the ones of our firts Step learning, so we can see the behaviour o J(theta) in the time
    fig.suptitle('Cost(J) in the time', fontsize=15)
    plt.xlabel('Number of Iterations', fontsize=8)
    plt.ylabel('Cost(J)', fontsize=8)
    
#plotting the linear fit(just for one-variable problems)
def plotLinearFit(X,y,theta_found):
    fig = plt.figure()
    plt.plot(X[:,1],y,'bx')
    plt.plot(X[:,1],X.dot(theta_found),'r')
    fig.suptitle('Training Data: Profit x Population', fontsize=15)
    plt.xlabel('Population of City in 10,000s', fontsize=8)
    plt.ylabel('Profit in $10,000s', fontsize=8)

plotData(X, y)