import numpy as np
from matplotlib import pyplot as plt 
import scipy.optimize as opt

#opening the data
data=open('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex2\ex2\ex2data1.txt','r').readlines()

#dumping the first 2 columns in X
X=[]
for i in data:
    X.append(i.split(",")[:-1])
for i in X:
    for e in range(len(i)):
        i[e]=float(i[e])
#add an extra column of 1's to X
for m in X:
    m.insert(0,1.0)    
X=np.array(X,dtype=np.float64)

#dumping the las column in y   
y=[]
for i in data:
    y.append([i.split(",")[-1]])
for i in y:
    for e in range(len(i)):
        i[e]=float(i[e])
y=np.array(y,dtype=np.float64)    


def sigmoid(z):
    return ((1)/(1+np.exp(-z)))    

def costFunction(theta, X, y):
    m=len(X)
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    h=sigmoid(X*theta.T)
    first = np.multiply(-y, np.log(h))
    second = np.multiply((1 - y), np.log(1 - h))
    return np.sum(first - second) / m


def gradient(theta, X, y):
    m=len(X)
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / m
    return grad


#set initial parameters to theta: 0's [0 0 0]
initial_theta=np.zeros((1,len(X[1])))
grad=gradient(initial_theta, X, y)
#WE WILL USE THE MINIMIZE FUNCTION FROM SCIPY.OPTIMIZE, WITH THE METHOD 'TNC'
result = opt.fmin_tnc(func=costFunction, x0=initial_theta, fprime=gradient, args=(X, y))

#we dump our optimal theta values in to a variable
theta_found=[]
theta_found.append(result[0])
theta_found=np.array(theta_found)


def predict(X,theta_found):
    p=sigmoid(X.dot(theta_found.T))
    #print(p)  #print probability of being Positive(1)
    for i in range(len(p)):
        if p[i][0]>=0.5:
            p[i][0]=1
        else:
            p[i][0]=0
    return p

X_test=[[45,85],[62,60]] #we can also load from a database, this is just an example
#add an extra column of 1's to X_test
for m in X_test:
    m.insert(0,1.0)    
X_test=np.array(X_test,dtype=np.float64)

p=predict(X_test,theta_found)  #here we have the predicted values

#Now we can calculate the acurracy of our model
acc=predict(X,theta_found)
counter=0
for i in range(len(y)):
    if y[i][0]==acc[i][0]:
        counter+=1

print("Acurracy of the model: {}%".format(counter))

#function to visualize the data
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
    positive=plt.plot(pos[:,1],pos[:,2], 'o', label='Positive')
    negative=plt.plot(neg[:,1],neg[:,2], 'rx', label='Negative')
    plt.xlabel('Test 1 Score', fontsize=8)
    plt.ylabel('Test 2 Score', fontsize=8)
    plt.legend(loc='upper right')


#function to Plot the Boundary
def plotBoundary(X,y,theta_found):
    plotData(X,y)
    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])
        # Calculate the decision boundary line
        plot_y = (-1. / theta_found[0][2]) * (theta_found[0][1] * plot_x + theta_found[0][0])
        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y,'black')
        # Legend, specific for the exercise

        plt.xlim([30, 100])
        plt.ylim([30, 100])
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))
        def mapFeatureForPlotting(X1, X2):
            degree = 6
            out = np.ones(1)
            for i in range(1, degree+1):
                for j in range(i+1):
                    out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))))
            return out
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = np.dot(mapFeatureForPlotting(u[i], v[j]), theta_found)

        countor=plt.contour(u,v,z,0,colors='black')
        plt.show()



plotBoundary(X,y,theta_found)







    

    
    
    
    
    