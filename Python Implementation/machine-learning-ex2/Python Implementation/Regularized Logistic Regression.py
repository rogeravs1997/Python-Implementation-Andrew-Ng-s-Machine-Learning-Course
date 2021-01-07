import numpy as np
from matplotlib import pyplot as plt 
import scipy.optimize as opt

#opening the data
data=open('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex2\ex2\ex2data2.txt','r').readlines()

#dumping the first 2 columns in X
X=[]
for i in data:
    X.append(i.split(",")[:-1])
for i in X:
    for e in range(len(i)):
        i[e]=float(i[e])
  
X=np.array(X,dtype=np.float64)

#dumping the las column in y   
y=[]
for i in data:
    y.append([i.split(",")[-1]])
for i in y:
    for e in range(len(i)):
        i[e]=float(i[e])
y=np.array(y,dtype=np.float64)    

def mapFeature(X1,X2):
    degree=6
    X1.shape = (X1.size, 1)
    X2.shape = (X2.size, 1)
    out = np.ones((len(X1), 1))
    for i in range(1,degree+1):
        for j in range(i+1):
            new=((X1**(i-j))*(X2**j))
            out=np.append(out,new,axis=1)
    return out

X=mapFeature(X[:,0],X[:,1])

def sigmoid(z):
    return ((1)/(1+np.exp(-z)))    

def costFunctionReg(theta, X, y, lmbda):
    m = len(y)
    temp1 = np.multiply(y, np.log(sigmoid(np.dot(X, theta))))
    temp2 = np.multiply(1-y, np.log(1-sigmoid(np.dot(X, theta))))
    return np.sum(temp1 + temp2) / (-m) + np.sum(theta[1:]**2) * lmbda / (2*m)

def gradRegularization(theta, X, y, lmbda):
    m = len(y)
    temp = sigmoid(np.dot(X, theta)) - y
    temp = np.dot(temp.T, X).T / m + theta * lmbda / m
    temp[0] = temp[0] - theta[0] * lmbda / m
    return temp

#set initial parameters
(m, n) = X.shape
initial_theta = np.zeros((n,1))
lam=1

#WE WILL USE THE MINIMIZE FUNCTION FROM SCIPY.OPTIMIZE, WITH THE METHOD 'TNC'
output = opt.fmin_tnc(func = costFunctionReg, x0 = initial_theta.flatten(), fprime = gradRegularization, \
                         args = (X, y.flatten(), lam))

#we dump our optimal theta values in to a variable
theta_found = output[0]
print("The regularized theta using ridge regression:\n",theta_found)
theta_found.shape=(len(theta_found),1)

def predict(X,theta_found):
    p=sigmoid(X.dot(theta_found))
    #print(p)  #print probability of being Positive(1)
    for i in range(len(p)):
        if p[i][0]>=0.5:
            p[i][0]=1
        else:
            p[i][0]=0
    return p


#Now we can calculate the acurracy of our model
acc=predict(X,theta_found)
counter=0
for i in range(len(y)):
    if y[i][0]==acc[i][0]:
        counter+=1

print("Acurracy of the model: {}%".format(counter/len(y)*100))

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
        plt.plot(plot_x, plot_y,'r')
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







    

    
    
    
    
    