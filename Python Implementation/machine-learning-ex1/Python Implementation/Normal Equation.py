import numpy as np

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

#defining Normal Equation Function
def normalEquation(X,y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

theta_found=normalEquation(X, y)

print(theta_found)