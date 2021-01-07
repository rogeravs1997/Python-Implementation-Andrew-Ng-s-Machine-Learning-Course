import numpy as np
from scipy.io import loadmat
import scipy.optimize as opt
import matplotlib.pyplot as plt
from PIL import Image  

#loading DATA
data=loadmat('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex4\ex4\ex4data1.mat')
X=data['X']
y = data['y']

# #loading weights for trials
weights=loadmat('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex4\ex4\ex4weights.mat')
Theta1=weights['Theta1']
Theta2=weights['Theta2']

# hidden_layer_size=25
# input_layer_size=400
# num_labels=10

#Unrolling parameters
nn_params=np.concatenate((Theta1.ravel(),Theta2.ravel()))
nn_params.shape=(np.size(nn_params,0),1)

#Re rolling parameters
# Theta11=np.reshape(nn_params[:hidden_layer_size*(1+input_layer_size),:],(hidden_layer_size,1+input_layer_size))
# Theta22=np.reshape(nn_params[hidden_layer_size*(1+input_layer_size):,:],(num_labels,1+hidden_layer_size))

#defining sigmoid function
def sigmoid(z):
    return ((1)/(1+np.exp(-z)))    

#defining sigmoid Gradient function
def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z))

#funtion to compute the cost J(theta)
def costFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda):
    """
    costFunction compute the cost J(theta) for an unrolled vector theta (from theta1 and theta2)
    it works just for a neurak network architecture with just one hidden layer.

    Parameters
    ----------
    X : Training features.
    y : Training labels.
    nn_params : theta1 and theta2 unrolled in one vector.
    Lambda : lambda.
    input_layer_size : number of units in input layer.
    hidden_layer_size : number of units in hidden layer.
    num_labels : number of units in output layer.

    Returns
    -------
    the cost error of nn_params (theta) with real values "y"

    """
    #unrolling nn_params into Theta1 and Theta2
    Theta1=np.reshape(nn_params[:hidden_layer_size*(1+input_layer_size)],(hidden_layer_size,1+input_layer_size))
    Theta2=np.reshape(nn_params[hidden_layer_size*(1+input_layer_size):],(num_labels,1+hidden_layer_size))
    
    #setup some useful variable
    m=np.size(X,0) #5000
    
    #FORWARD PROPAGATION
    a1=np.concatenate((np.ones((m,1)),X),axis=1) #(5000*401)
    a1=a1.T #(401*5000)
    z2=Theta1.dot(a1) #(25*401)*(401*5000)=(25*5000)
    a2=sigmoid(z2) #(25*5000)
    a2=np.concatenate((np.ones((1,m)),a2)) #(26*5000)
    z3=Theta2.dot(a2) #(10*26)*(26*5000)=(10*5000)
    a3=sigmoid(z3) #(10*5000)
    
    #Setting an equivalent representation of y where y=1 => [[1],[0],[0],[0],[0],[0],[0],[0],[0],[0]] and so on
    y_new=np.zeros((num_labels,m)) #(10*5000)
    
    for i in range(5000):
        y_new[int(y[i])-1,i]=1
    
    J=(1/m)*np.sum(np.sum((-1*y_new)*np.log(a3)-(1-y_new)*np.log(1-a3)))
    
    #adding regularization
    regTheta1=Theta1[:,1:]
    regTheta2=Theta2[:,1:]
    
    reg=(Lambda/(2*m))*(np.sum(np.sum(regTheta1**2))+np.sum(np.sum(regTheta2**2)))
    
    J=J+reg
    
    return J

def Gradient(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda):
    m=np.size(X,0) #5000
    #unrolling nn_params into Theta1 and Theta2
    Theta1=np.reshape(nn_params[:hidden_layer_size*(1+input_layer_size)],(hidden_layer_size,1+input_layer_size))
    Theta2=np.reshape(nn_params[hidden_layer_size*(1+input_layer_size):],(num_labels,1+hidden_layer_size))
    #Setting an equivalent representation of y where y=1 => [[1],[0],[0],[0],[0],[0],[0],[0],[0],[0]] and so on
    y_new=np.zeros((num_labels,m)) #(10*5000)
    for i in range(5000):
        y_new[int(y[i])-1,i]=1
    
    #BACK PROPAGATION 
    Theta1_grad=np.zeros((hidden_layer_size,input_layer_size+1)) #(25*401)
    Theta2_grad=np.zeros((num_labels,hidden_layer_size+1)) #(10*26)
    X=np.concatenate((np.ones((m,1)),X),axis=1)
    for t in range(m):
        #Step1: Forward propagation
        a1=X[t,:]
        a1=a1.T #(401*1)
        a1.shape=(np.size(a1,0),1)
        z2=Theta1.dot(a1) #(25*401)*(401*1)=(25*1)
        a2=sigmoid(z2) #(25*1)
        a2.shape=(np.size(a2,0),1)
        a2=np.concatenate((np.ones((1,1)),a2)) #(26*1)
        z3=Theta2.dot(a2) #(10*26)*(26*1)=(10*1)
        a3=sigmoid(z3) #(10*1)

            
        #Step2: Computing delta3
        lbl=y_new[:,t]
        lbl.shape=(np.size(lbl,0),1)
        delta3=a3-lbl #(10*1)

        #Step3: Computing delta2
        z2.shape=(np.size(z2,0),1)
        z2=np.concatenate((np.ones((1,1)),z2)) #(26*1)
        delta2=(Theta2.T.dot(delta3))*sigmoidGradient(z2) #(26*10)*(10*1)=(26*1)
        delta2 = delta2[1:] #skipping sigma2(0) (25*1)
        
        #Step4: Acumulate the gradient
        Theta1_grad+=(delta2.dot(a1.T)) #(25*1)*(1*401)=(25*401)
        Theta2_grad+=(delta3.dot(a2.T)) #(10*1)*(1*26)=(10*26)
        
    Theta1_grad=(1/m)*Theta1_grad #(25*401)
    Theta2_grad=(1/m)*Theta2_grad #(10*26)
    
    #ADDING REGULARIZATION
    Theta1_grad[:,1:]=Theta1_grad[:,1:]+((Lambda/m)*Theta1[:,1:])
    Theta2_grad[:,1:]=Theta2_grad[:,1:]+((Lambda/m)*Theta2[:,1:])  
    
    #finally unroll the gradient
    grad=np.concatenate((Theta1_grad.ravel(),Theta2_grad.ravel()))

    
    return grad

input_layer_size=400     
hidden_layer_size=25
num_labels=10  
Lambda=1
max_iter=500

#Defining Random Initialize function
def randomInitialize(input_layer_size,hidden_layer_size,num_labels):
    epsilon_init=0.12
    initial_Theta1=np.random.rand(hidden_layer_size,input_layer_size+1)*2*epsilon_init-epsilon_init
    initial_Theta2=np.random.rand(num_labels,hidden_layer_size+1)*2*epsilon_init-epsilon_init
    initial_nn_params=np.concatenate((initial_Theta1.ravel(),initial_Theta2.ravel()))
    # initial_nn_params.shape=(np.size(initial_nn_params,0),1)
    return initial_nn_params

initial_nn_params=randomInitialize(input_layer_size,hidden_layer_size,num_labels)

grad=Gradient(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
# cost=costFunction(X,y,initial_nn_params,Lambda,input_layer_size,hidden_layer_size,num_labels)
def checkGradient(initial_nn_params,grad,input_layer_size, hidden_layer_size, num_labels,myX,myy,mylambda=0.):
    myeps = 0.0001
    flattened = initial_nn_params
    flattenedDs = grad
    n_elems = len(flattened) 
    #Pick ten random elements, compute numerical gradient, compare to respective D's
    for i in range(10):
        x = int(np.random.rand()*n_elems)
        epsvec = np.zeros((n_elems,1))
        epsvec[x] = myeps

        cost_high = costFunction(flattened + epsvec.flatten(),input_layer_size, hidden_layer_size, num_labels,myX,myy,mylambda)
        cost_low  = costFunction(flattened - epsvec.flatten(),input_layer_size, hidden_layer_size, num_labels,myX,myy,mylambda)
        mygrad = (cost_high - cost_low) / float(2*myeps)
        print("Element: {0}. Numerical Gradient = {1:.9f}. BackProp Gradient = {2:.9f}.".format(x,mygrad,flattenedDs[x]))
        
#checkGradient(initial_nn_params, grad, input_layer_size, hidden_layer_size, num_labels, X, y,Lambda)


theta_opt = opt.fmin_cg(maxiter = max_iter, f = costFunction, x0 = initial_nn_params, fprime = Gradient, \
                        args = (input_layer_size, hidden_layer_size, num_labels, X, y.flatten(), Lambda))

Theta1=np.reshape(theta_opt[:hidden_layer_size*(1+input_layer_size)],(hidden_layer_size,1+input_layer_size))
Theta2=np.reshape(theta_opt[hidden_layer_size*(1+input_layer_size):],(num_labels,1+hidden_layer_size))

def predict(Theta1,Theta2,X):
    #adding a column of 1's to X before computing with Theta1
    X=np.concatenate((np.ones((len(X),1)),X),axis=1)
    a2=sigmoid(X.dot(Theta1.T))
    #adding a column of 1's to a2 before computing with Theta2
    a2=np.concatenate((np.ones((len(a2),1)),a2),axis=1)
    a3=sigmoid(a2.dot(Theta2.T))
    a3=a3.argmax(axis=1)+1
    a3=np.array(a3)
    return a3

prediction=predict(Theta1, Theta2, X)

def acurracy(prediction,y):
    counter=0
    for i in range(len(y)):
        if y[i][0]==prediction[i]:
            counter+=1
    acc=(counter/len(y)*100)
    print("Acurracy of the model: {}%".format(acc))
    return acc

acurracy(prediction, y)