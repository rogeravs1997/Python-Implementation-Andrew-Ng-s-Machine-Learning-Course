import matplotlib.pyplot as plt
import numpy as np
J_history=open(r'D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex4\Python Implementation\cost_history_1000its_lam0_01.txt','r').readlines()
for i in range(len(J_history)):
    J_history[i]=float(J_history[i])


def plotCost(J_history):
    fig = plt.figure()
    plt.plot(J_history)
    #plt.axis([0, num_iters+1, 4, 6.5]) #we modify this values for the ones of our firts Step learning, so we can see the behaviour o J(theta) in the time
    fig.suptitle('Cost(J) in the time', fontsize=15)
    plt.xlabel('Number of Iterations', fontsize=8)
    plt.ylabel('Cost(J)', fontsize=8)
    
plotCost(J_history)

