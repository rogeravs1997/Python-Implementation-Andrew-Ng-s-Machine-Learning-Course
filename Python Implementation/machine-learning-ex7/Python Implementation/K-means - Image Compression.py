# USEFUL LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import scipy.io #Used to load the OCTAVE *.mat files
import imageio

################################K-MEANS USING SCIKIT LEARN ##################################
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(X)
# idx = kmeans.predict(X)
# centroids = kmeans.cluster_centers_
#################################################################################################

# Loading data
X=imageio.imread(r'D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex7\ex7\bird_small.png')
X=X/255
X = X.reshape(-1, 3)


# DEFINING FUNCTION TO RANDOM INITIAIZE THE CENTROIDS
def kMeansInitCentroids(X,K):
    randidx=np.random.permutation(np.size(X,0))
    centroids=X[randidx[0:K],:]
    return centroids


# DEFINING FUNCTION TO COMPUTE CLOSEST CENTROID FOR EACH POINT

def findClosestCentroids(X,centroids):
    # set K (number of clusters)
    K= np.size(centroids,0) 
    # set m (number of training examples)
    m=np.size(X,0)
    # We need to return the following variable correctly.
    idx = np.zeros((m, 1))

    for i in range(m):
        init_error=np.sqrt(np.sum((X[i,:]-centroids[0,:])**2))
        idx[i]=1
        for k in range(1,K):
            error=np.sqrt(np.sum((X[i,:]-centroids[k,:])**2))
            if error <= init_error:
                init_error=error
                idx[i]=int(k+1)
    return idx


# DEFINING FUNCTION TO COMPUTE THE NEW CENTROID WITH POITNS ASSIGNED TO EACH CENTROID 

def computeCentroids(X,idx,K):
    [m,n]=X.shape
    centroids = np.zeros((K, n))
    
    for k in range(1,K+1):
        num_el=np.sum(idx==k)
        ind=np.where(idx==k)
        suma=np.zeros((1,n))
        for i in range(np.size(ind[0])):
            suma=suma+X[ind[0][i],:]
        new_centroid=suma/num_el
        centroids[k-1,:]=new_centroid

    return centroids


# DEFINING COST FUNCTION
def costFunction(X,K,idx,centroids):
    [m,n]=X.shape
    mu=np.zeros((m,n))
    for i in range(np.size(mu,0)):
        mu[i]=centroids[int(idx[i])-1]
    
    C=(1/m)*np.sum(np.sqrt(np.sum((X-mu)**2)))
    return C


############## TRIALS ############## 
                
# K = 3   #Centroids
# initial_centroids =kMeansInitCentroids(X, K)
# idx=findClosestCentroids(X,initial_centroids)
# centroids=computeCentroids(X,idx,K)
# c=costFunction(X,K,initial_centroids,idx,centroids)
    

# DEFINING FUNCTION TO OPTIMIZE CENTROIDS (RUN K-MEANS)
def runKmeans(X,K):
    max_iter=10
    initial_centroids = kMeansInitCentroids(X, K)
    error=0
    for i in range(max_iter):
        idx=findClosestCentroids(X,initial_centroids)
        centroids=computeCentroids(X,idx,K)
        c=costFunction(X,K,idx,centroids)
        if error!=c:
            initial_centroids=centroids
            error=c
        elif error==c:
            break
    idx=idx.reshape(-1)
    return centroids, idx

# K=3
# centroids,idx= runKmeans(X, K)


# DEFINING FUNCTION TO PLOT THE CLUSTERS WITH HIS CENTROID

def plotClusters(X,idx,centroids):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=idx, s=50, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.5);
    plt.title('Clusters with cost error = {}'.format(costFunction(X, 3, idx, centroids)))

K=8
centroids,idx= runKmeans(X, K)
# plotClusters(X,idx,centroids)


####################################  RECONSTRUCTING IMAGE  #########################################

X_recovered = X.copy()
for i in range(1,K+1):
    X_recovered[(idx==i).ravel(),:] = centroids[i-1]

# Reshape the recovered image into proper dimensions
X_recovered=X_recovered*255
X_recovered = X_recovered.reshape(128,128,3)
imageio.imsave(r'D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex7\ex7\bird_small_recovered.png', X_recovered)

##########################################################################################################

# DEFINING FUNCTION TO OPTIMIZE K-MEANS WITH RANDOM INITIALIZATION

def optimizeKmeans(X,K):
    max_iter=10
    centroids_g , idx_g= runKmeans(X, K)
    error = costFunction(X,K,idx_g,centroids_g)
    for i in range(max_iter):

        centroids,idx= runKmeans(X, K)
        c=costFunction(X,K,idx,centroids)
        if c<error:
            error=c
            centroids_g=centroids
            idx_g=idx
    idx_g=idx_g.reshape(-1)
    return centroids_g , idx_g

# K=16
# centroids,idx=optimizeKmeans(X, K)
# plotClusters(X,idx,centroids)