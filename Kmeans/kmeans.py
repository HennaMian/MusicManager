from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
# Load image
import imageio


# Set random seed so output is all same
np.random.seed(1)

def pairwise_dist(x, y):
    """
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
        dist: N x M array, where dist2[i, j] is the euclidean distance between 
        x[i, :] and y[j, :]
    """

    N = x.shape[0]
    M = y.shape[0]
    D = x.shape[1]

    xResize = np.tile(x, (1,M))
    yResize = np.tile(y.flatten(), (N,1))

    subtract = xResize - yResize
    squared = subtract**2
    
    makePairs = squared.reshape(int(squared.shape[0]*squared.shape[1]/D), D)
    
    sumPairs = np.sum(makePairs, axis = 1)

    sqrt = np.sqrt(sumPairs)
    
    dist = (sqrt.reshape(N,M))
    
    return dist

class KMeans(object):
    
    def __init__(self): #No need to implement
        pass

    def _init_centers(self, points, K, **kwargs): # [5 pts]
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers. 
        """
        N = points.shape[0]

        centerIndices = np.random.choice(N, K, replace=False)

        centers = points[centerIndices]

        return centers


    def _update_assignment(self, centers, points): # [10 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            points: NxD numpy array, the observations
        Return:
            cluster_idx: numpy array of length N, the cluster assignment for each point
            
        Hint: You could call pairwise_dist() function.
        """
        pairwiseDists = pairwise_dist(centers, points)
        cluster_inx = np.argmin(pairwiseDists, axis = 0)
        return cluster_inx

    def _update_centers(self, old_centers, cluster_idx, points): # [10 pts]
        """
        Args:
            old_centers: old centers KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            centers: new centers, K x D numpy array, where K is the number of clusters, and D is the dimension.
        """

        K = old_centers.shape[0]
        D = old_centers.shape[1]

        centers = np.empty((K, D))

        for i in range(K):
            indices = np.where(cluster_idx == i)
            centerPoints = points[indices, :]
            centers[i] = np.mean(centerPoints, axis=1)

        return centers


    def _get_loss(self, centers, cluster_idx, points): # [5 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans. 
        """

        N = cluster_idx.shape[0]

        dists = pairwise_dist(centers, points)
        zeroToN = np.arange(N)
        points = dists[cluster_idx, zeroToN]
        squared = points**2
        loss = np.sum(squared)
        return loss
        
    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False, **kwargs):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            verbose: boolean to set whether method should print loss (Hint: helpful for debugging)
            kwargs: any additional arguments you want
        Return:
            cluster assignments: Nx1 int numpy array
            cluster centers: K x D numpy array, the centers
            loss: final loss value of the objective function of KMeans
        """
        centers = self._init_centers(points, K, **kwargs)
        for it in range(max_iters):
            cluster_idx = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if verbose:
                print('iter %d, loss: %.4f' % (it, loss))
        return cluster_idx, centers, loss
    
def find_optimal_num_clusters(data, max_K=19): # [10 pts]
    np.random.seed(1)
    """Plots loss values for different number of clusters in K-Means

    Args:
        image: input image of shape(H, W, 3)
        max_K: number of clusters
    Return:
        List with loss values
    """
    kmeansObject = KMeans()

    indices = []
    lossValues = []
    for i in range(max_K):
        indices.append(i+1)
        a, b, loss = kmeansObject.__call__(data, i+1)
        lossValues.append(loss)
    print(indices)
    plt.plot(indices, lossValues)
    plt.show()

    return lossValues



# def find_optimal_num_clusters_data(data, max_K=19): # [10 pts]
#     np.random.seed(1)
#     """Plots loss values for different number of clusters in K-Means

#     Args:
#         image: input image of shape(H, W, 3)
#         max_K: number of clusters
#     Return:
#         List with loss values
#     """
#     kmeansObject = KMeans()

#     indices = []
#     lossValues = []
#     for i in range(max_K):
#         indices.append(i+1)
#         a, b, loss = kmeansObject.__call__(data, i+1)
#         lossValues.append(loss)
#     print(indices)
#     plt.plot(indices, lossValues)
#     plt.show()

#     return lossValues




def intra_cluster_dist(cluster_idx, data, labels): # [4 pts]
    """
    Calculates the average distance from a point to other points within the same cluster
    
    Args:
        cluster_idx: the cluster index (label) for which we want to find the intra cluster distance
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        intra_dist_cluster: 1D array where the i_th entry denotes the average distance from point i 
                            in cluster denoted by cluster_idx to other points within the same cluster
    """
    points = data[labels == cluster_idx]
    distance = pairwise_dist(points, points)
    intra_dist_cluster = np.sum(distance, axis = 1)
    intra_dist_cluster = intra_dist_cluster/(len(points)-1)

    return intra_dist_cluster


def inter_cluster_dist(cluster_idx, data, labels): # [4 pts]
    """
    Calculates the average distance from one cluster to the nearest cluster
    Args:
        cluster_idx: the cluster index (label) for which we want to find the intra cluster distance
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        inter_dist_cluster: 1D array where the i-th entry denotes the average distance from point i in cluster
                            denoted by cluster_idx to the nearest neighboring cluster
    """
    points = data[labels == cluster_idx]
    maxLabel = max(labels)
    mins = []

    for i in range(maxLabel+1):
        if (i != cluster_idx):
            dataPoints = data[labels == i]
            dists = pairwise_dist(points, dataPoints)
            sets = dists.shape[1]
            summed = np.sum(dists, axis = 1)
            average = summed / sets
            mins.append(average)

    mins = np.array(mins)
    inter_dist_cluster = np.min(mins, axis = 0)

    return inter_dist_cluster


def normalized_cut(data, labels): #[2 pts]
    """
    Finds the normalized_cut of the current cluster assignment
    
    Args:
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        normalized_cut: normalized cut of the current cluster assignment
    """
    normalized_cut = 0
    maxLabel = max(labels)

    for i in range(maxLabel + 1):
        intra = intra_cluster_dist(i, data, labels)
        inter = inter_cluster_dist(i, data, labels)
        summed = intra + inter
        div = inter/summed
        normalized_cut += np.sum(div)

    return normalized_cut

# load the feature matrix (NxC)
feature_matrix = np.load("./music_features_matrix.npy")


# load the genre truth values (N)
genre_truths = np.load("./meta_genre_truths.npy")

# load the recording IDs for the songs (N)
recording_ids = np.load("./meta_recording_ids.npy")

# load the recording titles (N)
music_titles = np.load("./meta_music_titles.npy")

# load the feature names (C)
feature_names = np.load("./music_feature_names.npy")


# find_optimal_num_clusters_data(feature_matrix)

kmeansObject = KMeans()

labelsSelected = 11

clustersIdx, centers, loss = kmeansObject.__call__(feature_matrix, labelsSelected)


k = centers.shape[0] 

print(k)

allLabels = []   

for i in range(k):
    index = np.where(clustersIdx == i)[0]
    
    minDist = None
    minVal = None

    for point in index:

        coords = feature_matrix[point]

        coords = coords.reshape(1, coords.shape[0])

        centerI = centers[i].reshape(1, centers[i].shape[0])

        dist = pairwise_dist(coords, centerI)

        if (minDist == None or dist < minDist):
            minDist = dist
            minVal = point

    allLabels.append(minVal)

finalLabels = genre_truths[allLabels]

print(finalLabels)


updatedFeatures  = np.copy(genre_truths)


for j in range(k):
    indexes = np.where(clustersIdx == j)[0]

    for indexe in indexes:
        updatedFeatures[indexe] = finalLabels[j]    

occurances = np.unique(updatedFeatures, return_counts=True)

# print(occurances)
sorted_indexes = np.argsort(occurances[1])[::-1]
# print(sorted_indexes)
occurance = occurances[0][sorted_indexes]
countss = occurances[1][sorted_indexes]
print(occurance)
print(countss)
# print(occurances)



accuracy=[]

for i in range(occurance.shape[0]):
    hit = 0
    total = 0
    for j in range(genre_truths.shape[0]):

        if(occurance[i]==genre_truths[j] and  occurance[i]== updatedFeatures[j]):
            hit+=1

    total = countss[i]
    point = hit/total

    accuracy.append(point)

accuracy = np.array(accuracy)

print(accuracy)




