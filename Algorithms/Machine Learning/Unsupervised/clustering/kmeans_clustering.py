from sklearn import cluster
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

"""
Step 1 
    We choose the number of clusters, k.
Step 2 
    Among the data points, we randomly choose k points as cluster centers.
Step 3
    Based on the selected distance measure, we iteratively compute the distance from each point in
    the problem space to each of the k cluster centers. Based on the size of the dataset, this may be a
    time-consuming step—for example, if there are 10,000 points in the cluster and k = 3, this means
    that 30,000 distances need to be calculated.
Step 4 
    We assign each data point in the problem space to the nearest cluster center.
Step 5
    Now each data point in our problem space has an assigned cluster center. But we are not done,
    as the selection of the initial cluster centers was based on random selection. We need to verify
    that the current randomly selected cluster centers are actually the center of gravity of each
    cluster. We recalculate the cluster centers by computing the mean of the constituent data points
    of each of the k clusters. This step explains why this algorithm is called k-means.
Step 6
    If the cluster centers have shifted in step 5, this means that we need to recompute the cluster
    assignment for each data point. For this, we will go back to step 3 to repeat that computeintensive
    step. If the cluster centers have not shifted or if our predetermined stop condition (for
    example, the number of maximum iterations) has been satisfied, then we are done.
"""

dataset = pd.DataFrame({
'x': [11, 21, 28, 17, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 62, 70, 72, 10],
'y': [39, 36, 30, 52, 53, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 18, 7, 24, 10] })

myKmeans = cluster.KMeans(n_clusters=2)
myKmeans.fit(dataset)

centroids = myKmeans.cluster_centers_
labels = myKmeans.labels_

print(centroids)
print(labels)

plt.scatter(dataset['x'], dataset['y'], s=10)
plt.scatter(centroids[0], centroids[1], s=100)
plt.show()