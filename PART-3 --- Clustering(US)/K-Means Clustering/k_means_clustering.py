# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the elbow method to find the optimal number of clusters
# wcss
# run a loop of 1 to 10 clusters ..we take the elbow value at the end

wcss = []

for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', random_state=42)  # for escaping from random selection trap
    km.fit(X)
    wcss.append(km.inertia_)  # it calculates wcss value by implementing inertia method in KMeans lib and append it to the wcss list

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS Values')
plt.show()

# From the above Graph, taking i as 5 gives you the elbow point
# Training the K-Means model on the dataset

km = KMeans(n_clusters=5, init='k-means++', random_state=42)
Y_kmeans = km.fit_predict(X)
print(Y_kmeans)

# Visualising the clusters

plt.scatter(X[Y_kmeans == 0, 0], X[Y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')  # cluster 0 and its corresponding values from Y_kmeans
plt.scatter(X[Y_kmeans == 1, 0], X[Y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')  # cluster 1 and its corresponding values from Y_kmeans
plt.scatter(X[Y_kmeans == 2, 0], X[Y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')  # cluster 2 and its corresponding values from Y_kmeans
plt.scatter(X[Y_kmeans == 3, 0], X[Y_kmeans == 3, 1], s=100, c='yellow', label='Cluster 4')  # cluster 3 and its corresponding values from Y_kmeans
plt.scatter(X[Y_kmeans == 4, 0], X[Y_kmeans == 4, 1], s=100, c='black', label='Cluster 5')  # cluster 4 and its corresponding values from Y_kmeans

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=300, c='cyan', label='Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
