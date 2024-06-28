# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the dendrogram to find the optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(X , method= 'ward')) 
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidian Distance')
plt.show()
# Training the Hierarchical Clustering model on the dataset
hc = AgglomerativeClustering(n_clusters= 3 ,linkage= 'ward')
y_hc = hc.fit_predict(X) 
print(y_hc)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Cluster 1')  # cluster 0 and its corresponding values from y_hc
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')  # cluster 1 and its corresponding values from y_hc
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='Cluster 3')  # cluster 2 and its corresponding values from y_hc
# plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='yellow', label='Cluster 4')  # cluster 3 and its corresponding values from y_hc
# plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='black', label='Cluster 5')  # cluster 4 and its corresponding values from y_hc

plt.title('Clusters of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()