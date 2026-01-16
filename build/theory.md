Clustering is the task of grouping observations in such a way that members of the same cluster are more similar to each other and members of different clusters are very different from each other. Clustering is commonly used to explore a dataset to either identify the underlying patterns in it or to create a group of characteristics.

### 1. K-Means Clustering Algorithm
The K-Means clustering algorithm is an iterative process of moving the centers of clusters or centroids to the mean position of their constituent points, and reassigning instances to their closest clusters iteratively until there is no significant change in the number of cluster centers possible or number of iterations reached.

### 2. Cost Function
The cost function of K-Means is determined by the Euclidean distance (square-norm) between the observations belonging to that cluster with its respective centroid value. An intuitive way to understand the equation is, if there is only one cluster (k=1), then the distances between all the observations are compared with its single mean. Whereas, if number of clusters increases to 2 (k=2), then two-means are calculated and a few of the observations are assigned to cluster 1 and other observations are assigned to cluster two based on proximity. Subsequently, distances are calculated in cost functions by applying the same distance measure, but separately to their cluster centers.

Centroid distance calculations are performed by taking Euclidean distances. The Euclidean distance between two points A (X1, Y1) and B (X2, Y2) is shown as follows:

<div align="center" style="font-size: 1.2rem; margin: 20px 0;">
    <strong>Euclidean distance between A & B = √[(X2 − X1)² + (Y2 − Y1)²]</strong>
</div>

### 3. Elbow Method
To determine the optimal number of clusters, the Elbow method is used. The elbow method plots the value of the cost function produced by different values of k. The value of k at which improvement in distortion declines the most is called the elbow, at which we should stop dividing the data into further clusters. The elbow point indicates the most suitable number of clusters.

<div align="center" style="margin: 20px 0;">
    <img src="images/elbow.png" alt="Elbow Method Graph" style="max-width: 80%;">
</div>

The elbow method is used to determine the optimal number of clusters in k-means clustering. The elbow method plots the value of the cost function produced by different values of k. The value of k at which improvement in distortion declines the most is called the elbow, at which we should stop dividing the data into further clusters.

### 4. Merits of K-Means Clustering
- K-Means is simple to understand and easy to implement, making it suitable for clustering large datasets.
- It is computationally efficient and scales well when the number of clusters and features is relatively small.
- The algorithm provides clear and compact cluster representations using centroids, which are easy to interpret.

### 5. Demerits of K-Means Clustering
- It is sensitive to the initial selection of centroids and may converge to a local optimum.
- The algorithm performs poorly with non-spherical clusters, varying cluster sizes, and is highly sensitive to outliers.