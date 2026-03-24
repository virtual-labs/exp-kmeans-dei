### Step 1: Import Required Libraries
The experiment begins by importing the Python libraries required for data processing, visualization, clustering, and evaluation.

The following libraries are used:
- `pandas` for reading and managing the dataset
- `numpy` for numerical operations
- `matplotlib` for creating visualizations
- `KMeans` from scikit-learn to implement the clustering algorithm
- `StandardScaler` for feature scaling
- `silhouette_score` and `davies_bouldin_score` for evaluating clustering performance

### Step 2: Load the Dataset
Load the `Mall_Customers.csv` dataset using `pandas`.

### Step 3: Perform Data Analysis
Before applying the clustering algorithm, perform basic exploratory analysis to understand the dataset.

The following tasks should be performed:
- `head()` - used to display the first few records of the dataset
- `tail()` - shows the last few entries
- `describe()` - provides statistical summaries such as mean, minimum, and maximum values
- `info()` - displays column data types and general dataset information
- `isnull().sum()` - checks for missing values in each column
- `shape` - gives the number of rows and columns in the dataset

Although the dataset may contain several columns, only two features are used for clustering in this experiment:
- **Annual Income**
- **Spending Score**

These features represent the purchasing behavior of customers and are used to measure similarity between customers.

Before applying the clustering algorithm, a scatter plot is created to visualize how the data points are distributed.

The plot displays:
- **Annual Income** on the x-axis
- **Spending Score** on the y-axis

Each point in the graph represents a customer. This visualization provides a preliminary understanding of the data distribution and helps observe whether natural groupings might exist in the dataset.

### Step 4: Apply Feature Scaling
Apply feature scaling to normalize the data before clustering.

Feature scaling is necessary because the two features have different numerical ranges:
- **Annual Income** ranges approximately from 10 to 150
- **Spending Score** ranges from 1 to 100

Since K-Means uses distance-based calculations, features with larger ranges may dominate the clustering process. Applying scaling ensures that both features contribute equally to distance calculations.

Therefore, `StandardScaler` is applied to standardize the features.

This process transforms the data so that:
- The mean becomes 0
- The standard deviation becomes 1

The scaled data is stored in `X_scaled`, which is used for model training.

### Step 5: Determine the Optimal Number of Clusters using the Elbow Method
Determine the appropriate value of K (number of clusters) before training the K-Means model.

The K-Means algorithm requires the number of clusters to be specified in advance. In practice, the correct number of clusters is not known beforehand, so the Elbow Method is used to estimate an appropriate value.

The following steps should be performed:
1. Run the K-Means algorithm for multiple values of K (for example K = 1 to K = 10).
2. For each value of K, compute the Within-Cluster Sum of Squares (WCSS).
3. Store the WCSS value obtained for each value of K.
4. Plot a graph with:
    - Number of clusters (K) on the x-axis
    - WCSS values on the y-axis.

WCSS represents the sum of squared distances between each data point and the centroid of its assigned cluster.

As the value of K increases, WCSS decreases because clusters become smaller and more compact. However, after a certain point the decrease becomes minimal.

The point where the curve forms a distinct bend or “elbow” indicates the optimal number of clusters.

### Step 6: Train the K-Means Model
After determining the optimal value of K using the Elbow Method, train the K-Means clustering model using that value.

The training process includes the following steps:
1. Initialize the K-Means model with the selected value of K.
2. Randomly place the initial centroids in the feature space.
3. Assign each data point to the nearest centroid using distance calculations.
4. Compute the new centroid of each cluster by calculating the mean of all points belonging to that cluster.
5. Repeat the assignment and centroid update process until the centroids no longer change significantly.

After convergence, each data point is assigned a cluster label representing the group to which it belongs.

### Step 7: Visualize the Clusters
Visualize the clustering results using a scatter plot.

In the plot:
- Each cluster is represented using a different color.
- Cluster centroids are displayed using a distinct red “X” marker.

This visualization clearly shows how the algorithm segments customers into different groups based on purchasing behavior.

### Step 8: Evaluate the Clustering Results
Evaluate the clustering performance using the following metrics.

**Silhouette Score**

This metric measures how similar a data point is to its own cluster compared to other clusters.
- Range: −1 to 1
- Higher values indicate better clustering quality.

**Davies–Bouldin Score**

This metric measures the average similarity between clusters, considering both cluster compactness and separation.
- Lower values indicate better clustering.

**Calinski–Harabasz Score**

This metric measures the ratio of between-cluster dispersion to within-cluster dispersion.
- Higher values indicate better cluster separation.

These metrics can be computed using functions available in scikit-learn, and the obtained values help assess the quality of clustering.