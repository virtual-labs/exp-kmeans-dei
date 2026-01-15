The objective of this experiment is to segment customers into distinct groups based on their purchasing behavior using the K-Means clustering algorithm. The input to the model consists of independent variables representing customer attributes, specifically Annual Income and Spending Score, which are used to measure similarity among customers. Since K-Means is an unsupervised learning technique, there is no dependent or output variable involved in the clustering process.

1. **Import Required Libraries**: Import the essential Python libraries: `pandas`, `numpy`, `matplotlib`, and `scikit-learn`.
2. **Load the Dataset**: Load the `Mall_Customers.csv` dataset. The dataset consists of customer information with 4 columns, out of which **Annual Income** and **Spending Score** are used for analysis.
3. **Perform Data Analysis**: Analyze the dataset to understand its structure, identify missing values, and explore feature distributions.
4. **Apply Feature Scaling**: Apply feature scaling during data preprocessing to normalize the features. Feature scaling is necessary because **Annual Income** ranges from 10 to 150, while **Spending Score** ranges from 1 to 100.
5. **Model Training**: Train the K-Means model by performing the following steps:
    - Place initial centroids randomly.
    - Assign each data point to the nearest centroid.
    - Calculate the mean of each cluster and reassign the position of centroids.
    - Repeat until the centroids stop moving.
    - After this process, each data point will be assigned a cluster number.
6. **Model Evaluation**: Evaluate the model using the **Silhouette Score**, **Davies-Bouldin Score**, and **Calinski-Harabasz Score**.