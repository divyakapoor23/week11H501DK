# your code here
## Exercise 1

# Use Scikit-Learn and NumPy to write a function `kmeans(X, k)` that does the following:

# - performs k-means clustering on a numerical NumPy array `X`
# - returns a **tuple** `(centroids, labels)`, where
#     - `centroids` is a 2D array of shape `(k, n_features)` containing the cluster centroids, and 
#     - `labels` is a 1D array of shape `(n_samples,)` containing the index of the 
#  assigned cluster for each row of `X`.
import numpy as np
from sklearn.cluster import KMeans
def kmeans(X, k):
    kmeans_model = KMeans(n_clusters=k)
    kmeans_model.fit(X)
    centroids = kmeans_model.cluster_centers_
    labels = kmeans_model.labels_
    return centroids, labels

## Exercise 2
# Use Scikit-Learn and NumPy to write a function `pca(X, n_components)` that does the following:
# - performs Principal Component Analysis (PCA) on a numerical NumPy array `X`
# - returns a 2D NumPy array of shape `(n_samples, n_components)`
from sklearn.decomposition import PCA
def pca(X, n_components):
    pca_model = PCA(n_components=n_components)
    X_reduced = pca_model.fit_transform(X)
    return X_reduced

# Load diamonds dataset and extract numerical columns
import seaborn as sns
diamonds = sns.load_dataset('diamonds')
diamonds_numeric = diamonds.select_dtypes(include=[np.number])

def kmeans_diamonds(n, k):
    """
    Runs kmeans clustering on the first n rows of the numeric diamonds dataset.
    
    Parameters:
    - n: number of rows to use from the dataset
    - k: number of clusters
    
    Returns:
    - centroids: cluster centroids
    - labels: cluster assignments
    """
    X = diamonds_numeric.iloc[:n].values
    return kmeans(X, k)

## Exercise 3
# Write a function called `kmeans_timer(n, k, n_iter=5)` that does the following:
# - runs the function `kmeans_diamonds(n, k)` exactly `n_iter` times, and saves the runtime for each run.
# - returns the *average* time across the `n` runs, where "time" is in seconds.
from time import time

def kmeans_timer(n, k, n_iter=5):
    """
    Times the kmeans_diamonds function over multiple iterations.
    
    Parameters:
    - n: number of rows to use from the dataset
    - k: number of clusters
    - n_iter: number of iterations to run (default=5)
    
    Returns:
    - average time in seconds across all iterations
    """
    times = []
    for _ in range(n_iter):
        start = time()
        kmeans_diamonds(n, k)
        elapsed = time() - start
        times.append(elapsed)
    return np.mean(times)

## Exercise 3 (original)
# Use Scikit-Learn and NumPy to write a function `train_test_split(X,       
# y, test_size)` that does the following:
# - splits the numerical NumPy array `X` and the 1D NumPy array `y` into training and testing sets
# - returns a tuple `(X_train, X_test, y_train, y_test)`
from sklearn.model_selection import train_test_split as sk_train_test_split
def train_test_split(X, y, test_size):
    X_train, X_test, y_train, y_test = sk_train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test     
# - `X_train` and `y_train` are the training sets
# - `X_test` and `y_test` are the testing sets

## Bonus Exercise
# Binary search with step counting for time complexity analysis
step_count = 0

def bin_search(n):
    global step_count
    step_count = 0
    
    arr = np.arange(n)
    left = 0
    right = n-1
    x = n-1  # worst case: searching for the last element

    step_count += 1  # initialization
    while left <= right:
        step_count += 1  # comparison in while condition
        
        middle = left + (right - left) // 2
        step_count += 1  # middle calculation

        # check if x is present at mid
        if (arr[middle] == x):
            step_count += 1  # comparison
            return middle

        # if x greater, ignore left half
        if (arr[middle] < x):
            step_count += 1  # comparison
            left = middle + 1
            step_count += 1  # assignment
        # if x is smaller, ignore right half
        else:
            step_count += 1  # comparison (implicit from if-else)
            right = middle - 1
            step_count += 1  # assignment

    # if we reach here, then element was not present
    return -1  


