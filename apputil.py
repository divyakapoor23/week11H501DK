# your code here
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as sk_train_test_split
def kmeans(X, k):
    """
    Performs k-means clustering on the data X.
    Parameters:
    - X: 2D NumPy array of shape (n_samples, n_features)
    - k: number of clusters
    Returns:
    - centroids: 2D NumPy array of shape (k, n_features)
    - labels: 1D NumPy array of shape (n_samples,)
    """
    kmeans_model = KMeans(n_clusters=k)
    kmeans_model.fit(X)
    centroids = kmeans_model.cluster_centers_
    labels = kmeans_model.labels_
    return centroids, labels


def pca(X, n_components):
    """
    Performs Principal Component Analysis (PCA) on the data X.
    Parameters:
    - X: 2D NumPy array of shape (n_samples, n_features)
    - n_components: number of principal components to keep
    Returns:
    - X_reduced: 2D NumPy array of shape (n_samples, n_components)
    """
    pca_model = PCA(n_components=n_components)
    X_reduced = pca_model.fit_transform(X)
    return X_reduced

# Load diamonds dataset and extract numerical columns

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


def train_test_split(X, y, test_size):
    """
    Splits the numerical NumPy array `X` and the 1D NumPy array `y` into training and testing sets.
    Returns a tuple `(X_train, X_test, y_train, y_test)`.
    - `X_train` and `y_train` are the training sets
    - `X_test` and `y_test` are the testing sets
    """
    X_train, X_test, y_train, y_test = sk_train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test     

step_count = 0

def bin_search(n):
    """
    Performs binary search on an array of size n.
    """ 
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


