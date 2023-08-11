"""Implementation of k-Nearest Neighbors algorithm using only the NumPy library"""

from collections import Counter
import numpy as np

class KNN:
    """
    An implementation of the k-Nearest Neighbors (KNN) 
    algorithm via the NumPy library
    """
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        """
        Function used to train the KNN classifier
        """
        self.X_train = X 
        self.y_train = y
    
    def predict(self, X):
        """ 
        Function to predict a class based on input data 
        """
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        # Compute euclidean distances between X and all examples
        distances = [self._euclidean_distance(x1=x, x2=x_train) for x_train in self.X_train]
        # Sort by distances, and return indices of first k neighbors (sorted)
        k_idx = np.argsort(distances)[: self.k]
        # Extract the associated labels 
        k_labels = [self.y_train[i] for i in k_idx]
        # Return most common class label 
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
    
    def _euclidean_distance(self, x1, x2):
        """
        Helper function to calculate the 
        Euclidean distance between two datapoints
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))    