"""Implementation of Linear Regression using only the NumPy library"""

import numpy as np

class LinearRegression:
    """
    An implementation of the Linear Regression algorithm 
    using gradient descent via the NumPy library
    """
    def __init__(self, learning_rate=1e-3, n_iterations=1000):
        self.lr = learning_rate 
        self.n_iterations = n_iterations
        self.weights = None 
        self.bias = None 
    def fit(self, X, y):
        """
        Function used to train the LinerRegression classifier
        """
        n_samples, n_features = X.shape 

        # Initialize weights and bias parameters 
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Apply Gradient Descent 
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias 

            # Calculate Gradients 
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias parameters 
            self.weights -= self.lr * dw 
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Function to predict a new value from the trained
        classifier
        """
        y_predicted = np.dot(X, self.weights) + self.bias 
        return y_predicted 