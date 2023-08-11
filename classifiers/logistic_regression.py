"""Implementation of Logistic Regression using only the NumPy library"""

import numpy as np

class LogisticRegression:
    """
    An implementation of the Logistic Regression algorithm 
    using gradient descent via the NumPy library
    """
    def __init__(self, learning_rate=1e-3, n_iterations=1000, threshold=0.5):
        self.lr = learning_rate 
        self.n_iterations = n_iterations
        self.threshold = threshold
        self.weights = None 
        self.bias = None 
    def fit(self, X, y):
        """
        Function used to train the LogisticRegression classifier
        """
        n_samples, n_features = X.shape 

        # Initialize weights and bias parameters 
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Apply Gradient Descent 
        for _ in range(self.n_iterations):
            y_predicted = self._sigmoid(np.dot(X, self.weights) + self.bias)

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
        y_predicted = self._sigmoid(np.dot(X, self.weights) + self.bias)
        y_predicted_cls = [1 if i > self.threshold else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
    
    def predict_proba(self, X):
        """
        Function to get prediction probability for a new value from the trained
        classifier
        """        
        y_predicted = self._sigmoid(np.dot(X, self.weights) + self.bias)
        return np.array(y_predicted)

    def _sigmoid(self, x):
        """
        Helper function to get the sigmoid output
        from a given x
        """
        return 1 / (1 + np.exp(-x))