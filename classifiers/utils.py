""" Helper functions to use when training a Linear Regression model"""

# Import Libraries
import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Helper function to determine the 
    mean squared error (MSE) from the 
    trained model's predictions vs. actuals
    """
    return np.mean((y_true - y_pred)**2)

def r2_score(y_true, y_pred):
    """ 
    Helper function to determine the 
    R^2 score from the trained model's 
    predictions vs. actuals
    """
    correlation_coef = np.corrcoef(y_true, y_pred)
    corr = correlation_coef[0, 1]
    return corr ** 2