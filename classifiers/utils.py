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

def log_reg_metrics(y_true, y_pred):
    """
    Helper function to determine precision,
    recall, accuracy, and a confusion matrix
    from the trained model's predictions vs. actuals
    """
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    conf_mat = np.array([TN, FP, FN, TP], dtype=np.int64).reshape(2, 2)
    precision = TP / float(TP + FP)
    recall = TP / float(TP + FN)
    f1 = 2 * (precision * recall) / float(precision + recall)
    accuracy = float(TP + TN) / float(TP + TN + FP + FN)
    return precision, recall, f1, accuracy, conf_mat