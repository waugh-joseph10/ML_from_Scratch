""" Run a sample KNN model """

# Import Libraries 
import argparse
import sys 
sys.path.append('../classifiers') # Update the path to use the classifiers package
from knn import KNN # Import KNN Model 
from utils import accuracy # Import Helper Functions
from collections import Counter 
from sklearn import datasets
from sklearn.model_selection import train_test_split 

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process arguments for the KNN classifier')
    parser.add_argument('--k', 
                        type=float, 
                        default=3,
                        help='Number of classes')
    args = parser.parse_args()
    
    # Build classifier
    clf = KNN(k = args.k)
    
    # Generate a Classification Dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Perform a split of training and testing data with an 80/20 ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state = 4
    )

    clf.fit(X_train, y_train)

    # Generate Predictions from the Fitted Model
    predictions = clf.predict(X_test)
    
    # Generate and Print Output Metrics
    accuracy = accuracy(y_test, predictions)

    print(f'Accuracy: {accuracy:.2f}')