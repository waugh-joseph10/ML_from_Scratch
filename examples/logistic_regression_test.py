""" Run a sample logistic regression model """

# Import Libraries 
import argparse
import sys 
sys.path.append('../classifiers') # Update the path to use the classifiers package
from logistic_regression import LogisticRegression # Import LogisticRegression Model 
from utils import log_reg_metrics # Import Helper Functions
from sklearn import datasets
from sklearn.model_selection import train_test_split 

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process arguments for the LogisticRegression classifier')
    parser.add_argument('--lr', 
                        type=float, 
                        default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--n_iters', 
                        type=int, 
                        default=10000,
                        help='Number of Iterations')
    args = parser.parse_args()    

    # Build Classifier 
    clf = LogisticRegression(learning_rate = args.lr, n_iterations = args.n_iters)

    # Load the Breast Cancer Dataset
    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target

    # Perform a split of training and testing data with an 80/20 ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state = 4 
    )

    # Train a LinearRegression Classifier
    clf.fit(X_train, y_train)

    # Generate Predictions from the Fitted Model
    predictions = clf.predict(X_test)    

    # Calculate Regression Metrics (Precision, Recall F1-Score, Accuracy, Confusion Matrix)
    precision, recall, f1, accuracy, conf_mat = log_reg_metrics(y_test, predictions)

    # Print Metric Results
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Confusion Matrix:\n{conf_mat}')