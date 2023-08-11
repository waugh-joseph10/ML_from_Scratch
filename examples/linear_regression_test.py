""" Run a sample linear regression model """

# Import Libraries 
import argparse
import sys 
sys.path.append('../classifiers') # Update the path to use the classifiers package
from linear_regression import LinearRegression # Import LinearRegression Model 
from utils import mean_squared_error, r2_score # Import Helper Functions
from sklearn import datasets
from sklearn.model_selection import train_test_split 

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process arguments for the LinearRegression classifier')
    parser.add_argument('--lr', 
                        type=float, 
                        default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--n_iters', 
                        type=int, 
                        default=1000,
                        help='Number of Iterations')
    parser.add_argument('--noise', 
                        type=float, 
                        default=10, 
                        help='The standard deviation of the gaussian noise applied to the training data')
    args = parser.parse_args()
    
    # Build classifier
    clf = LinearRegression(learning_rate = args.lr, n_iterations=args.n_iters)
    
    # Generate a Regression Dataset
    X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=args.noise, random_state=4)

    # Perform a split of training and testing data with an 80/20 ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state = 4
    )
    # Train the classifier
    clf.fit(X_train, y_train)

    # Generate Predictions from the Fitted Model
    predictions = clf.predict(X_test)

    # Generate and Print Output Metrics
    mse = mean_squared_error(y_test, predictions)
    accuracy = r2_score(y_test, predictions)

    print(f'Mean Squared Error: {mse}')
    print(f'Accuracy: {accuracy}')        