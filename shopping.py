import csv
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    data = pd.read_csv(filename)

    # Print unique values in the 'Month' column
    print("Unique values in 'Month' column:", data['Month'].unique())

    # Map month names to numerical values
    month_mapping = {
        'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'Jun': 5,
        'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11
    }

    # Convert 'Month' to numerical values using the mapping
    data['Month'] = data['Month'].map(month_mapping)

    # Convert VisitorType to numerical values (1 for returning visitors, 0 otherwise)
    data['VisitorType'] = data['VisitorType'].apply(lambda x: 1 if x == 'Returning_Visitor' else 0)

    # Convert Weekend to numerical values (1 if true, 0 otherwise)
    data['Weekend'] = data['Weekend'].astype(int)

    # Handle missing values by filling NaN values with the mean
    data = data.fillna(data.mean())

    evidence = data.drop('Revenue', axis=1).values.tolist()
    labels = data['Revenue'].astype(int).values.tolist()

    return evidence, labels

def train_model(evidence, labels):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    tp = sum(1 for label, prediction in zip(labels, predictions) if label == 1 and prediction == 1)
    tn = sum(1 for label, prediction in zip(labels, predictions) if label == 0 and prediction == 0)

    sensitivity = tp / labels.count(1)
    specificity = tn / labels.count(0)

    return sensitivity, specificity


if __name__ == "__main__":
    main()
