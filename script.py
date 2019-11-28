import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors.classification import KNeighborsClassifier


df = pd.read_csv("tutorial_data_eval_Sonar.csv", delimiter=';')

# X = data without class
X = df.values[:, 0:-1].astype(float)

# y = just the class column
y = df['Class'].copy()

# we choose a test size of 30%, and not chosen randomly.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Set up the knn model with default values
clf = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=1)

# Fit the model
clf.fit(X_train, y_train)

# launch the prediction
predicted = clf.predict(X_test)

# results
print("Accuracy:")
print(accuracy_score(y_test, predicted))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted))
print("Classification Report:")
print(classification_report(y_test, predicted))
