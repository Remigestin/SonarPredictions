import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("tutorial_data_eval_Sonar.csv", delimiter=';')

# X = data without class column
X = df.values[:, 0:-1].astype(float)

# y = just the class column
y = df['Class'].copy()

# we choose a test size of 20%, and not chosen randomly.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Set up different methods of predictions
knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=1)
logreg = LogisticRegression(solver="lbfgs")
decisionTree = DecisionTreeClassifier()

# Fit the models
knn.fit(X_train, y_train)
logreg.fit(X_train, y_train)
decisionTree.fit(X_train, y_train)

# Get the odd ratios
print("________ ODD RATIOS _________")
print(np.exp(logreg.coef_))

# Launch the predictions
y_pred_knn = knn.predict(X_test)
y_pred_lr = logreg.predict(X_test)
y_pred_tree = decisionTree.predict(X_test)

# Results
print("_______________LOGISTIC REGRESSION______________")
print("Accuracy:")
print(accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
print("Classification Report:")
print(classification_report(y_test, y_pred_lr))

print("_______________KNN______________")
print("Accuracy:")
print(accuracy_score(y_test, y_pred_knn))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("Classification Report:")
print(classification_report(y_test, y_pred_knn))

print("_______________DECISION TREE______________")
print("Accuracy:")
print(accuracy_score(y_test, y_pred_tree))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))
print("Classification Report:")
print(classification_report(y_test, y_pred_tree))
