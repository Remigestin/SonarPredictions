import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

df = pd.read_csv("tutorial_data_eval_Sonar.csv", delimiter=';')

# X = data without class
X = df.values[:, 0:-1].astype(float)

# y = just the class column
y = df['Class'].copy()

# we choose a test size of 30%, and not chosen randomly.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Set up the knn model with default values
clf = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=1)
logreg = LogisticRegression(solver="lbfgs")
decisionTree = DecisionTreeClassifier()


# Fit the model
clf.fit(X_train, y_train)
logreg.fit(X_train, y_train)
decisionTree.fit(X_train, y_train)

# launch the prediction
y_pred_knn = clf.predict(X_test)
y_pred_lr = logreg.predict(X_test)
y_pred_tree = decisionTree.predict(X_test)

# results
print("Accuracy:")
print(accuracy_score(y_test, y_pred_knn))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("Classification Report:")
print(classification_report(y_test, y_pred_knn))


# results
print("Accuracy:")
print(accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
print("Classification Report:")
print(classification_report(y_test, y_pred_lr))

# results
print("Accuracy:")
print(accuracy_score(y_test, y_pred_tree))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))
print("Classification Report:")
print(classification_report(y_test, y_pred_tree))

dot_data = export_graphviz(decisionTree,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
