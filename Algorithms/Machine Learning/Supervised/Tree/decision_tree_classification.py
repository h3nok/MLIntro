from sklearn import tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

X = [[0,0], [1,1,]]
Y = [0, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)



# classification 
X, y = load_iris(return_X_y=True)
iris_clf = tree.DecisionTreeClassifier(criterion="entropy")
iris_clf.fit(X, y)

tree.plot_tree(iris_clf)
plt.show()

iris_clf.predict(X)
# regression 

