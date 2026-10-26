"""Small hand-checkable binary example with the public Python API."""
from scratch_tree import DecisionTreeClassifier, entropy, information_gain

X = [[1, 'solo'], [2, 'group'], [6, 'solo'], [8, 'group']]
y = ['no', 'no', 'yes', 'yes']
model = DecisionTreeClassifier(max_depth=2, max_leaf_nodes=3)
model.fit(X, y, feature_names=['hours', 'practice'])
print('Parent entropy:', entropy(y))
print('Perfect-split information gain:', information_gain(y, y[:2], y[2:]))
print(model.export_text())
print('Predictions:', model.predict([[1.5, 'solo'], [7, 'group']]))
