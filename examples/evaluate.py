"""Reproduce the report's depth comparison on the same validation split."""
import json
from pathlib import Path
from scratch_tree import DecisionTreeClassifier
from scratch_tree.cli import read_csv, stratified_split, convert_features, classification_metrics

root = Path(__file__).parent
rows, labels, names = read_csv(root / 'practice.csv', 'passed', [])
training, validation = stratified_split(labels, 0.2, 42)
rows, kinds = convert_features(rows, training, names, [])
X_train, y_train = [rows[i] for i in training], [labels[i] for i in training]
X_valid, y_valid = [rows[i] for i in validation], [labels[i] for i in validation]
results = []
for depth in [0, 1, 2, 3, 4, 6, 10]:
    model = DecisionTreeClassifier(max_depth=depth).fit(X_train, y_train, names, kinds)
    metrics = classification_metrics(y_valid, model.predict(X_valid), model.classes_)
    results.append({'max_depth': depth, 'leaves': model.n_leaves_,
                    'train_accuracy': model.score(X_train, y_train),
                    'validation_accuracy': metrics['accuracy'],
                    'validation_balanced_accuracy': metrics['balanced_accuracy_present_classes']})
output = json.dumps(results, indent=2) + '\n'
(root / 'depth_results.json').write_text(output)
print(output)
