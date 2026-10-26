"""Train and display a scratch tree on a user-supplied binary-classification CSV."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random

from .tree import DecisionTreeClassifier


def stratified_split(labels, fraction, seed):
    """Keep singleton classes in training; do not duplicate rows between splits."""
    if not 0 <= fraction < 1:
        raise ValueError("test-size must be between 0 (inclusive) and 1 (exclusive).")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)
    training, testing = [], []
    for indices in groups.values():
        rng.shuffle(indices)
        n_test = min(len(indices) - 1, max(1, round(len(indices) * fraction))) if fraction else 0
        testing.extend(indices[:n_test])
        training.extend(indices[n_test:])
    return sorted(training), sorted(testing)


def read_csv(path, target, drop):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or len(set(fields)) != len(fields) or target not in fields:
            raise ValueError("CSV needs unique headers and the requested target column.")
        if target in drop or any(c not in fields for c in drop):
            raise ValueError("Dropped columns must exist and cannot include the target.")
        features = [c for c in fields if c != target and c not in drop]
        rows, labels = [], []
        for record in reader:
            if None in record or any(value is None for value in record.values()):
                raise ValueError("CSV row width does not match its header.")
            label = record[target].strip()
            if not label:
                raise ValueError("Target labels cannot be empty.")
            rows.append([record[c].strip() or None for c in features])
            labels.append(label)
    if not rows or not features:
        raise ValueError("CSV must have data and at least one predictor.")
    return rows, labels, features


def convert_features(rows, training, names, forced_categories):
    """Infer numeric columns from training rows only; empty cells are missing."""
    if any(c not in names for c in forced_categories):
        raise ValueError("Unknown categorical column name.")
    kinds = []
    for j, name in enumerate(names):
        kind = 'numeric'
        if name in forced_categories:
            kind = 'categorical'
        else:
            for i in training:
                value = rows[i][j]
                if value is not None:
                    try:
                        float(value)
                    except ValueError:
                        kind = 'categorical'
                        break
        kinds.append(kind)
    converted = []
    for row in rows:
        result = []
        for value, kind in zip(row, kinds):
            if value is None or kind == 'categorical':
                result.append(value)
            else:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError("Use empty cells for missing numbers; infinity/NaN text is not accepted.")
                result.append(number)
        converted.append(result)
    return converted, kinds


def classification_metrics(actual, predicted, classes):
    per_class = {}
    for label in classes:
        support = sum(y == label for y in actual)
        correct = sum(y == label and p == label for y, p in zip(actual, predicted))
        per_class[label] = {'support': support, 'recall': correct / support if support else None}
    recalls = [v['recall'] for v in per_class.values() if v['recall'] is not None]
    return {
        'accuracy': sum(a == p for a, p in zip(actual, predicted)) / len(actual),
        'balanced_accuracy_present_classes': sum(recalls) / len(recalls),
        'per_class': per_class,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--drop', default='', help='Comma-separated columns to exclude, such as an ID.')
    parser.add_argument('--categorical', default='', help='Force comma-separated columns to be categorical.')
    parser.add_argument('--max-depth', type=int, default=5)
    parser.add_argument('--max-leaves', type=int)
    parser.add_argument('--min-leaf', type=int, default=1)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dot', type=Path, help='Save optional Graphviz source.')
    parser.add_argument('--metrics', type=Path, help='Save evaluation results as JSON.')
    args = parser.parse_args(argv)
    try:
        drop = [c.strip() for c in args.drop.split(',') if c.strip()]
        categorical = [c.strip() for c in args.categorical.split(',') if c.strip()]
        rows, labels, names = read_csv(args.csv, args.target, drop)
        if len(set(labels)) > 2:
            raise ValueError("Binary classification only: target must have at most two classes.")
        training, testing = stratified_split(labels, args.test_size, args.seed)
        rows, kinds = convert_features(rows, training, names, categorical)
        model = DecisionTreeClassifier(max_depth=args.max_depth, max_leaf_nodes=args.max_leaves,
                                       min_samples_leaf=args.min_leaf)
        model.fit([rows[i] for i in training], [labels[i] for i in training], names, kinds)
        summary = {'training_rows': len(training), 'test_rows': len(testing),
                   'depth': model.depth_, 'leaves': model.n_leaves_, 'seed': args.seed}
        if testing:
            actual = [labels[i] for i in testing]
            predicted = model.predict([rows[i] for i in testing])
            summary['evaluation'] = classification_metrics(actual, predicted, model.classes_)
        else:
            summary['evaluation'] = None
            summary['note'] = 'No held-out rows: no generalization score is reported.'
        print(json.dumps(summary, indent=2))
        print('\n' + model.export_text())
        if args.dot:
            args.dot.write_text(model.export_dot() + '\n')
        if args.metrics:
            args.metrics.write_text(json.dumps(summary, indent=2) + '\n')
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
