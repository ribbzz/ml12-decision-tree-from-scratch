"""A binary decision tree using entropy and information gain."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
import itertools
import json
import math
from numbers import Real


def entropy(labels):
    """Shannon entropy in bits; an empty or pure set has entropy zero."""
    return _entropy_counts(Counter(labels).values())

def _entropy_counts(counts):
    counts = list(counts)
    total = sum(counts)
    if total == 0:
        return 0.0

    result = 0.0
    for count in counts:
        if count > 0:
            probability = count / total
            result -= probability * math.log2(probability)
    return max(0.0, result)


def information_gain(parent, left, right):
    """Entropy reduction for a binary partition of parent labels."""
    parent = list(parent)
    left = list(left)
    right = list(right)
    if Counter(parent) != Counter(left) + Counter(right):
        raise ValueError("Children must partition the parent labels.")
    if not parent:
        return 0.0
    children_entropy = len(left) * entropy(left) + len(right) * entropy(right)
    return entropy(parent) - children_entropy / len(parent)



def _missing(value):
    if value is None:
        return True
    if isinstance(value, Real):
        return math.isnan(float(value))
    return False


@dataclass
class Node:
    """Class counts and, for internal nodes, a split and two children."""
    counts: tuple
    depth: int
    feature: int | None = None
    value: object = None
    kind: str | None = None
    missing_left: bool = True
    gain: float = 0.0
    left: Node | None = None
    right: Node | None = None

    @property
    def samples(self):
        return sum(self.counts)

    @property
    def is_leaf(self):
        return self.feature is None


class DecisionTreeClassifier:
    """A tree for two-class tabular data.

    X contains rows of features; y contains their labels. The root has depth 0.
    Missing values follow the route chosen during training. Tied predictions
    use the class that appeared first in y.
    """

    def __init__(self, max_depth=5, max_leaf_nodes=None, min_samples_split=2,
                 min_samples_leaf=1, min_gain=1e-12):
        for name, value, minimum in [
            ("max_depth", max_depth, 0), ("max_leaf_nodes", max_leaf_nodes, 1),
            ("min_samples_split", min_samples_split, 2),
            ("min_samples_leaf", min_samples_leaf, 1),
        ]:
            if value is None and name in {"max_depth", "max_leaf_nodes"}:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if not isinstance(min_gain, Real) or not math.isfinite(min_gain) or min_gain < 0:
            raise ValueError("min_gain must be finite and nonnegative.")
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain


    def fit(self, X, y, feature_names=None, feature_types=None):
        """Learn categories, class order, split questions and missing routes."""
        self.root_ = None  # A failed refit must not leave an old model usable.
        rows = [list(row) for row in X]
        labels = list(y)
        if not rows or not rows[0] or len(rows) != len(labels):
            raise ValueError("X and y must be nonempty, with matching lengths and at least one feature.")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("All feature rows must have the same length.")
        for label in labels:
            if _missing(label):
                raise ValueError("Target labels cannot be missing.")
            hash(label)
        self.classes_ = list(dict.fromkeys(labels))
        if len(self.classes_) > 2:
            raise ValueError("Binary classification only: y must contain at most two distinct classes.")
        self.n_features_in_ = width
        if feature_names is None:
            self.feature_names_ = [f"x{i}" for i in range(width)]
        else:
            self.feature_names_ = list(feature_names)
        if len(self.feature_names_) != width or len(set(self.feature_names_)) != width:
            raise ValueError("Provide one distinct feature name per column.")
        if feature_types is None:
            self.feature_types_ = []
            for column in range(width):
                kind = "numeric"
                for row in rows:
                    value = row[column]
                    if _missing(value):
                        continue
                    if not isinstance(value, Real) or isinstance(value, bool):
                        kind = "categorical"
                        break
                self.feature_types_.append(kind)
        else:
            self.feature_types_ = list(feature_types)
        if len(self.feature_types_) != width or any(t not in {"numeric", "categorical"} for t in self.feature_types_):
            raise ValueError("feature_types must contain 'numeric' or 'categorical' for each column.")
        rows = self._prepare(rows)
        mapping = {label: i for i, label in enumerate(self.classes_)}
        encoded = [mapping[label] for label in labels]
        size = len(self.classes_)

        def make_node(indices, depth):
            counts = Counter(encoded[i] for i in indices)
            return Node(tuple(counts[k] for k in range(size)), depth)

        self.root_ = make_node(range(len(rows)), 0)
        self.n_leaves_ = 1
        self.depth_ = 0
        queue = []
        counter = itertools.count()

        def offer(node, indices):
            if node.samples < self.min_samples_split:
                return
            if sum(count > 0 for count in node.counts) <= 1:
                return
            if self.max_depth is not None and node.depth >= self.max_depth:
                return
            split = self._best_split(rows, encoded, indices, node.counts)
            if split is not None:
                # The heap should choose the largest total entropy reduction first.
                priority = -split[0] * node.samples
                entry = (priority, next(counter), node, indices, split)
                heapq.heappush(queue, entry)

        offer(self.root_, list(range(len(rows))))
        while queue and (self.max_leaf_nodes is None or self.n_leaves_ < self.max_leaf_nodes):
            _, _, node, indices, split = heapq.heappop(queue)
            node.gain, node.feature, node.kind, node.value, node.missing_left = split
            left, right = [], []
            for i in indices:
                if self._go_left(rows[i], node):
                    left.append(i)
                else:
                    right.append(i)
            node.left = make_node(left, node.depth + 1)
            node.right = make_node(right, node.depth + 1)
            self.n_leaves_ += 1
            self.depth_ = max(self.depth_, node.depth + 1)
            offer(node.left, left)
            offer(node.right, right)
        return self

    def _prepare(self, rows):
        prepared = []
        for row in rows:
            if len(row) != self.n_features_in_:
                raise ValueError("Prediction rows must match the training feature count.")
            converted = []
            for value, kind in zip(row, self.feature_types_):
                if _missing(value):
                    converted.append(None)
                elif kind == "numeric":
                    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
                        raise ValueError("Numeric features must be finite real numbers or missing.")
                    converted.append(float(value))
                else:
                    hash(value)  # Categories must be scalar/hashable.
                    converted.append(value)
            prepared.append(converted)
        return prepared


    def _best_split(self, rows, labels, indices, parent_counts):
        """Scan numeric boundaries and categorical equality-vs-rest splits."""
        total, classes = len(indices), len(parent_counts)
        parent_entropy = _entropy_counts(parent_counts)
        best = None

        def consider(left_known, right_known, missing, feature, kind, value):
            nonlocal best
            for missing_left in (True, False):
                left = left_known.copy()
                right = right_known.copy()
                for k in range(classes):
                    if missing_left:
                        left[k] += missing[k]
                    else:
                        right[k] += missing[k]

                left_size = sum(left)
                right_size = sum(right)
                if min(left_size, right_size) < self.min_samples_leaf:
                    continue
                children_entropy = left_size * _entropy_counts(left) + right_size * _entropy_counts(right)
                gain = parent_entropy - children_entropy / total
                if gain > self.min_gain and (best is None or gain > best[0] + 1e-15):
                    best = (gain, feature, kind, value, missing_left)

        for feature, kind in enumerate(self.feature_types_):
            known, missing = [], [0] * classes
            for i in indices:
                value = rows[i][feature]
                if value is None:
                    missing[labels[i]] += 1
                else:
                    known.append((value, labels[i]))
            counts = [0] * classes
            for _, label in known:
                counts[label] += 1
            # Missingness itself can be predictive even if known values are constant.
            if known and sum(missing):
                consider([0] * classes, counts, missing, feature, "missing", None)
            if kind == "numeric":
                known.sort(key=lambda item: item[0])
                left, right = [0] * classes, counts.copy()
                for position in range(len(known) - 1):
                    value, label = known[position]
                    left[label] += 1
                    right[label] -= 1
                    following = known[position + 1][0]
                    if value == following:
                        continue
                    threshold = value / 2 + following / 2
                    if not value <= threshold < following:
                        threshold = value
                    consider(left, right, missing, feature, kind, threshold)
            else:
                groups = {}
                for value, label in known:
                    groups.setdefault(value, [0] * classes)[label] += 1
                for value, left in groups.items():
                    right = [counts[k] - left[k] for k in range(classes)]
                    consider(left, right, missing, feature, kind, value)
        return best

    @staticmethod
    def _go_left(row, node):
        value = row[node.feature]
        if value is None:
            return node.missing_left
        if node.kind == "missing":
            return False
        if node.kind == "numeric":
            return value <= node.value
        return value == node.value

    def _check_fitted(self):
        if getattr(self, "root_", None) is None:
            raise RuntimeError("Call fit before using the tree.")

    def _leaf(self, row):
        node = self.root_
        while not node.is_leaf:
            if self._go_left(row, node):
                node = node.left
            else:
                node = node.right
        return node


    def predict_proba(self, X):
        """Return leaf class proportions, ordered like classes_."""
        self._check_fitted()
        result = []
        for row in self._prepare([list(row) for row in X]):
            leaf = self._leaf(row)
            result.append([count / leaf.samples for count in leaf.counts])
        return result

    def predict(self, X):
        predictions = []
        for probabilities in self.predict_proba(X):
            best_class = probabilities.index(max(probabilities))
            predictions.append(self.classes_[best_class])
        return predictions

    def score(self, X, y):
        """Ordinary classification accuracy (not balanced accuracy)."""
        predicted = self.predict(X)
        actual = list(y)
        if not actual or len(predicted) != len(actual):
            raise ValueError("Scoring requires matching, nonempty predictions and labels.")
        correct = 0
        for prediction, label in zip(predicted, actual):
            if prediction == label:
                correct += 1
        return correct / len(actual)

    def _description(self, node):
        best_class = node.counts.index(max(node.counts))
        winner = self.classes_[best_class]
        detail = f"n={node.samples}, H={_entropy_counts(node.counts):.4f}, class={winner!r}, counts={node.counts}"
        if node.is_leaf:
            return "leaf: " + detail
        feature = self.feature_names_[node.feature]
        if node.kind == "missing":
            question = f"{feature} is missing"
        elif node.kind == "numeric":
            question = f"{feature} <= {node.value!r}"
        else:
            question = f"{feature} == {node.value!r}"
        return f"{question}? gain={node.gain:.4f}, missing={'yes' if node.missing_left else 'no'}; {detail}"


    def export_text(self):
        """Readable tree; yes follows the left branch, no follows the right."""
        self._check_fitted()
        lines = []
        stack = [(self.root_, "", "")]
        while stack:
            node, indent, branch = stack.pop()
            lines.append(indent + branch + self._description(node))
            if not node.is_leaf:
                stack.append((node.right, indent + "  ", "no -> "))
                stack.append((node.left, indent + "  ", "yes -> "))
        return "\n".join(lines)

    def export_dot(self):
        """Graphviz DOT source; rendering requires an optional Graphviz install."""
        self._check_fitted()
        lines = ["digraph Tree {", "  node [shape=box];"]
        stack = [(self.root_, 0)]
        counter = itertools.count(1)
        while stack:
            node, number = stack.pop()
            lines.append(f"  n{number} [label={json.dumps(self._description(node))}];")
            if not node.is_leaf:
                for child, answer in [(node.left, "yes"), (node.right, "no")]:
                    child_id = next(counter)
                    lines.append(f'  n{number} -> n{child_id} [label="{answer}"];')
                    stack.append((child, child_id))
        return "\n".join(lines + ["}"])
