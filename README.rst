Binary decision tree from scratch
=================================

An ML1.2 classification tree implemented using Python's standard library.
Entropy, information gain, split search, tree growth and prediction are written
from scratch. No scikit-learn, LightGBM or other tree implementation is used.

Repository: https://github.com/ribbzz/ml12-decision-tree-from-scratch

Quick start with uv
-------------------

From this folder::

    uv sync
    uv run python examples/demo.py
    uv run scratch-tree --csv examples/practice.csv --target passed --max-depth 4 --max-leaves 12
    uv run python -m unittest discover -s tests -v

Python 3.10 or newer is required. There are no runtime dependencies.

Your own binary dataset
-----------------------

Use a CSV with a header row and a target column containing two class labels::

    uv run scratch-tree --csv /path/to/data.csv --target outcome --max-depth 5 --drop id

Any nonempty strings can be target labels. A one-class sample is supported as a
constant classifier; more than two classes are explicitly rejected. Numeric
and categorical predictors may be mixed. Empty feature cells represent missing
values. Use ``--categorical postcode,status_code`` to keep numeric-looking
category codes categorical. Non-finite numeric strings are rejected.

The default evaluation is a reproducible, stratified 80/20 split. Preprocessing
is inferred from training rows only. Singleton classes remain in training;
classes with no evaluation rows have undefined recall and are excluded from
the printed average. ``--test-size 0`` fits all rows and reports no evaluation
score. ``--drop`` explicitly removes IDs or other unwanted predictors.

Tree size
---------

* ``--max-depth`` limits the longest root-to-leaf path. Zero means one leaf.
* ``--max-leaves`` places a hard limit on the number of terminal nodes.
* ``--min-leaf`` controls the minimum number of training examples per child.

The Python API additionally exposes ``min_samples_split`` and ``min_gain``.

Python API
----------

::

    from scratch_tree import DecisionTreeClassifier

    X = [[1, "solo"], [2, "group"], [6, "solo"], [8, "group"]]
    y = ["no", "no", "yes", "yes"]
    tree = DecisionTreeClassifier(max_depth=2, max_leaf_nodes=3)
    tree.fit(X, y, feature_names=["hours", "practice"])
    print(tree.predict([[7, "solo"]]))
    print(tree.predict_proba([[7, "solo"]]))
    print(tree.export_text())

Input rows are sequences of finite real numbers, hashable categorical values,
and ``None``/numeric NaN for missing features. Numeric features are represented
as floating-point values. DataFrames can be converted with
``frame[features].to_numpy().tolist()``; the package itself does not need pandas.

Display the tree
----------------

The command prints the tree, sample counts, entropy, gain and class counts.
It can also produce Graphviz source::

    mkdir -p outputs
    uv run scratch-tree --csv examples/practice.csv --target passed --max-depth 3 --dot outputs/tree.dot
    dot -Tsvg outputs/tree.dot -o outputs/tree.svg

The ``dot`` command requires an optional Graphviz installation. Text display
and DOT export work without it.

Files and limitations
---------------------

* ``src/scratch_tree/tree.py``: mathematical helpers and tree implementation.
* ``src/scratch_tree/cli.py``: CSV input, splitting, evaluation and display.
* ``tests/``: unit and end-to-end tests.
* ``examples/``: a hand-checkable example and an artificial 1,000-row dataset.
* ``report.md``: assignment description, derivation, measured results and limits.

This is binary, tabular classification, not regression or raw image/text
processing. Categorical splits test one category against the others; exhaustive
category subsets are not searched. Greedy positive-gain splits can miss pure
interactions such as XOR. No post-pruning or probability calibration is applied.
A Python implementation is intended for learning and moderate-sized datasets,
not as a replacement for optimized production tree libraries.
