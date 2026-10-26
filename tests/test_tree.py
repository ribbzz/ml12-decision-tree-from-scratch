"""Behavioral checks against hand-computed examples, without external ML code."""
import math
import unittest

from scratch_tree import DecisionTreeClassifier, entropy, information_gain


class EntropyTests(unittest.TestCase):
    def test_hand_calculated_entropy(self):
        self.assertEqual(entropy([]), 0)
        self.assertEqual(entropy(['a'] * 5), 0)
        self.assertEqual(entropy(['a', 'b']), 1)
        self.assertAlmostEqual(entropy(['a', 'a', 'a', 'b']), 0.8112781244591328)

    def test_hand_calculated_gain(self):
        self.assertEqual(information_gain(['a', 'a', 'b', 'b'], ['a', 'a'], ['b', 'b']), 1)
        self.assertEqual(information_gain(['a', 'b', 'a', 'b'], ['a', 'b'], ['a', 'b']), 0)
        with self.assertRaises(ValueError):
            information_gain(['a', 'b'], ['a'], ['a'])


class TreeTests(unittest.TestCase):
    def test_numeric_boundary_and_unseen_values(self):
        tree = DecisionTreeClassifier().fit([[1], [2], [4], [5]], ['a', 'a', 'b', 'b'])
        self.assertEqual(tree.root_.value, 3)
        self.assertEqual(tree.predict([[0], [3], [3.1], [8]]), ['a', 'a', 'b', 'b'])
        self.assertAlmostEqual(tree.root_.gain, 1)

    def test_categorical_and_unseen_category(self):
        tree = DecisionTreeClassifier().fit([['red'], ['red'], ['blue'], ['green']], [1, 1, 0, 0])
        self.assertEqual(tree.predict([['red'], ['purple']]), [1, 0])

    def test_binary_probabilities_and_tie(self):
        tree = DecisionTreeClassifier(max_depth=0).fit([[0], [1], [2], [3]], ['b', 'a', 'a', 'b'])
        self.assertEqual(tree.classes_, ['b', 'a'])
        self.assertEqual(tree.predict_proba([[20]]), [[0.5, 0.5]])
        tied = DecisionTreeClassifier(max_depth=0).fit([[0], [1]], ['z', 'a'])
        self.assertEqual(tied.predict([[2]]), ['z'])

    def test_missingness_can_be_the_only_signal(self):
        tree = DecisionTreeClassifier().fit([[None], [float('nan')], [7], [7]], ['a', 'a', 'b', 'b'])
        self.assertEqual(tree.predict([[None], [math.nan], [7], [8]]), ['a', 'a', 'b', 'b'])
        self.assertEqual(tree.n_leaves_, 2)

    def test_learned_missing_direction(self):
        tree = DecisionTreeClassifier().fit([[0], [1], [9], [10], [None]], [0, 0, 1, 1, 1])
        self.assertEqual(tree.score([[0], [1], [9], [10], [None]], [0, 0, 1, 1, 1]), 1)
        self.assertEqual(tree.predict([[None]]), [1])

    def test_tree_size_limits(self):
        X = [[i] for i in range(20)]
        y = [i % 2 for i in range(20)]
        for depth in [0, 1, 2, 4]:
            tree = DecisionTreeClassifier(max_depth=depth).fit(X, y)
            self.assertLessEqual(tree.depth_, depth)
            self.assertLessEqual(tree.n_leaves_, 2 ** depth)
        for leaves in [1, 2, 3, 5]:
            tree = DecisionTreeClassifier(max_depth=None, max_leaf_nodes=leaves).fit(X, y)
            self.assertLessEqual(tree.n_leaves_, leaves)

    def test_minimum_leaf_size(self):
        tree = DecisionTreeClassifier(max_depth=None, min_samples_leaf=3).fit([[i] for i in range(20)], [i % 2 for i in range(20)])
        stack = [tree.root_]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                self.assertGreaterEqual(node.samples, 3)
            else:
                stack.extend([node.left, node.right])

    def test_pure_constant_and_all_missing_features(self):
        for X, y in [([[1], [1]], [0, 1]), ([[None], [None]], [0, 1]), ([[1], [2]], [0, 0])]:
            tree = DecisionTreeClassifier().fit(X, y)
            self.assertEqual(tree.n_leaves_, 1)

    def test_adjacent_and_extreme_floats(self):
        for low, high in [(1e308, 1.1e308), (-1e308, 1e308), (1.0, math.nextafter(1.0, 2.0))]:
            tree = DecisionTreeClassifier().fit([[low], [high]], [0, 1])
            self.assertEqual(tree.predict([[low], [high]]), [0, 1])

    def test_display_and_dot_escaping(self):
        tree = DecisionTreeClassifier().fit([[0], [1]], ['a', 'b'], ['a"b'])
        self.assertIn('yes ->', tree.export_text())
        self.assertIn('gain=1.0000', tree.export_text())
        self.assertIn('digraph Tree', tree.export_dot())
        self.assertIn('a\\"b', tree.export_dot())

    def test_invalid_inputs(self):
        for kwargs in [{'max_depth': -1}, {'max_leaf_nodes': 0}, {'min_samples_leaf': 0}, {'max_depth': True}, {'min_gain': math.nan}]:
            with self.assertRaises(ValueError):
                DecisionTreeClassifier(**kwargs)
        tree = DecisionTreeClassifier()
        with self.assertRaises(RuntimeError):
            tree.predict([[1]])
        for X, y in [([], []), ([[1], [2, 3]], [0, 1]), ([[1]], []), ([[1]], [None]), ([[math.inf]], [0])]:
            with self.assertRaises(ValueError):
                tree.fit(X, y)
        tree.fit([[1], [2]], [0, 1])
        with self.assertRaises(ValueError):
            tree.predict([[1, 2]])
        with self.assertRaises(ValueError):
            tree.score([[1]], [])
        with self.assertRaises(ValueError):
            tree.fit([], [])
        with self.assertRaises(RuntimeError):
            tree.predict([[1]])

    def test_rejects_multiclass_targets(self):
        with self.assertRaisesRegex(ValueError, "Binary classification only"):
            DecisionTreeClassifier().fit([[0], [1], [2]], ["a", "b", "c"])

    def test_split_gain_matches_brute_force_partitions(self):
        import random
        rng = random.Random(104)
        for _ in range(30):
            X = [[rng.choice([None, 0, 1, 2, 3]), rng.choice([None, 'a', 'b', 'c'])] for _ in range(20)]
            y = [rng.randrange(2) for _ in X]
            best = 0.0
            for feature, values in [(0, [0.5, 1.5, 2.5]), (1, ['a', 'b', 'c'])]:
                for value in values:
                    for missing_left in [True, False]:
                        mask = [(missing_left if row[feature] is None else
                                 row[feature] <= value if feature == 0 else row[feature] == value) for row in X]
                        left = [label for label, flag in zip(y, mask) if flag]
                        right = [label for label, flag in zip(y, mask) if not flag]
                        if left and right:
                            best = max(best, information_gain(y, left, right))
                left = [label for label, row in zip(y, X) if row[feature] is None]
                right = [label for label, row in zip(y, X) if row[feature] is not None]
                if left and right:
                    best = max(best, information_gain(y, left, right))
            tree = DecisionTreeClassifier(max_depth=1).fit(X, y)
            self.assertAlmostEqual(tree.root_.gain, best)

    def test_deterministic_and_refit(self):
        X, y = [[i, str(i % 3)] for i in range(30)], [i % 2 for i in range(30)]
        a, b = DecisionTreeClassifier().fit(X, y), DecisionTreeClassifier().fit(X, y)
        self.assertEqual(a.export_text(), b.export_text())
        a.fit([[0], [1]], ['new', 'new'])
        self.assertEqual(a.predict([[10]]), ['new'])


if __name__ == '__main__':
    unittest.main()
