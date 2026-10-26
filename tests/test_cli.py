import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scratch_tree.cli import main, stratified_split, convert_features


class CLITests(unittest.TestCase):
    def test_disjoint_reproducible_split(self):
        labels = ['a'] * 8 + ['b'] * 6 + ['rare']
        train, test = stratified_split(labels, 0.2, 42)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(sorted(train + test), list(range(len(labels))))
        self.assertIn(14, train)
        self.assertEqual((train, test), stratified_split(labels, 0.2, 42))

    def test_feature_types_fit_on_training_only(self):
        with self.assertRaises(ValueError):
            convert_features([['1'], ['2'], ['new-text']], [0, 1], ['x'], [])
        rows, kinds = convert_features([['001'], ['002']], [0], ['code'], ['code'])
        self.assertEqual(kinds, ['categorical'])
        self.assertEqual(rows[0][0], '001')

    def test_csv_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv = root / 'data.csv'
            csv.write_text('id,number,color,label\n' + ''.join(f'{i},{i},{"red" if i < 10 else "blue"},{"A" if i < 10 else "B"}\n' for i in range(20)))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main(['--csv', str(csv), '--target', 'label', '--drop', 'id', '--dot', str(root / 'tree.dot'), '--metrics', str(root / 'metrics.json')])
            info = json.loads((root / 'metrics.json').read_text())
            self.assertEqual(info['training_rows'] + info['test_rows'], 20)
            self.assertEqual(info['evaluation']['accuracy'], 1)
            self.assertIn('leaf:', output.getvalue())
            self.assertIn('digraph', (root / 'tree.dot').read_text())

    def test_invalid_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.csv'
            path.write_text('x,label\n1,\n')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                main(['--csv', str(path), '--target', 'label'])
            self.assertEqual(exc.exception.code, 2)

    def test_rejects_three_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'multiclass.csv'
            path.write_text('x,label\n1,a\n2,b\n3,c\n')
            with contextlib.redirect_stderr(io.StringIO()) as output, self.assertRaises(SystemExit):
                main(['--csv', str(path), '--target', 'label'])
            self.assertIn('Binary classification only', output.getvalue())
