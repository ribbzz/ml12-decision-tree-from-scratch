"""Generate a reproducible, artificial binary-classification teaching dataset."""
import csv
from pathlib import Path
import random


def generate(path, n=1000, seed=2026):
    rng = random.Random(seed)
    with Path(path).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['hours_practiced', 'practice_type', 'previous_experience', 'passed'])
        for _ in range(n):
            hours = round(rng.uniform(0, 10), 2)
            practice = rng.choice(['solo', 'guided', 'group'])
            experience = rng.choice(['yes', 'no'])
            passed = hours >= (3 if practice == 'guided' else 5) or (experience == 'yes' and hours >= 2)
            if rng.random() < 0.04:
                passed = not passed
            observed_hours = '' if rng.random() < 0.03 else hours
            writer.writerow([observed_hours, practice, experience, 'yes' if passed else 'no'])


if __name__ == '__main__':
    generate(Path(__file__).with_name('practice.csv'))
