"""An educational decision tree built from scratch, with no runtime dependencies."""
from .tree import DecisionTreeClassifier, entropy, information_gain

__version__ = "0.1.0"
__all__ = ["DecisionTreeClassifier", "entropy", "information_gain"]
