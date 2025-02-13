# file: similarity.py

import math
from typing import List
import numpy as np


class Similarity:
    """
    Provides common methods for measuring distance/similarity between vectors.
    """

    def _ensure_same_length(
        self, vec_a: List[float], vec_b: List[float], method_name: str
    ) -> None:
        """
        Checks that vec_a and vec_b have the same length. Raises ValueError if not.
        """
        if len(vec_a) != len(vec_b):
            raise ValueError(
                f"Vectors must be the same length for {method_name}. "
                f"Got lengths {len(vec_a)} and {len(vec_b)}."
            )

    def calc_euclidean(self, vec_a: List[float], vec_b: List[float]) -> float:
        """
        Euclidean distance = sqrt( sum( (a_i - b_i)^2 ) )
        """
        self._ensure_same_length(vec_a, vec_b, "Euclidean distance")
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))

    def calc_manhattan(self, vec_a: List[float], vec_b: List[float]) -> float:
        """
        Manhattan distance = sum( |a_i - b_i| )
        """
        self._ensure_same_length(vec_a, vec_b, "Manhattan distance")
        return sum(abs(a - b) for a, b in zip(vec_a, vec_b))

    def calc_cosine(self, vec_a: List[float], vec_b: List[float]) -> float:
        """
        Cosine similarity = (A • B) / (||A|| * ||B||)
        Returns a value in [-1, 1], where 1 = very similar, -1 = opposite.
        """
        self._ensure_same_length(vec_a, vec_b, "Cosine similarity")

        arr_a = np.array(vec_a)
        arr_b = np.array(vec_b)

        dot = np.dot(arr_a, arr_b)
        norm_a = np.linalg.norm(arr_a)
        norm_b = np.linalg.norm(arr_b)

        if norm_a == 0 or norm_b == 0:
            # Avoid division by zero if one vector is all zeros
            return 0.0

        return dot / (norm_a * norm_b)
