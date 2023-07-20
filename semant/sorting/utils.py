"""Constants and utility functions for sorting

Date -- 16.07.2023
Author -- Martin Kostelnik
"""

from typing import List
from itertools import pairwise


def compare_regions(true_region: List[str], pred_region: List[str]) -> int:
    hits = 0
    for pair in pairwise(true_region):
        if pair in pairwise(pred_region):
            hits += 1

    return hits

