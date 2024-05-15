"""Constants and utility functions for sorting

Date -- 16.07.2023
Author -- Martin Kostelnik
"""

from typing import List
from itertools import pairwise
import random


def compare_regions(true_region: List[str], pred_region: List[str]) -> int:
    hits = 0
    for pair in pairwise(true_region):
        if pair in pairwise(pred_region):
            hits += 1

    return hits


def split_into_regions(text: str | List[str], region_size: int, shuffle: bool = False) -> List[str]:
    lines = text.strip().split("\n") if isinstance(text, str) else text

    regions = [lines[i:i+region_size] for i in range(0, len(lines), region_size)]

    if shuffle:
        for region in regions:
            random.shuffle(region)

    return ["\n".join(region) for region in regions]
    
