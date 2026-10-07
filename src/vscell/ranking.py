"""Candidate anonymization, strict ranking parsing, and Borda aggregation."""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from typing import Iterable, Sequence


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ORDINALS = (
    "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"
)


def shuffled_mapping(crop_id: str, repetition: int, models: Sequence[str]) -> dict[str, str]:
    if len(models) > len(LETTERS):
        raise ValueError("too many candidate models")
    material = f"vscell\0{crop_id}\0{int(repetition)}".encode()
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    shuffled = list(models)
    random.Random(seed).shuffle(shuffled)
    return dict(zip(LETTERS, shuffled))


def parse_ranking(text: str, mapping: dict[str, str]) -> list[str]:
    labels = list(mapping)
    if not labels or labels != list(LETTERS[: len(labels)]):
        raise ValueError("mapping labels must be consecutive letters beginning at A")
    ordinals = ORDINALS[: len(labels)]
    label_class = "".join(labels)
    pattern = r"RANKING:\s*" + r"\s+".join(
        rf"{ordinal}=\[?([{label_class}])\]?" for ordinal in ordinals
    )
    match = re.search(pattern, text or "", flags=re.IGNORECASE)
    if match is None:
        raise ValueError("response does not contain a valid strict RANKING line")
    ranked_labels = [match.group(index + 1).upper() for index in range(len(labels))]
    if len(set(ranked_labels)) != len(labels) or set(ranked_labels) != set(labels):
        raise ValueError("ranking must contain every candidate exactly once")
    return [mapping[label] for label in ranked_labels]


def borda_aggregate(
    rankings: Iterable[Sequence[str]], candidates: Sequence[str] | None = None
) -> tuple[list[str], dict[str, int]]:
    rows = [list(row) for row in rankings]
    if not rows:
        raise ValueError("at least one ranking is required")
    names = list(candidates if candidates is not None else rows[0])
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    expected = set(names)
    scores = {name: 0 for name in names}
    for row in rows:
        if len(row) != len(names) or set(row) != expected:
            raise ValueError("every ranking must be a permutation of the candidates")
        for position, name in enumerate(row):
            # Matches the paper pipeline: K, K-1, ..., 1 points.
            scores[name] += len(names) - position
    order = sorted(names, key=lambda name: (-scores[name], name))
    return order, scores


def pairwise_vote_probabilities(
    rankings: Iterable[Sequence[str]], candidates: Sequence[str]
) -> list[list[float]]:
    rows = [list(row) for row in rankings]
    index = {name: idx for idx, name in enumerate(candidates)}
    matrix = [[0.5 for _ in candidates] for _ in candidates]
    for left, left_name in enumerate(candidates):
        for right in range(left + 1, len(candidates)):
            right_name = candidates[right]
            votes = Counter(
                row.index(left_name) < row.index(right_name) for row in rows
            )
            probability = votes[True] / len(rows)
            matrix[left][right] = probability
            matrix[right][left] = 1.0 - probability
    return matrix
