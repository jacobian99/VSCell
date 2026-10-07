"""Target-aware ranking prompts derived from the validated VSCell runs."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Sequence

from .config import Target, normalize_modality, preset_target
from .ranking import LETTERS, ORDINALS


TARGET_PHRASES: dict[Target, tuple[str, str, str, str]] = {
    "nuclear": ("nucleus", "nuclei", "true nuclei", "cell-nucleus"),
    "whole_cell": (
        "cell",
        "cells",
        "true cells -- whole-cell bodies",
        "whole-cell",
    ),
}

MODALITY_DESCRIPTIONS = {
    "DAPI": (
        "This is a DAPI-stained nuclear fluorescence microscopy crop. Nuclei "
        "appear as bright blobs on a dark background; the target is nuclei only."
    ),
    "ssDNA": (
        "This is an ssDNA-stained nuclear microscopy crop (single-stranded DNA "
        "marker; nuclei appear bright on a dark background, similar to DAPI); "
        "the target is nuclei only."
    ),
    "H&E": (
        "This is an H&E histology crop rendered as INVERTED grayscale so nuclei "
        "appear bright on a dark background (matching DAPI/ssDNA polarity); the "
        "target is nuclei only."
    ),
    "mIF": (
        "This is an mIF (multiplex immunofluorescence) membrane-stained microscopy "
        "crop. UNLIKE DAPI/ssDNA, mIF stains the cell MEMBRANE: BRIGHT outlines "
        "mark the membrane and the DARK INTERIORS inside them are the cell bodies. "
        "Nuclei are NOT visible in this channel. The target is WHOLE-CELL instance "
        "segmentation -- each segmented region should fill ONE cell body, bounded "
        "by the membrane that encloses it. Yellow fill that traces membranes "
        "themselves (rather than the dark interiors) is WRONG."
    ),
    "phase_contrast": (
        "This is a label-free phase-contrast microscopy crop. Cells generally "
        "appear as textured or granular regions against a smoother background and "
        "may exhibit bright/dark refractive halos near their boundaries. The target "
        "is whole-cell instance segmentation."
    ),
    "fluorescence": (
        "This is a fluorescence microscopy crop. Cells generally appear as coherent "
        "colored or grayscale regions with fluorescence contrast against the "
        "surrounding field, although signal intensity, polarity, and labeling may "
        "vary. Diffuse or punctate background fluorescence may also be present. The "
        "target is whole-cell instance segmentation."
    ),
    "brightfield": (
        "This is a stained brightfield microscopy crop. Cells generally appear as "
        "colored, textured bodies against a lighter field, although stain intensity "
        "and contrast may vary. Other stained structures or debris may also be "
        "present. The target is whole-cell instance segmentation."
    ),
    "DIC": (
        "This is a label-free differential interference contrast (DIC) microscopy "
        "crop. Cells generally appear as textured, relief-like regions with paired "
        "bright/dark edge shading against a smoother background; shading direction "
        "and contrast may vary. The target is whole-cell instance segmentation."
    ),
}


def modality_description(
    modality: str,
    target: Target,
    custom_description: str | None = None,
) -> str:
    if custom_description:
        return custom_description.strip()
    canonical = normalize_modality(modality)
    expected_target = preset_target(canonical)
    if expected_target is not None and expected_target != target:
        raise ValueError(
            f"built-in modality {canonical!r} is validated for target "
            f"{expected_target!r}, not {target!r}; provide an explicit custom "
            "modality description to intentionally override the preset"
        )
    if canonical in MODALITY_DESCRIPTIONS:
        return MODALITY_DESCRIPTIONS[canonical]
    if target == "whole_cell":
        return (
            f"This is a {modality} microscopy crop. The target is WHOLE-CELL "
            "instance segmentation; each segmented region should fill one cell body."
        )
    return (
        f"This is a {modality} microscopy crop. The target is NUCLEAR instance "
        "segmentation; each segmented region should contain one nucleus."
    )


def _labels(count: int) -> list[str]:
    if count < 2 or count > min(len(LETTERS), len(ORDINALS)):
        raise ValueError("candidate count must be between 2 and 10")
    return list(LETTERS[:count])


def _ranking_template(count: int) -> str:
    return " ".join(f"{ordinal}=[letter]" for ordinal in ORDINALS[:count])


def system_prompt(candidate_count: int, target: Target) -> str:
    labels = _labels(candidate_count)
    last_two = ", ".join(labels[:-1]) + " and " + labels[-1]
    letter_set = ", ".join(labels)
    number = {4: "FOUR", 5: "FIVE", 6: "SIX"}.get(
        candidate_count, str(candidate_count)
    )
    target_phrase = TARGET_PHRASES[target][3]
    return (
        f"You are an expert evaluator of {target_phrase} instance segmentation. "
        "You will see a reference microscopy image, then "
        f"{number} candidate segmentations of the SAME image labelled {last_two}, "
        f"produced by {candidate_count} unknown methods. You have no information "
        "about which method produced which result and must NOT assume any is better "
        "a priori -- judge ONLY from the images. Rank the candidates from best to "
        "worst by how faithfully each segments the true target. You MUST produce a "
        "strict ranking with NO ties and NO repeats. End your response with EXACTLY "
        f"one line: 'RANKING: {_ranking_template(candidate_count)}' where each "
        f"[letter] is one of {letter_set}."
    )


def data_url(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def user_content(
    reference_path: str | Path,
    candidate_paths: dict[str, str | Path],
    mapping: dict[str, str],
    modality: str,
    target: Target,
    custom_modality_description: str | None = None,
) -> list[dict[str, object]]:
    count = len(mapping)
    labels = _labels(count)
    if list(mapping) != labels or set(mapping.values()) != set(candidate_paths):
        raise ValueError("candidate paths and anonymized mapping are inconsistent")
    singular, plural, ground_truth_phrase, _ = TARGET_PHRASES[target]
    label_csv = ", ".join(labels[:-1]) + ", " + labels[-1]
    number = {4: "four", 5: "five", 6: "six"}.get(count, str(count))
    criteria = (
        f"The next {number} images each show the SAME reference with ONE method's "
        f"segmentation overlaid: a single translucent YELLOW fill marks all pixels "
        f"the method assigned to a {singular}, and a hard GREEN outline traces every "
        f"segmented {singular} boundary (a green line between two adjacent yellow "
        f"regions means the method split them into separate {plural}). Candidates "
        f"appear in alphabetical order {label_csv}. Judge each overlay against the "
        f"real {plural} visible in the reference.\n\n"
        f"Rank the candidates by how faithfully each matches the TRUE {plural}, "
        "weighing ALL of the following together, with no single criterion dominating:\n"
        f"  (1) Boundary fit: does each region's outline closely follow the actual "
        f"{singular} extent (not markedly too large, too small, or ragged)?\n"
        f"  (2) Completeness: is every real {singular} found? (missed {plural} = "
        "false negatives)\n"
        f"  (3) Correctness: are there colored regions where there is no real "
        f"{singular}? (false positives)\n"
        f"  (4) Separation: are touching {plural} split into distinct regions (not "
        f"merged), and single {plural} not over-split?\n\n"
        f"Give a one-line assessment of each of the {number} candidates "
        f"({', '.join(labels)}), then produce the strict best-to-worst ranking."
    )
    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": modality_description(
                modality, target, custom_description=custom_modality_description
            ),
        },
        {
            "type": "text",
            "text": (
                "REFERENCE IMAGE -- study this first; treat the "
                f"{ground_truth_phrase} here as the ground truth for what should be "
                "segmented:"
            ),
        },
        {"type": "image_url", "image_url": {"url": data_url(reference_path)}},
        {"type": "text", "text": criteria},
    ]
    for label in labels:
        model = mapping[label]
        content.extend(
            [
                {"type": "text", "text": f"Candidate {label}:"},
                {
                    "type": "image_url",
                    "image_url": {"url": data_url(candidate_paths[model])},
                },
            ]
        )
    content.append(
        {
            "type": "text",
            "text": (
                "End your response with EXACTLY one line: "
                f"'RANKING: {_ranking_template(count)}' where each [letter] is one "
                f"of {', '.join(labels)} (no repeats)."
            ),
        }
    )
    return content


def expected_candidate_names(mapping: dict[str, str]) -> Sequence[str]:
    return tuple(mapping.values())
