"""Validated configuration and built-in modality metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


Target = Literal["nuclear", "whole_cell"]

DEFAULT_MODELS = ("cellpose", "stardist", "cellsam", "cellpose_sam")

# These routes reproduce the target semantics in the frozen LLM inference
# prompts. In particular, mIF is membrane-stained whole-cell segmentation;
# inversion in a model-specific preprocessing path does not make it nuclear.
MODALITY_TARGETS: dict[str, Target] = {
    "dapi": "nuclear",
    "ssdna": "nuclear",
    "h&e": "nuclear",
    "he": "nuclear",
    "mif": "whole_cell",
    "phase_contrast": "whole_cell",
    "phc": "whole_cell",
    "fluorescence": "whole_cell",
    "fluorescent": "whole_cell",
    "brightfield": "whole_cell",
    "dic": "whole_cell",
}

MODEL_IDS = {
    "gpt5.5": "gpt-5.5",
    "gpt-5.5": "gpt-5.5",
    "gpt6": "gpt-6-astra",
    "gpt-6": "gpt-6-astra",
    "gpt-6-astra": "gpt-6-astra",
    "claude": "claude-fable-5-1",
    "claude-fable-5.1": "claude-fable-5-1",
    "claude-fable-5-1": "claude-fable-5-1",
}

# Empirical dollars per ranked patch, already including three repetitions.
# These are historical planning values; actual usage is reported separately.
HISTORICAL_COST_PER_PATCH = {
    "gpt-5.5": 20.87682 / 100.0,
    "gpt-6-astra": 4.5877 / 40.0,
    "claude-fable-5-1": 3.7301 / 40.0,
}


def normalize_modality(value: str) -> str:
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "dapi": "DAPI",
        "ssdna": "ssDNA",
        "he": "H&E",
        "h&e": "H&E",
        "mif": "mIF",
        "phc": "phase_contrast",
        "phasecontrast": "phase_contrast",
        "phase_contrast": "phase_contrast",
        "fluorescent": "fluorescence",
        "fluorescence": "fluorescence",
        "brightfield": "brightfield",
        "dic": "DIC",
    }
    return aliases.get(key, value.strip())


def preset_target(modality: str) -> Target | None:
    key = modality.strip().lower().replace("-", "_").replace(" ", "_")
    return MODALITY_TARGETS.get(key)


@dataclass(frozen=True)
class GridConfig:
    """Controls for candidate-density-aware crop planning.

    ``target_cells_per_core`` is deliberately a soft target.  The planner first
    preserves the source-image aspect ratio (an exactly square image receives a
    square grid), then selects an occupancy within the acceptable range that is
    closest to the target.  Limits are applied afterwards and may only add
    splits; they never make a core larger.
    """

    target_cells_per_core: float = 5.0
    minimum_expected_cells_per_core: float = 3.0
    maximum_expected_cells_per_core: float = 10.0
    max_core_height: int = 2048
    max_core_width: int = 2048
    max_core_area: int = 4_194_304
    max_patch_count: int = 4096
    vlm_panel_size: int = 256
    count_disagreement_ratio: float = 2.0
    padding_fraction: float = 0.10

    def __post_init__(self) -> None:
        if self.target_cells_per_core <= 0:
            raise ValueError("target_cells_per_core must be positive")
        if self.minimum_expected_cells_per_core <= 0:
            raise ValueError("minimum_expected_cells_per_core must be positive")
        if self.maximum_expected_cells_per_core < self.minimum_expected_cells_per_core:
            raise ValueError(
                "maximum_expected_cells_per_core must be at least the minimum"
            )
        if not (
            self.minimum_expected_cells_per_core
            <= self.target_cells_per_core
            <= self.maximum_expected_cells_per_core
        ):
            raise ValueError("target_cells_per_core must lie within the expected range")
        for name in (
            "max_core_height",
            "max_core_width",
            "max_core_area",
            "max_patch_count",
            "vlm_panel_size",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.count_disagreement_ratio < 1:
            raise ValueError("count_disagreement_ratio must be at least 1")
        if not 0 <= self.padding_fraction < 1:
            raise ValueError("padding_fraction must be in [0, 1)")
@dataclass(frozen=True)
class EnsembleConfig:
    minimum_area: int = 1
    support_required: int = 2
    anchor_support_iou: float = 0.50
    residual_complete_link_iou: float = 0.50
    anchor_null_iou: float = 0.10
    refinement_iou: float = 0.50
    refinement_other_peers: int = 2

    def __post_init__(self) -> None:
        if self.minimum_area < 1 or self.support_required < 1:
            raise ValueError("minimum_area and support_required must be positive")
        for name in (
            "anchor_support_iou",
            "residual_complete_link_iou",
            "anchor_null_iou",
            "refinement_iou",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class RunConfig:
    input_dir: Path
    output_dir: Path
    target: Target
    modality: str
    modality_description: str | None = None
    teacher_fraction: float = 1.0
    vlm_model: str = "gpt-5.5"
    repetitions: int = 3
    models: tuple[str, ...] = field(default_factory=lambda: DEFAULT_MODELS)
    grid: GridConfig = field(default_factory=GridConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    seed: int = 120912

    def __post_init__(self) -> None:
        if self.target not in ("nuclear", "whole_cell"):
            raise ValueError("target must be 'nuclear' or 'whole_cell'")
        if not 0 <= self.teacher_fraction <= 1:
            raise ValueError("teacher_fraction must be in [0, 1]")
        if self.repetitions != 3:
            raise ValueError("the validated VSCell pipeline uses exactly 3 repetitions")
        if len(self.models) < 3 or len(set(self.models)) != len(self.models):
            raise ValueError("models must contain at least 3 unique candidates")
        canonical = MODEL_IDS.get(self.vlm_model.lower(), self.vlm_model)
        object.__setattr__(self, "vlm_model", canonical)
        object.__setattr__(self, "modality", normalize_modality(self.modality))
        object.__setattr__(self, "input_dir", Path(self.input_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["input_dir"] = str(self.input_dir)
        value["output_dir"] = str(self.output_dir)
        value["models"] = list(self.models)
        return value

    def estimated_teacher_cost(self, patch_count: int) -> float | None:
        per_patch = HISTORICAL_COST_PER_PATCH.get(self.vlm_model)
        if per_patch is None:
            return None
        teacher_patches = round(int(patch_count) * self.teacher_fraction)
        return teacher_patches * per_patch
