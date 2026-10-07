"""Optional identity-blind student ranker distilled from VLM rankings.

The architecture and losses are ported from the validated CB ranking student.
Candidate identities are never embedded; the same encoder scores each candidate
and uses mean/max set context, so a checkpoint is tied to the candidate set only
through its recorded compatibility metadata.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class StudentExample:
    patch_id: str
    image_id: str
    reference_path: str
    candidate_mask_paths: dict[str, str]
    modality: str
    target: str
    teacher_order: list[str]
    borda_scores: dict[str, int]
    pairwise_votes: list[list[float]]

    @property
    def candidate_models(self) -> list[str]:
        return list(self.candidate_mask_paths)


def _torch():
    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
    except ImportError as error:
        raise RuntimeError("install vscell[student] to use the student ranker") from error
    return torch, nn, functional


def instance_maps(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D instance mask, got {mask.shape}")
    foreground = mask > 0
    different = np.zeros(mask.shape, dtype=bool)
    different[:-1] |= mask[:-1] != mask[1:]
    different[1:] |= mask[:-1] != mask[1:]
    different[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    different[:, 1:] |= mask[:, :-1] != mask[:, 1:]
    boundary = different & foreground
    return np.stack([foreground, boundary], axis=0).astype(np.float32)


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    if mask.shape == (size, size):
        return mask.astype(np.int64, copy=False)
    return np.asarray(
        Image.fromarray(mask.astype(np.int32), mode="I").resize(
            (size, size), Image.Resampling.NEAREST
        )
    ).astype(np.int64)


def _resize_reference(path: str | Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def build_model_class():
    torch, nn, functional = _torch()

    class ConvBlock(nn.Sequential):
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
            super().__init__(
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
                nn.SiLU(inplace=True),
            )

    class StudentRanker(nn.Module):
        def __init__(
            self,
            width: int = 16,
            embedding_dim: int = 128,
            dropout: float = 0.15,
            num_modalities: int = 1,
            num_targets: int = 1,
        ) -> None:
            super().__init__()
            self.image_encoder = nn.Sequential(
                ConvBlock(3, width, stride=2),
                ConvBlock(width, 2 * width, stride=2),
                ConvBlock(2 * width, 4 * width, stride=2),
                ConvBlock(4 * width, 6 * width, stride=2),
            )
            self.mask_encoder = nn.Sequential(
                ConvBlock(2, width, stride=2),
                ConvBlock(width, 2 * width, stride=2),
                ConvBlock(2 * width, 2 * width, stride=2),
                ConvBlock(2 * width, 2 * width, stride=2),
            )
            self.fusion = nn.Sequential(
                ConvBlock(8 * width, 4 * width),
                nn.Conv2d(4 * width, 4 * width, 1, bias=False),
                nn.GroupNorm(_group_count(4 * width), 4 * width),
                nn.SiLU(inplace=True),
            )
            self.modality_embedding = nn.Embedding(num_modalities, 8)
            self.target_embedding = nn.Embedding(num_targets, 4)
            self.candidate_projector = nn.Sequential(
                nn.Linear(8 * width + 12, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.score_head = nn.Sequential(
                nn.Linear(4 * embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, 1),
            )

        def forward(self, reference, mask_maps, modality, target_kind):
            batch, candidates, channels, height, width = mask_maps.shape
            image_features = self.image_encoder(reference)
            mask_features = self.mask_encoder(
                mask_maps.reshape(batch * candidates, channels, height, width)
            )
            if mask_features.shape[-2:] != image_features.shape[-2:]:
                mask_features = functional.interpolate(
                    mask_features,
                    size=image_features.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            image_features = (
                image_features[:, None]
                .expand(-1, candidates, -1, -1, -1)
                .reshape(batch * candidates, image_features.shape[1], *image_features.shape[-2:])
            )
            fused = self.fusion(torch.cat([image_features, mask_features], dim=1))
            pooled = torch.cat(
                [
                    functional.adaptive_avg_pool2d(fused, 1),
                    functional.adaptive_max_pool2d(fused, 1),
                ],
                dim=1,
            ).flatten(1)
            metadata = torch.cat(
                [self.modality_embedding(modality), self.target_embedding(target_kind)], dim=1
            )
            metadata = metadata[:, None].expand(-1, candidates, -1).reshape(
                batch * candidates, -1
            )
            candidate = self.candidate_projector(torch.cat([pooled, metadata], dim=1))
            candidate = candidate.reshape(batch, candidates, -1)
            context_mean = candidate.mean(dim=1, keepdim=True).expand_as(candidate)
            context_max = candidate.max(dim=1, keepdim=True).values.expand_as(candidate)
            score_input = torch.cat(
                [candidate, context_mean, context_max, candidate - context_mean], dim=-1
            )
            return self.score_head(score_input).squeeze(-1)

    return StudentRanker


def _load_tensor_example(
    example: StudentExample,
    candidate_models: Sequence[str],
    modality_vocabulary: Sequence[str],
    target_vocabulary: Sequence[str],
    image_size: int,
) -> dict[str, Any]:
    import tifffile

    torch, _, _ = _torch()
    if list(example.candidate_mask_paths) != list(candidate_models):
        raise ValueError(f"{example.patch_id}: incompatible candidate set/order")
    reference = _resize_reference(example.reference_path, image_size)
    maps = []
    for model in candidate_models:
        mask = tifffile.imread(example.candidate_mask_paths[model])
        maps.append(instance_maps(_resize_mask(np.asarray(mask), image_size)))
    teacher_indices = [candidate_models.index(model) for model in example.teacher_order]
    return {
        "reference": torch.from_numpy(reference.transpose(2, 0, 1)).float().mul(2).sub(1),
        "mask_maps": torch.from_numpy(np.stack(maps)).float(),
        "modality": torch.tensor(modality_vocabulary.index(example.modality), dtype=torch.long),
        "target": torch.tensor(target_vocabulary.index(example.target), dtype=torch.long),
        "borda": torch.tensor(
            [example.borda_scores[model] for model in candidate_models], dtype=torch.float32
        ),
        "pairwise": torch.tensor(example.pairwise_votes, dtype=torch.float32),
        "teacher_order": torch.tensor(teacher_indices, dtype=torch.long),
    }


def _loss(scores, pairwise, borda, listwise_weight: float = 0.25, temperature: float = 2.0):
    torch, _, functional = _torch()
    candidates = scores.shape[1]
    upper = torch.triu(
        torch.ones((candidates, candidates), dtype=torch.bool, device=scores.device),
        diagonal=1,
    )
    differences = scores[:, :, None] - scores[:, None, :]
    pair_loss = functional.binary_cross_entropy_with_logits(
        differences[:, upper], pairwise[:, upper]
    )
    target_distribution = functional.softmax(borda / temperature, dim=1)
    list_loss = -(
        target_distribution * functional.log_softmax(scores, dim=1)
    ).sum(dim=1).mean()
    return pair_loss + listwise_weight * list_loss


def train_student(
    examples: Sequence[StudentExample],
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
    epochs: int = 60,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    image_size: int = 256,
    width: int = 16,
    embedding_dim: int = 128,
    dropout: float = 0.15,
    batch_size: int = 8,
    seed: int = 120912,
) -> dict[str, Any]:
    """Train from teacher-ranked patches without loading segmentation GT."""
    torch, _, _ = _torch()
    examples = list(examples)
    if len(examples) < 4:
        raise ValueError("at least four teacher-ranked patches are required")
    candidate_sets = {tuple(example.candidate_models) for example in examples}
    if len(candidate_sets) != 1:
        raise ValueError("one student run requires one consistent candidate set")
    candidates = list(next(iter(candidate_sets)))
    modalities = sorted({example.modality for example in examples})
    targets = sorted({example.target for example in examples})
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    groups: dict[str, list[StudentExample]] = {}
    for example in examples:
        groups.setdefault(example.image_id, []).append(example)
    image_ids = sorted(groups)
    if len(image_ids) < 2:
        raise ValueError(
            "training from scratch requires teacher crops from at least two source "
            "images so validation is image-grouped without leakage"
        )
    random.Random(seed).shuffle(image_ids)
    validation_group_count = max(1, round(len(image_ids) * 0.10))
    validation_ids = set(image_ids[:validation_group_count])
    validation = [
        example for example in examples if example.image_id in validation_ids
    ]
    training = [
        example for example in examples if example.image_id not in validation_ids
    ]
    tensors = {
        "train": [
            _load_tensor_example(item, candidates, modalities, targets, image_size)
            for item in training
        ],
        "validation": [
            _load_tensor_example(item, candidates, modalities, targets, image_size)
            for item in validation
        ],
    }
    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else device
    )
    if resolved_device == "auto":
        resolved_device = "cpu"
    Model = build_model_class()
    model_config = {
        "width": width,
        "embedding_dim": embedding_dim,
        "dropout": dropout,
        "num_modalities": len(modalities),
        "num_targets": len(targets),
    }
    model = Model(**model_config).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    history: list[dict[str, float | int]] = []

    def epoch(rows: list[dict[str, Any]], optimize: bool) -> float:
        model.train(optimize)
        total = 0.0
        order = list(range(len(rows)))
        rng = random.Random(seed + len(history) * 1009 + (1 if optimize else 0))
        if optimize:
            rng.shuffle(order)
        batches = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
        for indices in batches:
            reference = torch.stack([rows[index]["reference"] for index in indices])
            masks = torch.stack([rows[index]["mask_maps"] for index in indices])
            modality = torch.stack([rows[index]["modality"] for index in indices])
            target = torch.stack([rows[index]["target"] for index in indices])
            borda = torch.stack([rows[index]["borda"] for index in indices])
            pairwise = torch.stack([rows[index]["pairwise"] for index in indices])
            if optimize:
                # Historical synchronized D4-like augmentation: the reference
                # and every candidate receive the same spatial transform.
                rotations = rng.randrange(4)
                reference = torch.rot90(reference, rotations, dims=(-2, -1))
                masks = torch.rot90(masks, rotations, dims=(-2, -1))
                if rng.random() < 0.5:
                    reference = torch.flip(reference, dims=(-1,))
                    masks = torch.flip(masks, dims=(-1,))
                if rng.random() < 0.5:
                    reference = torch.flip(reference, dims=(-2,))
                    masks = torch.flip(masks, dims=(-2,))
                reference_01 = (reference + 1.0) / 2.0
                gain = rng.uniform(0.90, 1.10)
                bias = rng.uniform(-0.05, 0.05)
                reference = (reference_01 * gain + bias).clamp(0, 1).mul(2).sub(1)
                # Candidate order is randomized independently for each example;
                # scores cannot learn a fixed method position or identity.
                for batch_index in range(len(indices)):
                    permutation = torch.randperm(len(candidates))
                    masks[batch_index] = masks[batch_index, permutation]
                    borda[batch_index] = borda[batch_index, permutation]
                    pairwise[batch_index] = pairwise[batch_index][permutation][:, permutation]
            reference = reference.to(resolved_device)
            masks = masks.to(resolved_device)
            modality = modality.to(resolved_device)
            target = target.to(resolved_device)
            borda = borda.to(resolved_device)
            pairwise = pairwise.to(resolved_device)
            with torch.set_grad_enabled(optimize):
                scores = model(reference, masks, modality, target)
                value = _loss(scores, pairwise, borda)
                if optimize:
                    optimizer.zero_grad(set_to_none=True)
                    value.backward()
                    optimizer.step()
            total += float(value.detach().cpu()) * len(indices)
        return total / len(rows)

    best_loss = float("inf")
    best_state = None
    for epoch_index in range(1, epochs + 1):
        train_loss = epoch(tensors["train"], True)
        validation_loss = epoch(tensors["validation"], False)
        history.append(
            {"epoch": epoch_index, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    assert best_state is not None
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": best_state,
        "model_config": model_config,
        "candidate_models": candidates,
        "modality_vocabulary": modalities,
        "target_vocabulary": targets,
        "image_size": image_size,
        "batch_size": batch_size,
        "split_protocol": "source-image-grouped training/validation split",
        "training_image_ids": sorted({example.image_id for example in training}),
        "validation_image_ids": sorted({example.image_id for example in validation}),
        "best_validation_loss": best_loss,
        "history": history,
        "segmentation_ground_truth_used": False,
    }
    torch.save(payload, checkpoint_path)
    return {key: value for key, value in payload.items() if key != "model_state"}


def predict_student(
    examples: Sequence[StudentExample],
    checkpoint_path: str | Path,
    device: str = "auto",
) -> dict[str, list[str]]:
    torch, _, _ = _torch()
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    saved = torch.load(checkpoint_path, map_location=resolved_device)
    Model = build_model_class()
    model = Model(**saved["model_config"]).to(resolved_device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    candidates = list(saved["candidate_models"])
    result = {}
    with torch.no_grad():
        for example in examples:
            item = _load_tensor_example(
                example,
                candidates,
                saved["modality_vocabulary"],
                saved["target_vocabulary"],
                int(saved["image_size"]),
            )
            scores = model(
                item["reference"][None].to(resolved_device),
                item["mask_maps"][None].to(resolved_device),
                item["modality"][None].to(resolved_device),
                item["target"][None].to(resolved_device),
            )[0]
            order = scores.argsort(descending=True).cpu().tolist()
            result[example.patch_id] = [candidates[index] for index in order]
    return result


def save_examples(path: str | Path, examples: Sequence[StudentExample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.__dict__, sort_keys=True) + "\n")
