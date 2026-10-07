"""Hosted VLM adapters with strict, auditable three-repetition ranking."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import HISTORICAL_COST_PER_PATCH, MODEL_IDS, Target
from .prompts import system_prompt, user_content
from .ranking import borda_aggregate, parse_ranking, shuffled_mapping


@dataclass(frozen=True)
class VLMCall:
    crop_id: str
    repetition: int
    provider: str
    requested_model: str
    response_model: str
    mapping: dict[str, str]
    ranking: list[str]
    raw_text: str
    usage: dict[str, int]


def canonical_model(value: str) -> str:
    return MODEL_IDS.get(value.lower(), value)


def provider_for(model: str) -> str:
    return "anthropic" if model.startswith("claude-") else "openai"


def estimated_cost(model: str, patch_count: int, teacher_fraction: float = 1.0) -> float | None:
    model = canonical_model(model)
    if not 0 <= teacher_fraction <= 1:
        raise ValueError("teacher_fraction must be in [0, 1]")
    rate = HISTORICAL_COST_PER_PATCH.get(model)
    if rate is None:
        return None
    return round(round(patch_count * teacher_fraction) * rate, 4)


def _openai_content(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks = []
    for part in content:
        if part["type"] == "text":
            blocks.append({"type": "input_text", "text": part["text"]})
        else:
            blocks.append(
                {
                    "type": "input_image",
                    "image_url": part["image_url"]["url"],
                }
            )
    return blocks


def _call_openai(model: str, system: str, content: list[dict[str, Any]]) -> tuple[str, str, dict[str, int]]:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is unset")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("install vscell[vlm-openai] to use OpenAI models") from error
    parameters: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user", "content": _openai_content(content)},
        ],
        "reasoning": {"effort": "medium"},
    }
    # GPT-6 Astra did not accept temperature in the validated runs.
    if model == "gpt-5.5":
        parameters["temperature"] = 1.0
    response = OpenAI(api_key=key).responses.create(**parameters)
    usage_object = getattr(response, "usage", None)
    usage = {
        "input_tokens": int(getattr(usage_object, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage_object, "output_tokens", 0) or 0),
    }
    return str(response.output_text), str(getattr(response, "model", model)), usage


def _split_data_url(value: str) -> tuple[str, str]:
    head, payload = value.split(",", 1)
    media_type = head.split(";", 1)[0].split(":", 1)[1]
    base64.b64decode(payload, validate=True)
    return media_type, payload


def _anthropic_content(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks = []
    for part in content:
        if part["type"] == "text":
            blocks.append({"type": "text", "text": part["text"]})
        else:
            media_type, payload = _split_data_url(part["image_url"]["url"])
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": payload,
                    },
                }
            )
    return blocks


def _call_anthropic(model: str, system: str, content: list[dict[str, Any]]) -> tuple[str, str, dict[str, int]]:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is unset")
    try:
        import anthropic
    except ImportError as error:
        raise RuntimeError("install vscell[vlm-anthropic] to use Claude") from error
    response = anthropic.Anthropic(api_key=key).messages.create(
        model=model,
        system=system,
        messages=[{"role": "user", "content": _anthropic_content(content)}],
        temperature=1.0,
        max_tokens=8192,
    )
    text = "\n".join(
        str(block.text) for block in response.content if getattr(block, "type", "") == "text"
    )
    usage = {
        "input_tokens": int(getattr(response.usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(response.usage, "output_tokens", 0) or 0),
    }
    return text, str(getattr(response, "model", model)), usage


def rank_patch(
    crop_id: str,
    reference_path: str | Path,
    candidate_paths: dict[str, str | Path],
    modality: str,
    target: Target,
    model: str = "gpt-5.5",
    custom_modality_description: str | None = None,
    repetitions: int = 3,
) -> tuple[list[str], dict[str, int], list[VLMCall]]:
    if repetitions != 3:
        raise ValueError("validated VSCell inference uses exactly three repetitions")
    model = canonical_model(model)
    candidates = list(candidate_paths)
    system = system_prompt(len(candidates), target)
    calls: list[VLMCall] = []
    for repetition in range(repetitions):
        mapping = shuffled_mapping(crop_id, repetition, candidates)
        content = user_content(
            reference_path,
            candidate_paths,
            mapping,
            modality,
            target,
            custom_modality_description,
        )
        if provider_for(model) == "anthropic":
            raw, response_model, usage = _call_anthropic(model, system, content)
        else:
            raw, response_model, usage = _call_openai(model, system, content)
        ranking = parse_ranking(raw, mapping)
        calls.append(
            VLMCall(
                crop_id=crop_id,
                repetition=repetition,
                provider=provider_for(model),
                requested_model=model,
                response_model=response_model,
                mapping=mapping,
                ranking=ranking,
                raw_text=raw,
                usage=usage,
            )
        )
    final_order, scores = borda_aggregate(
        [call.ranking for call in calls], candidates
    )
    return final_order, scores, calls


def append_calls(path: str | Path, calls: list[VLMCall]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for call in calls:
            handle.write(json.dumps(asdict(call), sort_keys=True) + "\n")
