import argparse
import json
import threading
import time
from types import SimpleNamespace

from vscell import cli


def _manifest(count=6):
    models = ["a", "b", "c"]
    return {
        "models": models,
        "modality": "ssDNA",
        "target": "nuclear",
        "patches": [
            {
                "patch_id": f"patch-{index}",
                "image_id": "image",
                "reference_panel": f"ref-{index}.png",
                "candidate_panels": {
                    model: f"{model}-{index}.png" for model in models
                },
            }
            for index in range(count)
        ],
    }


def _args(output, workers=3, resume=False):
    return argparse.Namespace(
        manifest="manifest.json",
        output=str(output),
        teacher_fraction=1.0,
        vlm_model="gpt-5.5",
        modality_description=None,
        seed=120912,
        workers=workers,
        resume=resume,
        yes=True,
    )


def test_rank_teacher_workers_are_concurrent_and_jsonl_is_valid(tmp_path, monkeypatch):
    manifest = _manifest()
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def fake_rank_patch(patch_id, reference, panels, modality, target, **kwargs):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        ranking = list(panels)
        calls = [SimpleNamespace(ranking=ranking, repetition=index) for index in range(3)]
        return ranking, {model: len(ranking) - index for index, model in enumerate(ranking)}, calls

    monkeypatch.setattr(cli, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(
        cli,
        "select_teacher_patches",
        lambda loaded, fraction, seed: (
            [row["patch_id"] for row in loaded["patches"]],
            [],
        ),
    )
    monkeypatch.setattr(cli, "rank_patch", fake_rank_patch)
    output = tmp_path / "rankings.jsonl"
    assert cli._cmd_rank_teacher(_args(output)) == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 6
    assert len({row["patch_id"] for row in rows}) == 6
    assert all(len(row["calls"]) == 3 for row in rows)
    assert maximum_active == 3


def test_rank_teacher_resume_skips_completed_patch(tmp_path, monkeypatch):
    manifest = _manifest(count=3)
    output = tmp_path / "rankings.jsonl"
    output.write_text(
        json.dumps({"patch_id": "patch-0", "ranking": ["a", "b", "c"]}) + "\n"
    )
    called = []

    def fake_rank_patch(patch_id, reference, panels, modality, target, **kwargs):
        called.append(patch_id)
        ranking = list(panels)
        calls = [SimpleNamespace(ranking=ranking, repetition=index) for index in range(3)]
        return ranking, {model: len(ranking) - index for index, model in enumerate(ranking)}, calls

    monkeypatch.setattr(cli, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(
        cli,
        "select_teacher_patches",
        lambda loaded, fraction, seed: (
            [row["patch_id"] for row in loaded["patches"]],
            [],
        ),
    )
    monkeypatch.setattr(cli, "rank_patch", fake_rank_patch)
    assert cli._cmd_rank_teacher(_args(output, workers=2, resume=True)) == 0
    assert set(called) == {"patch-1", "patch-2"}
    assert len(output.read_text().splitlines()) == 3
