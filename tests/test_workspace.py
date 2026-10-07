import json

import numpy as np
import tifffile

from vscell.config import GridConfig
from vscell.workspace import finalize_workspace, load_manifest, prepare_workspace


def test_cellsam_explicit_model_path_preflight(tmp_path, monkeypatch):
    from vscell.backends import ensure_cellsam_credentials

    weight = tmp_path / "cellsam_general.pt"
    weight.write_bytes(b"validation-placeholder")
    monkeypatch.setenv("VSCELL_CELLSAM_MODEL_PATH", str(weight))
    monkeypatch.delenv("DEEPCELL_ACCESS_TOKEN", raising=False)
    ensure_cellsam_credentials()


def test_cellsam_missing_explicit_model_path_is_clear(tmp_path, monkeypatch):
    from vscell.backends import ensure_cellsam_credentials

    missing = tmp_path / "missing.pt"
    monkeypatch.setenv("VSCELL_CELLSAM_MODEL_PATH", str(missing))
    monkeypatch.delenv("DEEPCELL_ACCESS_TOKEN", raising=False)
    try:
        ensure_cellsam_credentials()
    except RuntimeError as error:
        assert "does not point to a file" in str(error)
    else:
        raise AssertionError("missing explicit CellSAM weight should fail")


def test_synthetic_workspace_round_trip(tmp_path):
    input_dir = tmp_path / "input"
    candidate_root = tmp_path / "candidates"
    workspace = tmp_path / "workspace"
    input_dir.mkdir()
    image = np.zeros((16, 16), dtype=np.uint8)
    image[4:12, 4:12] = 180
    tifffile.imwrite(input_dir / "example.tif", image)
    models = ("cellpose", "stardist", "cellsam", "cellpose_sam")
    for model in models:
        path = candidate_root / model
        path.mkdir(parents=True)
        mask = np.zeros((16, 16), dtype=np.uint16)
        mask[4:12, 4:12] = 1
        tifffile.imwrite(path / "example_masks.tif", mask)
    manifest_path = prepare_workspace(
        input_dir,
        candidate_root,
        workspace,
        target="nuclear",
        modality="ssDNA",
        grid=GridConfig(base_rows=2, base_cols=2),
    )
    manifest = load_manifest(manifest_path)
    assert len(manifest["patches"]) == 4
    rankings_path = workspace / "rankings.jsonl"
    rankings_path.write_text(
        "".join(
            json.dumps(
                {
                    "patch_id": row["patch_id"],
                    "source": "test",
                    "ranking": list(models),
                }
            )
            + "\n"
            for row in manifest["patches"]
        )
    )
    audit = finalize_workspace(manifest_path, rankings_path)
    assert audit["ground_truth_used"] is False
    final_path = workspace / "final_masks" / "example_vscell.tif"
    assert final_path.is_file()
    assert int(tifffile.imread(final_path).max()) == 1


def test_prepare_accepts_external_model_full_image_masks(tmp_path):
    input_dir = tmp_path / "input"
    candidate_root = tmp_path / "candidates"
    workspace = tmp_path / "workspace"
    input_dir.mkdir()
    image = np.zeros((12, 10), dtype=np.uint8)
    image[3:9, 2:8] = 180
    tifffile.imwrite(input_dir / "image01.tif", image)

    models = ("cellpose", "stardist", "my_new_model")
    for index, model in enumerate(models, start=1):
        model_dir = candidate_root / model
        model_dir.mkdir(parents=True)
        mask = np.zeros(image.shape, dtype=np.uint16)
        mask[3:9, 2:8] = index
        tifffile.imwrite(model_dir / "image01_masks.tif", mask)

    manifest_path = prepare_workspace(
        input_dir,
        candidate_root,
        workspace,
        target="nuclear",
        modality="DAPI",
        models=models,
        grid=GridConfig(base_rows=1, base_cols=1),
    )
    manifest = load_manifest(manifest_path)

    assert manifest["models"] == list(models)
    assert set(manifest["patches"][0]["candidate_masks"]) == set(models)
    assert manifest["images"][0]["full_candidate_masks"]["my_new_model"].endswith(
        "candidates/my_new_model/image01_masks.tif"
    )
