import numpy as np
import pytest
import tifffile
from PIL import Image

pytest.importorskip("torch")

from vscell.student import StudentExample, predict_student, train_student


def test_student_smoke_train_and_predict(tmp_path):
    models = ["cellpose", "stardist", "cellsam", "cellpose_sam"]
    examples = []
    for index in range(4):
        patch_id = f"patch{index}"
        reference = np.zeros((32, 32, 3), dtype=np.uint8)
        reference[8:24, 8:24] = 180
        reference_path = tmp_path / f"{patch_id}.png"
        Image.fromarray(reference).save(reference_path)
        mask_paths = {}
        for model_index, model in enumerate(models):
            mask = np.zeros((32, 32), dtype=np.uint16)
            offset = model_index % 2
            mask[8 + offset : 24 + offset, 8:24] = 1
            path = tmp_path / f"{patch_id}_{model}.tif"
            tifffile.imwrite(path, mask)
            mask_paths[model] = str(path)
        examples.append(
            StudentExample(
                patch_id=patch_id,
                image_id=f"image{index // 2}",
                reference_path=str(reference_path),
                candidate_mask_paths=mask_paths,
                modality="ssDNA",
                target="nuclear",
                teacher_order=models,
                borda_scores={model: 4 - position for position, model in enumerate(models)},
                pairwise_votes=[
                    [0.5 if left == right else float(left < right) for right in range(4)]
                    for left in range(4)
                ],
            )
        )
    checkpoint = tmp_path / "student.pt"
    audit = train_student(
        examples,
        checkpoint,
        device="cpu",
        epochs=1,
        image_size=32,
        width=4,
        embedding_dim=16,
        batch_size=2,
    )
    assert checkpoint.is_file()
    assert audit["split_protocol"].startswith("source-image-grouped")
    predictions = predict_student(examples, checkpoint, "cpu")
    assert set(predictions) == {example.patch_id for example in examples}
    assert all(set(order) == set(models) for order in predictions.values())
