import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from vscell import cli


MODELS = ("cellpose", "stardist", "cellsam", "cellpose_sam")


def test_guided_default_prompt_explicitly_explains_enter(monkeypatch):
    prompts = []

    def accept_default(prompt):
        prompts.append(prompt)
        return ""

    monkeypatch.setattr("builtins.input", accept_default)
    assert cli.wf.prompt_text("Imaging modality", "ssDNA") == "ssDNA"
    assert "press Enter to accept" in prompts[0]


def test_parallel_task_choices_do_not_label_default_as_recommended(
    monkeypatch, capsys
):
    prompts = []

    def accept_default(prompt):
        prompts.append(prompt)
        return ""

    monkeypatch.setattr("builtins.input", accept_default)
    selected = cli.wf.prompt_choice(
        "Segmentation target",
        (("nuclear", "segment nuclei"), ("whole_cell", "segment complete cells")),
        default="nuclear",
        mark_default_recommended=False,
    )
    assert selected == "nuclear"
    captured = capsys.readouterr()
    assert "recommended" not in captured.err
    assert "Enter option number (1-2)" in prompts[0]
    assert "default 1" in prompts[0]


def test_nuclear_modality_menu_only_lists_compatible_presets(monkeypatch, capsys):
    answers = iter(["3"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    modality, description = cli._prompt_guided_modality("nuclear")

    assert (modality, description) == ("H&E", None)
    displayed = capsys.readouterr().err
    assert "DAPI" in displayed
    assert "ssDNA" in displayed
    assert "H&E" in displayed
    assert "mIF" not in displayed
    assert "recommended" not in displayed


def test_whole_cell_modality_menu_accepts_custom_description(monkeypatch):
    answers = iter(
        [
            "6",
            "DAPI + membrane fluorescence",
            "Blue nuclei with a green membrane channel outlining complete cells.",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    modality, description = cli._prompt_guided_modality("whole_cell")

    assert modality == "DAPI + membrane fluorescence"
    assert description == (
        "Blue nuclei with a green membrane channel outlining complete cells."
    )


def test_guided_key_rejects_literal_surrounding_quotes(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", '"sk-test"')
    with pytest.raises(RuntimeError, match="surrounding quote"):
        with cli.wf.provider_key("gpt-5.5", interactive=False):
            pass


def test_guided_hidden_key_is_temporary(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(cli.wf.getpass, "getpass", lambda _prompt: "temporary-test-key")

    with cli.wf.provider_key("gpt-5.5", interactive=True) as name:
        assert name == "OPENAI_API_KEY"
        assert cli.wf.os.environ[name] == "temporary-test-key"

    assert "OPENAI_API_KEY" not in cli.wf.os.environ


def test_guided_hidden_claude_key_is_temporary(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli.wf.getpass, "getpass", lambda _prompt: "temporary-test-key")

    with cli.wf.provider_key("claude-fable-5-1", interactive=True) as name:
        assert name == "ANTHROPIC_API_KEY"
        assert cli.wf.os.environ[name] == "temporary-test-key"

    assert "ANTHROPIC_API_KEY" not in cli.wf.os.environ


def test_guided_existing_key_is_reused_without_prompt_and_preserved(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "existing-test-key")

    def unexpected_prompt(_prompt):
        raise AssertionError("an existing environment key must not trigger a prompt")

    monkeypatch.setattr(cli.wf.getpass, "getpass", unexpected_prompt)
    with cli.wf.provider_key("gpt-5.5", interactive=True) as name:
        assert name == "OPENAI_API_KEY"

    assert cli.wf.os.environ["OPENAI_API_KEY"] == "existing-test-key"


def _inputs(tmp_path):
    input_dir = tmp_path / "input"
    run_root = tmp_path / "run"
    candidates = run_root / "candidates"
    input_dir.mkdir()
    image = np.zeros((16, 16), dtype=np.uint8)
    image[4:12, 4:12] = 180
    tifffile.imwrite(input_dir / "example.tif", image)
    for model in MODELS:
        destination = candidates / model
        destination.mkdir(parents=True)
        mask = np.zeros(image.shape, dtype=np.uint16)
        mask[4:12, 4:12] = 1
        tifffile.imwrite(destination / "example_masks.tif", mask)
    return input_dir, run_root


def _modern_inputs(tmp_path):
    input_dir = tmp_path / "data"
    run_root = tmp_path / "run"
    candidates = tmp_path / "segmentation_results"
    input_dir.mkdir()
    image = np.zeros((16, 16), dtype=np.uint8)
    image[4:12, 4:12] = 180
    tifffile.imwrite(input_dir / "example.tif", image)
    for model in MODELS:
        destination = candidates / model
        destination.mkdir(parents=True)
        mask = np.zeros(image.shape, dtype=np.uint16)
        mask[4:12, 4:12] = 1
        tifffile.imwrite(destination / "example_masks.tif", mask)
    return input_dir, run_root


def _multi_image_inputs(tmp_path):
    input_dir = tmp_path / "input"
    run_root = tmp_path / "run"
    candidates = run_root / "candidates"
    input_dir.mkdir()
    for image_index, image_id in enumerate(("example_a", "example_b"), start=1):
        image = np.zeros((16, 16), dtype=np.uint8)
        image[3:13, 3:13] = 80 * image_index
        tifffile.imwrite(input_dir / f"{image_id}.tif", image)
        for model in MODELS:
            destination = candidates / model
            destination.mkdir(parents=True, exist_ok=True)
            mask = np.zeros(image.shape, dtype=np.uint16)
            mask[3:13, 3:13] = 1
            tifffile.imwrite(destination / f"{image_id}_masks.tif", mask)
    return input_dir, run_root


def test_guided_auto_detects_complete_existing_model_directories(tmp_path):
    input_dir = tmp_path / "data"
    candidate_root = tmp_path / "candidate_masks"
    input_dir.mkdir()
    image = np.zeros((16, 16), dtype=np.uint8)
    image[4:12, 4:12] = 180
    tifffile.imwrite(input_dir / "example.tif", image)
    for model in ("method_alpha", "method_beta", "method_gamma"):
        destination = candidate_root / model
        destination.mkdir(parents=True)
        mask = np.zeros(image.shape, dtype=np.uint16)
        mask[4:12, 4:12] = 1
        tifffile.imwrite(destination / "example_masks.tif", mask)

    args = cli.build_parser().parse_args(
        [
            "run",
            "--input",
            str(input_dir),
            "--output",
            str(tmp_path / "run"),
            "--candidates",
            str(candidate_root),
            "--target",
            "nuclear",
            "--modality",
            "DAPI",
            "--candidate-source",
            "existing",
            "--execution",
            "local",
            "--non-interactive",
        ]
    )
    config, _plan = cli._new_guided_config(args, interactive=False)

    assert config["models"] == ["method_alpha", "method_beta", "method_gamma"]
    assert config["guided_profile"] == "standard"
    assert config["teacher_fraction"] == 1.0


def test_guided_standard_profile_hides_expert_prompts(tmp_path, monkeypatch, capsys):
    input_dir, run_root = _modern_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.shutil, "which", lambda _name: None)
    answers = iter(("", "", "", ""))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    args = cli.build_parser().parse_args(
        [
            "run",
            "--input",
            str(input_dir),
            "--output",
            str(run_root),
            "--target",
            "nuclear",
            "--modality",
            "ssDNA",
            "--candidate-source",
            "existing",
        ]
    )

    config, _plan = cli._new_guided_config(args, interactive=True)

    assert config["models"] == list(MODELS)
    assert config["guided_profile"] == "standard"
    assert config["teacher_fraction"] == 1.0
    assert config["vlm_model"] == "gpt-5.5"
    assert config["workers"] == 32
    assert config["student_mode"] == "none"
    assert config["execution"] == "local"
    displayed = capsys.readouterr().err
    assert "Detected complete candidate-mask folders" in displayed
    assert "Fraction ranked by the hosted VLM" not in displayed
    assert "Concurrent VLM workers" not in displayed
    assert "Hosted VLM" in displayed


def test_guided_standard_profile_can_select_claude(tmp_path, monkeypatch):
    input_dir, run_root = _modern_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.shutil, "which", lambda _name: None)
    answers = iter(("", "", "3", ""))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    args = cli.build_parser().parse_args(
        [
            "run",
            "--input",
            str(input_dir),
            "--output",
            str(run_root),
            "--target",
            "nuclear",
            "--modality",
            "ssDNA",
            "--candidate-source",
            "existing",
        ]
    )

    config, _plan = cli._new_guided_config(args, interactive=True)

    assert config["vlm_model"] == "claude-fable-5-1"
    assert cli.wf.provider_key_name(config["vlm_model"]) == "ANTHROPIC_API_KEY"


def _fake_rank_patch(patch_id, reference, panels, modality, target, **kwargs):
    ranking = list(panels)
    scores = {model: len(ranking) - index for index, model in enumerate(ranking)}
    calls = [SimpleNamespace(ranking=ranking, repetition=index) for index in range(3)]
    return ranking, scores, calls


def _run_args(input_dir, run_root, *extra):
    return [
        "run",
        "--input",
        str(input_dir),
        "--output",
        str(run_root),
        "--target",
        "nuclear",
        "--modality",
        "ssDNA",
        "--candidate-source",
        "existing",
        "--teacher-fraction",
        "1",
        "--vlm-model",
        "gpt-5.5",
        "--workers",
        "2",
        "--base-rows",
        "1",
        "--base-cols",
        "1",
        "--non-interactive",
        "--yes",
        *extra,
    ]


def test_guided_local_run_continues_through_final_mask(
    tmp_path, monkeypatch, capsys
):
    input_dir, run_root = _inputs(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-test")
    monkeypatch.setattr(cli, "rank_patch", _fake_rank_patch)

    assert cli.main(_run_args(input_dir, run_root, "--execution", "local")) == 0

    final_mask = run_root / "workspace/final_masks/example_vscell.tif"
    state = json.loads((run_root / "workflow_state.json").read_text())
    state_text = (run_root / "workflow_state.json").read_text()
    assert final_mask.is_file()
    assert "dummy-key-for-test" not in state_text
    assert "OPENAI_API_KEY" not in state_text
    assert state["status"] == "complete"
    assert state["stages"]["rank"]["completed"] == 1
    assert state["stages"]["finalize"]["final_cells"] == {"example": 1}
    captured = capsys.readouterr()
    assert "VSCell guided-run summary" in captured.err
    assert "OPENAI_API_KEY detected in this process environment" in captured.err
    assert "dummy-key-for-test" not in captured.err
    assert "Workflow complete" in captured.err
    assert "Final stitched mask" in captured.err


def test_guided_run_discovers_recommended_segmentation_results_layout(
    tmp_path, monkeypatch
):
    input_dir, run_root = _modern_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-test")
    monkeypatch.setattr(cli, "rank_patch", _fake_rank_patch)

    assert cli.main(_run_args(input_dir, run_root, "--execution", "local")) == 0
    assert (run_root / "workspace/final_masks/example_vscell.tif").is_file()


def test_guided_preflight_stops_before_state_when_candidate_is_missing(
    tmp_path, capsys
):
    input_dir, run_root = _inputs(tmp_path)
    (run_root / "candidates/cellsam/example_masks.tif").unlink()

    assert cli.main(_run_args(input_dir, run_root, "--execution", "local")) == 1

    assert not (run_root / "workflow_state.json").exists()
    captured = capsys.readouterr()
    assert "existing candidate masks are incomplete" in captured.err
    assert "No hosted job or workflow state was created" in captured.err


def test_student_preflight_rejects_too_few_teacher_patches(tmp_path, capsys):
    input_dir, run_root = _inputs(tmp_path)

    assert cli.main(
        _run_args(
            input_dir,
            run_root,
            "--execution",
            "local",
            "--teacher-fraction",
            "0.2",
            "--base-rows",
            "4",
            "--base-cols",
            "4",
        )
    ) == 1

    assert not (run_root / "workflow_state.json").exists()
    captured = capsys.readouterr()
    assert "at least four hosted teacher patches" in captured.err
    assert "No hosted job or workflow state was created" in captured.err


def test_student_preflight_rejects_training_from_one_source_image(
    tmp_path, capsys
):
    input_dir, run_root = _inputs(tmp_path)

    assert cli.main(
        _run_args(
            input_dir,
            run_root,
            "--execution",
            "local",
            "--teacher-fraction",
            "0.5",
            "--base-rows",
            "4",
            "--base-cols",
            "4",
        )
    ) == 1

    assert not (run_root / "workflow_state.json").exists()
    captured = capsys.readouterr()
    assert "at least two source images" in captured.err
    assert "No hosted job or workflow state was created" in captured.err


def test_guided_slurm_detach_then_resume_continues_to_finalize(
    tmp_path, monkeypatch, capsys
):
    input_dir, run_root = _inputs(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-test")
    monkeypatch.setattr(cli.wf, "require_slurm", lambda: None)
    monkeypatch.setattr(cli.wf, "submit_slurm", lambda script: "12345")

    first = _run_args(
        input_dir,
        run_root,
        "--execution",
        "slurm",
        "--account",
        "example-account",
        "--cpu-partition",
        "example-cpu",
        "--environment",
        str(tmp_path / "venv"),
        "--detach",
    )
    assert cli.main(first) == 0
    state_path = run_root / "workflow_state.json"
    detached_state = json.loads(state_path.read_text())
    assert detached_state["stages"]["rank"]["status"] == "submitted"
    assert detached_state["stages"]["rank"]["job_id"] == "12345"
    assert not (run_root / "workspace/final_masks/example_vscell.tif").exists()

    def fake_wait(job_ids, *, poll_interval, poll_callback):
        manifest = json.loads((run_root / "workspace/run_manifest.json").read_text())
        rankings = run_root / "workspace/teacher_rankings.jsonl"
        rankings.write_text(
            "".join(
                json.dumps(
                    {
                        "patch_id": row["patch_id"],
                        "source": "test-slurm",
                        "ranking": list(MODELS),
                    }
                )
                + "\n"
                for row in manifest["patches"]
            )
        )
        poll_callback({"12345": "RUNNING"})
        return {"12345": "COMPLETED"}

    monkeypatch.setattr(cli.wf, "wait_for_slurm_jobs", fake_wait)
    assert cli.main(["run", "--output", str(run_root), "--resume"]) == 0

    resumed_state = json.loads(state_path.read_text())
    assert resumed_state["status"] == "complete"
    assert resumed_state["stages"]["rank"]["status"] == "complete"
    assert (run_root / "workspace/final_masks/example_vscell.tif").is_file()
    captured = capsys.readouterr()
    assert "Continue later with:" in captured.err
    assert "Resuming workflow" in captured.err
    assert "Hosted-VLM progress:" in captured.err
    assert "Workflow complete" in captured.err


def test_guided_slurm_student_gpu_job_continues_to_combined_finalization(
    tmp_path, monkeypatch, capsys
):
    input_dir, run_root = _multi_image_inputs(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-test")
    submitted_scripts = []
    job_ids = iter(("111", "222"))

    def fake_submit(script):
        submitted_scripts.append(script)
        return next(job_ids)

    monkeypatch.setattr(cli.wf, "submit_slurm", fake_submit)

    first = _run_args(
        input_dir,
        run_root,
        "--teacher-fraction",
        "0.5",
        "--base-rows",
        "2",
        "--base-cols",
        "2",
        "--execution",
        "slurm",
        "--account",
        "example-account",
        "--cpu-partition",
        "example-cpu",
        "--gpu-partition",
        "example-gpu",
        "--environment",
        str(tmp_path / "venv"),
        "--student-mode",
        "train",
        "--student-execution",
        "slurm",
        "--detach",
    )
    assert cli.main(first) == 0

    def fake_wait(job_ids_to_wait, *, poll_interval, poll_callback):
        manifest_path = run_root / "workspace/run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if job_ids_to_wait == ["111"]:
            teacher, _student = cli.select_teacher_patches(
                manifest, 0.5, 120912
            )
            (run_root / "workspace/teacher_rankings.jsonl").write_text(
                "".join(
                    json.dumps(
                        {
                            "patch_id": patch_id,
                            "source": "test-slurm",
                            "ranking": list(MODELS),
                        }
                    )
                    + "\n"
                    for patch_id in teacher
                )
            )
        else:
            assert job_ids_to_wait == ["222"]
            (run_root / "workspace/student.pt").write_bytes(b"test-checkpoint")
            (run_root / "workspace/all_rankings.jsonl").write_text(
                "".join(
                    json.dumps(
                        {
                            "patch_id": row["patch_id"],
                            "source": "student-job-test",
                            "ranking": list(MODELS),
                        }
                    )
                    + "\n"
                    for row in manifest["patches"]
                )
            )
        poll_callback({job_ids_to_wait[0]: "RUNNING"})
        return {job_ids_to_wait[0]: "COMPLETED"}

    monkeypatch.setattr(cli.wf, "wait_for_slurm_jobs", fake_wait)
    assert cli.main(["run", "--output", str(run_root), "--resume"]) == 0

    state = json.loads((run_root / "workflow_state.json").read_text())
    assert state["stages"]["rank"]["status"] == "complete"
    assert state["stages"]["student_train"]["status"] == "complete"
    assert state["stages"]["student_predict"]["status"] == "complete"
    assert state["stages"]["finalize"]["status"] == "complete"
    assert len(submitted_scripts) == 2
    student_script = Path(submitted_scripts[1]).read_text()
    assert "#SBATCH --gres=gpu:1" in student_script
    assert "train-student" in student_script
    assert "predict-student" in student_script
    assert len(list((run_root / "workspace/final_masks").glob("*_vscell.tif"))) == 2
    captured = capsys.readouterr()
    assert "Submitted student-training job 222" in captured.err
    assert "Student training complete" in captured.err
    assert "Student ranking complete" in captured.err
