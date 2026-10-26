from pathlib import Path

from vscell.slurm import (
    render_backend_scripts,
    render_rank_teacher_script,
    render_train_student_script,
)


def test_slurm_renderer_creates_usable_absolute_log_paths(tmp_path):
    input_dir = tmp_path / "input"
    candidate_dir = tmp_path / "candidates"
    output_dir = tmp_path / "slurm"
    input_dir.mkdir()
    paths = render_backend_scripts(
        output_dir,
        input_path=str(input_dir),
        candidate_root=str(candidate_dir),
        target="nuclear",
        modality="ssDNA",
    )
    assert len(paths) == 4
    assert (output_dir / "logs").is_dir()
    for path in paths:
        text = path.read_text()
        assert "#SBATCH --export=NONE" in text
        assert f"#SBATCH --output={output_dir.resolve()}/logs/%x_%j.out" in text
        assert f"--input {input_dir.resolve()}" in text
        assert f"--output {candidate_dir.resolve()}" in text


def test_rank_slurm_renderer_uses_workers_resume_and_no_embedded_key(tmp_path):
    environment = tmp_path / "venv"
    manifest = tmp_path / "workspace" / "run_manifest.json"
    rankings = tmp_path / "workspace" / "teacher_rankings.jsonl"
    path = render_rank_teacher_script(
        tmp_path / "slurm",
        manifest_path=str(manifest),
        rankings_path=str(rankings),
        teacher_fraction=1.0,
        vlm_model="gpt-5.5",
        workers=12,
        environment=str(environment),
    )
    text = path.read_text()
    assert "#SBATCH --cpus-per-task=12" in text
    assert "#SBATCH --export=ALL" in text
    assert "OPENAI_API_KEY is unset" in text
    assert "--workers" in text
    assert "--resume" in text
    assert "--yes" in text
    assert str(manifest.resolve()) in text
    assert str(rankings.resolve()) in text
    assert "sk-" not in text
    assert "\n+" not in text


def test_student_slurm_renderer_requests_gpu_and_uses_installed_cli(tmp_path):
    environment = tmp_path / "venv"
    manifest = tmp_path / "workspace" / "run_manifest.json"
    rankings = tmp_path / "workspace" / "teacher_rankings.jsonl"
    checkpoint = tmp_path / "workspace" / "student.pt"
    all_rankings = tmp_path / "workspace" / "all_rankings.jsonl"
    path = render_train_student_script(
        tmp_path / "slurm",
        manifest_path=str(manifest),
        teacher_rankings_path=str(rankings),
        teacher_fraction=0.2,
        checkpoint_path=str(checkpoint),
        all_rankings_path=str(all_rankings),
        epochs=60,
        device="cuda",
        environment=str(environment),
        cpus=4,
        account="example-account",
        gpu_partition="example-gpu",
    )
    text = path.read_text()
    assert "#SBATCH --partition=example-gpu" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --export=NONE" in text
    assert str(environment.resolve() / "bin/vscell") in text
    assert "train-student" in text
    assert "predict-student" in text
    assert "--device cuda" in text
    assert str(checkpoint.resolve()) in text
    assert str(all_rankings.resolve()) in text
    assert "API_KEY" not in text
    assert "\n+" not in text
