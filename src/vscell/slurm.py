"""Render portable SLURM scripts without site-specific accounts or partitions."""

from __future__ import annotations

import shlex
from pathlib import Path


DEFAULT_ENV_NAMES = {
    "cellpose": "vscell-cellpose",
    "cellpose_sam": "vscell-cpsam-cellsam",
    "cellsam": "vscell-cpsam-cellsam",
    "stardist": "vscell-stardist",
}


def render_backend_scripts(
    output_dir: str | Path,
    *,
    input_path: str,
    candidate_root: str,
    target: str,
    modality: str,
    account: str = "<ACCOUNT>",
    gpu_partition: str = "<GPU_PARTITION>",
    conda_init: str = "<CONDA_ROOT>/etc/profile.d/conda.sh",
    environments: dict[str, str] | None = None,
) -> list[Path]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    input_path = str(Path(input_path).resolve())
    candidate_root = str(Path(candidate_root).resolve())
    environments = {**DEFAULT_ENV_NAMES, **(environments or {})}
    paths = []
    for backend in ("cellpose", "cellpose_sam", "cellsam", "stardist"):
        body = f"""#!/bin/bash
#SBATCH --job-name=vscell_{backend}
#SBATCH --account={account}
#SBATCH --partition={gpu_partition}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output={logs_dir}/%x_%j.out
#SBATCH --error={logs_dir}/%x_%j.err

set -euo pipefail
source {conda_init}
conda activate {environments[backend]}

vscell backend \\
  --backend {backend} \\
  --input {input_path} \\
  --output {candidate_root} \\
  --target {target} \\
  --modality {modality} \\
  --device auto
"""
        path = output_dir / f"run_{backend}.slurm"
        path.write_text(body)
        paths.append(path)
    return paths


def render_rank_teacher_script(
    output_dir: str | Path,
    *,
    manifest_path: str,
    rankings_path: str,
    teacher_fraction: float,
    vlm_model: str,
    workers: int,
    environment: str,
    account: str = "<ACCOUNT>",
    cpu_partition: str = "<CPU_PARTITION>",
    time_limit: str = "04:00:00",
    memory: str = "8G",
    seed: int = 120912,
    modality_description: str | None = None,
) -> Path:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if not 0 <= teacher_fraction <= 1:
        raise ValueError("teacher_fraction must be in [0, 1]")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = str(Path(manifest_path).resolve())
    rankings_path = str(Path(rankings_path).resolve())
    executable = str(Path(environment).resolve() / "bin" / "vscell")
    required_key = (
        "ANTHROPIC_API_KEY" if vlm_model.startswith("claude-") else "OPENAI_API_KEY"
    )
    command = [
        executable,
        "rank-teacher",
        "--manifest",
        manifest_path,
        "--output",
        rankings_path,
        "--teacher-fraction",
        str(teacher_fraction),
        "--vlm-model",
        vlm_model,
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--resume",
        "--yes",
    ]
    if modality_description:
        command.extend(["--modality-description", modality_description])
    rendered_command = " \\\n  ".join(shlex.quote(item) for item in command)
    body = f"""#!/bin/bash
#SBATCH --job-name=vscell_rank
#SBATCH --account={account}
#SBATCH --partition={cpu_partition}
#SBATCH --cpus-per-task={workers}
#SBATCH --mem={memory}
#SBATCH --time={time_limit}
#SBATCH --export=ALL
#SBATCH --output={logs_dir}/%x_%j.out
#SBATCH --error={logs_dir}/%x_%j.err

set -euo pipefail
if [[ -z "${{{required_key}:-}}" ]]; then
  echo "{required_key} is unset; export it before sbatch submission" >&2
  exit 2
fi

{rendered_command}
"""
    path = output_dir / "run_rank_teacher.slurm"
    path.write_text(body)
    return path


def render_train_student_script(
    output_dir: str | Path,
    *,
    manifest_path: str,
    teacher_rankings_path: str,
    teacher_fraction: float,
    checkpoint_path: str,
    all_rankings_path: str,
    epochs: int,
    device: str,
    environment: str,
    cpus: int,
    account: str = "<ACCOUNT>",
    gpu_partition: str = "<GPU_PARTITION>",
    time_limit: str = "04:00:00",
    memory: str = "24G",
    seed: int = 120912,
) -> Path:
    """Render a secret-free one-GPU Slurm job for student distillation."""
    if cpus < 1:
        raise ValueError("cpus must be at least 1")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if not 0 < teacher_fraction < 1:
        raise ValueError("student training requires teacher_fraction in (0, 1)")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    executable = str(Path(environment).resolve() / "bin" / "vscell")
    command = [
        executable,
        "train-student",
        "--manifest",
        str(Path(manifest_path).resolve()),
        "--teacher-rankings",
        str(Path(teacher_rankings_path).resolve()),
        "--teacher-fraction",
        str(teacher_fraction),
        "--checkpoint",
        str(Path(checkpoint_path).resolve()),
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
    ]
    predict_command = [
        executable,
        "predict-student",
        "--manifest",
        str(Path(manifest_path).resolve()),
        "--teacher-rankings",
        str(Path(teacher_rankings_path).resolve()),
        "--teacher-fraction",
        str(teacher_fraction),
        "--checkpoint",
        str(Path(checkpoint_path).resolve()),
        "--output",
        str(Path(all_rankings_path).resolve()),
        "--device",
        device,
        "--seed",
        str(seed),
    ]
    body = f"""#!/bin/bash
#SBATCH --job-name=vscell_student
#SBATCH --account={account}
#SBATCH --partition={gpu_partition}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}
#SBATCH --time={time_limit}
#SBATCH --export=NONE
#SBATCH --output={logs_dir}/%x_%j.out
#SBATCH --error={logs_dir}/%x_%j.err

set -euo pipefail
{shlex.join(command)}
{shlex.join(predict_command)}
"""
    path = output_dir / "run_train_student.slurm"
    path.write_text(body)
    return path
