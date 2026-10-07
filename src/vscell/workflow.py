"""User-facing workflow helpers for guided, resumable VSCell runs."""

from __future__ import annotations

import getpass
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Iterator


STATE_VERSION = 1
FAILED_SLURM_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


def announce(message: str) -> None:
    """Write a consistent progress message without hiding it in JSON output."""
    print(f"[VSCell] {message}", file=sys.stderr, flush=True)


def prompt_text(label: str, default: str | None = None) -> str:
    suffix = (
        f" [{default}; press Enter to accept]"
        if default not in (None, "")
        else ""
    )
    while True:
        value = input(f"{label}{suffix}: ").strip()
        if value:
            return value
        if default not in (None, ""):
            return str(default)
        print("A value is required.", file=sys.stderr)


def prompt_choice(
    label: str,
    choices: Iterable[tuple[str, str]],
    *,
    default: str,
    mark_default_recommended: bool = True,
) -> str:
    options = list(choices)
    values = [value for value, _description in options]
    if default not in values:
        raise ValueError(f"default {default!r} is not one of {values}")
    print(f"{label}:", file=sys.stderr)
    for index, (value, description) in enumerate(options, start=1):
        marker = (
            " (recommended)"
            if mark_default_recommended and value == default
            else ""
        )
        print(f"  {index}) {value}{marker} — {description}", file=sys.stderr)
    default_index = values.index(default) + 1
    while True:
        answer = input(
            f"Enter option number (1-{len(options)}) "
            f"[default {default_index}; press Enter to accept]: "
        ).strip()
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return values[int(answer) - 1]
        if answer in values:
            return answer
        print(
            f"Enter a number from 1 to {len(options)}, or press Enter to use "
            f"the default ({default_index}).",
            file=sys.stderr,
        )


def prompt_int(label: str, default: int, *, minimum: int = 1) -> int:
    while True:
        raw = input(
            f"{label} [{default}; press Enter to accept]: "
        ).strip()
        try:
            value = default if not raw else int(raw)
        except ValueError:
            print("Enter a whole number.", file=sys.stderr)
            continue
        if value < minimum:
            print(f"Enter a value of at least {minimum}.", file=sys.stderr)
            continue
        return value


def prompt_fraction(label: str, default: float = 1.0) -> float:
    while True:
        raw = input(
            f"{label} [{default:g}; press Enter to accept]: "
        ).strip()
        try:
            value = default if not raw else float(raw)
        except ValueError:
            print("Enter a number between 0 and 1.", file=sys.stderr)
            continue
        if 0 <= value <= 1:
            return value
        print("Enter a number between 0 and 1.", file=sys.stderr)


def confirm(label: str) -> bool:
    return input(
        f"{label} [y/N; press Enter for N] "
    ).strip().lower() in {"y", "yes"}


def state_path(run_root: str | Path) -> Path:
    return Path(run_root).resolve() / "workflow_state.json"


def save_state(path: str | Path, state: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def load_state(path: str | Path) -> dict:
    source = Path(path)
    state = json.loads(source.read_text())
    if state.get("state_version") != STATE_VERSION:
        raise ValueError(f"unsupported workflow state version in {source}")
    return state


def resume_command(run_root: str | Path) -> str:
    return shlex.join(
        ["vscell", "run", "--output", str(Path(run_root).resolve()), "--resume"]
    )


def provider_key_name(vlm_model: str) -> str:
    return "ANTHROPIC_API_KEY" if vlm_model.startswith("claude-") else "OPENAI_API_KEY"


def _validate_key(name: str, value: str) -> None:
    if not value:
        raise RuntimeError(f"{name} is empty")
    if value[0] in {"'", '"'} or value[-1] in {"'", '"'}:
        raise RuntimeError(
            f"{name} contains a surrounding quote character; enter the raw key "
            "without single or double quotes"
        )


@contextmanager
def provider_key(vlm_model: str, *, interactive: bool) -> Iterator[str]:
    """Make a provider key available temporarily without persisting it to state."""
    name = provider_key_name(vlm_model)
    existing = os.environ.get(name)
    added = False
    if existing is None:
        if not interactive:
            raise RuntimeError(
                f"{name} is unset. Export it before running non-interactively, or "
                "run from an interactive terminal for a hidden prompt."
            )
        existing = getpass.getpass(
            f"{name} (raw key, no surrounding quotes; input hidden): "
        ).strip()
        _validate_key(name, existing)
        os.environ[name] = existing
        added = True
    else:
        _validate_key(name, existing)
    try:
        yield name
    finally:
        if added:
            os.environ.pop(name, None)


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def require_slurm() -> None:
    missing = [name for name in ("sbatch", "squeue") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            "Slurm execution was selected, but these commands are unavailable: "
            + ", ".join(missing)
        )


def submit_slurm(script: str | Path) -> str:
    require_slurm()
    result = _run_command(["sbatch", "--parsable", str(Path(script).resolve())])
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Slurm submission failed: {detail}")
    job_id = result.stdout.strip().split(";", maxsplit=1)[0]
    if not re.fullmatch(r"\d+(?:_\d+)?", job_id):
        raise RuntimeError(f"could not parse Slurm job ID from: {result.stdout!r}")
    return job_id


def queue_states(job_ids: Iterable[str]) -> dict[str, str]:
    ids = [str(job_id) for job_id in job_ids]
    if not ids:
        return {}
    result = _run_command(
        ["squeue", "--noheader", "--jobs", ",".join(ids), "--format=%i|%T"]
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"could not query Slurm queue: {detail}")
    states = {}
    for line in result.stdout.splitlines():
        if "|" not in line:
            continue
        job_id, state = line.strip().split("|", maxsplit=1)
        states[job_id] = state.upper()
    return states


def accounting_state(job_id: str) -> str:
    if shutil.which("sacct") is None:
        return "UNKNOWN"
    result = _run_command(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--jobs",
            str(job_id),
            "--format=JobIDRaw,State",
        ]
    )
    if result.returncode != 0:
        return "UNKNOWN"
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 2 and fields[0] == str(job_id):
            return fields[1].split("+", maxsplit=1)[0].upper()
    return "UNKNOWN"


def wait_for_slurm_jobs(
    job_ids: Iterable[str],
    *,
    poll_interval: float = 10.0,
    poll_callback: Callable[[dict[str, str]], None] | None = None,
) -> dict[str, str]:
    ids = [str(job_id) for job_id in job_ids]
    last: dict[str, str] | None = None
    while True:
        current = queue_states(ids)
        if current != last:
            summary = ", ".join(
                f"{job_id}={current.get(job_id, 'leaving queue')}" for job_id in ids
            )
            announce(f"Slurm status: {summary}")
            last = current
        if poll_callback is not None:
            poll_callback(current)
        if not current:
            break
        time.sleep(max(0.1, poll_interval))
    return {job_id: accounting_state(job_id) for job_id in ids}


def failed_jobs(states: dict[str, str]) -> dict[str, str]:
    return {
        job_id: state
        for job_id, state in states.items()
        if state.split(maxsplit=1)[0] in FAILED_SLURM_STATES
    }


def count_jsonl_records(path: str | Path) -> int:
    source = Path(path)
    if not source.is_file():
        return 0
    count = 0
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                continue
            count += 1
    return count
