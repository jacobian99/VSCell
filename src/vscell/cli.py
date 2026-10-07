"""VSCell command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path

from . import workflow as wf
from .backends import BACKENDS, backend_environment_hint, run_backend
from .config import DEFAULT_MODELS, MODEL_IDS, EnsembleConfig, GridConfig
from .grid import GridReference, build_grid
from .ranking import pairwise_vote_probabilities
from .slurm import (
    render_backend_scripts,
    render_rank_teacher_script,
    render_train_student_script,
)
from .student import StudentExample, predict_student, train_student
from .vlm import canonical_model, estimated_cost, rank_patch
from .workspace import (
    analyze_inputs,
    finalize_workspace,
    load_manifest,
    load_rankings,
    prepare_workspace,
    select_teacher_patches,
    write_teacher_split,
)


def _models(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(result) < 3:
        raise argparse.ArgumentTypeError("provide at least three comma-separated models")
    return result


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return result


def _grid_from_args(args: argparse.Namespace) -> GridConfig:
    return GridConfig(
        base_rows=args.base_rows,
        base_cols=args.base_cols,
        lower_deadband=args.lower_deadband,
        upper_deadband=args.upper_deadband,
        padding_fraction=args.padding_fraction,
    )


def _add_grid(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-rows", type=int, default=4)
    parser.add_argument("--base-cols", type=int, default=4)
    parser.add_argument("--lower-deadband", type=float, default=0.75)
    parser.add_argument("--upper-deadband", type=float, default=1.50)
    parser.add_argument(
        "--padding-fraction",
        type=float,
        default=0.10,
        help="Historical total side-length increase; 0.10 is about 5%% per side.",
    )


def _cmd_plan(args: argparse.Namespace) -> int:
    analysis = analyze_inputs(args.input, _grid_from_args(args))
    teacher_patch_count = round(analysis["patch_count"] * args.teacher_fraction)
    analysis["teacher_fraction"] = args.teacher_fraction
    analysis["teacher_patch_count"] = teacher_patch_count
    analysis["vlm_model"] = args.vlm_model
    analysis["repetitions_per_teacher_patch"] = 3
    analysis["api_call_count"] = 3 * teacher_patch_count
    analysis["estimated_total_vlm_cost_usd"] = estimated_cost(
        args.vlm_model, analysis["patch_count"], args.teacher_fraction
    )
    analysis["cost_estimate_note"] = (
        "Historical empirical total for all three repetitions per teacher patch; "
        "not a live provider quote."
    )
    print(json.dumps(analysis, indent=2))
    return 0


def _cmd_backend(args: argparse.Namespace) -> int:
    results = run_backend(
        args.backend,
        args.input,
        args.output,
        args.target,
        args.modality,
        args.device,
        args.overwrite,
    )
    print(
        json.dumps(
            {
                "backend": args.backend,
                "environment": backend_environment_hint(args.backend),
                "processed_or_reused": len(results),
                "masks": [str(item.mask) for item in results],
            },
            indent=2,
        )
    )
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    path = prepare_workspace(
        args.input,
        args.candidates,
        args.output,
        target=args.target,
        modality=args.modality,
        models=args.models,
        grid=_grid_from_args(args),
    )
    print(path)
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    path = write_teacher_split(args.manifest, args.teacher_fraction, args.seed)
    print(path)
    return 0


def _rank_teacher_record(
    manifest: dict,
    row: dict,
    patch_id: str,
    vlm_model: str,
    modality_description: str | None,
) -> dict:
    candidates = list(manifest["models"])
    candidate_panels = {
        model: row["candidate_panels"][model] for model in candidates
    }
    order, scores, calls = rank_patch(
        patch_id,
        row["reference_panel"],
        candidate_panels,
        manifest["modality"],
        manifest["target"],
        model=vlm_model,
        custom_modality_description=modality_description,
    )
    repetitions = [call.ranking for call in calls]
    return {
        "patch_id": patch_id,
        "image_id": row["image_id"],
        "source": "vlm_teacher",
        "model": vlm_model,
        "candidate_order": candidates,
        "ranking": order,
        "borda_scores": scores,
        "repetitions": repetitions,
        "pairwise_votes": pairwise_vote_probabilities(repetitions, candidates),
        "calls": [call.__dict__ for call in calls],
    }


def _cmd_rank_teacher(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    teacher, _ = select_teacher_patches(manifest, args.teacher_fraction, args.seed)
    projected = estimated_cost(args.vlm_model, len(manifest["patches"]), args.teacher_fraction)
    preflight = (
        f"teacher_patches={len(teacher)} repetitions=3 api_calls={len(teacher) * 3} "
        f"workers={args.workers} "
        "historical_estimated_total_cost_usd="
        f"{projected if projected is not None else 'unknown'} "
        "(covers_all_3_repetitions=true)"
    )
    print(preflight, file=sys.stderr)
    if teacher and not args.yes:
        if not sys.stdin.isatty():
            raise RuntimeError(
                "paid VLM inference requires explicit confirmation; inspect the "
                "estimate above and rerun with --yes"
            )
        answer = input(
            "Proceed with hosted VLM inference? [y/N; press Enter for N] "
        ).strip().lower()
        if answer not in ("y", "yes"):
            print("cancelled", file=sys.stderr)
            return 2
    patch_rows = {row["patch_id"]: row for row in manifest["patches"]}
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output}; use --resume")
    done = set()
    if output.exists():
        done = set(load_rankings(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = [patch_id for patch_id in teacher if patch_id not in done]
    completed = sum(patch_id in done for patch_id in teacher)
    if completed:
        print(
            f"resuming with {completed}/{len(teacher)} teacher patches complete",
            file=sys.stderr,
        )
    failures: list[tuple[str, Exception]] = []
    with output.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _rank_teacher_record,
                    manifest,
                    patch_rows[patch_id],
                    patch_id,
                    args.vlm_model,
                    args.modality_description,
                ): patch_id
                for patch_id in pending
            }
            cancel_requested = False
            for future in as_completed(futures):
                patch_id = futures[future]
                try:
                    record = future.result()
                except CancelledError:
                    continue
                except Exception as error:
                    failures.append((patch_id, error))
                    print(f"[failed] {patch_id}: {error}", file=sys.stderr)
                    if not cancel_requested:
                        for other in futures:
                            if other is not future:
                                other.cancel()
                        cancel_requested = True
                    continue
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                completed += 1
                print(f"[{completed}/{len(teacher)}] {patch_id}", file=sys.stderr)
    if failures:
        patch_id, error = failures[0]
        raise RuntimeError(
            f"teacher ranking failed for {len(failures)} active patch(es); "
            f"first failure was {patch_id}: {error}; rerun with --resume"
        ) from error
    print(
        f"teacher ranking complete: {completed}/{len(teacher)} patches -> {output}",
        file=sys.stderr,
    )
    return 0


def _student_examples(
    manifest: dict,
    ranking_rows: dict[str, dict],
    patch_ids: list[str],
    require_teacher: bool,
) -> list[StudentExample]:
    patches = {row["patch_id"]: row for row in manifest["patches"]}
    candidates = list(manifest["models"])
    examples = []
    for patch_id in patch_ids:
        row = patches[patch_id]
        teacher = ranking_rows.get(patch_id)
        if teacher is None and require_teacher:
            raise ValueError(f"{patch_id}: teacher ranking missing")
        order = teacher["ranking"] if teacher else candidates
        scores = teacher.get("borda_scores") if teacher else {
            model: len(candidates) - index for index, model in enumerate(candidates)
        }
        pairwise = teacher.get("pairwise_votes") if teacher else [
            [0.5 for _ in candidates] for _ in candidates
        ]
        examples.append(
            StudentExample(
                patch_id=patch_id,
                image_id=row["image_id"],
                reference_path=row["reference_panel"],
                candidate_mask_paths=row["candidate_masks"],
                modality=manifest["modality"],
                target=manifest["target"],
                teacher_order=order,
                borda_scores=scores,
                pairwise_votes=pairwise,
            )
        )
    return examples


def _cmd_train_student(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    rankings = load_rankings(args.teacher_rankings)
    teacher, _ = select_teacher_patches(manifest, args.teacher_fraction, args.seed)
    examples = _student_examples(manifest, rankings, teacher, True)
    audit = train_student(
        examples,
        args.checkpoint,
        device=args.device,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(json.dumps(audit, indent=2, default=str))
    return 0


def _cmd_predict_student(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    teacher_rankings = load_rankings(args.teacher_rankings)
    teacher, student = select_teacher_patches(manifest, args.teacher_fraction, args.seed)
    examples = _student_examples(manifest, {}, student, False)
    predictions = predict_student(examples, args.checkpoint, args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [teacher_rankings[patch_id] for patch_id in teacher]
    rows.extend(
        {"patch_id": patch_id, "source": "student", "ranking": ranking}
        for patch_id, ranking in sorted(predictions.items())
    )
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(output)
    return 0


def _cmd_finalize(args: argparse.Namespace) -> int:
    config = EnsembleConfig(minimum_area=args.minimum_area)
    audit = finalize_workspace(args.manifest, args.rankings, ensemble=config)
    print(json.dumps({"status": audit["status"], "images": audit["images"]}, indent=2))
    return 0


def _cmd_slurm(args: argparse.Namespace) -> int:
    paths = render_backend_scripts(
        args.output,
        input_path=args.input,
        candidate_root=args.candidates,
        target=args.target,
        modality=args.modality,
        account=args.account,
        gpu_partition=args.gpu_partition,
        conda_init=args.conda_init,
    )
    wf.announce(
        f"Created {len(paths)} backend Slurm scripts; no jobs were submitted. "
        "Use `vscell run` for guided submission and monitoring."
    )
    print("\n".join(str(path) for path in paths))
    return 0


def _cmd_rank_slurm(args: argparse.Namespace) -> int:
    path = render_rank_teacher_script(
        args.output,
        manifest_path=args.manifest,
        rankings_path=args.rankings,
        teacher_fraction=args.teacher_fraction,
        vlm_model=args.vlm_model,
        workers=args.workers,
        environment=args.environment,
        account=args.account,
        cpu_partition=args.cpu_partition,
        time_limit=args.time,
        memory=args.memory,
        seed=args.seed,
        modality_description=args.modality_description,
    )
    wf.announce(
        f"Created ranking script with model={args.vlm_model}, "
        f"workers={args.workers}, teacher_fraction={args.teacher_fraction}; "
        "no job was submitted. Use `vscell run` for guided selection, "
        "confirmation, monitoring, and automatic finalization."
    )
    print(path)
    return 0


def _default_conda_init() -> str | None:
    conda_executable = shutil.which("conda")
    if not conda_executable:
        return None
    candidate = Path(conda_executable).resolve().parent.parent / "etc/profile.d/conda.sh"
    return str(candidate) if candidate.is_file() else None


def _required_run_value(
    value: str | None,
    label: str,
    *,
    interactive: bool,
    default: str | None = None,
) -> str:
    if value:
        return value
    if interactive:
        return wf.prompt_text(label, default)
    if default:
        return default
    raise RuntimeError(f"{label} is required for this non-interactive run")


def _prompt_guided_modality(target: str) -> tuple[str, str | None]:
    if target == "nuclear":
        choices = (
            ("DAPI", "DAPI nuclear fluorescence"),
            ("ssDNA", "single-stranded-DNA nuclear stain"),
            ("H&E", "hematoxylin-and-eosin histology"),
            ("custom", "another or multimodal acquisition"),
        )
        default = "ssDNA"
    else:
        choices = (
            ("mIF", "membrane-stained multiplex immunofluorescence"),
            ("phase_contrast", "phase-contrast microscopy"),
            ("fluorescence", "general fluorescence microscopy"),
            ("brightfield", "brightfield microscopy"),
            ("DIC", "differential interference contrast"),
            ("custom", "another or multimodal acquisition"),
        )
        default = "mIF"
    modality = wf.prompt_choice(
        f"Imaging modality for {target} segmentation",
        choices,
        default=default,
        mark_default_recommended=False,
    )
    if modality != "custom":
        return modality, None
    custom_name = wf.prompt_text("Custom modality name")
    custom_description = wf.prompt_text(
        "Describe the image appearance and segmentation-relevant signals"
    )
    return custom_name, custom_description


def _missing_candidate_masks(
    candidate_root: Path, models: tuple[str, ...], analysis: dict
) -> list[str]:
    missing = []
    for image in analysis["images"]:
        stem = image["image_id"]
        for model in models:
            choices = (
                candidate_root / model / f"{stem}_masks.tif",
                candidate_root / model / f"{stem}.tif",
                candidate_root / f"{stem}__{model}.tif",
            )
            if not any(path.is_file() for path in choices):
                missing.append(f"{stem}/{model}")
    return missing


def _teacher_source_ids(
    analysis: dict,
    grid: GridConfig,
    fraction: float,
    seed: int,
) -> set[str]:
    reference = GridReference(**analysis["grid_reference"])
    patch_sources = {}
    patch_rows = []
    for image in analysis["images"]:
        for patch in build_grid(
            image["image_id"], tuple(image["shape"]), reference, grid
        ):
            patch_rows.append({"patch_id": patch.patch_id})
            patch_sources[patch.patch_id] = image["image_id"]
    teacher, _student = select_teacher_patches(
        {"patches": patch_rows}, fraction, seed
    )
    return {patch_sources[patch_id] for patch_id in teacher}


def _new_guided_config(args: argparse.Namespace, interactive: bool) -> tuple[dict, dict]:
    run_root = Path(args.output).resolve()
    input_value = args.input
    if not input_value:
        if not interactive:
            raise RuntimeError("--input is required when starting a non-interactive run")
        input_value = wf.prompt_text(
            "Raw-image path (.tif/.tiff/.png/.jpg/.jpeg/.bmp)", "data"
        )
    input_path = Path(input_value).resolve()

    target = args.target
    if target is None:
        target = (
            wf.prompt_choice(
                "Segmentation target",
                (
                    ("nuclear", "segment nuclei"),
                    ("whole_cell", "segment complete cells"),
                ),
                default="nuclear",
                mark_default_recommended=False,
            )
            if interactive
            else "nuclear"
        )
    modality = args.modality
    modality_description = args.modality_description
    if modality is None:
        if interactive:
            modality, prompted_description = _prompt_guided_modality(target)
            if modality_description is None:
                modality_description = prompted_description
        else:
            modality = "ssDNA" if target == "nuclear" else "mIF"

    standard_candidates = Path("segmentation_results")
    legacy_candidates = run_root / "candidates"
    if args.candidates:
        candidate_default = args.candidates
    elif any(standard_candidates.glob("*/*.tif")):
        candidate_default = str(standard_candidates)
    elif any(legacy_candidates.glob("*/*.tif")):
        candidate_default = str(legacy_candidates)
    else:
        candidate_default = str(standard_candidates)
    candidates = Path(candidate_default).resolve()
    candidate_source = args.candidate_source
    if candidate_source is None:
        if interactive:
            existing_hint = any(candidates.glob("*/*.tif"))
            default_source = "existing" if existing_hint else "slurm"
            candidate_source = wf.prompt_choice(
                "Candidate segmentation masks",
                (
                    ("existing", "use precomputed masks organized by model"),
                    ("slurm", "submit the four built-in segmentation models"),
                ),
                default=default_source,
            )
        else:
            candidate_source = "existing"

    if args.candidates is None and interactive:
        path_label = (
            "Existing segmentation-results path"
            if candidate_source == "existing"
            else "Segmentation-results output path"
        )
        candidates = Path(wf.prompt_text(path_label, candidate_default)).resolve()

    if candidate_source == "slurm":
        models = tuple(DEFAULT_MODELS)
    elif args.models is not None:
        models = tuple(args.models)
    elif interactive:
        models = _models(
            wf.prompt_text("Candidate model names", ",".join(DEFAULT_MODELS))
        )
    else:
        models = tuple(DEFAULT_MODELS)

    teacher_fraction = args.teacher_fraction
    if teacher_fraction is None:
        teacher_fraction = (
            wf.prompt_fraction("Fraction ranked by the hosted VLM", 1.0)
            if interactive
            else 1.0
        )
    if not 0 <= teacher_fraction <= 1:
        raise ValueError("teacher fraction must be in [0, 1]")

    vlm_model = canonical_model(args.vlm_model or "gpt-5.5")
    if args.vlm_model is None and interactive and teacher_fraction > 0:
        vlm_model = wf.prompt_choice(
            "Hosted VLM",
            (
                ("gpt-5.5", "validated OpenAI default"),
                ("gpt-6-astra", "validated OpenAI alternative"),
                ("claude-fable-5-1", "validated Anthropic alternative"),
            ),
            default="gpt-5.5",
        )
    supported_models = set(MODEL_IDS.values())
    if vlm_model not in supported_models:
        raise ValueError(
            f"unsupported VLM {vlm_model!r}; choose from {sorted(supported_models)}"
        )

    workers = args.workers
    if workers is None:
        workers = (
            wf.prompt_int("Concurrent VLM workers", 8)
            if interactive and teacher_fraction > 0
            else 8
        )
    if workers < 1:
        raise ValueError("workers must be at least 1")

    execution = args.execution
    if teacher_fraction == 0:
        execution = "local"
    elif execution is None:
        if interactive:
            slurm_available = shutil.which("sbatch") is not None
            execution = wf.prompt_choice(
                "Hosted-VLM execution",
                (
                    ("slurm", "persistent cluster job; monitored automatically"),
                    ("local", "run in this terminal process"),
                ),
                default="slurm" if slurm_available else "local",
            )
        else:
            execution = "local"

    workspace = (run_root / "workspace").resolve()
    student_mode = args.student_mode
    student_execution = "none"
    student_checkpoint = Path(
        args.student_checkpoint or workspace / "student.pt"
    ).resolve()
    if teacher_fraction < 1:
        if student_mode is None:
            if teacher_fraction == 0:
                student_mode = "existing"
            elif interactive:
                student_mode = wf.prompt_choice(
                    "Student ranker",
                    (
                        ("train", "train from this run's teacher-ranked patches"),
                        ("existing", "use an existing compatible checkpoint"),
                    ),
                    default="train",
                )
            else:
                student_mode = "train"
        if student_mode == "existing" and args.student_checkpoint is None:
            if interactive:
                student_checkpoint = Path(
                    wf.prompt_text("Existing student checkpoint path")
                ).resolve()
            else:
                raise RuntimeError(
                    "--student-checkpoint is required with --student-mode existing"
                )
        if student_mode == "train":
            student_execution = args.student_execution
            if student_execution is None:
                if interactive:
                    slurm_available = shutil.which("sbatch") is not None
                    student_execution = wf.prompt_choice(
                        "Student-training execution",
                        (
                            (
                                "slurm",
                                "monitored one-GPU cluster job",
                            ),
                            ("local", "run in this terminal process"),
                        ),
                        default="slurm" if slurm_available else "local",
                    )
                else:
                    student_execution = (
                        "slurm" if execution == "slurm" else "local"
                    )
    else:
        student_mode = "none"

    needs_slurm = (
        candidate_source == "slurm"
        or execution == "slurm"
        or student_execution == "slurm"
    )
    if args.detach and not needs_slurm:
        raise ValueError("--detach is only meaningful when a Slurm stage is selected")
    account = args.account
    if needs_slurm:
        account = _required_run_value(
            account or os.environ.get("SLURM_ACCOUNT"),
            "Slurm account",
            interactive=interactive,
        )
    cpu_partition = args.cpu_partition
    if execution == "slurm":
        cpu_partition = _required_run_value(
            cpu_partition,
            "Slurm CPU partition",
            interactive=interactive,
        )
    gpu_partition = args.gpu_partition
    if candidate_source == "slurm" or student_execution == "slurm":
        gpu_label = (
            "Slurm GPU partition for student training"
            if candidate_source != "slurm"
            else "Slurm GPU partition"
        )
        gpu_partition = _required_run_value(
            gpu_partition,
            gpu_label,
            interactive=interactive,
        )
    conda_init = args.conda_init
    if candidate_source == "slurm":
        conda_init = _required_run_value(
            conda_init,
            "Conda initialization script",
            interactive=interactive,
            default=_default_conda_init(),
        )

    analysis = analyze_inputs(input_path, _grid_from_args(args))
    if candidate_source == "existing":
        missing = _missing_candidate_masks(candidates, models, analysis)
        if missing:
            shown = ", ".join(missing[:5])
            remainder = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
            raise FileNotFoundError(
                f"existing candidate masks are incomplete under {candidates}: "
                f"{shown}{remainder}"
            )
    teacher_count = round(analysis["patch_count"] * teacher_fraction)
    if student_mode == "train":
        if teacher_count < 4:
            raise ValueError(
                "student training requires at least four hosted teacher patches; "
                "increase the teacher fraction or use an existing compatible "
                "student checkpoint"
            )
        teacher_sources = _teacher_source_ids(
            analysis,
            _grid_from_args(args),
            teacher_fraction,
            args.seed,
        )
        if len(teacher_sources) < 2:
            raise ValueError(
                "student training requires teacher patches from at least two "
                "source images for leakage-free validation; use teacher fraction "
                "1.0 for a one-image run or use an existing compatible student "
                "checkpoint"
            )
    projected_cost = estimated_cost(
        vlm_model, analysis["patch_count"], teacher_fraction
    )
    config = {
        "run_root": str(run_root),
        "input": str(input_path),
        "candidates": str(candidates),
        "workspace": str(workspace),
        "slurm_dir": str((run_root / "slurm").resolve()),
        "target": target,
        "modality": modality,
        "modality_description": modality_description,
        "models": list(models),
        "candidate_source": candidate_source,
        "teacher_fraction": teacher_fraction,
        "vlm_model": vlm_model,
        "workers": workers,
        "execution": execution,
        "account": account,
        "cpu_partition": cpu_partition,
        "gpu_partition": gpu_partition,
        "conda_init": conda_init,
        "environment": str(Path(args.environment or sys.prefix).resolve()),
        "rank_time": args.rank_time,
        "rank_memory": args.rank_memory,
        "seed": args.seed,
        "minimum_area": args.minimum_area,
        "grid": _grid_from_args(args).__dict__,
        "student_mode": student_mode,
        "student_execution": student_execution,
        "student_checkpoint": str(student_checkpoint),
        "student_epochs": args.student_epochs,
        "student_device": (
            "cuda" if student_execution == "slurm" else args.student_device
        ),
        "student_cpus": args.student_cpus,
        "student_time": args.student_time,
        "student_memory": args.student_memory,
    }
    plan = {
        **analysis,
        "teacher_patch_count": teacher_count,
        "api_call_count": teacher_count * 3,
        "estimated_total_vlm_cost_usd": projected_cost,
    }
    return config, plan


def _print_guided_summary(config: dict, plan: dict) -> None:
    estimate = plan["estimated_total_vlm_cost_usd"]
    estimate_text = "unknown" if estimate is None else f"${estimate:.4f} historical estimate"
    print("\nVSCell guided-run summary", file=sys.stderr)
    print(f"  Raw images:       {config['input']} ({plan['image_count']} image(s))", file=sys.stderr)
    print(
        f"  Candidate masks:  {config['candidate_source']} -> {config['candidates']}",
        file=sys.stderr,
    )
    print(f"  Models:           {', '.join(config['models'])}", file=sys.stderr)
    print(f"  Target/modality:  {config['target']} / {config['modality']}", file=sys.stderr)
    if config["modality_description"]:
        print(
            f"  Modality details:  {config['modality_description']}",
            file=sys.stderr,
        )
    print(f"  Patches:          {plan['patch_count']}", file=sys.stderr)
    print(
        f"  Hosted VLM:       {config['vlm_model']} on "
        f"{plan['teacher_patch_count']} patch(es), 3 repetitions each",
        file=sys.stderr,
    )
    print(f"  Hosted calls:     {plan['api_call_count']} ({estimate_text})", file=sys.stderr)
    if plan["teacher_patch_count"] > 0:
        key_name = wf.provider_key_name(config["vlm_model"])
        if key_name in os.environ:
            key_status = (
                f"{key_name} detected in this process environment; value hidden "
                "and not saved by VSCell"
            )
        else:
            key_status = (
                f"{key_name} will be requested with hidden input immediately "
                "before hosted inference"
            )
        print(f"  Provider key:     {key_status}", file=sys.stderr)
    print(
        f"  Execution:        {config['execution']} with {config['workers']} worker(s)",
        file=sys.stderr,
    )
    if config["execution"] == "slurm":
        print(
            f"  Ranking Slurm:    account={config['account']} "
            f"partition={config['cpu_partition']} cpus={config['workers']} "
            f"time={config['rank_time']} memory={config['rank_memory']}",
            file=sys.stderr,
        )
        print(f"  Ranking env:      {config['environment']}", file=sys.stderr)
    if config["candidate_source"] == "slurm":
        print(
            f"  Backend Slurm:    account={config['account']} "
            f"partition={config['gpu_partition']} (four 1-GPU jobs)",
            file=sys.stderr,
        )
    if config["student_mode"] != "none":
        print(
            f"  Student:          {config['student_mode']} -> "
            f"{config['student_checkpoint']}",
            file=sys.stderr,
        )
    if config["student_execution"] == "slurm":
        print(
            f"  Student Slurm:    account={config['account']} "
            f"partition={config['gpu_partition']} gpus=1 "
            f"cpus={config['student_cpus']} time={config['student_time']} "
            f"memory={config['student_memory']}",
            file=sys.stderr,
        )
    print(f"  Final output:     {config['workspace']}/final_masks", file=sys.stderr)
    print(
        "  Privacy:          source crops and candidate panels are sent only during "
        "hosted-VLM ranking; segmentation ground truth is never used",
        file=sys.stderr,
    )
    print(file=sys.stderr)


def _mark_workflow_stage(state_path: Path, state: dict, stage: str, status: str, **details) -> None:
    state.setdefault("stages", {})[stage] = {"status": status, **details}
    if status == "failed":
        state["status"] = "failed"
    elif stage == "finalize" and status == "complete":
        state["status"] = "complete"
    else:
        state["status"] = "running"
    wf.save_state(state_path, state)


def _stage_complete(state: dict, stage: str) -> bool:
    return state.get("stages", {}).get(stage, {}).get("status") == "complete"


def _rankings_complete(manifest_path: Path, rankings_path: Path, config: dict) -> tuple[bool, int, int]:
    manifest = load_manifest(manifest_path)
    expected, _ = select_teacher_patches(
        manifest, config["teacher_fraction"], config["seed"]
    )
    if not rankings_path.is_file():
        return False, 0, len(expected)
    actual = set(load_rankings(rankings_path))
    return actual == set(expected), len(actual & set(expected)), len(expected)


def _wait_or_detach(
    args: argparse.Namespace,
    state_path: Path,
    state: dict,
    *,
    stage: str,
    job_ids: list[str],
    callback=None,
) -> tuple[bool, dict[str, str]]:
    if args.detach:
        wf.announce("Jobs were submitted in detached mode; VSCell is not waiting.")
        wf.announce(f"Continue later with: {wf.resume_command(state['config']['run_root'])}")
        return True, {}
    wf.announce("Monitoring Slurm; this terminal will continue automatically when jobs finish.")
    wf.announce(
        "Press Ctrl+C only to stop monitoring—the submitted jobs will continue, and the run is resumable."
    )
    try:
        states = wf.wait_for_slurm_jobs(
            job_ids,
            poll_interval=args.poll_interval,
            poll_callback=callback,
        )
    except KeyboardInterrupt:
        wf.save_state(state_path, state)
        wf.announce("Monitoring stopped; submitted Slurm jobs were not cancelled.")
        wf.announce(f"Continue later with: {wf.resume_command(state['config']['run_root'])}")
        return True, {}
    failures = wf.failed_jobs(states)
    if failures:
        _mark_workflow_stage(
            state_path, state, stage, "failed", job_ids=job_ids, slurm_states=states
        )
        log_group = {
            "candidates": "backends",
            "rank": "ranking",
            "student_train": "student",
        }.get(stage, stage)
        log_dir = Path(state["config"]["slurm_dir"]) / log_group / "logs"
        raise RuntimeError(
            f"Slurm stage {stage!r} failed: {failures}; inspect logs under {log_dir}"
        )
    return False, states


def _execute_guided_run(args: argparse.Namespace, state_path: Path, state: dict) -> int:
    config = state["config"]
    run_root = Path(config["run_root"])
    input_path = Path(config["input"])
    candidates = Path(config["candidates"])
    workspace = Path(config["workspace"])
    slurm_dir = Path(config["slurm_dir"])
    manifest_path = workspace / "run_manifest.json"
    teacher_rankings = workspace / "teacher_rankings.jsonl"

    if not _stage_complete(state, "candidates"):
        wf.announce("Stage 1: candidate segmentation masks")
        if config["candidate_source"] == "existing":
            wf.announce(f"Using existing full-image masks from {candidates}")
            _mark_workflow_stage(
                state_path, state, "candidates", "complete", source="existing"
            )
        else:
            stage = state.get("stages", {}).get("candidates", {})
            jobs = stage.get("jobs") if stage.get("status") == "submitted" else None
            if not jobs or set(jobs) != set(BACKENDS):
                wf.announce("Rendering and submitting remaining built-in GPU backend jobs")
                scripts = render_backend_scripts(
                    slurm_dir / "backends",
                    input_path=str(input_path),
                    candidate_root=str(candidates),
                    target=config["target"],
                    modality=config["modality"],
                    account=config["account"],
                    gpu_partition=config["gpu_partition"],
                    conda_init=config["conda_init"],
                )
                jobs = dict(jobs or {})
                for script in scripts:
                    backend = script.stem.removeprefix("run_")
                    if backend in jobs:
                        continue
                    job_id = wf.submit_slurm(script)
                    jobs[backend] = job_id
                    wf.announce(f"Submitted {backend}: job {job_id}")
                    _mark_workflow_stage(
                        state_path,
                        state,
                        "candidates",
                        "submitted",
                        jobs=jobs,
                    )
            detached, states = _wait_or_detach(
                args,
                state_path,
                state,
                stage="candidates",
                job_ids=list(jobs.values()),
            )
            if detached:
                return 0
            _mark_workflow_stage(
                state_path,
                state,
                "candidates",
                "complete",
                jobs=jobs,
                slurm_states=states,
            )
            wf.announce("All segmentation backend jobs left the queue successfully")

    if not _stage_complete(state, "prepare"):
        wf.announce("Stage 2: validating masks and preparing full-image crops/panels")
        grid = GridConfig(**config["grid"])
        try:
            prepared = prepare_workspace(
                input_path,
                candidates,
                workspace,
                target=config["target"],
                modality=config["modality"],
                models=config["models"],
                grid=grid,
            )
        except Exception:
            if config["candidate_source"] == "slurm":
                previous = state.get("stages", {}).get("candidates", {})
                _mark_workflow_stage(
                    state_path,
                    state,
                    "candidates",
                    "failed",
                    jobs=previous.get("jobs", {}),
                    reason="candidate masks did not pass workspace validation",
                )
            raise
        _mark_workflow_stage(
            state_path, state, "prepare", "complete", manifest=str(prepared)
        )
        wf.announce(f"Prepared workspace manifest: {prepared}")

    if not _stage_complete(state, "split"):
        wf.announce("Stage 3: recording the deterministic teacher/student split")
        split_path = write_teacher_split(
            manifest_path, config["teacher_fraction"], config["seed"]
        )
        _mark_workflow_stage(
            state_path, state, "split", "complete", split=str(split_path)
        )
        wf.announce(f"Recorded split: {split_path}")

    if not _stage_complete(state, "rank"):
        complete, completed_count, expected_count = _rankings_complete(
            manifest_path, teacher_rankings, config
        )
        if complete:
            _mark_workflow_stage(
                state_path,
                state,
                "rank",
                "complete",
                completed=completed_count,
                expected=expected_count,
            )
        elif expected_count == 0:
            teacher_rankings.parent.mkdir(parents=True, exist_ok=True)
            teacher_rankings.touch(exist_ok=True)
            _mark_workflow_stage(
                state_path, state, "rank", "complete", completed=0, expected=0
            )
        elif config["execution"] == "local":
            wf.announce(
                f"Stage 4: hosted-VLM ranking locally "
                f"({expected_count} patches, {config['workers']} workers)"
            )
            with wf.provider_key(
                config["vlm_model"],
                interactive=sys.stdin.isatty() and not args.non_interactive,
            ):
                _cmd_rank_teacher(
                    argparse.Namespace(
                        manifest=str(manifest_path),
                        output=str(teacher_rankings),
                        teacher_fraction=config["teacher_fraction"],
                        vlm_model=config["vlm_model"],
                        modality_description=config["modality_description"],
                        seed=config["seed"],
                        workers=config["workers"],
                        resume=True,
                        yes=True,
                    )
                )
            complete, completed_count, expected_count = _rankings_complete(
                manifest_path, teacher_rankings, config
            )
            if not complete:
                raise RuntimeError(
                    f"hosted ranking ended with {completed_count}/{expected_count} patches"
                )
            _mark_workflow_stage(
                state_path,
                state,
                "rank",
                "complete",
                completed=completed_count,
                expected=expected_count,
            )
        else:
            wf.announce(
                f"Stage 4: hosted-VLM ranking on Slurm "
                f"({expected_count} patches, {config['workers']} workers)"
            )
            stage = state.get("stages", {}).get("rank", {})
            job_id = stage.get("job_id") if stage.get("status") == "submitted" else None
            if not job_id:
                script = render_rank_teacher_script(
                    slurm_dir / "ranking",
                    manifest_path=str(manifest_path),
                    rankings_path=str(teacher_rankings),
                    teacher_fraction=config["teacher_fraction"],
                    vlm_model=config["vlm_model"],
                    workers=config["workers"],
                    environment=config["environment"],
                    account=config["account"],
                    cpu_partition=config["cpu_partition"],
                    time_limit=config["rank_time"],
                    memory=config["rank_memory"],
                    seed=config["seed"],
                    modality_description=config["modality_description"],
                )
                with wf.provider_key(
                    config["vlm_model"],
                    interactive=sys.stdin.isatty() and not args.non_interactive,
                ):
                    job_id = wf.submit_slurm(script)
                _mark_workflow_stage(
                    state_path,
                    state,
                    "rank",
                    "submitted",
                    job_id=job_id,
                    script=str(script),
                    completed=completed_count,
                    expected=expected_count,
                )
                wf.announce(f"Submitted hosted-VLM ranking job {job_id}")

            progress_status = {"count": -1, "polls": 0}

            def show_progress(current_states):
                count = wf.count_jsonl_records(teacher_rankings)
                progress_status["polls"] += 1
                changed = count != progress_status["count"]
                heartbeat = progress_status["polls"] % 3 == 0
                if changed or heartbeat:
                    job_state = current_states.get(job_id, "leaving queue")
                    activity = "progress" if changed else "inference active"
                    wf.announce(
                        f"Hosted-VLM {activity}: {count}/{expected_count} patches "
                        f"complete; Slurm job {job_id}={job_state}"
                    )
                    progress_status["count"] = count

            detached, states = _wait_or_detach(
                args,
                state_path,
                state,
                stage="rank",
                job_ids=[job_id],
                callback=show_progress,
            )
            if detached:
                return 0
            complete, completed_count, expected_count = _rankings_complete(
                manifest_path, teacher_rankings, config
            )
            if not complete:
                error_log = slurm_dir / "ranking/logs" / f"vscell_rank_{job_id}.err"
                _mark_workflow_stage(
                    state_path,
                    state,
                    "rank",
                    "failed",
                    job_id=job_id,
                    slurm_states=states,
                    completed=completed_count,
                    expected=expected_count,
                    error_log=str(error_log),
                )
                raise RuntimeError(
                    f"hosted ranking job ended with {completed_count}/{expected_count} "
                    f"patches; inspect {error_log} and resume after correcting the issue"
                )
            _mark_workflow_stage(
                state_path,
                state,
                "rank",
                "complete",
                job_id=job_id,
                slurm_states=states,
                completed=completed_count,
                expected=expected_count,
            )
            wf.announce(f"Hosted-VLM ranking complete: {completed_count}/{expected_count}")

    rankings_for_finalize = teacher_rankings
    if config["teacher_fraction"] < 1:
        checkpoint = Path(config["student_checkpoint"])
        all_rankings = workspace / "all_rankings.jsonl"
        if config["student_mode"] == "train" and not _stage_complete(state, "student_train"):
            if config["student_execution"] == "slurm":
                wf.announce(
                    "Stage 5: training the student ranker in a monitored Slurm job"
                )
                stage = state.get("stages", {}).get("student_train", {})
                job_id = (
                    stage.get("job_id")
                    if stage.get("status") == "submitted"
                    else None
                )
                if not job_id:
                    script = render_train_student_script(
                        slurm_dir / "student",
                        manifest_path=str(manifest_path),
                        teacher_rankings_path=str(teacher_rankings),
                        teacher_fraction=config["teacher_fraction"],
                        checkpoint_path=str(checkpoint),
                        all_rankings_path=str(all_rankings),
                        epochs=config["student_epochs"],
                        device=config["student_device"],
                        environment=config["environment"],
                        cpus=config["student_cpus"],
                        account=config["account"],
                        gpu_partition=config["gpu_partition"],
                        time_limit=config["student_time"],
                        memory=config["student_memory"],
                        seed=config["seed"],
                    )
                    job_id = wf.submit_slurm(script)
                    _mark_workflow_stage(
                        state_path,
                        state,
                        "student_train",
                        "submitted",
                        job_id=job_id,
                        script=str(script),
                    )
                    wf.announce(f"Submitted student-training job {job_id}")

                student_polls = {"count": 0}

                def show_student_progress(current_states):
                    student_polls["count"] += 1
                    if student_polls["count"] == 1 or student_polls["count"] % 3 == 0:
                        job_state = current_states.get(job_id, "leaving queue")
                        activity = (
                            "student prediction"
                            if checkpoint.is_file()
                            else "student training"
                        )
                        wf.announce(
                            f"{activity.capitalize()} active: Slurm job "
                            f"{job_id}={job_state}"
                        )

                detached, states = _wait_or_detach(
                    args,
                    state_path,
                    state,
                    stage="student_train",
                    job_ids=[job_id],
                    callback=show_student_progress,
                )
                if detached:
                    return 0
                if not checkpoint.is_file() or not all_rankings.is_file():
                    error_log = (
                        slurm_dir
                        / "student/logs"
                        / f"vscell_student_{job_id}.err"
                    )
                    _mark_workflow_stage(
                        state_path,
                        state,
                        "student_train",
                        "failed",
                        job_id=job_id,
                        slurm_states=states,
                        error_log=str(error_log),
                    )
                    raise RuntimeError(
                        "student training/prediction job completed without all "
                        "expected outputs; "
                        f"inspect {error_log} and resume after correcting the issue"
                    )
                _mark_workflow_stage(
                    state_path,
                    state,
                    "student_train",
                    "complete",
                    checkpoint=str(checkpoint),
                    job_id=job_id,
                    slurm_states=states,
                )
                wf.announce(f"Student training complete: {checkpoint}")
                _mark_workflow_stage(
                    state_path,
                    state,
                    "student_predict",
                    "complete",
                    rankings=str(all_rankings),
                    job_id=job_id,
                    slurm_states=states,
                )
                wf.announce(
                    f"Student ranking complete: {all_rankings}"
                )
            else:
                wf.announce(
                    "Stage 5: training the student ranker in this terminal process"
                )
                _cmd_train_student(
                    argparse.Namespace(
                        manifest=str(manifest_path),
                        teacher_rankings=str(teacher_rankings),
                        teacher_fraction=config["teacher_fraction"],
                        checkpoint=str(checkpoint),
                        device=config["student_device"],
                        epochs=config["student_epochs"],
                        seed=config["seed"],
                    )
                )
                _mark_workflow_stage(
                    state_path,
                    state,
                    "student_train",
                    "complete",
                    checkpoint=str(checkpoint),
                )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"student checkpoint does not exist: {checkpoint}")
        if not _stage_complete(state, "student_predict"):
            wf.announce("Stage 6: ranking remaining patches with the student")
            _cmd_predict_student(
                argparse.Namespace(
                    manifest=str(manifest_path),
                    teacher_rankings=str(teacher_rankings),
                    teacher_fraction=config["teacher_fraction"],
                    checkpoint=str(checkpoint),
                    output=str(all_rankings),
                    device=config["student_device"],
                    seed=config["seed"],
                )
            )
            _mark_workflow_stage(
                state_path,
                state,
                "student_predict",
                "complete",
                rankings=str(all_rankings),
            )
        rankings_for_finalize = all_rankings

    if not _stage_complete(state, "finalize"):
        wf.announce("Final stage: consensus selection and full-image stitching")
        audit = finalize_workspace(
            manifest_path,
            rankings_for_finalize,
            ensemble=EnsembleConfig(minimum_area=config["minimum_area"]),
        )
        images = audit["images"]
        final_masks = [row["final_mask_path"] for row in images]
        final_cells = {
            row["image_id"]: row["stitch"]["final_cells"] for row in images
        }
        _mark_workflow_stage(
            state_path,
            state,
            "finalize",
            "complete",
            final_masks=final_masks,
            final_cells=final_cells,
            audit=str(workspace / "final_audit.json"),
        )
    final_stage = state["stages"]["finalize"]
    wf.announce("Workflow complete")
    for mask in final_stage["final_masks"]:
        wf.announce(f"Final stitched mask: {mask}")
    wf.announce(f"Audit: {final_stage['audit']}")
    wf.announce(
        "Accuracy is not reported because VSCell inference does not accept ground truth. "
        "Evaluate against held-out ground truth separately when available."
    )
    print(json.dumps({"status": "complete", **final_stage}, indent=2))
    return 0


def _cmd_run_impl(args: argparse.Namespace) -> int:
    run_root = Path(args.output).resolve()
    state_path = wf.state_path(run_root)
    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(
                f"no guided workflow state found at {state_path}; omit --resume to start"
            )
        state = wf.load_state(state_path)
        wf.announce(f"Resuming workflow from {state_path}")
        return _execute_guided_run(args, state_path, state)

    if state_path.exists():
        raise FileExistsError(
            f"a guided run already exists at {state_path}; use --resume or choose "
            "a different --output directory"
        )
    interactive = sys.stdin.isatty() and not args.non_interactive
    config, plan = _new_guided_config(args, interactive)
    _print_guided_summary(config, plan)
    if not args.yes:
        if not interactive:
            raise RuntimeError(
                "review the summary and rerun with --yes to authorize the workflow"
            )
        if not wf.confirm("Start this workflow?"):
            wf.announce("Cancelled before any jobs or hosted calls were started")
            return 2
    state = {
        "state_version": wf.STATE_VERSION,
        "status": "running",
        "config": config,
        "plan": plan,
        "stages": {},
    }
    wf.save_state(state_path, state)
    wf.announce(f"Saved resumable workflow state: {state_path}")
    return _execute_guided_run(args, state_path, state)


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        return _cmd_run_impl(args)
    except Exception as error:
        if os.environ.get("VSCELL_DEBUG") == "1":
            raise
        wf.announce(f"Workflow stopped: {type(error).__name__}: {error}")
        state_path = wf.state_path(args.output)
        if state_path.is_file():
            wf.announce("Completed work was preserved; correct the issue and resume with:")
            wf.announce(wf.resume_command(args.output))
        else:
            wf.announce("No hosted job or workflow state was created.")
        wf.announce("Set VSCELL_DEBUG=1 to show a developer traceback.")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vscell")
    parser.add_argument("--version", action="version", version="VSCell 0.1.0.dev0")
    sub = parser.add_subparsers(dest="command", required=True)

    guided = sub.add_parser(
        "run",
        help="run the guided, resumable workflow with visible progress",
        description=(
            "Guide and orchestrate candidate masks, workspace preparation, hosted "
            "ranking, optional student inference, and final stitching. Slurm jobs "
            "are monitored by default; use --detach only when desired."
        ),
    )
    guided.add_argument("--input", help="Raw-image file or directory; prompted if omitted")
    guided.add_argument(
        "--output",
        default="run",
        help="Run root containing candidates, workspace, Slurm files, and state",
    )
    guided.add_argument(
        "--candidates",
        help=(
            "Full-image candidate-mask root (default: segmentation_results; "
            "legacy OUTPUT/candidates is auto-detected)"
        ),
    )
    guided.add_argument("--target", choices=("nuclear", "whole_cell"))
    guided.add_argument("--modality")
    guided.add_argument("--modality-description")
    guided.add_argument(
        "--models",
        type=_models,
        metavar="MODEL1,MODEL2,MODEL3,...",
        help="Candidate names when using existing masks",
    )
    guided.add_argument(
        "--candidate-source",
        choices=("existing", "slurm"),
        help="Reuse existing masks or submit the four built-in GPU jobs",
    )
    guided.add_argument("--teacher-fraction", type=float)
    guided.add_argument("--vlm-model")
    guided.add_argument("--workers", type=_positive_int)
    guided.add_argument(
        "--execution",
        choices=("local", "slurm"),
        help="Where to execute hosted-VLM ranking",
    )
    guided.add_argument("--account", help="Slurm account")
    guided.add_argument("--cpu-partition", help="Slurm partition for hosted ranking")
    guided.add_argument(
        "--gpu-partition",
        help="Slurm GPU partition for segmentation or student-training jobs",
    )
    guided.add_argument("--conda-init", help="Path to conda.sh for backend jobs")
    guided.add_argument(
        "--environment",
        help="Environment root containing bin/vscell for the ranking job",
    )
    guided.add_argument("--rank-time", default="04:00:00")
    guided.add_argument("--rank-memory", default="8G")
    guided.add_argument("--student-mode", choices=("train", "existing"))
    guided.add_argument(
        "--student-execution",
        choices=("local", "slurm"),
        help="Where to train a new student; Slurm requests one GPU",
    )
    guided.add_argument("--student-checkpoint")
    guided.add_argument("--student-epochs", type=_positive_int, default=60)
    guided.add_argument("--student-cpus", type=_positive_int, default=4)
    guided.add_argument("--student-time", default="04:00:00")
    guided.add_argument("--student-memory", default="24G")
    guided.add_argument(
        "--student-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    guided.add_argument("--seed", type=int, default=120912)
    guided.add_argument("--minimum-area", type=_positive_int, default=1)
    guided.add_argument(
        "--detach",
        action="store_true",
        help="Return after Slurm submission instead of monitoring and continuing",
    )
    guided.add_argument(
        "--resume",
        action="store_true",
        help="Resume from OUTPUT/workflow_state.json",
    )
    guided.add_argument(
        "--yes",
        action="store_true",
        help="Accept the displayed summary without the final confirmation prompt",
    )
    guided.add_argument(
        "--non-interactive",
        action="store_true",
        help="Use supplied flags/defaults without preference prompts",
    )
    guided.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help=argparse.SUPPRESS,
    )
    _add_grid(guided)
    guided.set_defaults(func=_cmd_run)

    plan = sub.add_parser("plan", help="analyze image sizes and estimate VLM cost")
    plan.add_argument("--input", required=True)
    plan.add_argument("--teacher-fraction", type=float, default=1.0)
    plan.add_argument("--vlm-model", default="gpt-5.5")
    _add_grid(plan)
    plan.set_defaults(func=_cmd_plan)

    backend = sub.add_parser("backend", help="run one full-image segmentation backend")
    backend.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    backend.add_argument("--input", required=True)
    backend.add_argument("--output", required=True)
    backend.add_argument("--target", choices=("nuclear", "whole_cell"), required=True)
    backend.add_argument("--modality", required=True)
    backend.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    backend.add_argument("--overwrite", action="store_true")
    backend.set_defaults(func=_cmd_backend)

    prepare = sub.add_parser("prepare", help="crop masks and render VLM panels")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--candidates", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--target", choices=("nuclear", "whole_cell"), required=True)
    prepare.add_argument("--modality", required=True)
    prepare.add_argument(
        "--models",
        type=_models,
        default=DEFAULT_MODELS,
        metavar="MODEL1,MODEL2,MODEL3,...",
        help=(
            "Comma-separated candidate names. For an external model named MODEL, "
            "place one full-image instance mask per input image under "
            "CANDIDATES/MODEL/<image_stem>_masks.tif. At least three candidates "
            "are required."
        ),
    )
    _add_grid(prepare)
    prepare.set_defaults(func=_cmd_prepare)

    split = sub.add_parser("split", help="create deterministic teacher/student split")
    split.add_argument("--manifest", required=True)
    split.add_argument("--teacher-fraction", type=float, required=True)
    split.add_argument("--seed", type=int, default=120912)
    split.set_defaults(func=_cmd_split)

    rank = sub.add_parser("rank-teacher", help="rank the teacher subset with a hosted VLM")
    rank.add_argument("--manifest", required=True)
    rank.add_argument("--output", required=True)
    rank.add_argument("--teacher-fraction", type=float, required=True)
    rank.add_argument("--vlm-model", default="gpt-5.5")
    rank.add_argument("--modality-description")
    rank.add_argument("--seed", type=int, default=120912)
    rank.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="Concurrent patch-ranking workers; each worker makes one API call at a time.",
    )
    rank.add_argument("--resume", action="store_true")
    rank.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the displayed estimate and allow paid hosted inference.",
    )
    rank.set_defaults(func=_cmd_rank_teacher)

    train = sub.add_parser("train-student", help="distill a student from teacher rankings")
    train.add_argument("--manifest", required=True)
    train.add_argument("--teacher-rankings", required=True)
    train.add_argument("--teacher-fraction", type=float, required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--seed", type=int, default=120912)
    train.set_defaults(func=_cmd_train_student)

    predict = sub.add_parser("predict-student", help="rank non-teacher patches")
    predict.add_argument("--manifest", required=True)
    predict.add_argument("--teacher-rankings", required=True)
    predict.add_argument("--teacher-fraction", type=float, required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    predict.add_argument("--seed", type=int, default=120912)
    predict.set_defaults(func=_cmd_predict_student)

    finalize = sub.add_parser("finalize", help="run Solution 08 and stitch final masks")
    finalize.add_argument("--manifest", required=True)
    finalize.add_argument("--rankings", required=True)
    finalize.add_argument("--minimum-area", type=int, default=1)
    finalize.set_defaults(func=_cmd_finalize)

    slurm = sub.add_parser("make-slurm", help="render portable backend inference jobs")
    slurm.add_argument("--input", required=True)
    slurm.add_argument("--candidates", required=True)
    slurm.add_argument("--output", required=True)
    slurm.add_argument("--target", choices=("nuclear", "whole_cell"), required=True)
    slurm.add_argument("--modality", required=True)
    slurm.add_argument("--account", default="<ACCOUNT>")
    slurm.add_argument("--gpu-partition", default="<GPU_PARTITION>")
    slurm.add_argument("--conda-init", default="<CONDA_ROOT>/etc/profile.d/conda.sh")
    slurm.set_defaults(func=_cmd_slurm)

    rank_slurm = sub.add_parser(
        "make-rank-slurm", help="render a concurrent hosted-VLM ranking job"
    )
    rank_slurm.add_argument("--manifest", required=True)
    rank_slurm.add_argument("--rankings", required=True)
    rank_slurm.add_argument("--output", required=True)
    rank_slurm.add_argument("--teacher-fraction", type=float, required=True)
    rank_slurm.add_argument("--vlm-model", default="gpt-5.5")
    rank_slurm.add_argument("--modality-description")
    rank_slurm.add_argument("--seed", type=int, default=120912)
    rank_slurm.add_argument("--workers", type=_positive_int, default=8)
    rank_slurm.add_argument("--environment", default=sys.prefix)
    rank_slurm.add_argument("--account", default="<ACCOUNT>")
    rank_slurm.add_argument("--cpu-partition", default="<CPU_PARTITION>")
    rank_slurm.add_argument("--time", default="04:00:00")
    rank_slurm.add_argument("--memory", default="8G")
    rank_slurm.set_defaults(func=_cmd_rank_slurm)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
