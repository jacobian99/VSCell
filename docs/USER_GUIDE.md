# VSCell User Guide

This is the complete installation, data-preparation, guided-workflow, Slurm,
student-ranker, and advanced-command reference. For the short introduction and
quick start, see the [main README](../README.md).

## Contents

- [Pipeline](#pipeline)
- [Targets and modality presets](#targets-and-modality-presets)
- [Installation](#installation)
- [Model weights and licenses](#model-weights-and-licenses)
- [Prepare data before running](#prepare-data-before-running-offline)
- [Guided workflow](#guided-workflow-recommended)
- [Advanced manual workflow](#advanced-manual-workflow)
- [Local five-image demo](#local-five-image-cellbindb-demo)
- [Scientific defaults](#scientific-defaults)
- [Tests](#tests)

VSCell ranks multiple candidate cell-segmentation masks with a vision-language
model (VLM) or a distilled student ranker, applies the paper's GT-free Solution
08 consensus ensemble, and stitches local results into a globally labelled
full-image instance mask.

The package is an integration of the validated research pipeline. Segmentation
models run once on each full image; their masks are cropped afterward. VSCell
does not rerun heavy segmentation models for every local crop and never needs
segmentation ground truth at inference time.

[`PROVENANCE.md`](PROVENANCE.md) maps each packaged module to the validated experiment code
from which it was ported and lists the few intentional GT-free integration
changes.
[`VALIDATION.md`](VALIDATION.md) records the local package-level and no-API end-to-end
validation performed before release.

> Development status: this is a pre-release package. Raw validation/demo image
> files are not redistributed. The repository includes five approved rendered
> CellBinDB display panels under `examples/example_results/`.

## Pipeline

1. Run the built-in Cellpose, StarDist, CellSAM, and Cellpose-SAM backends, or
   provide full-image instance masks from at least three segmentation models.
2. Analyze the dataset's image-size distribution and create adaptive square-like
   local crops. Typical images use 4×4; unusually large or rectangular images
   scale each axis independently.
3. Enlarge each grid cell by the historical `PAD=0.10`: 10% total side-length
   increase, approximately 5% on each side.
4. Render the reference crop and anonymous yellow-fill/green-boundary candidate
   overlays.
5. Rank a user-selected random fraction with GPT-5.5 (default), GPT-6 Astra, or
   Claude Fable 5.1. Each patch is shuffled and ranked three times, then combined
   by Borda score.
6. Optionally train the identity-blind student ranker from teacher-ranked crops
   and predict the remaining crops. A 0% teacher run requires a compatible
   pretrained student checkpoint.
7. Apply Solution 08 anchor-consensus selection and boundary refinement locally.
8. Use the image-level Borda winner for cells spanning multiple crop cores, use
   local VSCell results elsewhere, merge, and assign new global instance labels.

## Targets and modality presets

The user selects `nuclear` or `whole_cell` in the guided menu (or supplies
`--target` in a non-interactive command). VSCell then fills the prompt's target
wording automatically. It discovers the candidate models, calculates candidate
count `N`, generates anonymous labels, and constructs the strict ranking format;
users never edit `<TARGET>`, `<N>`, labels, or other prompt placeholders.

These target-aware descriptions are built into the software:

| Target | Modality | Prompt description in brief |
|---|---|---|
| Nuclear | DAPI | Bright nuclear fluorescence blobs on a dark background |
| Nuclear | ssDNA | Bright single-stranded-DNA nuclear signal, similar in appearance to DAPI |
| Nuclear | H&E | Inverted-grayscale histology route in which nuclei appear bright |
| Whole-cell | mIF | Bright membrane outlines enclosing dark cell interiors; nuclei are not the target |
| Whole-cell | Phase contrast | Textured/granular cells with refractive boundary halos |
| Whole-cell | Fluorescence | Coherent fluorescence regions with variable signal, polarity, and background |
| Whole-cell | Brightfield | Stained, textured cell bodies against a lighter field |
| Whole-cell | DIC | Relief-like textured cells with paired bright/dark edge shading |

In `vscell run`, the modality menu is filtered by the target selected in the
previous prompt. A final `custom` choice asks for both the modality name and a
plain-language description of the image appearance and segmentation-relevant
signals. One modality selection applies to the entire run; place images from
different modalities in separate runs. A single RGB/RGBA composite can use the
custom choice, but arbitrary multichannel stacks are not accepted by this 2-D
workflow.

### Datasets containing multiple modalities

Organize and run different modalities separately. Do not mix images from
different modalities in one `data/` directory because the selected modality
and target apply to every image in that VSCell run. For example, CellBinDB
contains four staining groups rather than four channels in every image. A clear
project layout is:

```text
cellbindb_project/
├── DAPI/
│   ├── data/
│   └── segmentation_results/
├── HE/
│   ├── data/
│   └── segmentation_results/
├── mIF/
│   ├── data/
│   └── segmentation_results/
└── ssDNA/
    ├── data/
    └── segmentation_results/
```

Run `vscell run` once inside each modality directory. Select `nuclear` for
DAPI, H&E (`HE` in the CellBinDB folder name), and ssDNA. Select `whole_cell`
for mIF. Each run creates its own `run/` directory. The same organization
applies to any heterogeneous dataset: make one project/run per modality and
segmentation target.

The validated mIF prompt is whole-cell: bright membrane outlines enclose dark
cell interiors, and nuclei are not visible in that channel. StarDist receives an
inverted mIF input in the corrected model-specific preprocessing path, but input
polarity does not change the target from whole-cell to nuclear.

Unknown modalities are allowed. Supply `--modality-description` for the VLM or
let VSCell generate a generic target-aware description.
If a target conflicts with a built-in modality preset, VSCell stops before an
API call unless an explicit custom modality description is supplied; this
prevents internally contradictory prompts.

For DAPI and H&E nuclear images, a raw-image-only run is supported: choose the
`slurm` candidate source and VSCell submits Cellpose, StarDist, CellSAM, and
Cellpose-SAM once per full image before ranking. The isolated backend
environments and authorized model-weight access must already be prepared.
H&E uses its validated H&E-specific preprocessing/model route; DAPI follows the
nuclear-fluorescence route. The same four-backend interface is available for
the whole-cell presets, with the backend limitations and warnings documented
below.

## Installation

Python 3.10 or newer is required. Install the dependency-light core first. Add
only the hosted-provider client needed for ranking; this avoids downloading the
optional PyTorch student stack for VLM-only use:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[vlm-openai]'
```

Use `.[vlm-anthropic]` for Claude, `.[student]` for student training and
inference, and `.[test]` for development tests. `.[all,test]` intentionally
installs both provider clients, PyTorch, and the test tools and is therefore a
substantially larger installation.

The validated backends use separate environments. This is intentional: classic
Cellpose and Cellpose-SAM use incompatible Cellpose APIs, and StarDist's
TensorFlow stack should not be forced into the CellSAM/PyTorch environment.

| Backend | Validated environment | Model behavior |
|---|---|---|
| Cellpose | `cellpose==3.1.1.3` | `nuclei` for nuclear; `cyto` for whole-cell |
| Cellpose-SAM | `cellpose==4.1.1` | original `cpsam`, default inference parameters |
| StarDist | `stardist==0.9.2`, `csbdeep==0.8.2` | H&E versatile model; fluorescence model otherwise |
| CellSAM | commit `35bc6e0d…`, official v1.2 weights | `cellsam_general`, no target-mode switch |

Environment specifications are under `envs/` in the repository and source
distribution. From that source root:

```bash
conda env create -f envs/cellpose.yml
conda env create -f envs/cpsam-cellsam.yml
conda env create -f envs/stardist.yml
```

Install a CUDA-enabled PyTorch/TensorFlow build appropriate for the machine if
GPU inference is desired. CPU execution is supported where the upstream model
supports it, but full-image inference can be substantially slower. Student
inference is lightweight on CPU; GPU is recommended for student training.

## Model weights and licenses

VSCell does not copy third-party weights into this repository.

- Cellpose/Cellpose-SAM and StarDist use their official download/cache paths.
- CellSAM v1.2 weights have non-commercial academic-use terms and require access
  through the DeepCell portal. Set `DEEPCELL_ACCESS_TOKEN` in the CellSAM
  environment; do not commit it. If an authorized copy is already stored outside
  the default DeepCell cache, set
  `VSCELL_CELLSAM_MODEL_PATH=/path/to/cellsam_general.pt`; the file is loaded in
  place and is never copied into the VSCell repository.
- VSCell's integration code is MIT-licensed. Dependencies, weights, datasets,
  and hosted APIs retain their own terms. See `THIRD_PARTY_NOTICES.md`.

For shared/offline caches, the upstream variables
`CELLPOSE_LOCAL_MODELS_PATH` and `KERAS_HOME` can likewise point Cellpose and
StarDist to existing authorized model caches.

Upstream projects: [Cellpose](https://github.com/MouseLand/cellpose),
[StarDist](https://github.com/stardist/stardist), and
[CellSAM](https://github.com/vanvalenlab/cellSAM).

## Prepare data before running (offline)

Do this filesystem and environment preparation before starting `vscell run`.
The directory names below are recommendations, not hard-coded names; pass a
different path through `--input`, `--candidates`, or `--output` when needed.

The user creates a local project directory and its `data/` folder, then copies
or saves the source microscopy images into that folder. VSCell does not create
or download the source images. Run `vscell run` from inside the project
directory; relative paths such as `data` are resolved from that location.

For example:

```bash
mkdir my_vscell_project
cd my_vscell_project
mkdir data
# Copy your microscopy images into data/ before continuing.
vscell run
```

VSCell creates `run/` automatically after preflight and confirmation. If the
user already has candidate masks, the user also creates and populates
`segmentation_results/<model>/`. A ready-to-copy empty layout is provided in
`examples/project_template/`.

### Raw images

Put only the source microscopy images intended for this run in one directory:

```text
project/
└── data/
    ├── image01.tif
    └── image02.tif
```

The input may instead be one image file. Directory discovery is non-recursive.
Supported extensions are `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, and `.bmp`.
Each image must be either:

- a 2-D grayscale image; or
- a channels-last RGB or RGBA image. RGBA input is read as RGB.

Do not place ground-truth masks, candidate masks, panels, or unrelated images
in the data directory: every supported image file directly inside it is
treated as a source image. Volumetric stacks and channels-first arrays are not
accepted by this 2-D workflow. Use short, stable, unique filenames because the
filename stem, such as `image01`, identifies the image throughout the run.

TIFF is recommended for microscopy because it preserves lossless pixel values
and common higher bit depths. PNG is also lossless and accepted, but users do
not need to convert valid TIFF data to PNG. JPEG is accepted for convenience
but is lossy and is therefore not recommended for quantitative microscopy.

### Option A: start with raw images only

VSCell can submit the four built-in segmentation backends before VLM ranking.
Create the three backend environments from `envs/` first and arrange authorized
model-weight access as described above:

```bash
conda env create -f envs/cellpose.yml
conda env create -f envs/cpsam-cellsam.yml
conda env create -f envs/stardist.yml
```

Then start the guided command and choose `slurm` when it asks how to obtain
candidate segmentation masks:

```bash
vscell run --input data --output run
```

The guided summary shows the four GPU jobs and site settings before anything is
submitted. The backend jobs operate once on each complete source image; VSCell
later crops their masks and does not rerun a backend for every patch.

### Option B: start with existing candidate masks

Provide at least three candidate models for every source image. The recommended
layout is:

```text
project/
├── data/
│   ├── image01.tif
│   └── image02.tif
└── segmentation_results/
    ├── cellpose/
    │   ├── image01_masks.tif
    │   └── image02_masks.tif
    ├── stardist/
    │   ├── image01_masks.tif
    │   └── image02_masks.tif
    └── my_model/
        ├── image01_masks.tif
        └── image02_masks.tif
```

Each candidate mask must be:

- a single 2-D TIFF with exactly the same height and width as its source image;
- a nonnegative integer instance-label image, preferably `uint16` or `uint32`;
- `0` for background; and
- one positive integer label per segmented instance.

Labels do not need to be consecutive. If an upstream label contains multiple
disconnected 8-connected components, VSCell safely assigns them separate
instance IDs during output relabeling. Do not provide a binary foreground mask
unless the entire foreground genuinely represents one instance.

For a model directory named `<model>` and source image stem `<stem>`, VSCell
accepts these mask locations:

```text
segmentation_results/<model>/<stem>_masks.tif   # recommended
segmentation_results/<model>/<stem>.tif
segmentation_results/<stem>__<model>.tif
```

With the recommended `data/` and `segmentation_results/` layout, first change
into the project directory and then start the interactive command without path
flags:

```bash
cd my_vscell_project
vscell run
```

VSCell displays `data` and `segmentation_results` as explicit defaults and asks
the user to press Enter to accept them. It creates the separate `run/` output
directory; users should not place source images or third-party masks there.

If the masks are stored elsewhere, pass that location explicitly:

```bash
vscell run \
  --input data \
  --candidates /path/to/candidates \
  --output run
```

Before creating workflow state or making a hosted call, the guided preflight
checks that every requested image/model mask is present. During workspace
preparation it also validates mask dimensionality and source-image shape.
Segmentation ground truth is neither requested nor accepted for inference.

## Guided workflow (recommended)

For a new run, start with one command:

```bash
vscell run
```

The default interactive path asks only for the decisions most users need:

1. raw-image file or folder;
2. nuclear or whole-cell segmentation;
3. a target-compatible built-in imaging modality, or a custom modality and
   description;
4. existing candidate masks or four built-in segmentation jobs on Slurm;
5. a ranking quality/cost profile;
6. a hosted VLM (GPT-5.5, GPT-6 Astra, or Claude Fable 5.1); and
7. local or Slurm hosted-ranking execution.

For existing masks, VSCell automatically detects model directories that have a
complete mask set for the chosen images. The recommended `standard` profile
ranks every patch with the selected VLM and does not use a student. The
`budget` profile ranks half of the patches with the selected VLM and trains a
student for the remainder. Choose `advanced` in the ranking-profile menu, pass
`--advanced`, or use command-line flags to set the teacher fraction, student
behavior, and worker count explicitly. Slurm account, partition, and environment
prompts remain conditional on selecting Slurm.

Every menu asks the user to enter an option number. When a default is available,
the prompt explicitly labels its number and says `press Enter to accept`;
entering another option number overrides that default.

### Prompt-by-prompt guide

Only prompts relevant to the selected path are shown. Paths such as `data` and
`segmentation_results` are relative to the directory from which `vscell run`
was started.

| Prompt | Meaning and expected input |
|---|---|
| `Raw-image folder or file` | File or directory containing the user's source microscopy images. Press Enter for `data`, or type another path. VSCell does not create or download these inputs. |
| `Segmentation target` | Enter the option number for nuclear or whole-cell instances. This choice also filters the modality menu and selects target-aware model/prompt behavior. |
| `Imaging modality` | Enter the option number for a compatible built-in preset. Choose `custom` to provide both a modality name and an image/signal description. One modality applies to every image in the run. |
| `Candidate segmentation masks` | Choose `existing` to reuse masks already organized by model, or `slurm` to submit the four built-in segmentation backends. |
| `Candidate-mask folder` | Root containing `<model>/<image_stem>_masks.tif`. Press Enter for `segmentation_results`. VSCell announces the complete model directories it finds. An incomplete folder stops the run rather than being silently omitted; use `--models` only to deliberately select a complete subset. |
| `Ranking quality and cost` | Choose `standard` (all patches hosted-ranked with the VLM selected next), `budget` (half hosted-ranked, then student-ranked), or `advanced` (expert controls below). |
| `Hosted VLM` | Choose GPT-5.5, GPT-6 Astra, or Claude Fable 5.1. GPT models use `OPENAI_API_KEY`; Claude uses `ANTHROPIC_API_KEY`. The matching key is requested through hidden input immediately before hosted inference. This prompt is omitted only when the teacher fraction is `0`. |
| `Where should hosted ranking run` | Choose a monitored CPU Slurm job or execution in the current terminal. Slurm is normally appropriate on a cluster. |
| `Fraction ranked by the hosted VLM` | Advanced only. Proportion of patches ranked directly by the hosted model, from `0` to `1`. See the calculation below. |
| `Concurrent VLM workers` | Advanced only. Maximum simultaneous patch-ranking workers. The Slurm ranking job requests the same number of CPUs. |
| `Student ranker` | Shown when the teacher fraction is below `1`. Choose training from this run's teacher labels or provide an existing compatible checkpoint. |
| `Student-training execution` | When training a new student, choose a monitored one-GPU Slurm job or the current process. Slurm is the cluster default. The GPU job performs both training and remaining-patch prediction. |
| `Existing student checkpoint path` | Required only when reusing a checkpoint. Its candidate models, modality vocabulary, and target vocabulary must match the run. |
| `Slurm account` | Cluster project/allocation charged for compute time, not the login username or API account. Ask the cluster administrator if unknown. |
| `Slurm CPU partition` | Queue for the CPU hosted-VLM ranking job. It is requested only when hosted ranking uses Slurm. |
| `Slurm GPU partition` | Queue for built-in segmentation jobs and/or the one-GPU student train/predict job. |
| `Conda initialization script` | Path to `conda.sh`, needed only when VSCell must launch the isolated built-in segmentation environments. |
| `Start this workflow?` | Final authorization after reviewing images, patches, calls, historical cost, resources, privacy, and output paths. Pressing Enter selects the safe default `N`; no job or hosted call starts. |
| Provider API key | Requested through a hidden prompt immediately before hosted inference. Paste the raw key without single or double quote characters. It is not written to workflow state, scripts, or logs. |

Advanced resource overrides are available as command-line flags, including
`--workers`, `--rank-time`, `--rank-memory`, `--student-cpus`,
`--student-time`, and `--student-memory`. The complete choices are repeated in
the final summary before authorization.

### Hosted-VLM teacher fraction

`Fraction ranked by the hosted VLM` is a proportion from `0` to `1`, not a
number of images. VSCell first divides every source image into patches. If
there are `P` patches and the selected fraction is `f`, approximately
`round(P × f)` patches are ranked directly by the hosted VLM. Each hosted patch
is ranked three times, so the planned hosted-call count is:

```text
3 × round(number_of_patches × teacher_fraction)
```

For example, a 16-patch run with fraction `1.0` uses all 16 patches as teacher
patches and plans 48 hosted calls. Fraction `0.5` uses about 8 teacher patches
and plans 24 hosted calls. All patches still contribute to the final mask.

When the fraction is below `1`, the remaining patches must be ranked by the
student ranker. A fraction between `0` and `1` can train that student from
the hosted teacher subset or use a compatible existing checkpoint; this
requires the `student` optional dependencies. Fraction `0` requires an existing
compatible student checkpoint because there are no teacher labels from which
to train. Reducing the fraction lowers hosted calls but makes the result depend
on student generalization. Training from scratch requires at least four teacher
patches drawn from at least two source images so validation can be grouped by
source image without leakage. VSCell checks these requirements during preflight,
before creating workflow state or making hosted calls.

For a very small smoke test, such as one image producing only 16 patches, use
`1.0` to exercise the complete hosted-VLM path and avoid training a student on
too few examples. For a larger dataset, choose the fraction during preflight
after considering the displayed patch count, hosted-call estimate, cost,
privacy requirements, and whether a suitable student checkpoint is available.

Before any job submission or hosted call, it displays one summary containing
the image and patch counts, selected models, three-repetition call count,
historical cost estimate, worker count, Slurm resources, privacy behavior, and
final output path. Nothing starts until the user confirms that summary.

The preflight summary explicitly reports which provider environment-variable
name will be used, but never its value. If the variable is not already present,
VSCell asks for the key through a hidden prompt immediately before hosted
inference. Paste the raw key without surrounding quote characters. A key entered
through this prompt is added only to the running VSCell process, inherited by a
submitted Slurm job when needed, and removed from the parent process immediately
after submission/call setup. It is never written to workflow state, generated
Slurm scripts, or VSCell logs.

If the key variable was exported earlier in the same shell, VSCell deliberately
reuses it and therefore does not prompt again; the summary says that it was
detected. VSCell does not save such a key in a virtual environment. To guarantee
that a new user is prompted for their own credential, clear inherited provider
variables before starting:

```bash
unset OPENAI_API_KEY ANTHROPIC_API_KEY
vscell run
```

Every user must use a key authorized for their own provider account. Do not put
credentials in commands, configuration files, workflow state, Slurm scripts,
logs, issue reports, or a repository.

### Privacy and local identity

Hosted ranking sends the selected source-image crops, rendered anonymous
candidate panels, and generated target/modality instructions to the chosen VLM
provider. VSCell does not put local filenames, filesystem paths, login names,
Slurm accounts, or partition names into the VLM request. Provider SDK and API
terms still apply to the transmitted image content.

Local runtime files do contain operational information needed to resume a run:
`workflow_state.json` records absolute input/output paths, and generated Slurm
scripts record the user-selected account, partition, and executable paths. They
contain no provider key, but they should remain local rather than be committed
or shared if paths or cluster metadata are sensitive. Runtime outputs, logs,
credentials, unapproved local demo images, and third-party weights are excluded
by the repository ignore rules and are not part of the built wheel. The five
approved rendered display panels in `examples/example_results/` are static
GitHub documentation assets, not runtime data or package inputs.

Slurm execution is continuous by default. After displaying the submitted job
ID, VSCell remains in the terminal, reports queue and completed-patch progress,
and prints a periodic `Hosted-VLM inference active` heartbeat while calls are
still in flight. When a new student is requested, VSCell then submits a
monitored one-GPU job that trains the student and predicts rankings for the
remaining patches, with periodic training/prediction activity messages. It
automatically proceeds to consensus and stitching when all required rankings
exist. The command finishes by printing every final-mask path and the audit
path.

For a teacher fraction below `1`, `all_rankings.jsonl` contains the original
hosted-teacher rows plus one student row for every remaining patch. Final
consensus and stitching use that combined file, so teacher-ranked and
student-ranked patches both contribute to the final VSCell masks. With a
teacher fraction of `1`, finalization uses `teacher_rankings.jsonl` directly and
no student job is created.

To deliberately leave a Slurm job running in the background, add `--detach`:

```bash
vscell run --detach
```

VSCell then prints the exact continuation command. It is always equivalent to:

```bash
vscell run --output run --resume
```

The resumable record is `run/workflow_state.json`. Completed segmentation,
preparation, ranking, student, and finalization stages are not repeated. A
partially written teacher-ranking file is continued safely.

For scripts and automated environments, provide all preferences explicitly and
add `--non-interactive --yes`. For example, when full-image candidate masks
already exist:

```bash
read -rsp "OpenAI API key (raw key, no surrounding quotes): " OPENAI_API_KEY
echo
export OPENAI_API_KEY

vscell run \
  --input data \
  --candidates segmentation_results \
  --output run \
  --candidate-source existing \
  --target nuclear \
  --modality ssDNA \
  --teacher-fraction 1.0 \
  --vlm-model gpt-5.5 \
  --workers 16 \
  --execution slurm \
  --account '<ACCOUNT>' \
  --cpu-partition '<CPU_PARTITION>' \
  --non-interactive \
  --yes

unset OPENAI_API_KEY
```

Accuracy is not reported during inference because VSCell does not accept
segmentation ground truth. The completion report contains mask paths, instance
counts, and a full decision audit. Accuracy comparisons against held-out ground
truth remain a separate evaluation step.

## Advanced manual workflow

The individual commands below remain available for expert users, custom job
orchestration, and debugging. They expose the same stages used by `vscell run`
but do not form a continuous process by themselves.

Assume raw images are in `data/`, full-image candidate masks will be in
`segmentation_results/<model>/`, and outputs will be under `run/workspace/`.

### 1. Inspect crop counts and teacher cost

```bash
vscell plan \
  --input data \
  --teacher-fraction 0.25 \
  --vlm-model gpt-5.5
```

The displayed estimate is the projected total for the selected teacher subset,
not the cost of one repetition. Historical empirical costs already include all
three repetitions per teacher patch; actual token usage is recorded by the
provider adapter. The estimate is not a live provider quote.

### 2. Run the four full-image backends or provide existing masks

Run each command in its matching environment:

```bash
vscell backend --backend cellpose     --input data --output segmentation_results --target nuclear --modality ssDNA
vscell backend --backend stardist     --input data --output segmentation_results --target nuclear --modality ssDNA
vscell backend --backend cellsam      --input data --output segmentation_results --target nuclear --modality ssDNA
vscell backend --backend cellpose_sam --input data --output segmentation_results --target nuclear --modality ssDNA
```

All four calls operate on full images. Existing masks are reused unless
`--overwrite` is provided.

For a cluster, render portable scripts and replace the site placeholders:

```bash
vscell make-slurm \
  --input data \
  --candidates segmentation_results \
  --output run/slurm \
  --target nuclear \
  --modality ssDNA
```

The templates intentionally contain no site-specific account or partition
name.

#### Add masks from another segmentation model

VSCell can rank and combine masks from a model that is not built into the
package. Run that model independently on each complete source image and place
its full-image instance masks beside the built-in candidates. The recommended
layout is:

```text
data/
├── image01.tif
└── image02.tif

segmentation_results/
├── cellpose/
│   ├── image01_masks.tif
│   └── image02_masks.tif
├── stardist/
│   ├── image01_masks.tif
│   └── image02_masks.tif
└── my_new_model/
    ├── image01_masks.tif
    └── image02_masks.tif
```

Each mask must be a 2-D TIFF with the same height and width as its source
image. Use `0` for background and a distinct positive integer for each object.
VSCell accepts these filename layouts, where `<stem>` is the source-image
filename without its extension:

```text
segmentation_results/<model>/<stem>_masks.tif   # recommended
segmentation_results/<model>/<stem>.tif
segmentation_results/<stem>__<model>.tif
```

List every candidate, including the external model, when preparing the
workspace:

```bash
vscell prepare \
  --input data \
  --candidates segmentation_results \
  --output run/workspace \
  --target nuclear \
  --modality DAPI \
  --models cellpose,stardist,my_new_model
```

At least three candidate models are required. VSCell crops every listed mask,
automatically updates the candidate count (`N`), anonymizes and ranks the
candidates together, and includes all of them in consensus selection and final
stitching. The teacher/student split divides image patches, not models; every
listed model remains a candidate for every patch.

### 3. Prepare crops and panels

```bash
vscell prepare \
  --input data \
  --candidates segmentation_results \
  --output run/workspace \
  --target nuclear \
  --modality ssDNA

vscell split \
  --manifest run/workspace/run_manifest.json \
  --teacher-fraction 0.25
```

### 4. Rank the teacher subset

Provide keys only through environment variables. To keep the key out of shell
history, enter it with a hidden prompt. Paste the raw key itself, **without
surrounding single or double quote characters**:

```bash
read -rsp "OpenAI API key (raw key, no surrounding quotes): " OPENAI_API_KEY
echo
export OPENAI_API_KEY

# For Claude instead:
# read -rsp "Anthropic API key (raw key, no surrounding quotes): " ANTHROPIC_API_KEY
# echo
# export ANTHROPIC_API_KEY

vscell rank-teacher \
  --manifest run/workspace/run_manifest.json \
  --output run/workspace/teacher_rankings.jsonl \
  --teacher-fraction 0.25 \
  --vlm-model gpt-5.5 \
  --workers 8 \
  --resume

unset OPENAI_API_KEY
# or: unset ANTHROPIC_API_KEY
```

When `read` is used, pasted quote characters become part of the key and cause
an authentication failure. This differs from a direct shell assignment such as
`export OPENAI_API_KEY='sk-...'`, where the shell removes the syntactic quotes.

Supported model identifiers are `gpt-5.5`, `gpt-6-astra`, and
`claude-fable-5-1`. The latter two are the packaged GPT-6 and Claude
alternatives to the default. The selected provider account must have access to
the exact model identifier. Keys and raw images are sent only when this command
is run.

`--workers` is the maximum number of patches ranked concurrently. Each patch
still uses the validated three sequential, independently shuffled repetitions,
and only the main process writes completed records to JSONL. The default is one
worker for conservative compatibility. Increasing the value reduces wall time
but does not change the planned call count or estimated total cost; select a
value compatible with the provider account's rate limits.

For a persistent cluster run, render a CPU Slurm script from the activated
environment. Review the cost with `vscell plan` first and replace the site
placeholders:

```bash
vscell make-rank-slurm \
  --manifest run/workspace/run_manifest.json \
  --rankings run/workspace/teacher_rankings.jsonl \
  --output run/slurm \
  --teacher-fraction 0.25 \
  --vlm-model gpt-5.5 \
  --workers 8 \
  --environment .venv \
  --account '<ACCOUNT>' \
  --cpu-partition '<CPU_PARTITION>'

read -rsp "OpenAI API key (raw key, no surrounding quotes): " OPENAI_API_KEY
echo
export OPENAI_API_KEY
sbatch run/slurm/run_rank_teacher.slurm
unset OPENAI_API_KEY
```

The generated script contains no API key. Slurm inherits the exported key at
submission through the job environment; never paste a key into the script or a
log file. The script uses `--resume`, so completed patch records are preserved
across job restarts.

`make-rank-slurm` is an advanced, non-interactive script renderer: it does not
ask preferences, submit a job, monitor it, or run finalization. It reports the
model, worker count, and teacher fraction written into the script. `sbatch`
then returns immediately by Slurm design. A manual user must inspect
`squeue -j <JOB_ID>` and `run/slurm/logs/`, wait for successful completion, and
run the finalization command below. Use `vscell run` instead when confirmation,
monitoring, and automatic continuation are desired.

### 5. Optional student

For a mixed teacher/student run:

```bash
vscell train-student \
  --manifest run/workspace/run_manifest.json \
  --teacher-rankings run/workspace/teacher_rankings.jsonl \
  --teacher-fraction 0.25 \
  --checkpoint run/workspace/student.pt \
  --device auto

vscell predict-student \
  --manifest run/workspace/run_manifest.json \
  --teacher-rankings run/workspace/teacher_rankings.jsonl \
  --teacher-fraction 0.25 \
  --checkpoint run/workspace/student.pt \
  --output run/workspace/all_rankings.jsonl \
  --device auto
```

For 100% teacher, use `teacher_rankings.jsonl` directly and skip the student.
For 0% teacher, `train-student` is invalid; provide a checkpoint whose recorded
candidate models, modality vocabulary, and target vocabulary match the run.

### 6. Consensus and stitching

```bash
vscell finalize \
  --manifest run/workspace/run_manifest.json \
  --rankings run/workspace/all_rankings.jsonl
```

For a 100% teacher run, replace `all_rankings.jsonl` with
`teacher_rankings.jsonl`.

Primary outputs are:

- `teacher_rankings.jsonl` and/or `all_rankings.jsonl`;
- `vscell_patch_masks/*.tif`;
- `final_masks/*_vscell.tif`, with globally reassigned instance labels;
- `final_audit.json`, including every ranking, selector action, stitch decision,
  configuration value, and the explicit statement that segmentation GT was not
  used.

## Local five-image CellBinDB demo

Five 256×256 ssDNA images were selected from prior results across kidney, heart,
uterus, liver, and brain. They are available in the existing local CellBinDB
copy, so no download is needed:

```bash
python examples/prepare_cellbindb_demo.py \
  --cellbindb-root /path/to/CellBinDB \
  --index /path/to/cellbindb_index.csv
```

`examples/data/` is gitignored. The full raw CellBinDB demo files remain local.
The repository contains only the five separately approved rendered display
panels in `examples/example_results/`; their data reference and post-hoc-only
evaluation role are documented with the assets.

## Scientific defaults

Solution 08 uses anchor support IoU 0.50, residual complete-link IoU 0.50,
support from two distinct peers, anchor-null threshold 0.10, refinement IoU
0.50, and two other refining peers. `minimum_area=1` reproduces the GT-free
full-image Figure 2 run. No ground-truth mask is accepted by the inference API.

## Tests

```bash
python -m pytest
```

Tests cover adaptive grid examples, historical padding, deterministic ranking,
Solution 08 selection, boundary-aware stitching, prompts, and a synthetic
workspace round trip. Instance relabeling is also checked to guarantee that a
single numeric label cannot represent disconnected cells.
