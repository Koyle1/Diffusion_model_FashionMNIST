# Fashion-MNIST diffusion implementation

A PyTorch denoising-diffusion implementation with a timestep-conditioned residual
U-Net, attention blocks, noise-prediction loss and reverse sampling. This is an
implementation study, not a pretrained model or a claim of benchmark-quality
image generation.

## Setup

Use a Python version supported by your PyTorch installation. CPU verification
was run with Python 3.14, PyTorch 2.15.0, torchvision 0.29.2 and pytest 9.2.0.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. For a specific CUDA
installation, use the appropriate PyTorch installation command first.

## Quick smoke check

```sh
python train.py --smoke-test --device cpu
```

This uses a tiny version of the same architecture, synthetic 16-by-16 inputs,
four diffusion steps, one epoch and two batches per train/evaluation pass. It
does not download Fashion-MNIST. It performs real optimizer updates, saves a
checkpoint and writes a sample PNG under `artifacts/smoke/`. Those outputs are
separate from normal-training defaults so a smoke run cannot overwrite a normal
training checkpoint. A user-supplied output path is respected.

Smoke losses and generated images are integration evidence only, not
Fashion-MNIST performance or generative-quality results.

## Train from scratch

```sh
python train.py --device auto --epochs 20 \
  --checkpoint artifacts/fashion_run/model.pt \
  --sample-output artifacts/fashion_run/samples.png
```

Starting without `--resume` creates a new model; an existing checkpoint is not
required or loaded implicitly. The first normal run downloads Fashion-MNIST to
`data/`. Use `--data-dir PATH` to change that location and `--no-download` to
require a local copy. Without explicit output flags, normal training writes
`artifacts/diffusion_model.pt` and its adjacent sample PNG. Choose distinct
output paths for distinct experiments; explicitly chosen paths are overwritten.

`--device auto` selects CUDA, then Apple MPS, then CPU. Explicit unavailable
accelerators raise an error. Only CPU execution was verified in this repair.
Use `--max-batches N` and `--epochs 1` for a bounded dataset-backed trial.
`python train.py --help` lists all options.

Training data use the original resize, flip/rotation and normalisation pipeline.
The held-out split uses deterministic preprocessing without augmentation and
reports noise-prediction MSE. This loss alone does not establish sample quality;
no FID, visual-quality benchmark or downstream scientific result is asserted.

## Resume training

```sh
python train.py --resume artifacts/fashion_run/model.pt --epochs 5 \
  --checkpoint artifacts/fashion_run/resumed.pt \
  --sample-output artifacts/fashion_run/resumed_samples.png
```

`--epochs` means additional epochs. Resume restores the saved architecture,
diffusion schedule, AdamW hyperparameters and optimizer state bound to the
restored model parameters. Fresh-start `--lr` and `--weight-decay` do not replace
the resumed optimizer values. Conflicting explicit image size or schedule length
is rejected. Missing checkpoints fail before dataset construction/download.

If `--checkpoint` is omitted on a resume run, a unique output under
`artifacts/resume/` (or `artifacts/smoke/`) is chosen; the source checkpoint is
not overwritten. Pass the same source path explicitly only when overwriting it
is intended. Data-loader/global RNG states are not restored, so resume is **not**
a claim of bit-exact continuation of a shuffled or augmented training run.

Checkpoints contain tensors and primitive configuration dictionaries and are
loaded with `weights_only=True`. Legacy files containing pickled dataclass
objects are deliberately rejected; regenerate them using this version. Do not
enable unrestricted pickle loading for untrusted downloaded checkpoints. AdamW
is the supported resumable optimizer.

## Tests and complete bounded verification

```sh
python -m pytest -q tests --basetemp artifacts/pytest
python verify_repair.py
```

The verifier runs CPU tests plus separate fresh-start/resume CLI smoke commands,
checks safe checkpoint loading and advancing optimizer counters, then emits
`FASHION_REPAIR_VERIFIED` only after all checks pass. It uses temporary files
inside ignored `artifacts/`, writes a small verification summary there, and
requires no network or dataset download. Tests cover the default architecture's
training step as well as the tiny smoke architecture, safe checkpoint round trips,
the next resumed optimizer update, missing/incompatible checkpoints, finite
reverse samples, output-path isolation, evaluation behavior and device fallback.

Generated data, checkpoints, images and verification artifacts are not intended
for version control. Existing tracked files are not deleted by the ignore rules.
