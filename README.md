# Conditioning Stable Diffusion  
**Attribute-conditioned latent diffusion for face generation (CFG + cross-attention), with a reproducible, config-driven workflow.**

<p align="left">
  <a href="https://github.com/pablo-reyes8/conditioning-stable-diffusion"><img alt="Repo" src="https://img.shields.io/badge/GitHub-conditioning--stable--diffusion-181717?logo=github&logoColor=white"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-Research%20Only-yellow">
</p>

![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/conditioning-stable-diffusion)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/conditioning-stable-diffusion)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/conditioning-stable-diffusion)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/conditioning-stable-diffusion?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/conditioning-stable-diffusion?style=social)

**What this is:** a “from-scratch-ish” **latent diffusion** training + sampling stack (UNet + VAE latents) that generates faces **conditioned on binary attributes** (e.g., *Smiling*, *Young*, *Male*) using **Classifier-Free Guidance (CFG)**.  

The entire workflow is reproducible from versioned **YAML configs** and **CLIs** (data → training → inference → evaluation), not notebooks.

> **Training status (WIP):** this repo is actively training and improving results. The **512px model** is a ~**300M-parameter** setup that typically requires **≥40GB VRAM**, and it’s currently being trained on an **H100**. Early samples already show the model is learning facial structure and attribute-relevant cues, but **quality is still improving**. Current progress: **54 epochs** (and counting).

---

## Sample outputs

<p align="center">
  <img src="inference/samples_54epoc/samples_ddpm2.png" width="750" alt="Training samples (512px)"/>
</p>

> The grid above is emitted by the training loop as a checkpoint-time sanity check.  
> See more: [`training_progress/`](training_progress/)

---

## Table of contents

- [Why this project](#why-this-project)
- [Key features](#key-features)
- [How conditioning works](#how-conditioning-works)
- [Project layout](#project-layout)
- [Quick start](#quick-start)
- [Data provenance & ethics](#data-provenance--ethics)
- [Data pipeline](#data-pipeline)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Configuration system](#configuration-system)
- [Reproducibility checklist](#reproducibility-checklist)
- [Testing](#testing)
- [Docker](#docker)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)

---

## Why this project

Most diffusion repos optimize for quick demos. This one optimizes for **research workflow hygiene**:

- **Config-driven runs** (YAML) so you can re-run experiments exactly.
- **Explicit artifacts** (manifest, filtered archive, reports, checkpoints, samples, metrics).
- **Separation of concerns**: training does not “silently” evaluate; evaluation is a separate step.
- **Conditioning as a first-class primitive**: attributes are passed through a **label encoder** and used via **cross-attention + CFG**.

If you want to read this like a paper artifact, start at:  
**Data → Training → Inference → Evaluation**, each with a CLI entrypoint.

---

## Key features

### Modeling
- **Latent diffusion**: images are encoded with a pre-trained VAE (Stable Diffusion VAE), and the UNet denoises in latent space.
- **Attribute conditioning**:
  - binary attribute vector (e.g., `Smiling: 1`, `Young: 1`)
  - **Classifier-Free Guidance (CFG)** with configurable guidance scale
  - conditioning injected via **cross-attention** (UNet has `use_cross_attn: true`, `context_dim: 256`)
- **Training stability**
  - EMA weights
  - mixed precision (bf16 preferred on A100-class GPUs)
  - gradient clipping + optional OOM skip behavior

### Engineering
- CLI pipeline:
  - `scripts/data.py` → manifests, filtering, reports
  - `scripts/train.py` → train loop + checkpoints + sample grids
  - `scripts/infer.py` → DDPM/DDIM sampling (+ CFG) for fixed attribute settings
  - `scripts/evaluate.py` → FID/KID/IS + face detectability (MTCNN)
- Modular code layout under `src/` with tests under `tests/`
- Docker images for each stage (data/training/inference/evaluation)

---

## How conditioning works

At a high level, the model learns **p(x | a)** where:
- **x** is a face image (in VAE latent space during training)
- **a** is a binary attribute vector (e.g., 11 attributes)

### (1) Conditioning signal
Attributes are encoded into a continuous representation (context) and fed to the UNet via cross-attention.

### (2) Classifier-Free Guidance (CFG)
During training, we randomly drop the conditioning vector with probability `cfg_drop_prob` (unconditional branch).  
At sampling time, we run:

$$
\epsilon_\text{cfg}(z_t, a) = \epsilon_\theta(z_t, \varnothing) + s \cdot \big(\epsilon_\theta(z_t, a) - \epsilon_\theta(z_t, \varnothing)\big),
$$

where:
- $\epsilon_\theta(z_t, a)$ is the conditional noise prediction
- $\epsilon_\theta(z_t, \varnothing)$ is the unconditional prediction
- $s$ is `guidance_scale`

**Practical intuition:** higher `guidance_scale` enforces the attributes harder, but too high can reduce diversity / introduce artifacts.

---

## Project layout

```text
config/
  data/         Dataset provenance and ingestion settings
  models/       UNet architecture presets
  training/     Training experiments
  inference/    Inference presets
  evaluation/   Evaluation presets
data/
  raw/          Original metadata and source archives
  processed/    Balanced manifests and filtered archives
  reports/      JSON artifacts emitted by the data CLI
docker/
  data.Dockerfile
  training.Dockerfile
  inference.Dockerfile
  evaluation.Dockerfile
scripts/
  data.py       Data ingestion CLI
  train.py      Training CLI
  infer.py      Inference CLI
  evaluate.py   Evaluation CLI
src/
  data/         Ingestion logic and dataset constants
  model/        Diffusion, UNet, attention, label encoder, VAE wrapper
  training/     Schedules, checkpoints, EMA, train loop
  inference/    DDPM and DDIM samplers (+ CFG)
  evaluation/   FID/KID/IS evaluation and face detection pipeline
tests/
  data/
  model/
  training/
  inference/
  evaluation/
training_progress/
  training_samples_256/   Sample grids emitted during training
  training_samples_512/
```

---

## Quick start

### 1) Install dependencies

Full environment:

```bash
python3 -m pip install -r requirements.txt
```

Or stage-specific installs:

```bash
python3 -m pip install -r requirements/data.txt
python3 -m pip install -r requirements/training.txt
python3 -m pip install -r requirements/inference.txt
python3 -m pip install -r requirements/evaluation.txt
```

### 2) Run the pipeline (end-to-end)

```bash
# (A) Build a balanced manifest from MAAD-Face metadata
python3 scripts/data.py build-manifest --config config/data/maad_face.yaml

# (B) Filter the VGGFace2 archive to the manifest subset
python3 scripts/data.py filter-archive --config config/data/maad_face.yaml

# (C) Train latent diffusion (UNet in latent space, CFG dropout, EMA)
python3 scripts/train.py --config config/training/maad_256.yaml

# (D) Sample with DDIM + CFG for a chosen attribute set
python3 scripts/infer.py --config config/inference/ddim_256.yaml

# (E) Evaluate generated images offline (FID/KID/IS + MTCNN detectability)
python3 scripts/evaluate.py --config config/evaluation/maad_face_eval.yaml
```

---

## Data provenance & ethics

This repo is designed to **operate locally** on two upstream resources:

| Component | Role in this repo | Where to obtain |
| --- | --- | --- |
| **MAAD-Face** | Attribute annotations & metadata used to build a balanced manifest | https://github.com/pterhoer/MAAD-Face |
| **VGGFace2 (archive)** | Source images to assemble the training subset | https://www.kaggle.com/datasets/hearfool/vggface2 |

Expected local paths (defaults):

- Metadata CSV: `data/raw/metadata/MAAD_Face.csv`
- Image archive: `data/raw/downloads/vggface2.zip`

> **Important (portfolio / publication):** Please review the upstream dataset licenses/terms before redistributing any images, metadata, or derived artifacts.  
> This repository intentionally ignores large datasets, filtered archives, checkpoints, and generated images in `.gitignore`.

---

## Data pipeline

Artifacts are explicit and versionable:

1. Raw metadata → `data/raw/metadata/`
2. Raw archive → `data/raw/downloads/`
3. Balanced manifest → `data/processed/manifests/`
4. Filtered ZIP archive → `data/processed/archives/`
5. JSON reports → `data/reports/`

Commands:

```bash
python3 scripts/data.py download --config config/data/maad_face.yaml --url "<DATASET_URL>"
python3 scripts/data.py build-manifest --config config/data/maad_face.yaml
python3 scripts/data.py filter-archive --config config/data/maad_face.yaml
```

The data CLI emits **machine-readable reports** (balancing + archive filter) to make dataset handling auditable.

---

## Training

Default experiment (256px) config:

- Training: `config/training/maad_256.yaml`
- Model: `config/models/unet_latent_256.yaml`

Run:

```bash
python3 scripts/train.py --config config/training/maad_256.yaml
```

Before a serious run, check:

- `device: cuda` (in config)
- `data.archive_path` and `data.manifest_path`
- checkpoint directory is writable (`checkpoint.dir`)
- VAE weights are accessible (Hugging Face model name is set in the config)
- choose AMP dtype:
  - **bf16** recommended on A100/modern GPUs
  - fp16 if bf16 unavailable

---

## Inference

Presets:

```bash
python3 scripts/infer.py --config config/inference/ddim_256.yaml
python3 scripts/infer.py --config config/inference/ddpm_256.yaml
```

What inference does:

- loads the configured checkpoint
- restores EMA (if enabled)
- samples **n** images using DDIM or DDPM
- applies **CFG** via `guidance_scale`
- writes a grid to `out_path` (and optionally individual images)

**Editing attributes (example):**

In `config/inference/ddim_256.yaml`:

```yaml
inference:
  guidance_scale: 7.5
  attributes:
    Smiling: 1
    Young: 1
```

---

## Evaluation

Evaluation is separate on purpose: it scores already-generated images without touching training code.

Supported metrics:

- FID
- KID
- Inception Score (IS)
- Face detection pass-through with a pre-trained MTCNN detector

Run:

```bash
python3 scripts/evaluate.py --config config/evaluation/maad_face_eval.yaml
```

The evaluation config declares:

- `generated_dir`: generated images
- `real_dir`: reference images
- `distribution_metrics`: FID/KID/IS
- `face_detection`: MTCNN options
- `output_path`: JSON summary artifact

---

## Configuration system

The repository is **configuration-driven**:

- `config/data/maad_face.yaml` → paths, provenance, attribute list, sampling/balancing knobs  
- `config/models/*.yaml` → UNet architecture presets  
- `config/training/*.yaml` → training hyperparameters, EMA, CFG dropout  
- `config/inference/*.yaml` → sampler (DDPM/DDIM), steps/eta, guidance scale, attribute dict  
- `config/evaluation/*.yaml` → metrics + face detectability checks  

This makes notebook experimentation optional—not the source of truth.

---

## Reproducibility checklist

If you want a run to be re-creatable by someone else:

- [ ] commit the YAML configs used  
- [ ] log the exact checkpoint path + git commit hash  
- [ ] store the emitted JSON reports (`data/reports/`)  
- [ ] store the evaluation JSON summary (`output_path`)  
- [ ] record sampling seed + sampler (DDIM/DDPM) + steps + guidance scale  

---

## Testing

```bash
python3 -m pytest -s
```

Tests cover:

- ingestion + artifact generation
- diffusion utilities + model blocks
- checkpointing + training loop behavior
- DDPM/DDIM sampling helpers
- evaluation path handling + summaries

---

## Docker

Build:

```bash
docker build -f docker/data.Dockerfile -t csd-data .
docker build -f docker/training.Dockerfile -t csd-train .
docker build -f docker/inference.Dockerfile -t csd-infer .
docker build -f docker/evaluation.Dockerfile -t csd-eval .
```

Run (examples):

```bash
docker run --rm -v "$(pwd)/data:/app/data" csd-data build-manifest --config config/data/maad_face.yaml
docker run --gpus all --rm -v "$(pwd):/app" csd-train --config config/training/maad_256.yaml
docker run --gpus all --rm -v "$(pwd):/app" csd-infer --config config/inference/ddim_256.yaml
docker run --gpus all --rm -v "$(pwd):/app" csd-eval --config config/evaluation/maad_face_eval.yaml
```

---

## Roadmap

Ideas that would make this repo even stronger as a research artifact:

- [ ] log metrics + samples to a tracker (Weights & Biases / MLflow)
- [ ] add a small “ablation” suite:
  - CFG scale sweep
  - DDIM eta sweep
  - conditioning dropout sweep
- [ ] add attribute-consistency metrics (classifier-based or CLIP-based)
- [ ] add a small model card describing limitations, intended use, and risks
- [ ] export a minimal inference package (single command, minimal deps)

---

## Acknowledgements

This project is inspired by foundational diffusion work and the Stable Diffusion / Latent Diffusion ecosystem:

- Denoising Diffusion Probabilistic Models (DDPM)
- DDIM sampling
- Classifier-Free Guidance (CFG)
- Latent Diffusion Models / Stable Diffusion tooling (VAE latents)

Datasets used operationally (local-only ingestion):

- MAAD-Face (attributes/metadata)
- VGGFace2 (images)

---

### Repo notes

- Large datasets, filtered archives, checkpoints, generated images, and reports are ignored in `.gitignore`.
- `data/filter_data.py` is a compatibility wrapper; the maintained workflow is the CLI in `scripts/data.py`.
