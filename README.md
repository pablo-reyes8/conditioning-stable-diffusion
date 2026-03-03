# Conditioning Stable Diffusion

Attribute-conditioned latent diffusion pipeline for face generation, dataset ingestion, training, inference, and evaluation. The repository is organized so the operational workflow is reproducible from config files instead of notebooks: data preparation, experiment settings, sampling, and evaluation all live in versioned code and YAML.

## Table of Contents

1. [Overview](#overview)
2. [Project Layout](#project-layout)
3. [Quick Start](#quick-start)
4. [Data Provenance](#data-provenance)
5. [Data Pipeline](#data-pipeline)
6. [Training](#training)
7. [Inference](#inference)
8. [Evaluation](#evaluation)
9. [Configuration](#configuration)
10. [Testing](#testing)
11. [Docker](#docker)
12. [Repository Notes](#repository-notes)

## Overview

This project contains:

- A structured data ingestion workflow for MAAD-Face metadata and the local VGGFace2 image archive
- A latent diffusion training stack with configurable DDPM/DDIM sampling utilities
- Command-line entrypoints for data preparation, training, inference, and evaluation
- A dedicated evaluation pipeline for standard generative metrics and face detectability checks with a pre-trained detector

## Project Layout

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
  inference/    DDPM and DDIM samplers
  evaluation/   FID/KID/IS evaluation and face detection pipeline
tests/
  data/
  model/
  training/
  inference/
  evaluation/
```

## Quick Start

Install the full environment:

```bash
python3 -m pip install -r requirements.txt
```

Or install only the part you need:

```bash
python3 -m pip install -r requirements/data.txt
python3 -m pip install -r requirements/training.txt
python3 -m pip install -r requirements/inference.txt
python3 -m pip install -r requirements/evaluation.txt
```

Common commands:

```bash
python3 scripts/data.py build-manifest --config config/data/maad_face.yaml
python3 scripts/train.py --config config/training/maad_256.yaml
python3 scripts/infer.py --config config/inference/ddim_256.yaml
python3 scripts/evaluate.py --config config/evaluation/maad_face_eval.yaml
```

## Data Provenance

This repository uses two external data sources in the workflow:

| Component | Role in this repository | Source |
| --- | --- | --- |
| MAAD-Face | Attribute annotations and metadata used to build the balanced manifest | [MAAD-Face official repository](https://github.com/pterhoer/MAAD-Face) |
| VGGFace2 archive | Source images used to assemble the training archive | [Kaggle VGGFace2 mirror](https://www.kaggle.com/datasets/hearfool/vggface2) |

Practical notes:

- The project expects the metadata table at `data/raw/metadata/MAAD_Face.csv`.
- The image archive is expected locally as `data/raw/downloads/vggface2.zip`.
- The Kaggle archive is treated as the local operational source for this repository, but you should still verify the upstream usage terms and licensing before redistribution or publication.
- Dataset provenance fields are also tracked in [`config/data/maad_face.yaml`](config/data/maad_face.yaml).

## Data Pipeline

The data workflow is split into explicit artifacts:

1. Raw metadata in `data/raw/metadata/`
2. Raw archive in `data/raw/downloads/`
3. Balanced manifest in `data/processed/manifests/`
4. Filtered ZIP archive in `data/processed/archives/`
5. JSON reports in `data/reports/`

Commands:

```bash
python3 scripts/data.py download --config config/data/maad_face.yaml --url "<DATASET_URL>"
python3 scripts/data.py build-manifest --config config/data/maad_face.yaml
python3 scripts/data.py filter-archive --config config/data/maad_face.yaml
```

The data CLI now emits machine-readable reports for sampling balance and archive filtering instead of relying on an ad-hoc notebook script.

## Training

The default 256px training experiment is defined in [`config/training/maad_256.yaml`](config/training/maad_256.yaml) and references [`config/models/unet_latent_256.yaml`](config/models/unet_latent_256.yaml).

```bash
python3 scripts/train.py --config config/training/maad_256.yaml
```

Before running a real experiment:

- Switch `device` from `cpu` to `cuda`
- Confirm `data.archive_path` and `data.manifest_path`
- Set a writable checkpoint directory in `checkpoint.dir`
- Ensure the configured VAE weights are available locally or through Hugging Face

## Inference

Available presets:

```bash
python3 scripts/infer.py --config config/inference/ddim_256.yaml
python3 scripts/infer.py --config config/inference/ddpm_256.yaml
```

Both presets load the checkpoint path declared in YAML, restore EMA when configured, and write image grids into the output path defined in the inference config.

## Evaluation

The evaluation flow is intentionally separate from training and inference. It is designed to score already-generated images without modifying the training codepath.

The evaluation module currently supports:

- Frechet Inception Distance (FID)
- Kernel Inception Distance (KID)
- Inception Score (IS)
- Face detection pass-through using a pre-trained MTCNN detector

Default preset:

```bash
python3 scripts/evaluate.py --config config/evaluation/maad_face_eval.yaml
```

The evaluation config lets you declare:

- `generated_dir`: folder with generated images
- `real_dir`: reference real-image folder
- `distribution_metrics`: standard generative metrics
- `face_detection`: pre-trained face detector settings
- `output_path`: JSON summary artifact

This CLI is only wired here; nothing in the repository runs evaluation automatically.

## Configuration

The repository is configuration-driven:

- [`config/data/maad_face.yaml`](config/data/maad_face.yaml) defines data provenance, paths, and ingestion parameters
- [`config/models/`](config/models/) contains architecture presets
- [`config/training/`](config/training/) defines experiment settings
- [`config/inference/`](config/inference/) defines sampling presets
- [`config/evaluation/`](config/evaluation/) defines offline evaluation runs

This keeps notebook experimentation optional instead of being the source of truth.

## Testing

Run the suite with:

```bash
python3 -m pytest -s
```

The tests cover:

- Ingestion and artifact generation
- Diffusion utilities and model blocks
- Checkpointing and training loop behavior
- DDPM/DDIM sampling helpers
- Evaluation path handling and summaries

## Docker

Each workflow has its own image:

```bash
docker build -f docker/data.Dockerfile -t csd-data .
docker build -f docker/training.Dockerfile -t csd-train .
docker build -f docker/inference.Dockerfile -t csd-infer .
docker build -f docker/evaluation.Dockerfile -t csd-eval .
```

Example usage:

```bash
docker run --rm -v "$(pwd)/data:/app/data" csd-data build-manifest --config config/data/maad_face.yaml
docker run --gpus all --rm -v "$(pwd):/app" csd-train --config config/training/maad_256.yaml
docker run --gpus all --rm -v "$(pwd):/app" csd-infer --config config/inference/ddim_256.yaml
docker run --gpus all --rm -v "$(pwd):/app" csd-eval --config config/evaluation/maad_face_eval.yaml
```

## Repository Notes

- Large datasets, filtered archives, checkpoints, generated images, and reports are ignored explicitly in `.gitignore`.
- [`data/filter_data.py`](data/filter_data.py) remains only as a compatibility wrapper; the maintained workflow is the CLI under `scripts/data.py`.
- Notebooks are still available for exploration, but they are no longer the operational entrypoint for the repository.
