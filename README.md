# Stable Diffusion Face Attribute Project

Attribute-conditioned latent diffusion pipeline for face generation, including dataset ingestion, model training, and inference CLIs. The repository is structured to be source-control friendly: configuration is externalized to YAML, heavy artifacts are ignored selectively, and operational workflows are separated into data, training, and inference entrypoints.

## Repository structure

```text
config/
  data/         Dataset ingestion settings
  models/       UNet architecture presets
  training/     Training experiment definitions
  inference/    Inference presets
data/
  raw/          Source metadata and downloaded archives
  processed/    Generated manifests and filtered archives
  reports/      JSON reports emitted by the data CLI
scripts/
  data.py       Dataset download and ingestion CLI
  train.py      Training CLI
  infer.py      Inference CLI
src/
  data/         Ingestion logic and dataset constants
  model/        Diffusion, UNet, attention, VAE wrappers
  training/     Training loop, EMA, schedules, checkpoints
  inference/    DDPM and DDIM samplers
tests/
  data/         Dataset ingestion tests
  model/        Core model and diffusion tests
  training/     Scheduler and checkpoint tests
  inference/    Sampler tests
docker/
  data.Dockerfile
  training.Dockerfile
  inference.Dockerfile
```

## Setup

Install the full development environment:

```bash
python3 -m pip install -r requirements.txt
```

If you only need a subset:

```bash
python3 -m pip install -r requirements/data.txt
python3 -m pip install -r requirements/training.txt
python3 -m pip install -r requirements/inference.txt
```

## Data workflow

1. Configure paths in [config/data/maad_face.yaml](/mnt/c/Users/alejo/OneDrive/Escritorio/Pablo/Profesional/Modelaje/Modelos%20de%20DL/stable-difussion/config/data/maad_face.yaml).
2. Download the source archive:

```bash
python3 scripts/data.py download --config config/data/maad_face.yaml --url "<DATASET_URL>"
```

3. Build the balanced manifest:

```bash
python3 scripts/data.py build-manifest --config config/data/maad_face.yaml
```

4. Create the filtered training archive:

```bash
python3 scripts/data.py filter-archive --config config/data/maad_face.yaml
```

The CLI generates JSON reports under `data/reports/` for reproducibility and review.

## Training

The default experiment is defined in [config/training/maad_256.yaml](/mnt/c/Users/alejo/OneDrive/Escritorio/Pablo/Profesional/Modelaje/Modelos%20de%20DL/stable-difussion/config/training/maad_256.yaml), which references [config/models/unet_latent_256.yaml](/mnt/c/Users/alejo/OneDrive/Escritorio/Pablo/Profesional/Modelaje/Modelos%20de%20DL/stable-difussion/config/models/unet_latent_256.yaml).

```bash
python3 scripts/train.py --config config/training/maad_256.yaml
```

Key runtime concerns:

- Update `device` to `cuda` before GPU training.
- Point `checkpoint.dir` to a writable artifact directory.
- Ensure `vae.name` is reachable from the local environment or Hugging Face cache.

## Inference

DDIM preset:

```bash
python3 scripts/infer.py --config config/inference/ddim_256.yaml
```

DDPM preset:

```bash
python3 scripts/infer.py --config config/inference/ddpm_256.yaml
```

Both presets load the checkpoint path declared in the YAML and optionally apply EMA weights during sampling.

## Testing

```bash
python3 -m pytest
```

The tests focus on deterministic utilities, checkpointing, ingestion logic, and inference/training behavior that can run without the full dataset.

## Docker

Each operational surface has its own image:

```bash
docker build -f docker/data.Dockerfile -t stable-diffusion-data .
docker build -f docker/training.Dockerfile -t stable-diffusion-train .
docker build -f docker/inference.Dockerfile -t stable-diffusion-infer .
```

Typical usage:

```bash
docker run --rm -v "$(pwd)/data:/app/data" stable-diffusion-data build-manifest --config config/data/maad_face.yaml
docker run --gpus all --rm -v "$(pwd):/app" stable-diffusion-train --config config/training/maad_256.yaml
docker run --gpus all --rm -v "$(pwd):/app" stable-diffusion-infer --config config/inference/ddim_256.yaml
```

## Notes

- Large datasets, generated archives, checkpoints, and sample outputs are ignored explicitly in `.gitignore`.
- The legacy [data/filter_data.py](/mnt/c/Users/alejo/OneDrive/Escritorio/Pablo/Profesional/Modelaje/Modelos%20de%20DL/stable-difussion/data/filter_data.py) file now delegates to the new CLI.
- Notebook experimentation is preserved under `notebooks/`, but repository workflows no longer depend on notebooks.
