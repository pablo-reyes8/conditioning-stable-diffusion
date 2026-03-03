# Evaluation Inputs

Place real reference images for offline evaluation under `data/evaluation/reference_images/`.

The evaluation CLI compares:

- `generated_dir`: generated images from inference or training samples
- `real_dir`: reference real images for FID, KID, and face detection baselines

Example:

```bash
python3 scripts/evaluate.py --config config/evaluation/maad_face_eval.yaml
```
