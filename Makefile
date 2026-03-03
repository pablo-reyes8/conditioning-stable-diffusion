PYTHON ?= python3

.PHONY: test data-manifest data-filter train infer-ddim infer-ddpm evaluate

test:
	$(PYTHON) -m pytest

data-manifest:
	$(PYTHON) scripts/data.py build-manifest --config config/data/maad_face.yaml

data-filter:
	$(PYTHON) scripts/data.py filter-archive --config config/data/maad_face.yaml

train:
	$(PYTHON) scripts/train.py --config config/training/maad_256.yaml

infer-ddim:
	$(PYTHON) scripts/infer.py --config config/inference/ddim_256.yaml

infer-ddpm:
	$(PYTHON) scripts/infer.py --config config/inference/ddpm_256.yaml

evaluate:
	$(PYTHON) scripts/evaluate.py --config config/evaluation/maad_face_eval.yaml
