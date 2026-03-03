FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements/base.txt requirements/training.txt requirements/inference.txt /tmp/requirements/
RUN pip install --no-cache-dir -r /tmp/requirements/inference.txt

COPY src ./src
COPY scripts ./scripts
COPY config ./config

ENTRYPOINT ["python", "scripts/infer.py"]
