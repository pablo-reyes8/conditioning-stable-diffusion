FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements/base.txt requirements/data.txt /tmp/requirements/
RUN pip install --no-cache-dir -r /tmp/requirements/data.txt

COPY src ./src
COPY scripts ./scripts
COPY config ./config
COPY data ./data

ENTRYPOINT ["python", "scripts/data.py"]
