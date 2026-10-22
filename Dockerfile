# Package the CPU inference service with tested dependencies.
# Run as a non-root user and expose a readiness health check.

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HOME=/home/appuser

WORKDIR /app

# Install the CPU runtime library and create the application user.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

# Install dependencies before copying source to reuse dependency layers.
COPY requirements-tested.txt ./

RUN python -m pip install --no-cache-dir \
        "torch==2.14.1+cpu" \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir \
        -r requirements-tested.txt

COPY pyproject.toml ./
COPY src/ ./src/

RUN python -m pip install --no-cache-dir --no-deps . \
    && python -m pip check

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).close()"

ENTRYPOINT ["python", "-m", "adaptive_inference.serve"]

CMD ["--host", "0.0.0.0", "--port", "8000", "--batch-size", "8", "--batch-wait-ms", "0", "--torch-threads", "1"]