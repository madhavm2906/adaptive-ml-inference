# Adaptive ML Inference Service

[![Python CI](https://github.com/madhavm2906/adaptive-ml-inference/actions/workflows/ci.yml/badge.svg)](https://github.com/madhavm2906/adaptive-ml-inference/actions/workflows/ci.yml)

A CPU inference service with bounded request queues, deadline-aware dynamic
batching, automated correctness tests, Docker packaging, and reproducible
HTTP performance experiments.

The service combines individual HTTP requests into model batches and returns
each score vector to its original caller.

The current model is a reproducible, untrained PyTorch neural network.
It provides a numerical workload for evaluating serving infrastructure.
Its outputs are raw scores without trained class meaning or accuracy claims.

## Implemented features

- Strict validation of 64 numerical input features.
- Ten output scores per request.
- Reproducible CPU model initialization.
- One dedicated inference execution thread.
- Bounded waiting queue with immediate overload rejection.
- Configurable maximum batch size and collection window.
- Per-request deadlines covering queueing, batching, and execution.
- Skipping expired and cancelled requests before execution.
- Correct result routing across concurrent requests.
- Worker recovery after processor failures.
- Graceful queue draining during shutdown.
- Separate liveness and readiness endpoints.
- Server configuration and environment metadata.
- Numerically validated HTTP benchmark records.
- GitHub Actions tests and downloadable test evidence.
- Non-root Docker image with a readiness health check.

Batch sizes vary with available requests within configured limits.
The service does not automatically tune the configured batch limit or
waiting window based on traffic.

## Architecture

```text
HTTP clients
    |
    v
FastAPI request validation
    |
    v
Bounded FIFO waiting queue
    |
    v
Batch collection: size limit, waiting window, request deadlines
    |
    v
One dedicated execution thread
    |
    v
PyTorch CPU model
    |
    v
Per-request result futures
    |
    v
Individual HTTP responses
```

The neural network has layers:

```text
64 inputs -> 256 hidden -> ReLU -> 256 hidden -> ReLU -> 10 scores
```

Inference uses evaluation mode with gradient tracking disabled.

## Requirements

- Linux, including Ubuntu under WSL 2.
- Python 3.12.
- CPU-only PyTorch.
- Docker for container execution.

The tested environment used Ubuntu 24.04 under WSL 2.

## Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

python -m pip install "torch==2.14.1+cpu" \
  --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
python -m pip check
```

`requirements-tested.txt` records the installed dependency versions used
during verification. It is a package-version snapshot, not a hash-verified
cross-platform lockfile.

## Run the tests

```bash
python -m pytest -v
```

The current suite contains 124 test cases.

Coverage includes model correctness, individual/batched equivalence,
request validation, overload handling, deadlines, cancellation, shutdown,
batch result routing, failure recovery, launcher settings, and benchmark
recording and statistics.

## Run the service

### Single-request baseline

```bash
python -m adaptive_inference.serve \
  --port 8000 \
  --queue-capacity 32 \
  --request-timeout 5 \
  --batch-size 1 \
  --batch-wait-ms 0 \
  --torch-threads 1
```

### Batching with a collection window

```bash
python -m adaptive_inference.serve \
  --port 8000 \
  --queue-capacity 32 \
  --request-timeout 5 \
  --batch-size 8 \
  --batch-wait-ms 2 \
  --torch-threads 1
```

### Batching without deliberate waiting

Use `--batch-size 8 --batch-wait-ms 0`.

This combines requests already waiting in the queue without waiting for
additional arrivals.

The launcher uses one server process, one inference execution thread,
one PyTorch interop thread, and the requested PyTorch intra-op thread count.

## HTTP interface

| Endpoint | Purpose |
|---|---|
| `GET /health` | Application liveness |
| `GET /ready` | Model loaded and queue accepting submissions |
| `GET /info` | Launch settings, environment, source state, and server identity |
| `POST /predict` | One feature vector, returning ten scores |
| `GET /docs` | Interactive API documentation |

`/info` is provided by the configurable launcher.

Readiness does not become false merely because the waiting queue is full.
Queue overload is reported by prediction requests.

### Send a prediction

With a server running on port 8000:

```python
import httpx

response = httpx.post(
    "http://127.0.0.1:8000/predict",
    json={
        "features": [
            (index - 32) / 128.0
            for index in range(64)
        ]
    },
    timeout=10,
)
response.raise_for_status()
print(response.json())
```

Features must be finite numbers between -1,000,000 and 1,000,000.
Numeric strings, booleans, extra fields, and incorrect feature counts
are rejected.

### Response behavior

| Status | Meaning |
|---|---|
| `200` | Prediction completed |
| `422` | Request validation failed |
| `503`, `queue_full` | Waiting queue is full |
| `503`, `model_not_ready` | Model is unavailable |
| `503`, `service_unavailable` | Queue is unavailable or closing |
| `504`, `request_timeout` | Prediction deadline expired |
| `500`, `inference_failed` | Processor execution failed |

Queue-full responses include `Retry-After: 1`.

## Queue, deadline, and shutdown semantics

Queue capacity counts entries waiting in the FIFO queue.

The worker additionally holds at most one batch while assembling or
executing it. Thus capacity 32 with batch limit 8 permits up to 32 queued
entries plus eight entries held by the worker.

This bounds inference queue entries, not all HTTP connections or request
body memory.

Request deadlines begin at scheduler submission, after HTTP validation.
They include queue waiting, batch assembly, and model execution.

Expired and cancelled queued requests are skipped before model execution.
Their queue entries remain occupied until the worker consumes them.

A timeout cannot interrupt a processor call already running in a thread.
The execution worker remains occupied until that call finishes.

Shutdown stops new submissions and drains accepted entries, skipping
expired or cancelled requests. It waits for running inference to finish;
it cannot forcibly terminate a hung processor thread.

## Measured performance

Measurements used:

- Intel Core i7-12650H.
- 16 logical CPUs available to WSL.
- Ubuntu 24.04 under WSL 2.
- Linux 5.15.167.4-microsoft-standard-WSL2.
- Python 3.12.3.
- PyTorch 2.14.1+cpu.
- FastAPI 0.159.1 and Uvicorn 0.54.0.
- One server process.
- One PyTorch intra-op thread and one interop thread.
- Queue capacity 32 and request deadline five seconds.
- HTTP loopback with the client and server sharing the machine.

These are WSL measurements, separate from the Docker functional check.

### Method

The benchmark uses a fixed-request, closed-loop workload. Each client worker
sends its next request after its previous request has completed and been
validated.

Each configuration has three trials:

- One client: 200 warmups and 5,000 measured requests per trial.
- Eight clients: 500 warmups and 10,000 measured requests per trial.

Trials reuse the same server process. Each trial creates a fresh client
connection pool and performs warmup before measurement.

Reference predictions are computed before timing. Every successful
response is checked for model identity, finite scores, and numerical
agreement with its individual reference prediction.

Request latency includes client serialization, connection-pool waiting,
network transfer, and receiving the complete response body. JSON parsing
and numerical validation occur afterward. Trial throughput includes
client validation overhead.

p95 uses the nearest-rank definition within each trial.

The table reports the median of three trial statistics. Its p95 column
is the median of three trial p95 values, not a pooled percentile.

### Results

| Clients | Mode | Requests/sec | Median latency | p95 latency |
|---:|---|---:|---:|---:|
| 1 | Single request | 406.3 | 2.03 ms | 4.92 ms |
| 1 | Batch limit 8, zero wait | 395.0 | 2.02 ms | 4.75 ms |
| 1 | Batch limit 8, 2 ms wait | 210.4 | 4.11 ms | 8.33 ms |
| 8 | Single request | 427.7 | 16.19 ms | 37.47 ms |
| 8 | Batch limit 8, zero wait | 418.9 | 16.22 ms | 31.98 ms |
| 8 | Batch limit 8, 2 ms wait | 510.4 | 14.07 ms | 27.03 ms |

At eight concurrent clients, the 2 ms batching configuration increased
aggregate throughput by approximately 19%, from 427.7 to 510.4 requests/sec,
and reduced the aggregate p95 statistic by approximately 28%.

At one client, the same waiting window roughly doubled median latency.
Zero-wait batching kept one-client latency near the baseline but did not
reproduce the eight-client throughput improvement.

The published six configurations contain 135,000 measured requests.
All passed numerical validation with zero recorded failures.

These results support a configuration tradeoff, not universal improvement.
Configurations were measured in separate blocks rather than randomized
interleaved trials. Machine load and client overhead affect the results.

### Published evidence

Evidence is stored under [docs/experiments](docs/experiments).

Each experiment contains:

- Server and client metadata.
- Git revision and source cleanliness at launch.
- Input vectors and reference predictions.
- A workload SHA-256.
- Warmup request records.
- Raw measured request records.
- Per-trial summaries.
- A final experiment summary.

The comparison table is also recorded in
[comparison.json](docs/experiments/comparison.json).

The published measurements identify source revision `e41aa94`.
Later commits add CI and container packaging.

### Reproduce an experiment

Commit source changes before benchmarking, then restart the server.
The runner requires matching clean server/client source revisions and
checks server identity before and after each trial.

With the single-request server running:

```bash
python -m adaptive_inference.benchmark \
  --output results/reproduction-single-c8-v1 \
  --requests 10000 \
  --warmup 500 \
  --runs 3 \
  --concurrency 8 \
  --expected-batch-size 1 \
  --expected-batch-wait-ms 0 \
  --expected-torch-threads 1
```

For the batching experiment, restart the server with batch limit 8 and
2 ms waiting, select a new output directory, and change the expected
batch settings accordingly.

Existing output directories are rejected.

| Exit code | Meaning |
|---|---|
| `0` | All measured requests passed validation |
| `2` | At least one measured request failed |
| `1` | Experiment setup, warmup, or recording error |

Failures and invalid responses are retained as request outcomes rather
than silently removed from throughput accounting.

## Docker

Build and run:

```bash
docker build -t adaptive-ml-inference:local .

docker run --rm \
  --name adaptive-inference-demo \
  --stop-timeout 30 \
  --publish 127.0.0.1:8001:8000 \
  adaptive-ml-inference:local
```

The default container uses batch limit 8, zero deliberate batch waiting,
one CPU inference thread, and an unprivileged application user.

The image includes the tested dependency snapshot, including test tools.
Its health check requests `/ready`.

Open:

- http://127.0.0.1:8001/ready
- http://127.0.0.1:8001/docs

To select a 2 ms waiting window, supply explicit launcher arguments after
the image name:

```bash
docker run --rm \
  --publish 127.0.0.1:8001:8000 \
  adaptive-ml-inference:local \
  --host 0.0.0.0 \
  --port 8000 \
  --batch-size 8 \
  --batch-wait-ms 2 \
  --torch-threads 1
```

The container was built and its HTTP prediction verified against a local
reference. The image does not contain a Git checkout, so `/info` reports
Git metadata as unavailable.

The benchmark runner currently requires checkout-based Git provenance;
the published performance figures were collected outside Docker.

### WSL credential-helper troubleshooting

If Docker can reach the engine but fails to execute a Windows credential
helper, public-image operations can use a temporary client configuration:

```bash
docker_public_config=$(mktemp -d /tmp/adaptive-docker.XXXXXX)
printf '{"auths":{}}\n' > "$docker_public_config/config.json"

docker --config "$docker_public_config" build \
  -t adaptive-ml-inference:local .
```

Use the same `--config` option for subsequent Docker commands in that shell.
This configuration is separate from existing saved registry credentials.

## Continuous integration

GitHub Actions installs CPU PyTorch and the tested package versions on
Ubuntu with Python 3.12, then runs all 124 tests.

Each run uploads test results and installed-package information.

Docker packaging was verified locally. The current CI workflow runs the
Python test suite; it does not build or test the image.

## Current scope

- CPU inference using an untrained synthetic model.
- Fixed 64-feature input and ten-score output.
- Maximum model batch size of 128.
- One inference worker and one server process.
- Configurable dynamic batching without automatic parameter tuning.
- Local HTTP measurements with shared client/server CPU resources.
- Numerical serving correctness, not trained-model accuracy.
- No authentication, distributed serving, GPU execution, or failed-worker
  process recovery in this release.

A trained model, automatic traffic-policy selection, and separate-machine
load generation are possible extensions.

## Project layout

```text
src/adaptive_inference/
    model.py          CPU neural network and tensor validation
    scheduler.py      Queueing, deadlines, and dynamic batching
    api.py            HTTP validation, health checks, and error mapping
    serve.py          Configurable launcher and runtime metadata
    benchmark.py      HTTP workload, reference checks, and evidence recording

tests/                Model, queue, API, batching, and benchmark tests
docs/experiments/     Published raw evidence and comparison
.github/workflows/   Automated Python tests
Dockerfile            Non-root service image
requirements-tested.txt
```
