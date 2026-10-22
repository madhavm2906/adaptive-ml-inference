"""
Benchmark a running local inference server with bounded HTTP concurrency.

Generate reference predictions before timing. Warm up each trial, then
record a fixed number of requests using a fixed number of client workers.

Request latency includes serialization, connection-pool waiting, network
transfer, and receiving the response body. JSON validation happens afterward.
Trial throughput includes client validation overhead.

Trials reuse the same server process. They are repeated trials, not
independent server launches. Existing output directories are rejected.
"""

import argparse
import asyncio
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import httpx
import torch

from adaptive_inference.api import MODEL_ID
from adaptive_inference.model import (
    INPUT_FEATURES,
    OUTPUT_CLASSES,
    InferenceModel,
)
from adaptive_inference.serve import (
    nonnegative_float,
    positive_float,
    positive_integer,
    source_metadata,
)


def write_json(path, data):
    """Write a new JSON file without replacing existing evidence."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_samples(path, samples):
    """Write one complete request record per line."""
    with path.open("x", encoding="utf-8") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, allow_nan=False) + "\n")


def make_workload():
    """Prepare distinct inputs and individual reference predictions."""
    torch.set_num_threads(1)
    model = InferenceModel()
    workload = []

    for index in range(16):
        features = [
            (index + feature_index - 32) / 128.0
            for feature_index in range(INPUT_FEATURES)
        ]
        expected = model.predict(
            torch.tensor([features], dtype=torch.float32)
        )[0].tolist()

        workload.append({
            "features": features,
            "expected_scores": expected,
        })

    return workload


def validate_result(body, expected):
    """Require the correct model, finite scores, and numerical agreement."""
    if not isinstance(body, dict) or body.get("model_id") != MODEL_ID:
        raise ValueError("unexpected model identity")

    scores = body.get("scores")

    if not isinstance(scores, list) or len(scores) != OUTPUT_CLASSES:
        raise ValueError("invalid score vector")

    for actual, reference in zip(scores, expected, strict=True):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(actual)
        ):
            raise ValueError("scores must be finite numbers")

        if not math.isclose(
            actual,
            reference,
            rel_tol=1e-5,
            abs_tol=1e-6,
        ):
            raise ValueError("prediction differs from reference")

    return scores


async def run_phase(client, workload, count, concurrency, timeout):
    """Run bounded client workers and retain every request outcome."""
    indices = iter(range(count))
    samples = []

    async def worker():
        while True:
            request_id = next(indices, None)

            if request_id is None:
                return

            workload_id = request_id % len(workload)
            item = workload[workload_id]
            started = time.perf_counter()
            status = None
            scores = None

            try:
                # This adds a whole-request deadline to HTTPX's
                # connect/read/write/pool inactivity timeouts.
                async with asyncio.timeout(timeout):
                    response = await client.post(
                        "/predict",
                        json={"features": item["features"]},
                    )

                elapsed_ms = (time.perf_counter() - started) * 1000
                status = response.status_code

                if status != 200:
                    outcome = f"http_{status}"
                else:
                    try:
                        scores = validate_result(
                            response.json(),
                            item["expected_scores"],
                        )
                        outcome = "ok"
                    except (ValueError, TypeError, OverflowError):
                        outcome = "invalid_response"
                        scores = None

            except TimeoutError:
                elapsed_ms = (time.perf_counter() - started) * 1000
                outcome = "request_timeout"

            except httpx.RequestError as error:
                elapsed_ms = (time.perf_counter() - started) * 1000
                outcome = f"transport_{type(error).__name__}"

            samples.append({
                "request_id": request_id,
                "workload_id": workload_id,
                "latency_ms": elapsed_ms,
                "status_code": status,
                "outcome": outcome,
                "scores": scores,
            })

    started = time.perf_counter()

    await asyncio.gather(
        *(worker() for _ in range(min(concurrency, count)))
    )

    duration = time.perf_counter() - started
    samples.sort(key=lambda sample: sample["request_id"])

    return samples, duration


def summarize(samples, duration):
    """Report successful latency separately from all failure outcomes."""
    if not samples or not math.isfinite(duration) or duration <= 0:
        raise ValueError("summary requires samples and positive duration")

    latencies = sorted(
        sample["latency_ms"]
        for sample in samples
        if sample["outcome"] == "ok"
    )
    failures = Counter(
        sample["outcome"]
        for sample in samples
        if sample["outcome"] != "ok"
    )

    successful = len(latencies)
    failed = len(samples) - successful

    return {
        "attempted": len(samples),
        "successful": successful,
        "failed": failed,
        "failure_rate": failed / len(samples),
        "failure_outcomes": dict(failures),
        "duration_seconds": duration,
        "successful_requests_per_second": successful / duration,
        "completed_requests_per_second": len(samples) / duration,
        "successful_latency_median_ms": (
            statistics.median(latencies) if latencies else None
        ),
        "successful_latency_p95_ms": (
            latencies[math.ceil(0.95 * successful) - 1]
            if latencies else None
        ),
    }


def verify_server(info, local_source, args):
    """Reject untraceable source or an unexpected server configuration."""
    if (
        not local_source["available"]
        or local_source["dirty"] is not False
        or not local_source["git_revision"]
    ):
        raise ValueError("commit local source changes before benchmarking")

    remote_source = info["source_at_launch"]

    if (
        not remote_source["available"]
        or remote_source["dirty"] is not False
        or remote_source["git_revision"] != local_source["git_revision"]
    ):
        raise ValueError(
            "restart the server from the same clean committed source"
        )

    configuration = info["configuration"]
    environment = info["environment"]

    if info["model_id"] != MODEL_ID:
        raise ValueError("unexpected server model")

    if configuration["max_batch_size"] != args.expected_batch_size:
        raise ValueError("server batch size does not match the experiment")

    if not math.isclose(
        configuration["max_batch_wait_seconds"],
        args.expected_batch_wait_ms / 1000.0,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("server batch waiting window does not match")

    if (
        environment["torch_threads"] != args.expected_torch_threads
        or environment["torch_interop_threads"] != 1
    ):
        raise ValueError("server CPU thread settings do not match")

    if args.http_timeout <= configuration["request_timeout_seconds"]:
        raise ValueError("HTTP timeout must exceed the server deadline")


async def benchmark(args):
    """Record metadata, warmups, raw samples, and repeated trial summaries."""
    output = Path(args.output)
    local_source = source_metadata()

    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )

    # Fetch initial metadata before creating an experiment directory.
    async with httpx.AsyncClient(
        base_url=args.base_url,
        timeout=args.http_timeout,
        trust_env=False,
    ) as client:
        response = await client.get("/info")
        response.raise_for_status()
        server_info = response.json()

        verify_server(server_info, local_source, args)

        readiness = await client.get("/ready")
        readiness.raise_for_status()

    workload = make_workload()

    # Refuse to replace any existing experiment.
    output.mkdir(parents=True, exist_ok=False)

    try:
        encoded_workload = json.dumps(
            workload,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")

        write_json(output / "workload.json", workload)
        write_json(output / "metadata.json", {
            "schema_version": 1,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "method": "fixed-request closed-loop HTTP workload",
            "arguments": vars(args),
            "server": server_info,
            "client": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "httpx": version("httpx"),
                "torch": version("torch"),
                "torch_threads": torch.get_num_threads(),
                "source": local_source,
            },
            "workload_sha256": hashlib.sha256(
                encoded_workload
            ).hexdigest(),
            "validation_relative_tolerance": 1e-5,
            "validation_absolute_tolerance": 1e-6,
        })

        trials = []

        for trial in range(1, args.runs + 1):
            # Each trial gets a fresh connection pool and an untimed warmup.
            async with httpx.AsyncClient(
                base_url=args.base_url,
                limits=limits,
                timeout=args.http_timeout,
                trust_env=False,
            ) as client:
                before = await client.get("/info")
                before.raise_for_status()

                if before.json() != server_info:
                    raise ValueError("server changed before a trial")

                warmup, _ = await run_phase(
                    client,
                    workload,
                    args.warmup,
                    args.concurrency,
                    args.http_timeout,
                )
                write_samples(
                    output / f"run-{trial:03d}.warmup.jsonl",
                    warmup,
                )

                if any(row["outcome"] != "ok" for row in warmup):
                    raise ValueError(
                        "warmup failed; inspect saved warmup records"
                    )

                samples, duration = await run_phase(
                    client,
                    workload,
                    args.requests,
                    args.concurrency,
                    args.http_timeout,
                )
                write_samples(
                    output / f"run-{trial:03d}.samples.jsonl",
                    samples,
                )

                after = await client.get("/info")
                after.raise_for_status()

                if after.json() != server_info:
                    raise ValueError("server changed during a trial")

                if source_metadata() != local_source:
                    raise ValueError("local source changed during the run")

            summary = {"trial": trial, **summarize(samples, duration)}
            write_json(
                output / f"run-{trial:03d}.summary.json",
                summary,
            )
            trials.append(summary)

            print(json.dumps(summary, indent=2))

        all_valid = all(trial["failed"] == 0 for trial in trials)

        write_json(output / "summary.json", {
            "schema_version": 1,
            "all_requests_valid": all_valid,
            "trials": trials,
        })

        return 0 if all_valid else 2

    except Exception as error:
        write_json(output / "failure.json", {
            "error_type": type(error).__name__,
            "message": str(error),
        })
        raise


def parse_args(argv=None):
    """Parse one experiment's workload and expected server settings."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--requests", type=positive_integer, default=1000)
    parser.add_argument("--warmup", type=positive_integer, default=100)
    parser.add_argument("--runs", type=positive_integer, default=3)
    parser.add_argument("--concurrency", type=positive_integer, default=8)
    parser.add_argument("--http-timeout", type=positive_float, default=10.0)
    parser.add_argument(
        "--expected-batch-size",
        type=positive_integer,
        required=True,
    )
    parser.add_argument(
        "--expected-batch-wait-ms",
        type=nonnegative_float,
        required=True,
    )
    parser.add_argument(
        "--expected-torch-threads",
        type=positive_integer,
        default=1,
    )

    return parser.parse_args(argv)


def main():
    """Return 0 for valid runs, 2 for measured failures, and 1 for errors."""
    args = parse_args()

    try:
        return asyncio.run(benchmark(args))
    except (OSError, ValueError, RuntimeError, KeyError, httpx.HTTPError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())