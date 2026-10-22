"""
Launch a reproducibly configured local inference server.

Set CPU thread limits before model startup. Expose configuration,
package versions, machine information, and source state through /info.

Use one server process with explicit asyncio and HTTP implementations.
Disable per-request access logging to avoid adding that benchmark cost.
"""

import argparse
import math
import os
import platform
import subprocess
import uuid
from importlib.metadata import version
from pathlib import Path

import torch
import uvicorn

from adaptive_inference.api import MODEL_ID, create_app
from adaptive_inference.model import MAX_BATCH_SIZE


def positive_integer(text):
    """Parse an integer greater than zero."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None

    if value < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")

    return value


def nonnegative_float(text):
    """Parse a finite number greater than or equal to zero."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number") from None

    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")

    return value


def positive_float(text):
    """Parse a finite number greater than zero."""
    value = nonnegative_float(text)

    if value == 0:
        raise argparse.ArgumentTypeError("must be greater than zero")

    return value


def parse_args(argv=None):
    """Validate launch settings before starting the server."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--port", type=positive_integer, default=8000)
    parser.add_argument("--queue-capacity", type=positive_integer, default=32)
    parser.add_argument("--request-timeout", type=positive_float, default=5.0)
    parser.add_argument("--batch-size", type=positive_integer, default=1)
    parser.add_argument(
        "--batch-wait-ms",
        type=nonnegative_float,
        default=0.0,
    )
    parser.add_argument("--torch-threads", type=positive_integer, default=1)

    args = parser.parse_args(argv)

    if args.port > 65535:
        parser.error("--port must be between 1 and 65535")

    if args.batch_size > MAX_BATCH_SIZE:
        parser.error(f"--batch-size cannot exceed {MAX_BATCH_SIZE}")

    return args


def source_metadata():
    """Record source state at launch; report unavailable Git explicitly."""
    directory = Path(__file__).resolve().parent

    def git(*arguments):
        return subprocess.run(
            ["git", "-C", str(directory), *arguments],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()

    try:
        revision = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
    except (OSError, subprocess.SubprocessError):
        return {
            "git_revision": None,
            "dirty": None,
            "available": False,
        }

    return {
        "git_revision": revision,
        "dirty": dirty,
        "available": True,
    }


def cpu_model():
    """Read the Linux CPU model when available."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, separator, value = line.partition(":")

            if separator and key.strip() == "model name":
                return value.strip()
    except OSError:
        pass

    return platform.processor() or "unknown"


def build_app(args):
    """Build the application and capture its launch metadata."""
    app = create_app(
        queue_capacity=args.queue_capacity,
        request_timeout=args.request_timeout,
        max_batch_size=args.batch_size,
        max_batch_wait=args.batch_wait_ms / 1000.0,
    )

    try:
        available_cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available_cpus = None

    metadata = {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "model_id": MODEL_ID,
        "configuration": {
            "queue_capacity": args.queue_capacity,
            "request_timeout_seconds": args.request_timeout,
            "max_batch_size": args.batch_size,
            "max_batch_wait_seconds": args.batch_wait_ms / 1000.0,
            "requested_torch_threads": args.torch_threads,
            "server_workers": 1,
            "event_loop": "asyncio",
            "http_implementation": "h11",
            "access_log": False,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_model": cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "available_cpu_ids": available_cpus,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "packages": {
                name: version(name)
                for name in ("torch", "numpy", "fastapi", "uvicorn")
            },
        },
        "source_at_launch": source_metadata(),
    }

    @app.get("/info")
    async def info():
        """Return metadata captured when this server instance was built."""
        return metadata

    return app


def main():
    """Configure CPU execution and launch one local server process."""
    args = parse_args()

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)

    app = build_app(args)

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        workers=1,
        loop="asyncio",
        http="h11",
        access_log=False,
    )


if __name__ == "__main__":
    main()