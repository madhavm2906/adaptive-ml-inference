"""
Verify launcher argument validation and reported server configuration.

Build test applications without changing process-wide PyTorch thread
settings. The command-line entry point applies those settings separately.
"""

import uuid

import pytest
import torch
from fastapi.testclient import TestClient

from adaptive_inference.model import MAX_BATCH_SIZE
from adaptive_inference.serve import build_app, parse_args


@pytest.mark.parametrize(
    "arguments",
    [
        ["--port", "0"],
        ["--port", "65536"],
        ["--batch-size", "0"],
        ["--batch-size", str(MAX_BATCH_SIZE + 1)],
        ["--batch-wait-ms", "-1"],
        ["--batch-wait-ms", "nan"],
        ["--torch-threads", "0"],
        ["--queue-capacity", "0"],
        ["--request-timeout", "0"],
        ["--request-timeout", "inf"],
    ],
)
def test_invalid_launch_settings_are_rejected(arguments):
    with pytest.raises(SystemExit) as error:
        parse_args(arguments)

    assert error.value.code == 2


def test_metadata_reports_configuration_and_stable_server_identity():
    args = parse_args(
        [
            "--batch-size", "8",
            "--batch-wait-ms", "2",
            "--queue-capacity", "64",
            "--request-timeout", "3",
        ]
    )

    with TestClient(build_app(args)) as client:
        first = client.get("/info")
        second = client.get("/info")
        ready = client.get("/ready")

    assert first.status_code == 200
    assert ready.status_code == 200

    data = first.json()

    assert data == second.json()
    assert str(uuid.UUID(data["run_id"])) == data["run_id"]

    configuration = data["configuration"]

    assert configuration["max_batch_size"] == 8
    assert configuration["max_batch_wait_seconds"] == 0.002
    assert configuration["queue_capacity"] == 64
    assert configuration["request_timeout_seconds"] == 3.0
    assert configuration["server_workers"] == 1
    assert configuration["access_log"] is False

    assert data["environment"]["torch_threads"] == torch.get_num_threads()
    assert "torch" in data["environment"]["packages"]
    assert "git_revision" in data["source_at_launch"]


def test_each_server_instance_has_a_different_identity():
    args = parse_args([])

    with TestClient(build_app(args)) as first:
        first_id = first.get("/info").json()["run_id"]

    with TestClient(build_app(args)) as second:
        second_id = second.get("/info").json()["run_id"]

    assert first_id != second_id