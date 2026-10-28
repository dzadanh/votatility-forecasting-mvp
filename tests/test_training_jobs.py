"""Uploaded CSV jobs remain isolated and report their outcome."""
from pathlib import Path
import json
import threading
import time

import pytest

from src.config import ROOT
from src.pipeline import training_jobs
from src.pipeline import train_pipeline


CSV = (ROOT / "data" / "raw" / "acb.csv").read_bytes()


def _wait_for_terminal(root: Path, run_id: str, timeout: float = 10) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = training_jobs.read_job(root, run_id)
        if job["state"] in ("succeeded", "failed"):
            return job
        time.sleep(0.02)
    pytest.fail("Training job did not reach a terminal state")


def _save_fake_outputs(**kwargs):
    assert Path(kwargs["csv_path"]).read_bytes() == CSV
    assert Path(kwargs["config_path"]).is_file()
    kwargs["progress_callback"]("Đang đánh giá", 0.8)
    output = Path(kwargs["artifact_dir"]) / kwargs["ticker"]
    output.mkdir(parents=True)
    for name in training_jobs.REQUIRED_OUTPUTS:
        (output / name).write_text("{}", encoding="utf-8")


def test_uploaded_job_saves_isolated_run_and_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(train_pipeline, "run_training", _save_fake_outputs)
    root = tmp_path / "artifacts"
    submitted = training_jobs.submit_uploaded_training(CSV, " acb ", "xgboost", True, root=root)
    job = _wait_for_terminal(root, submitted["run_id"])

    assert job["state"] == "succeeded"
    assert job["progress"] == 1.0
    assert job["ticker"] == "ACB"
    assert job["input_summary"]["test_rows"] > 0
    assert (root / "runs" / job["run_id"] / "input.csv").read_bytes() == CSV
    assert training_jobs.list_jobs(root)[0]["run_id"] == job["run_id"]
    assert not (root / "ACB").exists()


def test_upload_rejects_unsafe_ticker_and_short_data(tmp_path):
    with pytest.raises(ValueError, match="Ticker"):
        training_jobs.submit_uploaded_training(CSV, "../ACB", root=tmp_path)
    with pytest.raises(ValueError, match="Not enough labeled rows"):
        training_jobs.submit_uploaded_training(CSV.splitlines()[0] + b"\n" +
                                               b"2020-01-01,1,1,1,1,1\n", "ACB", root=tmp_path)
    assert not (tmp_path / "runs").exists()


def test_local_worker_rejects_second_simultaneous_job(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_runner(**kwargs):
        started.set()
        assert release.wait(8)
        _save_fake_outputs(**kwargs)

    monkeypatch.setattr(train_pipeline, "run_training", slow_runner)
    root = tmp_path / "artifacts"
    first = training_jobs.submit_uploaded_training(CSV, "ACB", "garch", root=root)
    try:
        assert started.wait(5)
        with pytest.raises(training_jobs.TrainingBusyError):
            training_jobs.submit_uploaded_training(CSV, "ACB", "garch", root=root)
    finally:
        release.set()
    assert _wait_for_terminal(root, first["run_id"])["state"] == "succeeded"


def test_incomplete_job_from_previous_process_is_marked_interrupted(tmp_path):
    run_id = "20261028T010000Z-1234abcd"
    path = tmp_path / "runs" / run_id / "status.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"run_id": run_id, "state": "running", "worker_pid": -1,
                                "created_at": "2026-10-28T01:00:00+00:00"}), encoding="utf-8")

    job = training_jobs.read_job(tmp_path, run_id)
    assert job["state"] == "interrupted"
    assert training_jobs.list_jobs(tmp_path)[0]["state"] == "interrupted"


@pytest.mark.slow
def test_uploaded_garch_runs_real_pipeline(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    submitted = training_jobs.submit_uploaded_training(CSV, "ACB", "garch", root=root)
    job = _wait_for_terminal(root, submitted["run_id"], timeout=120)
    assert job["state"] == "succeeded", job.get("error")
    folder = root / "runs" / job["run_id"] / "ACB"
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["garch"]["samples"] > 0
    assert (folder / "train_predictions.csv").is_file()

    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(training_jobs, "artifact_root", lambda config=None: root)
    app = AppTest.from_file(str(ROOT / "app" / "app.py")).run(timeout=30)
    assert not app.exception
    view = next(button for button in app.button if button.label == "Xem kết quả trên dashboard")
    app = view.click().run(timeout=30)
    assert not app.exception
    assert app.sidebar.radio[0].value == "So sánh"

    # A committed run may contain an absolute path from another machine.
    metadata_path = folder / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_csv"] = r"Z:\old-machine\input.csv"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    app = app.sidebar.radio[0].set_value("Tổng quan").run(timeout=30)
    assert not app.exception
    assert any(header.value == "Tổng quan dữ liệu" for header in app.header)
    assert not any("Chưa có CSV" in message.value for message in app.info)
