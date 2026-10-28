"""Local, single-worker training jobs for uploaded offline CSV data.

The JSON status and immutable run folders let the Streamlit UI survive reruns.
The worker is process-local; a later cloud deployment can replace this module
without changing the model pipeline or the dashboard's run format.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import os
import re
import threading
import traceback
import uuid

import pandas as pd

from src.config import DEFAULT_CONFIG, ROOT, load_config
from src.pipeline.dataset import prepare_dataset


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_ROWS = 50_000
MODELS = ("all", "garch", "xgboost", "lstm")
TICKER_PATTERN = re.compile(r"[A-Z0-9]{1,12}")
RUN_ID_PATTERN = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{8}")
REQUIRED_OUTPUTS = ("metadata.json", "metrics.json", "warning.json", "train_predictions.csv")

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="offline-training")
_submit_lock = threading.Lock()
_status_lock = threading.RLock()
_active_future: Future | None = None


class TrainingBusyError(RuntimeError):
    """The local worker is already training another dataset."""


def artifact_root(config: dict | None = None) -> Path:
    settings = config or load_config()
    root = Path(settings["paths"]["artifacts"])
    return root if root.is_absolute() else ROOT / root


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _status_path(root: Path, run_id: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("Mã lần huấn luyện không hợp lệ.")
    return root / "runs" / run_id / "status.json"


def _write_status(path: Path, status: dict) -> None:
    with _status_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".status-{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)


def read_job(root: Path, run_id: str) -> dict:
    path = _status_path(Path(root), run_id)
    with _status_lock:
        job = json.loads(path.read_text(encoding="utf-8"))
        if job.get("state") in ("queued", "running") and job.get("worker_pid") != os.getpid():
            job.update(state="interrupted", stage="Tiến trình huấn luyện đã dừng",
                       error="Ứng dụng đã khởi động lại trước khi lần huấn luyện này hoàn tất.",
                       updated_at=_now())
            _write_status(path, job)
        return job


def list_jobs(root: Path) -> list[dict]:
    jobs = []
    folder = Path(root) / "runs"
    if not folder.exists():
        return jobs
    for path in folder.glob("*/status.json"):
        if not RUN_ID_PATTERN.fullmatch(path.parent.name):
            continue
        try:
            job = read_job(Path(root), path.parent.name)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if job.get("run_id") == path.parent.name:
            jobs.append(job)
    return sorted(jobs, key=lambda job: job["created_at"], reverse=True)


def _update_job(root: Path, run_id: str, **changes) -> None:
    path = _status_path(root, run_id)
    with _status_lock:
        status = json.loads(path.read_text(encoding="utf-8"))
        status.update(changes)
        status["updated_at"] = _now()
        _write_status(path, status)


def _run_job(root: Path, run_id: str, ticker: str, model: str, fast: bool) -> None:
    run_dir = root / "runs" / run_id
    _update_job(root, run_id, state="running", stage="Đang chuẩn bị dữ liệu", progress=0.02)

    def progress(stage: str, fraction: float) -> None:
        _update_job(root, run_id, stage=stage, progress=round(float(fraction), 3))

    try:
        from src.pipeline.train_pipeline import run_training

        run_training(ticker=ticker, model=model, fast=fast,
                     csv_path=run_dir / "input.csv", config_path=run_dir / "config.yaml",
                     artifact_dir=run_dir, progress_callback=progress)
        output = run_dir / ticker
        missing = [name for name in REQUIRED_OUTPUTS if not (output / name).is_file()]
        if missing:
            raise RuntimeError(f"Thiếu artifact sau huấn luyện: {', '.join(missing)}")
        _update_job(root, run_id, state="succeeded", stage="Hoàn tất", progress=1.0,
                    result_dir=str(output))
    except Exception as exc:
        (run_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        _update_job(root, run_id, state="failed", stage="Huấn luyện thất bại",
                    error=str(exc)[:500])


def submit_uploaded_training(csv_bytes: bytes, ticker: str, model: str = "xgboost",
                             fast: bool = False, *, root: Path | None = None,
                             config_path: Path = DEFAULT_CONFIG) -> dict:
    """Validate a CSV, snapshot it and the config, then submit one local job."""
    global _active_future

    ticker = ticker.strip().upper()
    if not TICKER_PATTERN.fullmatch(ticker):
        raise ValueError("Ticker chỉ gồm 1–12 chữ cái hoặc chữ số.")
    if model not in MODELS:
        raise ValueError("Mô hình không hợp lệ.")
    if not csv_bytes or len(csv_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("CSV phải có dữ liệu và không vượt quá 10 MB.")

    config_bytes = Path(config_path).read_bytes()
    with TemporaryDirectory(prefix="volatility-upload-") as directory:
        temporary = Path(directory)
        input_path = temporary / "input.csv"
        input_path.write_bytes(csv_bytes)
        config_snapshot = temporary / "config.yaml"
        config_snapshot.write_bytes(config_bytes)
        config = load_config(config_snapshot)
        try:
            raw_rows = len(pd.read_csv(BytesIO(csv_bytes)))
        except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError(f"Không đọc được CSV: {exc}") from exc
        if raw_rows > MAX_UPLOAD_ROWS:
            raise ValueError(f"CSV vượt quá {MAX_UPLOAD_ROWS:,} dòng.")
        dataset = prepare_dataset(ticker, config, input_path)

    root = Path(root) if root is not None else artifact_root(config)
    summary = {"raw_rows": raw_rows, "clean_rows": len(dataset.frame),
               "train_rows": len(dataset.train), "test_rows": len(dataset.test),
               "first_date": str(dataset.frame.date.min().date()),
               "last_date": str(dataset.frame.date.max().date())}

    with _submit_lock:
        if _active_future is not None and not _active_future.done():
            raise TrainingBusyError("Đang có một lần huấn luyện khác. Hãy chờ lần đó hoàn tất.")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        run_dir = root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "input.csv").write_bytes(csv_bytes)
        (run_dir / "config.yaml").write_bytes(config_bytes)
        status = {"run_id": run_id, "ticker": ticker, "model": model, "fast": bool(fast),
                  "state": "queued", "stage": "Đang chờ huấn luyện", "progress": 0.0,
                  "created_at": _now(), "updated_at": _now(),
                  "worker_pid": os.getpid(),
                  "input_sha256": hashlib.sha256(csv_bytes).hexdigest(),
                  "input_summary": summary, "error": None, "result_dir": None}
        _write_status(_status_path(root, run_id), status)
        try:
            _active_future = _executor.submit(_run_job, root, run_id, ticker, model, bool(fast))
        except RuntimeError as exc:
            _update_job(root, run_id, state="failed", stage="Không thể khởi chạy", error=str(exc))
            raise
    return status
