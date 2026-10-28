"""Upload an existing CSV and train an isolated offline experiment."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import json

import pandas as pd
import streamlit as st

from src.pipeline.training_jobs import (
    MAX_UPLOAD_BYTES, TrainingBusyError, list_jobs, read_job, submit_uploaded_training,
)


@st.fragment(run_every="3s")
def _job_panel(root_string: str, run_id: str) -> None:
    root = Path(root_string)
    try:
        job = read_job(root, run_id)
    except (FileNotFoundError, ValueError):
        st.error("Không tìm thấy lần huấn luyện này.")
        return

    st.subheader(f"Lần huấn luyện {run_id}")
    st.write(f"**{job['ticker']} · {job['model']}** · "
             f"{'chạy nhanh' if job['fast'] else 'chạy đầy đủ'} · {job['state']}")
    st.progress(float(job.get("progress", 0)), text=job.get("stage", ""))
    summary = job.get("input_summary", {})
    if summary:
        st.caption(f"CSV: {summary['raw_rows']} dòng gốc, {summary['clean_rows']} phiên sau làm sạch "
                   f"({summary['first_date']} đến {summary['last_date']}); "
                   f"train {summary['train_rows']}, test {summary['test_rows']} origin.")

    if job["state"] in ("queued", "running"):
        st.info("Huấn luyện đang chạy trên máy chủ. Có thể chuyển trang và quay lại xem trạng thái.")
        return
    if job["state"] in ("failed", "interrupted"):
        st.error(job.get("error") or "Huấn luyện thất bại. Hãy kiểm tra dữ liệu và thử lại.")
        return
    if job["state"] != "succeeded":
        return

    if job["fast"]:
        st.warning("Kết quả chạy nhanh chỉ dùng kiểm tra luồng; không dùng kết luận chất lượng mô hình.")
    folder = root / "runs" / run_id / job["ticker"]
    try:
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(folder / "train_predictions.csv")
    except (OSError, ValueError) as exc:
        st.error(f"Không đọc được kết quả đã lưu: {exc}")
        return

    rows = []
    for name, values in metrics.items():
        high = values["high_warning"]
        rows.append({"Mô hình": name, "MAE": values["mae"], "RMSE": values["rmse"],
                     "Precision HIGH": high["precision"], "Recall HIGH": high["recall"],
                     "Mẫu test": values["samples"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    latest = predictions[predictions["partition"] == "latest"]
    if not latest.empty:
        st.write("Dự báo mới nhất (chưa có thực tế của các phiên tiếp theo):")
        st.dataframe(latest[["date", "model", "predicted_volatility", "risk_level"]],
                     hide_index=True)
    if st.button("Xem kết quả trên dashboard", key=f"view-{run_id}"):
        st.session_state["navigate_to_run"] = {"ticker": job["ticker"], "run_id": run_id}
        st.rerun()


def render(root: Path, default_ticker: str = "ACB") -> None:
    st.header("Huấn luyện từ CSV đã có")
    st.caption("Tải CSV OHLCV từ máy của bạn lên để huấn luyện offline. Trang này không tìm kiếm "
               "hay tải dữ liệu từ nhà cung cấp.")
    uploaded = st.file_uploader("CSV OHLCV", type=["csv"], accept_multiple_files=False,
                                help="Các cột cần có: date/time, open, high, low, close, volume.")
    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_BYTES:
            st.error("CSV vượt quá giới hạn 10 MB.")
        else:
            try:
                preview = pd.read_csv(BytesIO(uploaded.getvalue()), nrows=5)
                st.dataframe(preview, hide_index=True)
            except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                st.error(f"Không đọc được CSV: {exc}")

    with st.form("offline-training"):
        ticker = st.text_input("Mã chứng khoán", value=default_ticker, max_chars=12)
        model = st.selectbox("Mô hình", ["xgboost", "garch", "lstm", "all"],
                             format_func=lambda name: "Cả ba mô hình" if name == "all" else name.upper())
        fast = st.checkbox("Chạy nhanh để kiểm tra luồng (metrics không dùng để kết luận)")
        submitted = st.form_submit_button("Bắt đầu huấn luyện")

    if submitted:
        if uploaded is None:
            st.error("Hãy tải một CSV lên trước khi huấn luyện.")
        else:
            try:
                with st.spinner("Đang kiểm tra CSV và chia dữ liệu theo thời gian..."):
                    job = submit_uploaded_training(uploaded.getvalue(), ticker, model, fast, root=root)
                st.session_state["training_run_id"] = job["run_id"]
                st.rerun()
            except (ValueError, TrainingBusyError, OSError) as exc:
                st.error(str(exc))

    jobs = list_jobs(root)
    if jobs:
        ids = [job["run_id"] for job in jobs]
        active = st.session_state.get("training_run_id")
        selected = st.selectbox("Theo dõi lần huấn luyện", ids,
                                index=ids.index(active) if active in ids else 0,
                                format_func=lambda run_id: next(
                                    f"{job['ticker']} · {job['model']} · {run_id}"
                                    for job in jobs if job["run_id"] == run_id))
        _job_panel(str(root), selected)
