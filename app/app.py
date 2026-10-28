"""Streamlit UI for uploaded offline training and saved experiments."""
import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, load_config
from src.pipeline.training_jobs import artifact_root, list_jobs
from pages import overview, forecast, comparison, explanation, training


@st.cache_data(max_entries=32)
def read_outputs(path: str, version: tuple[int, ...]):
    """Use file versions in the cache key so a retrained result is refreshed."""
    folder = Path(path)
    predictions = pd.read_csv(folder / "train_predictions.csv", parse_dates=["date"])
    files = {name: json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
             for name in ("metrics", "metadata", "warning")}
    shap_path = folder / "shap.json"
    files["shap"] = (json.loads(shap_path.read_text(encoding="utf-8"))
                     if shap_path.exists() and "xgboost" in files["metadata"].get("model_settings", {}) else None)
    return predictions, files


def _results(base: Path) -> list[dict]:
    results = []
    for job in list_jobs(base):
        if job.get("state") != "succeeded":
            continue
        folder = base / "runs" / job["run_id"] / job["ticker"]
        if (folder / "metadata.json").is_file():
            results.append({"id": job["run_id"], "ticker": job["ticker"],
                            "folder": folder,
                            "label": f"{job['run_id']} · {job['model']}"
                                     + (" · nhanh" if job["fast"] else ""),
                            "fast": job["fast"]})
    if base.exists():
        for folder in base.iterdir():
            if folder.is_dir() and (folder / "metadata.json").is_file():
                results.append({"id": f"legacy:{folder.name}", "ticker": folder.name,
                                "folder": folder, "label": "Kết quả CLI mặc định",
                                "fast": None})
    return results


def _version(folder: Path) -> tuple[int, ...]:
    names = ("train_predictions.csv", "metrics.json", "metadata.json", "warning.json", "shap.json")
    return tuple((folder / name).stat().st_mtime_ns if (folder / name).exists() else 0
                 for name in names)


def _overview_source(selected: dict | None, raw: Path, ticker: str) -> Path:
    """Resolve uploaded input from its run folder, independent of the host OS."""
    if selected and not selected["id"].startswith("legacy:"):
        uploaded = selected["folder"].parent / "input.csv"
        if uploaded.is_file():
            return uploaded
    return raw / f"{ticker.lower()}.csv"


def main():
    st.set_page_config(page_title="Volatility Forecasting", layout="wide")
    navigation = st.session_state.pop("navigate_to_run", None)
    if navigation:
        st.session_state["selected_ticker"] = navigation["ticker"]
        st.session_state["selected_run"] = navigation["run_id"]
        st.session_state["selected_page"] = "So sánh"
    st.title("Giám sát độ biến động · bản mẫu nghiên cứu")
    st.caption("Huấn luyện từ CSV được tải lên; không cập nhật thời gian thực. "
               "Không phải lời khuyên đầu tư.")
    config = load_config()
    base = artifact_root(config)
    raw = Path(config["paths"]["raw"])
    raw = raw if raw.is_absolute() else ROOT / raw
    results = _results(base)
    raw_tickers = {path.stem.upper() for path in raw.glob("*.csv")} if raw.exists() else set()
    tickers = sorted(raw_tickers | {item["ticker"] for item in results} |
                     {config["data"]["ticker"]})
    st.sidebar.selectbox("Mã chứng khoán", tickers, key="selected_ticker")
    page = st.sidebar.radio("Trang", ["Huấn luyện", "Tổng quan", "Dự báo & Cảnh báo",
                                         "So sánh", "Giải thích"], key="selected_page")
    if page == "Huấn luyện":
        training.render(base, st.session_state["selected_ticker"])
        return

    ticker = st.session_state["selected_ticker"]
    matching = [item for item in results if item["ticker"] == ticker]
    selected = None
    if matching:
        ids = [item["id"] for item in matching]
        if st.session_state.get("selected_run") not in ids:
            st.session_state["selected_run"] = ids[0]
        choice = st.sidebar.selectbox("Lần huấn luyện", ids, key="selected_run",
                                      format_func=lambda run_id: next(
                                          item["label"] for item in matching if item["id"] == run_id))
        selected = next(item for item in matching if item["id"] == choice)

    if page == "Tổng quan":
        try:
            source = _overview_source(selected, raw, ticker)
            overview.render(ticker, str(source))
        except FileNotFoundError:
            st.info(f"Chưa có CSV cho {ticker}. Hãy tải CSV lên ở trang Huấn luyện.")
        return
    if selected is None:
        st.info(f"Chưa có kết quả cho {ticker}. Hãy tải CSV lên và huấn luyện trước.")
        return

    try:
        predictions, files = read_outputs(str(selected["folder"]), _version(selected["folder"]))
    except FileNotFoundError:
        st.error("Kết quả chưa đầy đủ. Hãy chọn một lần huấn luyện khác hoặc chạy lại.")
        return
    if files["metadata"].get("fast"):
        st.warning("Lần huấn luyện nhanh chỉ để kiểm tra luồng; metrics không dùng để kết luận chất lượng.")
    if page == "Dự báo & Cảnh báo":
        forecast.render(predictions, files["warning"], files["metadata"])
    elif page == "So sánh":
        comparison.render(predictions, files["metrics"], files["metadata"])
    else:
        explanation.render(files["shap"])


if __name__ == "__main__":
    main()
