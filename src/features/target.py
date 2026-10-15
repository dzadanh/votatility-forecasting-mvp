import pandas as pd


def add_future_volatility_target(
    data: pd.DataFrame,
    horizon: int = 5,
    return_column: str = "return_1d",
    target_column: str | None = None,
) -> pd.DataFrame:
    """Gắn nhãn độ lệch chuẩn return trong horizon phiên giao dịch kế tiếp.

    Target tại t dùng return t+1 đến t+horizon, nên chỉ dùng làm nhãn huấn luyện
    hoặc đánh giá, tuyệt đối không dùng làm feature đầu vào tại t. Các dòng cuối
    không đủ kỳ hạn được giữ với giá trị NaN.
    """
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon phải là số nguyên dương.")
    if return_column not in data.columns:
        raise ValueError(f"Không tìm thấy cột return: {return_column}")

    result = data.copy()
    name = target_column or f"target_volatility_{horizon}d"
    future_returns = result[return_column].shift(-1)
    result[name] = future_returns.iloc[::-1].rolling(
        horizon, min_periods=horizon
    ).std().iloc[::-1]
    return result
