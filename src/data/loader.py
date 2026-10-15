from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_data(company_name: str) -> pd.DataFrame:
    """Load a company's raw stock data from ``data/raw``.

    The company name is treated as a ticker symbol, so values such as ``ACB``
    and `` acb `` resolve to the same CSV file.
    """
    if not isinstance(company_name, str) or not company_name.strip():
        raise ValueError("Không tìm thấy tên công ty.")

    symbol = company_name.strip().casefold()
    input_path = RAW_DATA_DIR / f"{symbol}.csv"
    if not input_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {input_path}")

    return pd.read_csv(input_path)


def main() -> None:
    data = load_data("ACB")
    print(data.head())
    


if __name__ == "__main__":
    main()
