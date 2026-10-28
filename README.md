# Offline Volatility Forecasting Prototype

Bản mẫu nghiên cứu dự báo độ biến động cổ phiếu trong 5 phiên giao dịch tiếp theo bằng GARCH(1,1), XGBoost và LSTM; CLI mặc định dùng ACB từ `data/raw/acb.csv`. Pipeline chạy offline sau khi cài thư viện. Huấn luyện, đánh giá, test và dashboard không tự tải dữ liệu thị trường từ provider; trang Streamlit chỉ nhận CSV do người dùng upload. Đây không phải hệ thống giao dịch hay lời khuyên đầu tư.

## Cấu trúc dự án

- `data/raw/`: CSV OHLCV cục bộ; repo có sẵn ACB, FPT, HPG và VIC.
- `src/data/`, `src/features/`, `src/pipeline/`: làm sạch dữ liệu, tạo đặc trưng/nhãn tương lai, chia tập theo thời gian và điều phối huấn luyện/dự báo.
- `src/models/`, `src/risk/`, `src/evaluation/`, `src/explainability/`: mô hình, cảnh báo, đánh giá và SHAP.
- `scripts/`: các lệnh CLI; `app/`: dashboard Streamlit và trang upload/train offline; `tests/`: kiểm thử; `notebooks/`: khám phá và thử nghiệm.
- `config.yaml`: cấu hình dùng chung; `artifacts/<TICKER>/`: kết quả huấn luyện có thể commit để dashboard cloud hiển thị dữ liệu mẫu.

Luồng chính: CSV thô (có sẵn hoặc do người dùng upload) → kiểm tra/làm sạch → đặc trưng từ hiện tại và quá khứ → nhãn biến động tương lai → chia train/validation/test theo thời gian → huấn luyện và đánh giá → lưu artifact → CLI dự báo hoặc dashboard.

## Cài đặt (Python 3.10+)

Chạy mọi lệnh trong README tại thư mục gốc `votatility-forecasting/` (nơi có `config.yaml`). Ví dụ trên Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install -r requirements.txt
```

Nếu `py -3.11` không tìm thấy runtime, gọi trực tiếp bản Python 3.11 đã cài để tạo `.venv` và kiểm tra bằng `.\.venv\Scripts\python.exe --version`; tránh dùng lệnh `python` khi nó trỏ tới một phiên bản khác chưa được kiểm thử. Nếu PowerShell chặn lệnh kích hoạt, có thể gọi trực tiếp `.\.venv\Scripts\python.exe -m ...` thay cho `python -m ...` trong các ví dụ bên dưới.

## Dữ liệu đầu vào và cấu hình

Mặc định pipeline đọc `data/raw/acb.csv`. Với mã khác, đặt CSV tại `data/raw/<ticker viết thường>.csv` rồi truyền `--ticker <TICKER>`. CSV cần các cột `date, open, high, low, close, volume`; cột ngày cũng chấp nhận tên `time`, `datetime` hoặc `timestamp` như file ACB hiện có. Ngày phải đọc được; giá OHLC phải dương và hợp lệ (`high` không thấp hơn `open`/`close`, `low` không cao hơn chúng); `volume` không âm. Bộ làm sạch sắp xếp ngày, loại dòng thiếu/sai định dạng và ngày trùng (giữ dòng cuối), nhưng từ chối quan hệ OHLC hoặc giá trị âm không hợp lệ. Dữ liệu cần đủ dài cho 5 fold validation, gap và tập test trong `config.yaml`.

`config.yaml` đặt horizon 5 phiên, cửa sổ LSTM 20 phiên, thứ tự 14 feature, tỷ lệ test 15%, 5 fold validation (`test_size=125`, `gap=5`), ngưỡng cảnh báo, tham số mô hình, seed và các đường dẫn. `split.gap` phải ít nhất bằng `data.horizon` để tránh nhãn tương lai chồng qua ranh giới. Các script CLI mặc định dùng ticker `ACB` nếu không truyền `--ticker`; thay `data.ticker` trong YAML không đổi mặc định của CLI. Khi thay horizon, feature hoặc thứ tự feature, cần huấn luyện lại vì CLI dự báo kiểm tra tính tương thích của artifact.

Có thể truyền `--csv đường/dẫn/file.csv` cho lệnh `train` và `predict` để dùng CSV cục bộ khác; đường dẫn tương đối được tính từ gốc repo. Nếu cần tái tạo bảng feature/target để kiểm tra, chạy `python -m scripts.build_dataset --ticker ACB`; file trong `data/processed/` là đầu ra trung gian, không phải đầu vào huấn luyện mặc định.

## Chạy CLI và kiểm thử

Lần chạy nhanh để kiểm tra luồng offline:

```powershell
python -m scripts.train --ticker ACB --model all --fast
python -m scripts.evaluate --ticker ACB
python -m scripts.predict --ticker ACB --model xgboost
```

`train` ghi kết quả vào `artifacts/ACB/` và ghi đè các output cùng tên của ticker đó; dùng `--artifact-dir` để giữ thí nghiệm cũ. `evaluate` chỉ in `metrics.json` đã lưu; `predict` nạp model đã lưu và tái tạo feature từ CSV để dự báo origin mới nhất. `--fast` giới hạn XGBoost ở hai cấu hình thử và LSTM ở tối đa ba epoch; chế độ này chỉ để kiểm tra luồng chạy, không dùng kết luận chất lượng cảnh báo. Bỏ cờ `--fast` để dùng lưới XGBoost trong `config.yaml` (15 cấu hình, 5 fold; 75 lượt fit cho bước chọn tham số) và số epoch LSTM đã cấu hình. Lưới có cả cấu hình 5 cây cũ và các cấu hình 50–300 cây. XGBoost dùng GridSearchCV trên các fold thời gian; số luồng của estimator được cố định và việc song song hóa chỉ diễn ra ở tầng tìm kiếm. Ngưỡng HIGH phục vụ chọn tham số được tính từ train của fold đầu tiên, trước mọi validation. Bộ chọn ưu tiên average precision của HIGH trong các cấu hình có RMSE validation không quá 5% so với RMSE tốt nhất; metadata lưu cả hai chỉ số và kết quả từng cấu hình. Có thể thay `--model all` bằng `garch`, `xgboost`, hoặc `lstm`; để so sánh đủ ba mô hình trên cùng ngày test, chạy `all`.

Kiểm thử nhanh bỏ qua các bài tích hợp đánh dấu `slow`; chạy toàn bộ khi cần kiểm tra cả huấn luyện offline (có thể mất nhiều thời gian):

```powershell
python -m pytest -q -m "not slow"
python -m pytest -q
```

Test `test_offline_acb_end_to_end` tự chạy ba mô hình ở chế độ nhanh. Dữ liệu mới phải được kiểm định riêng trước khi diễn giải kết quả.

Để kiểm tra XGBoost mới mà vẫn giữ artifact chính, có thể dùng `python -m scripts.train --ticker ACB --model xgboost --artifact-dir artifacts/xgboost_trial` rồi `python -m scripts.evaluate --ticker ACB --artifact-dir artifacts/xgboost_trial`. Thay ticker cho FPT, HPG, VIC; mỗi lần chạy chỉ lưu kết quả mô hình được chọn vào thư mục thử nghiệm.

## Chạy dashboard Streamlit

Từ thư mục gốc repo, sau khi cài thư viện, chạy:

```powershell
python -m streamlit run app/app.py
```

Mở Local URL mà Streamlit in trong terminal (thường là `http://localhost:8501`). Nhấn `Ctrl+C` để dừng server. Trang **Huấn luyện** cho tải CSV OHLCV đã có từ máy lên, nhập ticker, chọn XGBoost/GARCH/LSTM hoặc cả ba và bấm **Bắt đầu huấn luyện**. CSV được kiểm tra theo cùng pipeline của CLI trước khi nhận job; giới hạn 10 MB và 50.000 dòng. Không có chức năng tìm kiếm/tải dữ liệu thị trường từ giao diện.

Mỗi lần train có `run_id` riêng tại `artifacts/runs/<run_id>/`: `input.csv`, bản chụp `config.yaml`, `status.json` và thư mục `<TICKER>/` chứa kết quả. Trang Huấn luyện tự cập nhật trạng thái, tiến độ theo **bước** (không phải ước tính thời gian), metrics test và dự báo mới nhất khi job thành công. Có thể đổi trang rồi quay lại chọn lần train trong danh sách. Worker xử lý **một job tại một thời điểm**; job khác sẽ được từ chối cho đến khi job hiện tại xong. Nếu tiến trình Streamlit bị dừng giữa chừng, job được đánh dấu gián đoạn và không tự phục hồi. Bản MVP chưa có xác thực hoặc cách ly dữ liệu giữa người dùng: mọi người truy cập cùng một app có thể xem các run trên server.

Thanh bên cho chọn mã chứng khoán, lần huấn luyện và các trang:

- **Huấn luyện:** upload CSV đã có, train offline và xem kết quả của run.
- **Tổng quan:** đồ thị giá đóng cửa và biến động lịch sử 20 phiên từ CSV đã upload của run được chọn hoặc từ `data/raw/<ticker>.csv` nếu chưa train.
- **Dự báo & Cảnh báo:** dự báo mới nhất, mức NORMAL/WARNING/HIGH và đồ thị dự báo so với thực tế trên test của mô hình đã chọn. Dự báo mới nhất chưa có nhãn thực tế của 5 phiên tiếp theo.
- **So sánh:** MAE/RMSE, chỉ số cảnh báo HIGH và chỉ số ba mức rủi ro trên các ngày test được căn chung. Nếu chỉ train một model, trang này chỉ có model đó.
- **Giải thích:** SHAP toàn cục và theo ngày của XGBoost khi artifact `shap.json` có mặt; GARCH/LSTM chưa có SHAP.

Dashboard vẫn đọc được artifact CLI cũ tại `<paths.artifacts>/<TICKER>/`, đồng thời cho chọn các run do giao diện tạo trong `<paths.artifacts>/runs/`. Artifact CLI tạo bằng `--artifact-dir` ở thư mục khác sẽ không xuất hiện nếu `paths.artifacts` chưa trỏ tới thư mục đó. Các trang kết quả chỉ đọc file đã lưu; không train lại khi đổi trang. Cache kết quả được phân biệt theo phiên bản file để hiện metrics mới sau khi train.

## Triển khai bản demo trên Streamlit Community Cloud

1. Kiểm tra các file trong `artifacts/` trước khi commit, nhất là `runs/*/input.csv` và metadata: repo công khai sẽ công khai cả dữ liệu upload và đường dẫn ghi trong metadata. Commit các artifact mẫu muốn hiển thị cùng mã nguồn, `config.yaml`, `data/raw/` và `.streamlit/config.toml`, rồi push lên GitHub. `artifacts/` hiện không bị Git ignore; Git **không tự thêm** file mới nếu chưa `git add`.
2. Trong [Streamlit Community Cloud](https://share.streamlit.io/), tạo app từ repo/branch đã push, chọn **Main file path** là `app/app.py` và **Python 3.11** trong Advanced settings. Cloud sẽ dùng `app/requirements.txt` nằm cạnh entrypoint; file này chốt các phiên bản đã chạy cục bộ, dùng PyTorch CPU cho LSTM và bỏ `pytest` của môi trường phát triển. Chạy từ thư mục gốc repo để `.streamlit/config.toml` ẩn menu trang tự động như khi chạy local.
3. Mở link được cấp, chọn **Tổng quan**, **Dự báo & Cảnh báo**, **So sánh** hoặc **Giải thích** để xem artifact đã commit. Trang **Huấn luyện** vẫn nhận CSV upload và tạo run trên máy chủ Cloud, nhưng file mới sinh chỉ tồn tại tạm trong phiên chạy; muốn chúng thành dữ liệu demo cố định sau khi app khởi động lại, cần commit/push các artifact đó từ môi trường phát triển.

Community Cloud có giới hạn tài nguyên; huấn luyện đủ ba mô hình, nhất là LSTM và SHAP, có thể chạy chậm hoặc vượt giới hạn. Demo công khai hiện dùng chung một worker, không có tài khoản hay vùng dữ liệu riêng cho từng người xem. Chỉ upload CSV có thể chia sẻ công khai và tránh dùng kết quả sinh trên Cloud như kho lưu trữ lâu dài.

Chế độ **chạy nhanh** chỉ kiểm tra luồng và được đánh dấu trên giao diện; không dùng metrics đó để kết luận chất lượng. Nếu hiện “Chưa có kết quả”, hãy upload/train cho đúng ticker hoặc chọn artifact CLI đã có. Nếu hiện “Chưa có SHAP”, hãy train XGBoost. Dashboard không phát cảnh báo ra email/Telegram hay cập nhật thời gian thực.

## Hợp đồng dữ liệu và mô hình

`src/pipeline/dataset.py` luôn gọi preprocessing, feature builder và target builder trên CSV thô. Dữ liệu được sắp theo ngày, loại dòng feature không hợp lệ khi huấn luyện, nhưng giữ các ngày gần nhất có feature và chưa có target cho suy luận. Chia dữ liệu theo thời gian: test cuối chiếm 15% số origin có target hợp lệ; phần trước test dùng `TimeSeriesSplit(n_splits=5, test_size=125, gap=5, max_train_size=None)`. Gap 5 origin được giữ trước mỗi fold và trước test để target tương lai không chạm sang đoạn kế tiếp. Các fold validation mở rộng tập train theo thời gian; tuning và hiệu chỉnh cảnh báo dùng dự báo out-of-fold. LSTM có thể dùng feature quá khứ làm ngữ cảnh; scaler chỉ fit trên train. XGBoost chỉ nhận danh sách 14 feature trong `config.yaml` theo đúng thứ tự đã lưu; các cột khác trong frame gốc được bộ chọn feature loại bỏ có chủ đích.

`return_1d` là log return nhân 100, đơn vị **điểm phần trăm**. `target_volatility_5d` là độ lệch chuẩn **mẫu** của năm return ở các phiên `t+1…t+5`, cũng theo điểm phần trăm; không annualize. Target không được làm đầu vào. `data/processed/acb_features.csv` là bản cũ có return xấp xỉ `-0.026` tại phiên 2018-12-17, trong khi builder hiện nay cho xấp xỉ `-2.616`; vì vậy pipeline không dùng bản processed này.

Mỗi model nhận frame lịch sử và danh sách origin date, xuất `date`, `model`, `predicted_volatility` cùng đơn vị không âm. XGBoost và LSTM còn lưu `raw_prediction`; giá trị hồi quy âm được chặn ở 0 cho cảnh báo. SHAP giải thích **đầu ra thô** XGBoost, trước khi chặn. GARCH khớp tham số trên train rồi lọc return quan sát đến từng origin với tham số cố định. Căn bậc hai trung bình phương sai dự báo nhiều bước chỉ là **xấp xỉ** target độ lệch chuẩn mẫu thực tế, không phải cùng một đại lượng theo toán học.

Ngưỡng NORMAL/WARNING và nhãn HIGH thực tế lấy quantile **0.50/0.70** của target train thực tế. Ngưỡng dự báo HIGH được hiệu chỉnh riêng từng model trên validation, ưu tiên recall trong giới hạn tỷ lệ cảnh báo và precision tối thiểu. Giới hạn mặc định là 15% số ngày validation; `alert_policy.model_overrides.xgboost` cho phép XGBoost dùng tối đa **30%** để ưu tiên phát hiện HIGH. Đây là giới hạn trên validation, không bảo đảm cùng tỷ lệ trên test; tăng cảnh báo có thể làm tăng báo động sai. Nếu không có ngưỡng thỏa điều kiện, dự báo theo độ biến động dùng lại quantile HIGH train; LSTM có đầu ra điểm HIGH riêng (chỉ khi bật qua cấu hình) dùng ngưỡng fallback `1.0` trên điểm đó. `high_alert_event` đánh dấu lần đầu bước vào HIGH trong chuỗi dự báo đã lưu để lớp thông báo bên ngoài có thể gộp các ngày HIGH liên tiếp. Trong pipeline, test chỉ dùng để báo cáo, không dùng chọn tham số hay ngưỡng. GARCH và LSTM không có SHAP trong bản mẫu.

Các giai đoạn test hiện có đã được xem trong quá trình phát triển, kể cả khi so ngân sách cảnh báo XGBoost 15% và 30%. Vì thế kết quả trên những giai đoạn này là **thử nghiệm lịch sử**, chưa phải bằng chứng trên holdout chưa từng xem. Cần một giai đoạn dữ liệu mới để xác nhận mức cải thiện và số báo động sai.

## Artifact ổn định

`artifacts/<TICKER>/` của CLI hoặc `artifacts/runs/<run_id>/<TICKER>/` của Streamlit (có thể đưa lên Git để làm dữ liệu demo) gồm:

| File | Nội dung |
|---|---|
| `metadata.json` | ticker, target, horizon, danh sách feature, đơn vị, seed, cấu hình model, khoảng ngày và số mẫu mỗi split, số ngày test khớp, chính sách và kết quả hiệu chỉnh cảnh báo trên validation |
| `warning.json` | ngưỡng WARNING/HIGH train và ngưỡng HIGH vận hành theo từng model |
| `<model>.joblib` | model đã fit; chỉ tải artifact đáng tin cậy |
| `metrics.json` | MAE, RMSE, R², số mẫu; precision, recall, F1 và support cho NORMAL/WARNING/HIGH, confusion matrix 3×3 (hàng là mức thực tế, cột là mức dự báo), cùng metric cảnh báo HIGH |
| `train_predictions.csv` | dự báo test/latest được tạo trong pipeline train, mức rủi ro và `high_alert_event` đánh dấu lần chuyển mới sang HIGH; actual của latest trống |
| `shap.json` | global mean absolute SHAP và giá trị local, feature input, base value, raw output, clipped forecast |

Các trang kết quả của dashboard chỉ đọc output đã lưu và CSV cục bộ; trang Huấn luyện chỉ chạy khi người dùng bấm nút. CLI `predict` tái tạo feature từ CSV thô và từ chối model lưu có schema/target/horizon không khớp.

## Thử nghiệm và tài liệu liên quan

Ngoài luồng CLI chính, `scripts/lstm_tune.py` tìm cấu hình LSTM theo các fold thời gian và có thể fit ứng viên đã chọn vào thư mục artifact riêng; `scripts/lstm_walk_forward.py` chạy thí nghiệm LSTM tập trung vào cảnh báo HIGH. Hai luồng này dùng CSV cục bộ và không phải bước bắt buộc để mở dashboard mặc định. Chi tiết và giới hạn diễn giải kết quả có trong [ghi chú XGBoost](docs/XGBOOST_IMPROVEMENTS.md), [ghi chú LSTM](docs/LSTM_IMPROVEMENTS.md) và [báo cáo LSTM walk-forward](reports/lstm_high_walkforward_2026-10-26.md).

## Phạm vi tiếp theo

`scripts/download_data.py` giữ nguyên hành vi provider cũ và không được gọi bởi pipeline hay Streamlit. Việc lựa chọn, kiểm định và tích hợp nguồn dữ liệu khác được để lại cho giai đoạn sau. `.venv/` bị Git ignore; `artifacts/` có thể được đưa lên Git, nhưng cần kiểm tra dữ liệu và metadata trước khi push. Khi chuyển máy vẫn cần tạo lại môi trường Python.
