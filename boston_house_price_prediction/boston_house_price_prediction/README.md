# So sánh PyTorch MLP và các mô hình hồi quy truyền thống – Boston Housing

Dự án học phần **Máy học (Machine Learning)**. Mục tiêu là dự đoán giá nhà (`MEDV`, đơn vị nghìn USD) trên bộ dữ liệu Boston Housing bằng mạng nơ-ron nhiều lớp (MLP) viết bằng **PyTorch**, sau đó so sánh với ba mô hình hồi quy truyền thống của Scikit-learn: Linear Regression, Decision Tree và Random Forest.

Dự án được tổ chức theo hai phần dùng chung một nguồn xử lý:

- `src/`: mã nguồn Python chia module để tải/tiền xử lý dữ liệu, khai báo mô hình, huấn luyện MLP, đánh giá và so sánh.
- `main.py` và `demo.ipynb`: hai cách chạy cùng một quy trình. `main.py` chạy toàn bộ pipeline bằng một lệnh; `demo.ipynb` chạy từng bước để minh họa/thuyết trình. Cả hai đều import từ `src/` để tránh viết trùng logic.

## 1. Cấu trúc thư mục

```text
.
├── README.md
├── requirements.txt
├── main.py                   # chạy toàn bộ pipeline
├── demo.ipynb                # notebook demo từng bước
├── data/
│   └── BostonHousing.csv     # dữ liệu gốc
├── src/
│   ├── data_loader.py        # tải, tiền xử lý, chia train/test, tạo DataLoader
│   ├── models.py             # MLPRegressor và get_traditional_models()
│   ├── train.py              # huấn luyện MLP (train_mlp)
│   └── evaluate.py           # tính MSE, MAE, R², so sánh và vẽ biểu đồ
└── outputs/
    ├── saved_models/         # trọng số MLP (tạo khi chạy)
    ├── plots/                # biểu đồ (tạo khi chạy)
    └── metrics_comparison.csv
```

> Lưu ý: tên và vị trí file có thể khác đôi chút tùy cách nhóm tổ chức thư mục. Hãy chỉnh lại mục này cho khớp với repo cuối cùng.

## 2. Cài đặt môi trường

Khuyến nghị Python 3.9 trở lên.

```bash
python -m venv venv

# Windows PowerShell / CMD
venv\Scripts\activate

# macOS / Linux
# source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Thư viện sử dụng: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `torch`.

## 3. Chạy source code

Chạy từ thư mục gốc của dự án:

```bash
python main.py
```

Lệnh này thực hiện:

1. Cố định `torch.manual_seed(42)` để kết quả tái lập được.
2. Tải dữ liệu Boston Housing và tiền xử lý.
3. Chia dữ liệu Train/Test theo tỷ lệ 80/20 với `random_state=42`, tạo DataLoader với `batch_size=32`.
4. Khởi tạo `MLPRegressor` với số đầu vào bằng số đặc trưng (13).
5. Huấn luyện MLP 100 epoch với `lr=0.01`, hàm mất mát MSE.
6. Huấn luyện ba mô hình baseline bằng Scikit-learn (`random_state=42`).
7. Tính MSE, MAE, R² trên tập kiểm tra và in bảng so sánh.

Kết quả (biểu đồ, trọng số, bảng so sánh) được lưu trong `outputs/`.

## 4. Chạy Notebook

```bash
jupyter lab
```

Mở `demo.ipynb`, chọn kernel Python đã cài `requirements.txt`, rồi chọn **Restart Kernel and Run All**.

Notebook gồm ba bước: tải dữ liệu, khởi tạo và huấn luyện MLP, đánh giá và so sánh với các mô hình truyền thống.

## 5. Dữ liệu và phương pháp

- Dataset Boston Housing: 506 mẫu, 13 đặc trưng đầu vào, 1 biến mục tiêu (giá trị trung vị của nhà, nghìn USD).
- Chia 80/20 với `test_size=0.2`, `random_state=42`.
- Đánh giá trên tập kiểm tra bằng ba chỉ số: **MSE**, **MAE** và **R-squared**.

**Kiến trúc PyTorch MLP (`MLPRegressor`):**

```text
Linear(13 -> 32) -> ReLU -> Linear(32 -> 16) -> ReLU -> Linear(16 -> 1)
```

| Tham số | Giá trị |
|---|---|
| Số epoch (`main.py`) | 100 |
| Learning rate | 0.01 |
| Batch size | 32 |
| Hàm mất mát | MSE |
| Seed | 42 |

**Mô hình baseline (Scikit-learn):** Linear Regression, Decision Tree, Random Forest.

## 6. Kết quả tham chiếu

Kết quả trên tập kiểm tra trong bản `demo.ipynb` (có thể có sai khác ở các chữ số cuối do phiên bản thư viện và seed):

| Mô hình | MSE | MAE | R² |
|---|---:|---:|---:|
| PyTorch MLP | 11.467 | 2.148 | 0.8436 |
| Linear Regression | 24.291 | 3.189 | 0.6688 |
| Decision Tree | 17.816 | 2.725 | 0.7571 |
| Random Forest | **7.890** | **2.038** | **0.8924** |

Nhận xét:

- Random Forest cho kết quả tốt nhất trên cả ba chỉ số.
- PyTorch MLP đứng thứ hai, vượt rõ rệt Linear Regression và Decision Tree, cho thấy mạng nơ-ron nắm bắt được quan hệ phi tuyến trong dữ liệu.
- Linear Regression kém nhất do giả định quan hệ tuyến tính quá đơn giản.
- Với bộ dữ liệu nhỏ (506 mẫu), các mô hình dạng cây thường rất cạnh tranh so với mạng nơ-ron. Không nên kết luận chắc chắn chỉ từ một lần chia Train/Test.
- Trong `demo.ipynb`, hàm `train_mlp` được gọi 2 lần liên tiếp nên MLP thực tế đã huấn luyện 200 epoch. Khi chạy `main.py` (100 epoch), kết quả có thể chênh lệch nhẹ so với bảng trên.

## 7. Đầu ra

Sau khi chạy, các file sau được tạo trong thư mục `outputs/`:

- `saved_models/mlp_model.pth`: trọng số mô hình MLP đã huấn luyện.
- `plots/training_history.png`: đường cong MSE và MAE của tập train và validation theo epoch.
- `plots/pytorch_mlp_actual_vs_predicted.png`: biểu đồ giá trị thực tế so với giá trị dự đoán.
- `metrics_comparison.csv`: bảng so sánh hiệu năng các mô hình.

## 8. Lưu ý

- Kết quả phụ thuộc vào dữ liệu, seed và phiên bản thư viện. Hãy chạy lại `main.py` hoặc notebook trong môi trường của nhóm trước khi demo.
- Chạy các lệnh từ thư mục gốc để các import `from src...` hoạt động đúng.
- Boston Housing là bộ dữ liệu cũ, chỉ dùng cho mục đích học tập.

## 9. Tài liệu tham khảo

- PyTorch Documentation: https://pytorch.org/docs/
- Scikit-learn Documentation: https://scikit-learn.org/stable/
- Harrison, D. and Rubinfeld, D.L. (1978). Hedonic prices and the demand for clean air. *Journal of Environmental Economics and Management*.
