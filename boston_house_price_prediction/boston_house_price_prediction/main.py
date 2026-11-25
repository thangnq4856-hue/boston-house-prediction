import torch
from src.data_loader import load_and_preprocess_data
from src.models import MLPRegressor, get_traditional_models
from src.train import train_mlp
from src.evaluate import evaluate_and_compare

def main():
    # Cố định seed để kết quả tái lập được (reproducible)
    torch.manual_seed(42)

    print("--- BƯỚC 1: TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU ---")
    data_dict = load_and_preprocess_data(test_size=0.2, batch_size=32, random_state=42)
    input_dim = data_dict["X_train_np"].shape[1]

    print("\n--- BƯỚC 2: KHỞI TẠO MÔ HÌNH PYTORCH MLP ---")
    mlp_model = MLPRegressor(input_dim=input_dim)
    print(mlp_model)

    print("\n--- BƯỚC 3: HUẤN LUYỆN MÔ HÌNH PYTORCH MLP ---")
    train_mlp(mlp_model, data_dict, epochs=100, lr=0.01)

    print("\n--- BƯỚC 4: ĐÁNH GIÁ VÀ SO SÁNH CÁC MÔ HÌNH ---")
    traditional_models = get_traditional_models(random_state=42)
    comparison_df = evaluate_and_compare(mlp_model, traditional_models, data_dict)

    print("=== BẢNG SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH ===")
    print(comparison_df.to_string(index=False))

if __name__ == "__main__":
    main()