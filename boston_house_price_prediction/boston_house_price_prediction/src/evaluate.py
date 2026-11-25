import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_and_compare(mlp_model, traditional_models, data_dict, output_dir="outputs"):
    """
    Đánh giá mô hình PyTorch MLP, vẽ biểu đồ và so sánh với các mô hình sklearn.
    """
    X_test_tensor = data_dict["X_test_tensor"]
    y_test = data_dict["y_test_np"]
    X_train = data_dict["X_train_np"]
    y_train = data_dict["y_train_np"]
    X_test = data_dict["X_test_np"]

    # 1. Dự đoán với PyTorch MLP
    mlp_model.eval()
    with torch.no_grad():
        mlp_preds = mlp_model(X_test_tensor).numpy().flatten()

    results = []
    results.append({
        "Model": "PyTorch MLP",
        "MSE": mean_squared_error(y_test, mlp_preds),
        "MAE": mean_absolute_error(y_test, mlp_preds),
        "R-squared": r2_score(y_test, mlp_preds)
    })

    # 2. Vẽ biểu đồ Actual vs Predicted cho PyTorch MLP
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "pytorch_mlp_actual_vs_predicted.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, mlp_preds, alpha=0.7, edgecolors="k", label="Dự đoán (MLP)")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Đường lý tưởng (y = x)")
    plt.xlabel("Giá thực tế (MEDV)")
    plt.ylabel("Giá dự đoán (MEDV)")
    plt.title("PyTorch MLP: Giá thực tế vs Giá dự đoán")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Đã lưu biểu đồ tại: {plot_path}")

    # 3. Huấn luyện và đánh giá các mô hình truyền thống
    for name, model in traditional_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results.append({
            "Model": name,
            "MSE": mean_squared_error(y_test, preds),
            "MAE": mean_absolute_error(y_test, preds),
            "R-squared": r2_score(y_test, preds)
        })

    # 4. Lưu và trả về bảng so sánh
    comparison_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metrics_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"[INFO] Đã lưu bảng so sánh tại: {csv_path}")

    return comparison_df