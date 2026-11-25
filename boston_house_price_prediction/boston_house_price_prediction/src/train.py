import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def train_mlp(model, data_dict, epochs=100, lr=0.01, save_dir="outputs"):
    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = data_dict["train_loader"]
    X_train_tensor = data_dict["X_train_tensor"]
    y_train_tensor = data_dict["y_train_tensor"]
    X_test_tensor = data_dict["X_test_tensor"]
    y_test_tensor = data_dict["y_test_tensor"]

    history = {
        "train_mse": [],
        "val_mse": [],
        "train_mae": [],
        "val_mae": []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

        # Đánh giá trên toàn bộ tập Train và Test (Validation) sau mỗi epoch
        model.eval()
        with torch.no_grad():
            train_preds = model(X_train_tensor)
            train_mse = criterion(train_preds, y_train_tensor).item()
            train_mae = mae_criterion(train_preds, y_train_tensor).item()

            val_preds = model(X_test_tensor)
            val_mse = criterion(val_preds, y_test_tensor).item()
            val_mae = mae_criterion(val_preds, y_test_tensor).item()

        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch [{epoch:3d}/{epochs}] | Train MSE: {train_mse:.4f}, MAE: {train_mae:.4f} | Val MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")

    # Lưu trọng số mô hình
    model_dir = os.path.join(save_dir, "saved_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "mlp_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n[INFO] Đã lưu trọng số mô hình PyTorch tại: {model_path}")

    # Vẽ biểu đồ MSE và MAE theo epoch
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    history_plot_path = os.path.join(plots_dir, "training_history.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Biểu đồ 1: MSE theo epoch
    axes[0].plot(history["train_mse"], label="Train")
    axes[0].plot(history["val_mse"], label="Test (validation)")
    axes[0].set_title("MSE theo epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Biểu đồ 2: MAE theo epoch
    axes[1].plot(history["train_mae"], label="Train")
    axes[1].plot(history["val_mae"], label="Test (validation)")
    axes[1].set_title("MAE theo epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(history_plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Đã lưu biểu đồ quá trình huấn luyện tại: {history_plot_path}")

    return history