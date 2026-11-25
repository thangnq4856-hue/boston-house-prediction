import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

class MLPRegressor(nn.Module):
    """
    Mô hình Multilayer Perceptron (MLP) cho bài toán hồi quy bằng PyTorch.
    """
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Đầu ra tuyến tính (linear) cho hồi quy
        )

    def forward(self, x):
        return self.network(x)

def get_traditional_models(random_state=42):
    """
    Các mô hình hồi quy truyền thống để so sánh ở Task 6.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state)
    }