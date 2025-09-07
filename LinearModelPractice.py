import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义线性模型
def forward(x, w, b):
    """线性模型: y_pred = x + w + b"""
    return x + w + b

# 定义损失函数（MSE）
def compute_loss(x_data, y_data, w, b):
    """计算均方误差损失"""
    total_loss = 0
    for x, y_true in zip(x_data, y_data):
        y_pred = forward(x, w, b)
        total_loss += (y_pred - y_true) ** 2
    return total_loss / len(x_data)

# 训练数据
x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([4.0, 6.0, 8.0])  # 真实关系: y = x + 3

# 创建参数网格
w_values = np.arange(0.0, 4.1, 0.1)  # w从0到4
b_values = np.arange(0.0, 4.1, 0.1)  # b从0到4
W, B = np.meshgrid(w_values, b_values)

# 使用向量化计算MSE损失
# 扩展维度以便广播计算
X_data = x_data[:, np.newaxis, np.newaxis]
Y_data = y_data[:, np.newaxis, np.newaxis]

# 计算所有数据点的预测值 (向量化)
Y_pred = X_data + W + B

# 计算MSE (向量化)
MSE = np.mean((Y_pred - Y_data) ** 2, axis=0)

# 创建3D图形
fig = plt.figure(figsize=(15, 6))

# 3D曲面图
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(W, B, MSE, cmap='viridis', alpha=0.8,
                       linewidth=0, antialiased=True)
ax1.set_xlabel('Weight (ω)', fontsize=12, labelpad=10)
ax1.set_ylabel('Bias (b)', fontsize=12, labelpad=10)
ax1.set_zlabel('MSE Loss', fontsize=12, labelpad=10)
ax1.set_title('3D Cost Surface: MSE vs (ω, b)', fontsize=14, pad=15)
fig.colorbar(surf, ax=ax1, shrink=0.6, label='MSE Value')

# 等高线图
ax2 = fig.add_subplot(122, projection='3d')
contour = ax2.contour3D(W, B, MSE, 20, cmap='plasma')
ax2.set_xlabel('Weight (ω)', fontsize=12, labelpad=10)
ax2.set_ylabel('Bias (b)', fontsize=12, labelpad=10)
ax2.set_zlabel('MSE Loss', fontsize=12, labelpad=10)
ax2.set_title('3D Contour Plot', fontsize=14, pad=15)
fig.colorbar(contour, ax=ax2, shrink=0.6, label='MSE Value')

plt.tight_layout()
plt.show()

# 找到最小MSE对应的参数
min_idx = np.unravel_index(np.argmin(MSE), MSE.shape)
optimal_w = W[min_idx]
optimal_b = B[min_idx]
min_mse = MSE[min_idx]

print("=" * 50)
print("线性模型: ŷ = x + ω + b")
print("训练数据:")
for i, (x, y) in enumerate(zip(x_data, y_data)):
    print(f"  样本 {i+1}: x={x}, y={y}")

print(f"\n最优参数: ω = {optimal_w:.2f}, b = {optimal_b:.2f}")
print(f"最小MSE: {min_mse:.4f}")

print("\n验证最优参数:")
for x, y_true in zip(x_data, y_data):
    y_pred = forward(x, optimal_w, optimal_b)
    error = y_pred - y_true
    print(f"  x={x}: y_true={y_true}, ŷ={y_pred:.2f}, error={error:.2f}")