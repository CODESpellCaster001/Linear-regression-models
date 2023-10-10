import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
# 输入数据
X = np.array([58.0, 55.0, 85.0, 75.0, 70.0, 78.0, 77.0, 83.0, 80.0, 85.0], dtype=np.float32)
Y = np.array([168, 165, 175, 178, 180, 170, 173, 183, 179, 190], dtype=np.float32)

# 归一化数据
X_mean = X.mean()
X_std = X.std()
Y_mean = Y.mean()
Y_std = Y.std()

X = (X - X_mean) / X_std
Y = (Y - Y_mean) / Y_std

# 转换为PyTorch张量
X = torch.from_numpy(X)
Y = torch.from_numpy(Y)

# 定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 一个输入特征，一个输出特征

    def forward(self, x):
        return self.linear(x)

# 创建模型实例
model = LinearRegression()

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X.view(-1, 1))
    loss = criterion(outputs, Y.view(-1, 1))

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# 预测新的数据点
X_pred = torch.tensor([(76.0 - X_mean) / X_std], dtype=torch.float32)
Y_pred = model(X_pred.view(-1, 1))

# 反归一化预测结果
Y_pred = Y_pred * Y_std + Y_mean

print("预测结果:", Y_pred.item())

# 绘制线性拟合图
plt.scatter(X.numpy(), Y.numpy(), label='sourse data')
plt.plot(X.numpy(), outputs.detach().numpy(), label='analyze', color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("linear fitting")
plt.legend()
plt.show()
