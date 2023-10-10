import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
# 读取数据
data = pd.read_excel('E://homework//Folds5x2_pp.xlsx')  # 替换'your_data.xlsx'为数据文件的实际路径

# 划分特征和目标变量
X = data[['AT', 'V', 'AP', 'RH']]
y = data['PE']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 打印参数
theta_0 = model.intercept_
theta_1, theta_2, theta_3, theta_4 = model.coef_
print(f"θ_0 (intercept): {theta_0}")
print(f"θ_1: {theta_1}")
print(f"θ_2: {theta_2}")
print(f"θ_3: {theta_3}")
print(f"θ_4: {theta_4}")

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算均方根误差
mse = mean_squared_error(y_test, y_pred)
print(f"均方根误差 (MSE): {mse}")
# 计算平均绝对误差（MAE）
mae = mean_absolute_error(y_test, y_pred)
print(f"平均绝对误差 (MAE): {mae}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='red', marker='x', label='real')
plt.scatter(y_test, y_test, color='blue', marker='o', label='predict')

plt.xlabel('rael_value')
plt.ylabel('predict_value')
plt.title('linear_regression')
plt.legend()
plt.grid(True)
plt.show()