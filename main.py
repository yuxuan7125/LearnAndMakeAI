import pandas as pd
import numpy as np

# 讀取資料
data = pd.read_csv("train.csv")

x = data["X"].to_numpy(dtype=float)/10
y = data["Y"].to_numpy(dtype=float)

# 初始參數
a = 0.0
b = 0.0
c = 0.0

# 學習率
learning_rate = 0.1

# 開始訓練
loss = 1.0
while(loss>=0.0001):

    # 模型預測
    prediction = a*x**2 + b*x + c

    # 計算誤差
    error = prediction - y

    # 計算梯度
    da = np.mean(2 * error * x**2)
    db = np.mean(2 * error * x)
    dc = np.mean(2 * error)

    # 更新參數
    a -= learning_rate * da
    b -= learning_rate * db
    c -= learning_rate * dc

    # 計算 loss
    loss = np.mean(error ** 2)

    print("loss = " , loss)

print("\n訓練完成")
print("a = " , a)
print("b = " , b)
print("c = " , c)


print("\n預測 x = ")
x_input = int(input())
print("y = ", a*(x_input/10)**2 + b*(x_input/10) + c)



