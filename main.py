import pandas as pd
import numpy as np

# 讀取資料
data = pd.read_csv("train.csv")

x = data["X"].to_numpy(dtype=float)/10
y = data["Y"].to_numpy(dtype=float)

# 初始化資料
parms = np.array([0.0]*3)
features = np.array([
    x**2,
    x**1,
    x**0
]).T

# 學習率
learning_rate = 0.1

# 開始訓練
loss = 1.0
while(loss>=0.0001):

    # 模型預測
    prediction = features @ parms

    # 計算誤差
    error = prediction - y

    # 計算梯度
    gradient = 2 * np.mean(features * error[:,None] , axis=0)

    # 更新參數
    parms -= gradient * learning_rate

    # 計算 loss
    loss = np.mean(error ** 2)

    print("loss = " , loss)

print("\n訓練完成")

x_test = int(input("\n預測 x = "))/10
text_features = np.array([
    x_test**2,
    x_test**1,
    x_test**0
]).T
text_prediction = text_features @ parms
print("y = ", int(text_prediction)+1)
