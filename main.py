import pandas as pd
import numpy as np

# 讀取資料
data = pd.read_csv("train.csv", header = None)

pre_inputs = data.iloc[:,:-1].to_numpy(dtype=float)
outputs = data.iloc[:,-1].to_numpy(dtype=float)
scale_factor = np.max(pre_inputs,axis=0)
inputs = pre_inputs / scale_factor
num_inputs = inputs.shape[1]

# 初始化資料
parms = np.array([0.0]*(num_inputs*2+1))
features_list = []
for i in range(num_inputs):
    features_list.append(inputs[:,i]**2)
    features_list.append(inputs[:,i]**1)
features_list.append(inputs[:,0]**0)
features = np.array(features_list).T

# 學習率
learning_rate = 0.1

# 開始訓練
max_times = 100000 
loss = 1.0
while(loss>=0.0001 and max_times):
    max_times -= 1

    # 模型預測
    prediction = features @ parms

    # 計算誤差
    error = prediction - outputs

    # 計算梯度
    gradient = 2 * np.mean(features * error[:,None] , axis=0)

    # 更新參數
    parms -= gradient * learning_rate

    # 計算 loss
    loss = np.mean(error ** 2)
    if max_times%1000 == 0:
        print("loss = " , loss)

print("\n訓練完成")

# text
text = [i for i in range(num_inputs)]
for i in range(num_inputs):
    text[i] = int(input(f"第{i+1}個輸入 = "))
text = np.array(text,dtype=float)
text /= scale_factor

text_features_list = []
for i in range(num_inputs):
    text_features_list.append(text[i]**2)
    text_features_list.append(text[i]**1)
text_features_list.append(text[0]**0)
text_features = np.array(text_features_list).T

text_prediction = text_features @ parms
print("answer = ", round(text_prediction))
