import os
import numpy as np
from core import *
import pandas as pd
import random

def ask() -> int:
    while True:
        print("""請選擇：
1 預測答案
2 訓練模型\n""")
        try:
            choice = int(input(""))
            print("\n")
            if choice in [i for i in range (1,3)]:
                return choice
            print("選擇錯誤\n")
        except ValueError:
            print("選擇錯誤\n")

def main():
    choice = ask() if check_train_result() else 2
    if choice == 1:
        predict()
    if choice == 2:
        # 讀取資料 & 整理資料
        train_data_filepath = "train_data/train_data.csv"
        # train_data_filepath = str(input("訓練檔案(.csv)："))
        if not os.path.exists(train_data_filepath):
            make_train_data(train_data_filepath)
        data = pd.read_csv(train_data_filepath, header=None).to_numpy(dtype=float)

        num_data = len(data)
        data_idx = [random.choices([0,1], weights=[20,80])[0] for i in range(num_data)]
        train_idx = []
        text_idx = []
        for i in range(num_data):
            if data_idx[i]:
                train_idx.append(i)
            else:
                text_idx.append(i)

        pre_inputs = data[train_idx, :-1]
        inputs_scale = np.max(pre_inputs, axis=0)
        inputs = pre_inputs / inputs_scale
        outputs = data[train_idx, -1]
        num_inputs = len(inputs_scale)
        max_degree = 2

        # 初始化資料
        parms = np.array([0.0] * (num_inputs * max_degree + 1))
        features_list = []
        for i in range(num_inputs):
            for dg in range(max_degree, 0, -1):
                features_list.append(inputs[:, i] ** dg)
        features_list.append(inputs[:, 0] ** 0)
        features = np.array(features_list).T

        # 學習率與訓練迴圈
        learning_rate = 0.1
        max_times = 100000
        loss = 10.0**100
        while loss >= 1.0 and max_times:
            max_times -= 1
            prediction = features @ parms
            error = prediction - outputs
            gradient = 2 * np.mean(features * error[:, None], axis=0)
            parms -= gradient * learning_rate
            loss = np.mean(error**2)
            if max_times % 1000 == 0:
                print("loss = ", loss)

        print("\n訓練完成")
        save_train_result(parms, inputs_scale, max_degree)
        text_loss = get_loss(data[text_idx])
        print(f"loss差值：{loss-text_loss}\n")

if __name__ == "__main__":
    while True:
        main()