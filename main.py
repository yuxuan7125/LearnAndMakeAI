import os
import concurrent.futures as con
import numpy as np
from core import *
import pandas as pd

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
        parms, num_inputs = load_train_result()
        predict(parms, num_inputs)
    if choice == 2:
        # 讀取資料
        train_data_filepath = "train_data/train_data.csv"
        # train_data_filepath = str(input("訓練檔案(.csv)："))
        if not os.path.exists(train_data_filepath):
            make_train_data(train_data_filepath)
        data = pd.read_csv(train_data_filepath, header=None)
        pre_inputs = data.iloc[:, :-1].to_numpy(dtype=float)
        inputs_scale = np.max(pre_inputs, axis=0)
        inputs = pre_inputs / inputs_scale
        outputs = data.iloc[:, -1].to_numpy(dtype=float)
        num_inputs = len(inputs_scale)

        # 初始化資料
        parms = np.array([0.0] * (num_inputs * 2 + 1))
        features_list = []
        for i in range(num_inputs):
            features_list.append(inputs[:, i] ** 2)
            features_list.append(inputs[:, i] ** 1)
        features_list.append(inputs[:, 0] ** 0)
        features = np.array(features_list).T

        # 學習率與訓練迴圈
        learning_rate = 0.1
        max_times = 100000
        loss = 1.0
        while loss >= 0.1 and max_times:
            max_times -= 1
            prediction = features @ parms
            error = prediction - outputs
            gradient = 2 * np.mean(features * error[:, None], axis=0)
            parms -= gradient * learning_rate
            loss = np.mean(error**2)
            if max_times % 1000 == 0:
                print("loss = ", loss)

        print("\n訓練完成\n")
        save_train_result(parms, inputs_scale)

if __name__ == "__main__":
    while True:
        main()