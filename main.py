import os
import concurrent.futures as con
import numpy as np
from core import check_train_file, load_train_data, make_train, predict, save_train_data


def ask() -> int:
    while True:
        print("""請選擇：
1 載入模型
2 重新訓練""")
        try:
            choice = int(input(""))
            if choice in [1, 2]:
                return choice
            print("選擇錯誤")
        except ValueError:
            print("選擇錯誤")

def main():
    choice = ask() if check_train_file() else 2
    if choice == 1:
        parms, scale_factor = load_train_data()
        print("模型載入完成")
        param_count = parms.size
        total_saved_values = parms.size + scale_factor.size
        print(f"參數量：{param_count}")
        print(f"總數值：{total_saved_values}")
    else:
        # 加速一下(?
        total = int(input("訓練次數："))
        if total >= 200:
            cpu = os.cpu_count() or 4
            print(f"With cpu: {cpu}")
            chunks = [total // cpu + (1 if i < total % cpu else 0) for i in range(cpu)]
            with con.ProcessPoolExecutor(max_workers=cpu) as executor:
                futures = [executor.submit(make_train, c) for c in chunks]
                results = []
                for i, future in enumerate(con.as_completed(futures), 1):
                    results.append(future.result())
                    print(f"Chunked to {i}")
                data = np.vstack(results).astype(float)
        else:
            data = np.array(make_train(),dtype=float)

        pre_inputs = data[:, :-1]
        outputs = data[:, -1]
        scale_factor = np.max(pre_inputs, axis=0)
        inputs = pre_inputs / scale_factor
        num_inputs = inputs.shape[1]

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

        print("\n訓練完成")
        save_train_data(parms, scale_factor)

    predict(parms, scale_factor)

if __name__ == "__main__":
    main()