import numpy as np
from core import load_train_result


def predict(filepath: str = "train_result/model_weights.npz") -> None:
    parms, num_inputs, degree = load_train_result()
    
    terms = []
    degree
    for i in range(num_inputs):
        for j,dg in enumerate(degree):
            w = parms[i * len(degree) + j]
            if i==0 and j==0:
                terms.append(f"{w:.4f} * x{i+1}^{dg}")
            else:
                terms.append(f"{w:+.4f} * x{i+1}^{dg}")

    bias = parms[-1]
    terms.append(f"{bias:+.4f}")

    print("預測公式：y = " + " ".join(terms) + "\n")

    text = [0.0] * num_inputs
    for i in range(num_inputs):
        text[i] = float(input(f"第{i+1}個輸入 = "))
    text = np.array(text, dtype=float)

    text_features_list = []
    for i in range(num_inputs):
        for dg in degree:
            text_features_list.append(text[i] ** dg)
    text_features_list.append(text[0] ** 0)
    text_features = np.array(text_features_list)

    text_prediction = text_features @ parms
    print(f"answer = {round(text_prediction)}\n")