import numpy as np


def predict(parms: np.ndarray, scale_factor: np.ndarray) -> None:
    num_inputs = len(scale_factor)
    terms = []
    for i in range(num_inputs):
        w2 = parms[i * 2] / (scale_factor[i] ** 2)
        w1 = parms[i * 2 + 1] / scale_factor[i]
        terms.append(f"{w2:.4f} * x{i+1}^2")
        terms.append(f"{w1:+.4f} * x{i+1}")

    bias = parms[-1]
    terms.append(f"{bias:+.4f}")

    print("預測公式：y = " + " ".join(terms))

    text = [0.0] * num_inputs
    for i in range(num_inputs):
        text[i] = float(input(f"第{i+1}個輸入 = "))
    text = np.array(text, dtype=float) / scale_factor

    text_features_list = []
    for i in range(num_inputs):
        text_features_list.append(text[i] ** 2)
        text_features_list.append(text[i] ** 1)
    text_features_list.append(text[0] ** 0)
    text_features = np.array(text_features_list).T

    text_prediction = text_features @ parms
    print("answer = ", round(text_prediction))