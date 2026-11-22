import numpy as np
from core import load_train_result


def predict(filepath: str = "train_result/model_weights.npz") -> None:
    parms, num_inputs, degree = load_train_result()
    
    terms = []
    first = True
    for i in range(num_inputs):
        for j,dg in enumerate(degree):
            w = round(parms[i * len(degree) + j])
            if w:
                if first:
                    first = False
                    terms.append(f"{w}*x{i+1}^{dg}")
                else:
                    terms.append(f"{w:+}*x{i+1}^{dg}")

    bias = round(parms[-1])
    if bias:
        terms.append(f"{bias:+}")

    print("預測公式：y = " + " ".join(terms).replace("1*","").replace("^1","").replace("+","+ ").replace("-","- ") + "\n")

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