import numpy as np
from core import load_train_result

def get_loss(data: np.ndarray, filepath: str = "train_result/model_weights.npz") -> float:
    inputs = data[:,:-1]
    outputs = data[:,-1]
    parms, num_inputs, max_degree = load_train_result(filepath)
    features_list = []
    for i in range(num_inputs):
        for dg in range(max_degree, 0, -1):
            features_list.append(inputs[:, i] ** dg)
    features_list.append(inputs[:, 0] ** 0)
    features = np.array(features_list).T
    prediction = features @ parms
    error = prediction - outputs
    loss = np.mean(error**2)

    return loss
