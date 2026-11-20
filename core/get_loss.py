import numpy as np
from core import load_train_result

def get_loss(data: np.ndarray, filepath: str = "train_result/model_weights.npz") -> float:
    inputs = data[:,:-1]
    outputs = data[:,-1]
    parms, num_inputs = load_train_result(filepath)
    features_list = []
    for i in range(num_inputs):
        features_list.append(inputs[:, i] ** 2)
        features_list.append(inputs[:, i] ** 1)
    features_list.append(inputs[:, 0] ** 0)
    features = np.array(features_list).T
    prediction = features @ parms
    error = prediction - outputs
    loss = np.mean(error**2)

    return loss
