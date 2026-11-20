import os
import numpy as np

def check_train_result(filepath: str = "train_result/model_weights.npz") -> bool:
    return os.path.exists(filepath)

def save_train_result(parms: np.ndarray, scale_factor: np.ndarray, max_degree: int, filepath: str = "train_result/model_weights.npz") -> None:
    num_inputs = len(scale_factor)
    for i in range(num_inputs):
        for dg in range(max_degree, 0, -1):
            parms[i * 2] /= (scale_factor[i] ** dg)

    folder = os.path.dirname(filepath)
    if folder:
        os.makedirs(folder, exist_ok=True)

    np.savez(filepath, parms=parms, num_inputs=num_inputs, max_degree=max_degree)
    print(f"train result save at {filepath}\n")


def load_train_result(filepath: str = "train_result/model_weights.npz", *, force: bool=False) -> tuple[np.ndarray, int, int]:
    if not os.path.exists(filepath):
        if not force:
            return np.array([]), np.array([])
        else:
            raise FileNotFoundError()
    
    with np.load(filepath) as data:
        return data["parms"], int(data["num_inputs"].item()), int(data["max_degree"].item())
