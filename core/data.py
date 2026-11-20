import os
import numpy as np

def check_train_result(filepath: str = "train_result/model_weights.npz") -> bool:
    return os.path.exists(filepath)

def save_train_result(parms: np.ndarray, scale_factor: np.ndarray, filepath: str = "train_result/model_weights.npz") -> None:
    num_inputs = len(scale_factor)
    for i in range(num_inputs):
        parms[i * 2] /= (scale_factor[i] ** 2)
        parms[i * 2 + 1] /= scale_factor[i]

    folder = os.path.dirname(filepath)
    if folder:
        os.makedirs(folder, exist_ok=True)

    np.savez(filepath, parms=parms, num_inputs=num_inputs)
    print(f"train result save at {filepath}\n")


def load_train_result(filepath: str = "train_result/model_weights.npz", *, force: bool=False) -> tuple[np.ndarray, int]:
    if not os.path.exists(filepath):
        if not force:
            return np.array([]), np.array([])
        else:
            raise FileNotFoundError()
    
    with np.load(filepath) as data:
        return data["parms"],int(data["num_inputs"].item())
