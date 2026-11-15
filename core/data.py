import os
import numpy as np

def check_train_file(filepath: str = "model_weights.npz") -> bool:
    return os.path.exists(filepath)

def save_train_data(parms: np.ndarray, scale_factor: np.ndarray, filepath: str = "model_weights.npz") -> None:
    folder = os.path.dirname(filepath)
    if folder:
        os.makedirs(folder, exist_ok=True)

    np.savez(filepath, parms=parms, scale_factor=scale_factor)
    print(f"Saves as {filepath}")


def load_train_data(filepath: str = "model_weights.npz", *, force: bool=False) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(filepath):
        if not force:
            return np.array([]), np.array([])
        else:
            raise FileNotFoundError()
    
    with np.load(filepath) as data:
        return data["parms"], data["scale_factor"]
