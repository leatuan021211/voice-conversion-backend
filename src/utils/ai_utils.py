from logging import getLogger
import torch


def get_device() -> str:
    """
    Get the device type (CPU or GPU) based on the availability of CUDA.
    
    Returns:
        str: Device type ('cuda' or 'cpu').
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device