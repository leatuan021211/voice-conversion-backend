from torch import Tensor


def normalize_tensor(tensor: Tensor, mean: float = 0.0, std: float = 1.0):
    """
    Normalize a mel-spectrogram tensor with fixed mean and std.

    Args:
        tensor (Tensor): Input mel tensor of shape [..., mel_bins]
        mean (float): Global mean
        std (float): Global std

    Returns:
        Tensor: Normalized tensor
    """
    return (tensor - mean) / std

def denormalize_tensor(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """
    Normalize a mel-spectrogram tensor with fixed mean and std.

    Args:
        tensor (Tensor): Input mel tensor of shape [..., mel_bins]
        mean (float): Global mean
        std (float): Global std

    Returns:
        Tensor: Normalized tensor
    """
    return (tensor * std) + mean