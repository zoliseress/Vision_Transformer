
import numpy as np
import torch


def get_device() -> torch.device:
    """
    Get the device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def get_positional_encodings(sequence_length: int, dim: int) -> torch.Tensor:
    """
    Positional encoding allows the model to understand where each patch would be placed
    in the original image. It adds high-frequency values to the first dimensions and
    low-frequency values to the latter dimensions.
    Basically it is a static function that maps the integer inputs to real-valued
    vectors in a way that captures the inherent relationships among the positions.
    See positional_embedding.png and visualize_positional_embeddings.png in doc folder.   

    Parameters
    ----------
    sequence_length : int
        Number of tokens.
    dim : int
        Dimensionality of each of the tokens.

    Returns
    -------
    torch.Tensor
        Its size is [`sequence_length`, `dim`].
    """

    result = torch.ones(sequence_length, dim)
    for i in range(sequence_length):
        for j in range(dim):
            if j % 2 == 0:
                result[i][j] = np.sin(i / (10000 ** (j / dim)))
            else:
                result[i][j] = np.cos(i / (10000 ** ((j - 1) / dim)))

    return result


def make_patches(images: torch.Tensor, num_patches: int = 7) -> torch.Tensor:
    """
    Extract patches from the input tensor.
    If `num_patches` is 7, the result will look like:
        (N, P², HWC/P²) = (N, 7x7, 4x4) = (N, 49, 16)
    The patches are stored flatten.

    Parameters
    ----------
    images : torch.Tensor
        The input sequence of 2D images that need to be splitted.
    num_patches : int
        The number of patches (in one image dimension).

    Returns
    -------
    torch.Tensor
        The patches. Its shape is (batch size, num_patches^2, patch_size^2).
    """

    n, c, h, w = images.shape  # n = batch size, c = channel, h = height, w = width

    assert h == w, "Only square images are supported."
    assert h % num_patches == 0, "`num_patches` must evenly divide image height/width."

    # If `num_patches` is 7, we break each (1, 28, 28) image into 7x7 patches
    # (each of size 4x4). At the end we will have 7x7=49 flattened sub-images.
    
    patch_size = h // num_patches

    # Extract sliding, non-overlapping blocks (patches).
    patches = torch.nn.functional.unfold(
        images, kernel_size=patch_size, stride=patch_size
    )
    # From (n, patch_size^2, num_patches^2) to (n, num_patches^2, patch_size^2).
    patches = patches.transpose(1, 2).contiguous()

    return patches