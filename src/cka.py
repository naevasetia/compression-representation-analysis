import numpy as np
import torch

def centering(K):
    """
    Applies centering to a kernel matrix K.
    Centering removes the mean from rows and columns, making the 
    similarity measure invariant to constant offsets in activation space.
    This is the 'C' in CKA — Centered Kernel Alignment.
    """
    n = K.shape[0]
    unit = torch.ones(n, n, device=K.device)
    I = torch.eye(n, device=K.device)
    H = I - unit / n
    return H @ K @ H

def linear_CKA(X, Y):
    """
    Computes linear CKA between activation matrices X and Y.
    
    X, Y: (n_samples, n_features) tensors
    
    Linear CKA uses the dot-product kernel: K = X @ X^T
    This is the variant used in Kornblith et al. (2019) for comparing 
    neural network representations - it's computationally efficient and 
    well-behaved for high-dimensional activations.
    
    Returns a scalar in [0, 1]. Higher = more similar representations.
    """
    X = X.to(torch.float32)
    Y = Y.to(torch.float32)

    # Kernel matrices: (n x n) gram matrices
    K = X @ X.T
    L = Y @ Y.T

    # Center both kernels
    K_c = centering(K)
    L_c = centering(L)

    # HSIC: Hilbert-Schmidt Independence Criterion
    # Measures statistical dependence between the two kernel matrices
    hsic     = (K_c * L_c).sum()
    norm_kk  = torch.sqrt((K_c * K_c).sum())
    norm_ll  = torch.sqrt((L_c * L_c).sum())

    return (hsic / (norm_kk * norm_ll)).item()