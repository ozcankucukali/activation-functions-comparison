import torch
import torch.nn as nn
import numpy as np

class S2LU(nn.Module):
    """
    S2LU Activation Function
    
    Formula: ((1 + (x + alpha) / sqrt(beta + x²)) / 2) * x
    
    Args:
        alpha (float): First parameter (default: 0.0025)
        beta (float): Second parameter (default: 5.0)
    """
    def __init__(self, alpha=0.0025, beta=5.0):
        super(S2LU, self).__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, x):
        result = ((1 + (x + self.alpha) / torch.sqrt(self.beta + x * x)) / 2) * x
        return result
    
    def extra_repr(self):
        return f'alpha={self.alpha}, beta={self.beta}'
