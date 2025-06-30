import torch
import torch.nn as nn
import numpy as np

class SSLU(nn.Module):
    """
    SSLU (Self-Stabilized Linear Unit) Activation Function
    
    Formula: ((1 + (x + alpha) / sqrt(beta + x²)) / 2) * x
    
    Args:
        alpha (float): First parameter (default: 0.0025)
        beta (float): Second parameter (default: 5.0)
    """
    def __init__(self, alpha=0.0025, beta=5.0):
        super(SSLU, self).__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, x):
        result = ((1 + (x + self.alpha) / torch.sqrt(self.beta + x * x)) / 2) * x
        return result
    
    def extra_repr(self):
        return f'alpha={self.alpha}, beta={self.beta}'

class CahLU(nn.Module):
    """
    CahLU (Custom Hyperbolic Linear Unit) Activation Function
    
    Formula: alpha * x * log(1.5 + atan(x) / π)
    
    Args:
        alpha (float): Scaling parameter (default: 1.444)
    """
    def __init__(self, alpha=1.444):
        super(CahLU, self).__init__()
        self.alpha = alpha
        self.pi = torch.tensor(np.pi)
    
    def forward(self, x):
        # Convert to tensor if numpy constant is used
        if not isinstance(self.pi, torch.Tensor):
            self.pi = torch.tensor(np.pi, device=x.device, dtype=x.dtype)
        
        result = self.alpha * x * torch.log(1.5 + torch.atan(x) / self.pi)
        return result
    
    def extra_repr(self):
        return f'alpha={self.alpha}'