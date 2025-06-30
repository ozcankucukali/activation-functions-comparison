import torch.nn as nn

# Import standard activations
from .standard_activations import ReLU, LeakyReLU, ELU, GELU, Mish, Swish

# Import custom activations  
from .custom_activations import SSLU, CahLU

def get_activation(activation_name, **kwargs):
    """Factory function to get activation functions by name"""
    
    activations = {
        'relu': ReLU,
        'leaky_relu': LeakyReLU,
        'elu': ELU,
        'gelu': GELU,
        'mish': Mish,
        'swish': Swish,
        'sslu': SSLU,
        'cahlu': CahLU,
    }
    
    activation_name = activation_name.lower()
    
    if activation_name not in activations:
        raise ValueError(f"Activation '{activation_name}' not supported. "
                        f"Available activations: {list(activations.keys())}")
    
    return activations[activation_name](**kwargs)

# Available activation functions
AVAILABLE_ACTIVATIONS = [
    'relu', 'leaky_relu', 'elu', 'gelu', 'mish', 'swish', 'sslu', 'cahlu'
]

__all__ = [
    'ReLU', 'LeakyReLU', 'ELU', 'GELU', 'Mish', 'Swish', 'SSLU', 'CahLU',
    'get_activation', 'AVAILABLE_ACTIVATIONS'
]