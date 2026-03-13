import torch.nn as nn
import torch


def load_activation(name):
    """Load activation function from name.
    Args:
        name (str): name of activation function.
    Returns:
        activation (function): activation function.
    """
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU()
    elif name == 'elu':
        return nn.ELU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'silu':
        return nn.SiLU()
    else:
        raise ValueError('Unknown activation function: {}'.format(name))


class swish(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        return x * x.sigmoid()


def activation_loader(name):
    if name == "swish":
        return swish()
    else:
        return getattr(nn, name)()


def get_distance(pos, index):
    pos_i = pos[index[0]]
    pos_j = pos[index[1]]
    return torch.norm(pos_i - pos_j, dim=-1)
