import torch.nn as nn
import torch


def load_encoder(config):
    from model import EncoderDict
    encoder_config = config.graph_encoder
    name = encoder_config["name"].lower()

    if name in EncoderDict:
        encoder = EncoderDict[name].from_config(encoder_config)
    else:
        raise ValueError(f"Unknown encoder: {name}")
    return encoder


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


def load_edge_encoder(config):
    """Get edge encoder from config.
    Args:
        config (dict): model-configuration.
    Returns:
        edge_encoder (nn.Module): edge encoder.
    """
    if config.edge_encoder.name == "mlp":
        from model.edge_encoders import MLPEdgeEncoder
        return MLPEdgeEncoder(
            hidden_dim=config.hidden_dim,
            activation=config.edge_encoder.activation,
            bond_dim=config.num_bond_type,
            append_coordinate=config.append_coordinate
        )
    if config.edge_encoder.name == "gaussian":
        from model.edge_encoders import GaussianEdgeEncoder
        return GaussianEdgeEncoder(
            hidden_dim=config.hidden_dim,
            cutoff=config.edge_encoder.cutoff,
            bond_dim=config.num_bond_type,
            append_coordinate=config.append_coordinate
        )
    else:
        raise ValueError("Unknown edge encoder: {}".format(config["edge_encoder"]["name"]))


def get_distance(pos, index):
    pos_i = pos[index[0]]
    pos_j = pos[index[1]]
    return torch.norm(pos_i - pos_j, dim=-1)
