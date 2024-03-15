import torch
import torch.nn.functional as F
import torch.nn as nn

from model.layers import MultiLayerPerceptron, GaussianSmearing


class EdgeEncoder(nn.Module):
    """
    Abstract class for edge encoder.
    """
    def __init__(self, hidden_dim=100, activation="relu", bond_dim=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bond_dim = bond_dim
        self.activation = activation

    def forward(self, edge_length_t, edge_length_T, edge_type):
        """
        Input:
            edge_length_t: The length of edges, shape=(E, 1).
            edge_length_T: The length of edges, shape=(E, 1).
            edge_type: The type pf edges, shape=(E,)
        Returns:
            edge_attr:  The representation of edges. (E, 2 * num_gaussians)
        """
        dist_emb = self.dist_emb(edge_length_t, edge_length_T)  # (num_edge, hidden_dim)
        bond_emb = self.bond_emb(edge_type)  # (num_edge, hidden_dim)
        return dist_emb * bond_emb

    @property
    def out_channels(self):
        return self.hidden_dim

    def dist_emb(self, edge_legnth_t, edge_length_T):
        raise NotImplementedError("This method should be implemented in the subclass.")

    def bond_emb(self, edge_type):
        raise NotImplementedError("This method should be implemented in the subclass.")


class MLPEdgeEncoder(EdgeEncoder):
    def __init__(self, hidden_dim=100, activation="relu", bond_dim=10):
        super().__init__(hidden_dim=hidden_dim, activation=activation, bond_dim=bond_dim)

        self.bond_func = nn.Embedding(bond_dim, embedding_dim=self.hidden_dim)
        self.dist_func1 = MultiLayerPerceptron(
            1, [self.hidden_dim, self.hidden_dim], activation=activation
        )
        self.dist_func2 = MultiLayerPerceptron(
            1, [self.hidden_dim, self.hidden_dim], activation=activation
        )
        self.dist_cat = MultiLayerPerceptron(
            2 * self.hidden_dim, [self.hidden_dim], activation=activation
        )
        self.attr_cat = MultiLayerPerceptron(
            2 * self.hidden_dim, [self.hidden_dim, self.hidden_dim], activation=activation
        )

    def bond_emb(self, edge_type):
        bond_emb = self.bond_func(edge_type)  # (num_edge, hidden_dim)
        return bond_emb

    def dist_emb(self, edge_length_t, edge_length_T):
        dist_emb1 = self.dist_func1(edge_length_t)  # (num_edge, hidden_dim)
        dist_emb2 = self.dist_func2(edge_length_T)  # (num_edge, hidden_dim)
        dist_emb = self.dist_cat(torch.cat([dist_emb1, dist_emb2], dim=-1))  # (num_edge, hidden)
        return dist_emb

    def cat_fn(self, attr_r, attr_p):
        attr = torch.cat([attr_r, attr_p], dim=-1)
        attr = self.attr_cat(attr)
        return attr


class GaussianSmearingEdgeEncoder(EdgeEncoder):
    def __init__(self, hidden_dim=100, cutoff=15.0, bond_dim=10, activation="relu"):
        super().__init__(hidden_dim=hidden_dim, bond_dim=bond_dim, activation=activation)

        self.bond_func = nn.Embedding(bond_dim, embedding_dim=self.hidden_dim)
        self.dist_func1 = GaussianSmearing(
            start=0.0, stop=cutoff, num_gaussians=self.hidden_dim
        )  # Larger `stop` to encode more cases
        self.dist_func2 = GaussianSmearing(
            start=0.0, stop=cutoff, num_gaussians=self.hidden_dim
        )  # Larger `stop` to encode more cases
        self.dist_cat = MultiLayerPerceptron(
            2 * self.hidden_dim, [self.hidden_dim], activation=self.activation
        )

    def dist_emb(self, edge_length_t, edge_length_T):
        dist_emb1 = self.dist_func1(edge_length_t)  # (num_edge, hidden_dim)
        dist_emb2 = self.dist_func2(edge_length_T)  # (num_edge, hidden_dim)
        dist_emb = self.dist_cat(torch.cat([dist_emb1, dist_emb2], dim=-1))  # (num_edge, hidden)
        return dist_emb

    def bond_emb(self, edge_type):
        bond_emb = self.bond_func(edge_type)
        return bond_emb
