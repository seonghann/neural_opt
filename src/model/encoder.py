import torch
from torch import nn
from torch.nn import functional as F

from src.manifold.graph import DynamicMolGraph
from src.model.layers import MultiLayerPerceptron, assemble_atom_pair_feature
from src.model.utils import get_distance, activation_loader
from src.model.edge import get_edge_encoder
from src.model.schnet import SchNetEncoder


EncoderDict = {
    "schnet": SchNetEncoder,
}


def load_encoder(config, encoder_type="global_encoder"):
    cfg = config.get(encoder_type)
    encoder = EncoderDict[cfg.name].from_config(cfg)
    return encoder


class GeoDiffEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_time_embedding = getattr(config, 'use_time_embedding', False)

        # If time embedding is enabled, reserve 1 dim for normalized time
        emb_dim = config.hidden_dim - 1 if self.use_time_embedding else config.hidden_dim

        self.edge_encoder = get_edge_encoder(config)

        self.atom_embedding = nn.Embedding(config.num_atom_type, emb_dim)
        self.encoder = load_encoder(config, "graph_encoder")  # graph neural network
        self.score_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1],
            activation=activation_loader(config.score_act),
        )  # Last MLP network for score function prediction

        self.layers = nn.ModuleList(
            [
                self.atom_embedding,
                self.edge_encoder,
                self.encoder,
                self.score_mlp,
            ]

        )

        if self.config.append_atom_feat:
            self.atom_feat_embedding = nn.Linear(
                config.num_atom_feat * 10, emb_dim, bias=False
            )
            self.layers.append(self.atom_feat_embedding)
        return

    def edge_embedding(
        self,
        graph: DynamicMolGraph,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        emb_type="bond_w_d"
    ):
        assert emb_type in ["bond_w_d", "bond_wo_d", "add_d"]
        _enc = self.edge_encoder

        pos = graph.pos
        edge_length = get_distance(pos, edge_index).unsqueeze(-1)

        if emb_type == "bond_wo_d":
            edge_attr = _enc.bond_emb(edge_type)
        elif emb_type == "bond_w_d":
            edge_attr = _enc(edge_length, edge_type)  # Embed edges
        else:
            raise ValueError()
        return edge_attr, edge_length

    def graph_encoding(
        self,
        graph: DynamicMolGraph,
        normalized_time: torch.Tensor = None,
        **kwargs,
    ):
        batch = graph.batch  # batch: batch index (N, )

        # 1) condensed atom embedding
        z = self.atom_embedding(graph.atom_type)

        if self.config.append_atom_feat:
            n_atoms = z.size(0)
            node_feat = F.one_hot(graph.node_feat, num_classes=10).reshape(n_atoms, -1)
            z = z + self.atom_feat_embedding(node_feat.float())

        # 1b) append time embedding if enabled
        if self.use_time_embedding and normalized_time is not None:
            time_per_node = normalized_time[batch].unsqueeze(-1)  # (N, 1)
            z = torch.cat([z, time_per_node], dim=-1)  # (N, hidden_dim)

        # 2) edge_embedding (using undirected-extended-edges)
        edge_index = graph.current_edge_index
        edge_feat = graph.current_edge_feat
        edge_attr, edge_length = self.edge_embedding(
            graph,
            edge_index,
            edge_feat,
        )

        # encoding geometric graph and atom-pair
        node = self.encoder(z, edge_index, edge_length, edge_attr)
        return node

    def forward(
        self,
        graph: DynamicMolGraph,
        normalized_time: torch.Tensor = None,
        **kwargs,
    ):
        """
        Args:
            graph (DynamicMolGraph): DynamicMolGraph object
            normalized_time (torch.Tensor, optional): Normalized time per graph in batch.
        """
        node = self.graph_encoding(graph, normalized_time=normalized_time, **kwargs)

        edge_index, edge_type = graph.full_edge(upper_triangle=True)
        edge, _ = self.edge_embedding(graph, edge_index, edge_type)
        h_pair = assemble_atom_pair_feature(node, edge_index, edge)  # (E, 2H)
        pred = self.score_mlp(h_pair)  # (E, 1)
        return pred
