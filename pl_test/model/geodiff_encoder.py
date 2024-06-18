import torch
from torch import nn
from torch.nn import functional as F

from utils.rxn_graph import DynamicMolGraph
from model.layers import MultiLayerPerceptron, assemble_atom_pair_feature
from model.utils import get_distance
from model_tsdiff.edge import get_edge_encoder
from model_tsdiff import load_encoder, activation_loader


class GeoDiffEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.edge_encoder = get_edge_encoder(config)

        self.atom_embedding = nn.Embedding(config.num_atom_type, config.hidden_dim)
        self.encoder = load_encoder(config, "graph_encoder")  # graph neural network
        self.score_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1],
            activation=activation_loader(config.score_act),
        )  # Last MLP network for score function prediction

        self.layers = nn.ModuleList(
            [
                self.atom_embedding,
                # self.atom_feat_embedding,
                self.edge_encoder,
                self.encoder,
                self.score_mlp,
            ]

        )

        if self.config.append_atom_feat:
            self.atom_feat_embedding = nn.Linear(
                config.num_atom_feat * 10, config.hidden_dim, bias=False
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
        **kwargs,
    ):
        batch = graph.batch  # batch: batch index (N, )

        # 1) condensed atom embedding
        z = self.atom_embedding(graph.atom_type)

        if self.config.append_atom_feat:
            n_atoms = z.size(0)
            node_feat = F.one_hot(graph.node_feat, num_classes=10).reshape(n_atoms, -1)
            z = z + self.atom_feat_embedding(node_feat.float())

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
        **kwargs,
    ):
        """
        Args: graph (DynamicMolGraph): DynamicMolGraph object
        """
        node = self.graph_encoding(graph, **kwargs)

        edge_index, edge_type = graph.full_edge(upper_triangle=True)
        edge, _ = self.edge_embedding(graph, edge_index, edge_type)
        h_pair = assemble_atom_pair_feature(node, edge_index, edge)  # (E, 2H)
        pred = self.score_mlp(h_pair)  # (E, 1)
        return pred
