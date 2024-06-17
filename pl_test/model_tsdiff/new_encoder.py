import torch
from torch import nn
from torch.nn import functional as F
# from torch_geometric.utils import to_dense_adj
# import numpy as np

# from utils.chem import BOND_TYPES_DECODER, BOND_TYPES_ENCODER
from utils.rxn_graph import RxnGraph
# from model.utils import load_activation
from model.layers import MultiLayerPerceptron, assemble_atom_pair_feature
# from model.utils import load_edge_encoder, load_encoder, get_distance
# from model.utils import load_encoder, get_distance
from model.utils import get_distance

from model_tsdiff.edge import get_edge_encoder
from model_tsdiff import load_encoder, activation_loader


class GeoDiffEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.edge_encoder = get_edge_encoder(config)

        self.atom_embedding = nn.Embedding(config.num_atom_type, config.hidden_dim)
        self.atom_feat_embedding = nn.Linear(
            config.num_atom_feat * 10, config.hidden_dim, bias=False
        )
        self.encoder = load_encoder(config)  # graph neural network
        self.score_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1],
            activation=activation_loader(config.mlp_act),
        )  # Last MLP network for score function prediction

        self.model_embedding = nn.ModuleList(
            [
                self.atom_embedding,
                self.atom_feat_embedding,
            ]
        )
        self.model = nn.ModuleList(
            [self.edge_encoder, self.encoder, self.score_mlp]
        )
        return

    def edge_embedding(
        self,
        rxn_graph: RxnGraph,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        emb_type="bond_w_d"
    ):
        assert emb_type in ["bond_w_d", "bond_wo_d", "add_d"]
        _enc = self.edge_encoder
        # _cat_fn = self.edge_cat

        pos = rxn_graph.pos  # FIX: no rxn_graph
        edge_length = get_distance(pos, edge_index).unsqueeze(-1)

        if emb_type == "bond_wo_d":
            edge_attr = _enc.bond_emb(edge_type)
        elif emb_type == "bond_w_d":
            edge_attr = _enc(edge_length, edge_type)  # Embed edges
        else:
            raise ValueError()

        return edge_attr, edge_length

    # def graph_encoding(self, rxn_graph, tt, pos, **kwargs):
    def graph_encoding(self, rxn_graph, **kwargs):
        """
        Args:
            rxn_graph: rxn graph object (atom_type, r_feat, p_feat, edge_type_r, edge_type_p, edge_index)
        """
        batch = rxn_graph.batch  # batch: batch index (N, )

        # 1) condensed atom embedding
        atom_emb = self.atom_embedding(rxn_graph.atom_type)
        n_atoms = atom_emb.size(0)
        node_feat = F.one_hot(rxn_graph.node_feat, num_classes=10).reshape(n_atoms, -1)
        z = self.atom_feat_embedding(node_feat.float())

        # 2) edge_embedding (using undirected-extended-edges)
        edge_index = rxn_graph.current_edge_index
        edge_feat = rxn_graph.current_edge_feat
        edge_attr, edge_length = self.condensed_edge_embedding(
            rxn_graph,
            edge_index,
            edge_feat,
        )

        # encoding TS geometric graph and atom-pair
        node = self.encoder(z, edge_index, edge_length, edge_attr)
        return node

    def forward(self, rxn_graph, **kwargs):
        """
        Args: rxn_graph (DynamicRxnGraph): rxn graph object
        """
        node = self.graph_encoding(rxn_graph, **kwargs)

        # edge_index, type_r, type_p = rxn_graph.full_edge(upper_triangle=True)
        edge_index, edge_type = rxn_graph.full_edge(upper_triangle=True)
        edge, _, _ = self.edge_embedding(rxn_graph, edge_index, edge_type)
        h_pair = assemble_atom_pair_feature(node, edge_index, edge)  # (E, 2H)
        pred = self.score_mlp(h_pair)  # (E, 1)
        return pred
