import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_adj
import numpy as np

from utils.chem import BOND_TYPES_DECODER, BOND_TYPES_ENCODER
from utils.rxn_graph import RxnGraph
from model.utils import load_activation
from model.layers import MultiLayerPerceptron, assemble_atom_pair_feature
from model.utils import load_edge_encoder, load_encoder, get_distance


class CondensedEncoderEpsNetwork(nn.Module):
    def __init__(self, config, solver):
        super().__init__()
        self.config = config  # model config
        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        self.edge_encoder = load_edge_encoder(config)
        assert config.hidden_dim % 2 == 0
        self.atom_embedding = nn.Embedding(config.num_atom_type, config.hidden_dim // 2)
        self.atom_feat_embedding = nn.Linear(
            # config.num_atom_feat, config.hidden_dim // 2, bias=False
            config.num_atom_feat * 10, config.hidden_dim // 2, bias=False
        )
        self.encoder = load_encoder(config)  # graph neural network
        self.score_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1],
            activation=load_activation(config.score_act),
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
        self.solver = solver

    def condensed_edge_embedding(
        self,
        rxn_graph: RxnGraph,
        edge_index: torch.Tensor,
        edge_type_r: torch.Tensor,
        edge_type_p: torch.Tensor,
        emb_type="bond_w_d"
    ):

        assert emb_type in ["bond_w_d", "bond_wo_d", "add_d"]
        _enc = self.edge_encoder
        _cat_fn = self.edge_encoder.cat_fn

        pos, pos_T = rxn_graph.pos, rxn_graph.pos_init
        if not self.config.append_pos_init:
            pos_T = torch.zeros_like(pos_T); print(f"Debug: pos_T is set to zeros in condensed_edge_embedding")

        if self.config.reduced_dimension:
            atom_type = rxn_graph.atom_type
            length_e = self.solver.compute_de(edge_index, atom_type).unsqueeze(-1)
            edge_length = self.solver.compute_d(edge_index, pos).unsqueeze(-1)
            edge_length_T = self.solver.compute_d(edge_index, pos_T).unsqueeze(-1)
            edge_length = edge_length / length_e
            edge_length_T = edge_length_T / length_e
        else:
            edge_length = get_distance(pos, edge_index).unsqueeze(-1)
            edge_length_T = get_distance(pos_T, edge_index).unsqueeze(-1)

        if emb_type == "bond_wo_d":
            edge_attr_r = _enc.bond_emb(edge_type_r)
            edge_attr_p = _enc.bond_emb(edge_type_p)
            edge_attr = _cat_fn(edge_attr_r, edge_attr_p)

        elif emb_type == "bond_w_d":
            edge_attr_r = _enc(edge_length, edge_length_T, edge_type_r)  # Embed edges
            edge_attr_p = _enc(edge_length, edge_length_T, edge_type_p)
            edge_attr = _cat_fn(edge_attr_r, edge_attr_p)

        elif emb_type == "add_d":
            edge_attr = _enc.mlp(edge_length, edge_length_T) * edge_attr

        if self.config.append_coordinate:
            q1 = torch.exp(- self.solver.alpha * (edge_length - 1))
            q2 = self.solver.beta / edge_length
            q3 = self.solver.gamma * edge_length
            q = torch.cat([q1, q2, q3], dim=-1)

            q1 = torch.exp(- self.solver.alpha * (edge_length_T - 1))
            q2 = self.solver.beta / edge_length_T
            q3 = self.solver.gamma * edge_length_T
            q_T = torch.cat([q1, q2, q3], dim=-1)

            edge_attr = torch.cat([edge_attr, q, q_T], dim=-1)

        return edge_attr, edge_length, edge_length_T

    def graph_encoding(self, rxn_graph, tt, pos, **kwargs):
        """
        Args:
            rxn_graph: rxn graph object (atom_type, r_feat, p_feat, edge_type_r, edge_type_p, edge_index)
            t: time parameter 0 <= t <= 1 (float) (G, )
            pos: structure of noisy structure (N, 3)
        """
        batch = rxn_graph.batch  # batch: batch index (N, )
        tt_node = tt.index_select(0, batch).unsqueeze(-1)  # Convert tt (G, ) to (N, 1)

        # 1) condensed atom embedding
        atom_emb = self.atom_embedding(rxn_graph.atom_type)
        n_atoms = atom_emb.size(0)
        r_feat = F.one_hot(rxn_graph.r_feat, num_classes=10).reshape(n_atoms, -1)
        p_feat = F.one_hot(rxn_graph.r_feat, num_classes=10).reshape(n_atoms, -1)
        atom_feat_emb_r = self.atom_feat_embedding(r_feat.float())
        atom_feat_emb_p = self.atom_feat_embedding(p_feat.float())

        # atom_feat_emb_r = self.atom_feat_embedding(rxn_graph.r_feat.float())
        # atom_feat_emb_p = self.atom_feat_embedding(rxn_graph.p_feat.float())
        z1 = atom_emb + atom_feat_emb_r
        z2 = atom_feat_emb_p - atom_feat_emb_r
        zz = torch.cat([z1, z2], dim=-1)

        # 2) edge_embedding (using undirected-extended-edges)
        edge_index = rxn_graph.current_edge_index
        edge_feat_r = rxn_graph.current_edge_feat_r
        edge_feat_p = rxn_graph.current_edge_feat_p
        # edge_length = get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)
        # edge_length_T = get_distance(pos_T, edge_index).unsqueeze(-1)  # (E, 1)
        edge_attr, edge_length, edge_length_T = self.condensed_edge_embedding(
            rxn_graph,
            edge_index,
            edge_feat_r,
            edge_feat_p,
        )

        # encoding TS geometric graph and atom-pair
        node = self.encoder(
            tt_node, zz, edge_index, edge_length, edge_attr, edge_length_T=edge_length_T
        )
        return node

    def forward(self, rxn_graph, **kwargs):
        """
        Args: rxn_graph (DynamicRxnGraph): rxn graph object
        """
        tt, pos, pos_T = rxn_graph.t, rxn_graph.pos, rxn_graph.pos_init
        # if config.model.append_pos_init is True, pos_T is used in self.condensed_edge_embedding()

        if not self.config.time_embedding:
            tt = torch.zeros_like(tt); print(f"Debug: tt is set to zeros (to remove t-variable from neural networks")

        node = self.graph_encoding(rxn_graph, tt, pos, **kwargs)

        edge_index, type_r, type_p = rxn_graph.full_edge(upper_triangle=True)
        edge, _, _ = self.condensed_edge_embedding(rxn_graph, edge_index, type_r, type_p)
        h_pair = assemble_atom_pair_feature(node, edge_index, edge)  # (E, 2H)
        pred = self.score_mlp(h_pair)  # (E, 1)
        return pred
