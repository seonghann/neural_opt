from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from utils.chem import BOND_TYPES_DECODER
from utils.rxn_graph import RxnGraph
from model.utils import load_edge_encoder, load_encoder, get_distance
from omegaconf.dictconfig import DictConfig


class AtomEncoder(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        assert config.hidden_dim % 2 == 0

        # self.atom_encoder = nn.Embedding(config.num_atom_type, config.hidden_dim // 2)
        bias = True
        self.atom_encoder = nn.Linear(
            config.num_atom_type, config.hidden_dim // 2, bias=bias
        )
        self.atom_feat_encoder = nn.Linear(
            config.num_atom_feat * 10, config.hidden_dim // 2, bias=bias
        )
        return

    def forward(
        self,
        atom_type: torch.Tensor,
        r_feat: torch.Tensor,
        p_feat: torch.Tensor,
    ) -> torch.Tensor:
        """args:
        atom_type: (natoms, num_atom_type) one-hot encoding of atom type
        r_feat: (natoms, num_atom_feat * 10) one-hot encoding of atom feat
        p_feat: (natoms, num_atom_feat * 10) one-hot encoding of atom feat

        z: (natoms, 2 * num_atom_feat * 10 + num_atom_type)
        """
        ## Refer to model.condensed_encoder.graph_encoding()

        atom_emb = self.atom_encoder(atom_type.float())
        atom_feat_emb_r = self.atom_feat_encoder(r_feat.float())
        atom_feat_emb_p = self.atom_feat_encoder(p_feat.float())

        z1 = atom_emb + atom_feat_emb_r
        z2 = atom_feat_emb_p - atom_feat_emb_r
        z = torch.cat([z1, z2], dim=-1)
        return z


class EquivariantEncoderEpsNetwork(nn.Module):
    def __init__(self, config: DictConfig, solver):
        super().__init__()
        self.config = config  # model config
        self.solver = solver
        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        self.check_encoder_config()

        self.node_encoder = AtomEncoder(config)
        self.edge_encoder = load_edge_encoder(config)
        self.encoder = load_encoder(config)  # graph neural network

        self.model = nn.ModuleList([self.node_encoder, self.edge_encoder, self.encoder])

        self.z_t_embedding = nn.Linear(config.hidden_dim + 1, config.hidden_channels)

        self.model_embedding = nn.ModuleList([self.z_t_embedding])
        return

    def check_encoder_config(self) -> None:
        ## Check config is valid
        available_graph_encoders = ["leftnet"]
        assert (
            self.config.graph_encoder.name.lower() in available_graph_encoders
        ), f"graph_encoder.name must be one of {available_graph_encoders}"
        assert self.config.hidden_dim % 2 == 0, "hidden_dim must be an even number"
        assert (
            self.config.hidden_channels == self.config.graph_encoder.hidden_channels
        ), "hidden_channels must match graph_encoder.hidden_channels"
        assert (
            self.config.hidden_channels == self.config.hidden_dim
        ), "hidden_channels must match hidden_dim"
        return

    def atom_embedding(self, rxn_graph: RxnGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        natoms = len(rxn_graph.atom_type)
        num_atom_type = self.config.num_atom_type
        num_atom_feat = self.config.num_atom_feat

        ## 1) One-hot encoding
        atom_type = F.one_hot(rxn_graph.atom_type, num_classes=num_atom_type)
        r_feat = F.one_hot(rxn_graph.r_feat, num_classes=10)
        p_feat = F.one_hot(rxn_graph.p_feat, num_classes=10)

        atom_type = atom_type.reshape(natoms, num_atom_type)
        r_feat = r_feat.reshape(natoms, num_atom_feat * 10)
        p_feat = p_feat.reshape(natoms, num_atom_feat * 10)

        ## 2) Embedding to z
        z = self.node_encoder(atom_type, r_feat, p_feat)  # shape==(natoms, hidden_dim)

        ## 3) Concat one-hot encoded values
        h = torch.cat([atom_type, r_feat, p_feat], dim=-1)
        # h.shape==(natoms, num_atom_type + num_atom_feat * 10 * 2)
        return z, h

    def bond_embedding(
        self,
        rxn_graph: RxnGraph,
        emb_type: str = "bond_w_d",
    ) -> torch.Tensor:
        ## Refer to condensed_encoder.condensed_edge_embedding()
        available_option = ["bond_w_d", "bond_wo_d", "add_d"]
        assert (
            emb_type in available_option
        ), f"{emb_type} must be one of {available_option}"

        _enc = self.edge_encoder
        _cat_fn = self.edge_encoder.cat_fn

        edge_index, edge_type_r, edge_type_p = rxn_graph.full_edge()
        pos, pos_T = rxn_graph.pos, rxn_graph.pos_init

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
        else:
            raise ValueError

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

    def forward(
        self,
        rxn_graph: RxnGraph,
        **kwargs,
    ) -> torch.Tensor:
        """
        args: rxn_graph (DynamicRxnGraph): rxn graph object
        """
        z, h = self.atom_embedding(rxn_graph)

        ## Time feature concat and embedding
        t_node = rxn_graph.t.index_select(0, rxn_graph.batch).unsqueeze(-1)
        h_t = torch.cat([h, t_node], dim=-1)
        z_t = torch.cat([z, t_node], dim=-1)
        z_t = self.z_t_embedding(z_t)
        # h_t.shape==(natoms, num_atom_type + num_atom_feat * 10 * 2)
        # z_t.shape==(natoms, hidden_channels)

        pos = rxn_graph.pos
        pos_T = rxn_graph.pos_init
        edge_index = rxn_graph.full_edge()[0]
        edge_attr, dist, dist_T = self.bond_embedding(rxn_graph)

        pred = self.encoder(h_t, z_t, pos, pos_T, dist, dist_T, edge_index, edge_attr)  # (n_atoms, 3)
        return pred
