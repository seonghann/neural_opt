import torch
from torch import nn
from torch.nn import functional as F

from utils.chem import BOND_TYPES_DECODER
from model.utils import load_edge_encoder, load_encoder, get_distance


class AtomEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.hidden_dim % 2 == 0
        self.atom_encoder = nn.Embedding(config.num_atom_type, config.hidden_dim // 2)
        self.atom_feat_encoder = nn.Linear(config.num_atom_feat * 10, config.hidden_dim // 2, bias=False)

    def forward(self, atom_type, r_feat, p_feat):
        """args:
        atom_type: (n_atom)
        r_feat: (n_atom, n_feat) one-hot encoding of atom feat
        p_feat: (n_atom, n_feat) one-hot encoding of atom feat
        """
        ## Refer to model.condensed_encoder.graph_encoding()

        atom_emb = self.atom_encoder(atom_type)
        atom_feat_emb_r = self.atom_feat_encoder(r_feat.float())
        atom_feat_emb_p = self.atom_feat_encoder(p_feat.float())
        h1 = atom_emb * atom_feat_emb_r
        h2 = atom_emb * atom_feat_emb_p
        h = torch.cat([h1, h2], dim=-1)  # shape==(natoms, hidden_dim)
        return h


class EquivariantEncoderEpsNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # model config
        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        self.node_encoder = AtomEncoder(config)
        self.edge_encoder = load_edge_encoder(config)
        assert config.hidden_dim % 2 == 0
        self.encoder = load_encoder(config)  # graph neural network

        self.model = nn.ModuleList(
            [self.node_encoder, self.edge_encoder, self.encoder]
        )

        self.z_t_embedding = nn.Linear(
            config.hidden_dim + 1, config.hidden_channels
        )

        self.model_embedding = nn.ModuleList([self.z_t_embedding])
        return

    def atom_embedding(self, rxn_graph):
        ## !!! CondenseEncoderEpsNetwork와는 방식에 차이 있음.
        ## atom_type_embedding과 atom_feat_embedding을 따로 진행.
        ## 여기서는, 한번에 진행.
        atom_type = rxn_graph.atom_type
        n_atoms = atom_type.size(0)
        r_feat = F.one_hot(rxn_graph.r_feat, num_classes=10).reshape(n_atoms, -1)
        p_feat = F.one_hot(rxn_graph.p_feat, num_classes=10).reshape(n_atoms, -1)

        z = self.node_encoder(atom_type, r_feat, p_feat)  # shape==(natoms, hidden_dim)
        h = torch.cat([atom_type.unsqueeze(dim=-1), r_feat, p_feat], dim=-1)
        # h.shape==(natoms, num_atom_feat * 10 * 2 + num_atom_type + 1)
        return z, h

    def bond_embedding(self, rxn_graph, emb_type="bond_w_d"):
        ## Refer to condensed_encoder.condensed_edge_embedding()
        available_option = ["bond_w_d", "bond_wo_d", "add_d"]
        assert emb_type in available_option, f"{emb_type} must be one of {available_option}"

        _enc = self.edge_encoder
        _cat_fn = self.edge_encoder.cat_fn

        edge_index, edge_type_r, edge_type_p = rxn_graph.full_edge()
        pos, pos_T = rxn_graph.pos, rxn_graph.pos_init

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

        return edge_attr

    def forward(self, rxn_graph, **kwargs):
        """
        args: rxn_graph (DynamicRxnGraph): rxn graph object
        """
        z, h = self.atom_embedding(rxn_graph)

        ## Time embedding
        t_node = rxn_graph.t.index_select(0, rxn_graph.batch).unsqueeze(-1)
        h_t = torch.cat([h, t_node], dim=-1)  # shape==(natoms, 2 * 10 * num_atom_feat + 2)
        z_t = torch.cat([z, t_node], dim=-1)
        z_t = self.z_t_embedding(z_t)  # shape==(natoms, hidden_channels)

        pos = rxn_graph.pos
        pos_T = rxn_graph.pos_init
        edge_index = rxn_graph.full_edge()[0]
        edge_attr = self.bond_embedding(rxn_graph)

        pred = self.encoder(h_t, z_t, pos, pos_T, edge_index, edge_attr)   # (n_atoms, 3)
        return pred
