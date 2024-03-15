import torch
from torch import nn
from torch.nn import functional as F

from utils.chem import BOND_TYPES_DECODER
from model.utils import load_edge_encoder, load_encoder


class AtomEncoder(nn.Module):
    def __init__(self, config):
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
        atom_emb = self.atom_encoder(atom_type)
        atom_feat_emb_r = self.atom_feat_encoder(r_feat)
        atom_feat_emb_p = self.atom_feat_encoder(p_feat)
        h1 = atom_emb * atom_feat_emb_r
        h2 = atom_emb * atom_feat_emb_p
        h = torch.cat([h1, h2], dim=-1)
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

    def atom_embedding(self, rxn_graph):
        atom_type = rxn_graph.atom_type
        n_atoms = atom_type.size(0)
        r_feat = F.one_hot(rxn_graph.r_feat, num_classes=10).reshape(n_atoms, -1)
        p_feat = F.one_hot(rxn_graph.p_feat, num_classes=10).reshape(n_atoms, -1)
        h = self.atom_encoder(atom_type, r_feat, p_feat)

        raw_h = torch.cat([atom_type, r_feat, p_feat], dim=-1)
        t = rxn_graph.t.index_select(0, rxn_graph.batch).unsqueeze(-1)
        raw_h = torch.cat([raw_h, t], dim=-1)  #  dim = num_atom_feat * 10 * 2 + num_atom_type + 1
        return h, raw_h

    def bond_embedding(self, rxn_graph):
        # edge_index = rxn_graph.current_edge_index
        edge_index, edge_type_r, edge_type_p = rxn_graph.full_edge()
        i, j = edge_index
        pos, pos_T = rxn_graph.pos, rxn_graph.pos_init
        dist = (pos[i] - pos[j]).norm(dim=-1)
        dist_T = (pos_T[i] - pos_T[j]).norm(dim=-1)

        # edge_type_r = rxn_graph.current_edge_feat_r
        # edge_type_p = rxn_graph.current_edge_feat_p

        edge_attr_r = self.edge_encoder.bond_emb(dist, dist_T, edge_type_r)
        edge_attr_p = self.edge_encoder.bond_emb(dist, dist_T, edge_type_p)
        edge_attr = self.edge_encoder.cat_fn(edge_attr_r, edge_attr_p)
        return edge_attr

    def forward(self, rxn_graph, **kwargs):
        """
        args: rxn_graph (DynamicRxnGraph): rxn graph object
        """
        h, raw_h = self.atom_embedding(rxn_graph)

        t_node = rxn_graph.t.index_select(0, rxn_graph.batch).unsqueeze(-1)
        h = torch.cat([h, t_node], dim=-1)
        z_emb = self.node_embedding(h)

        pos = rxn_graph.pos
        pos_T = rxn_graph.pos_init
        edge_index = rxn_graph.full_edge()[0]
        edge_attr = self.bond_embedding(self, rxn_graph)

        pred = self.model(raw_h, z_emb, pos, pos_T, edge_index, edge_attr)   # (n_atoms, 3)
        return pred
