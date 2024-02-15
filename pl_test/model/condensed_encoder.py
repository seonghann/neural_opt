import torch
from torch import nn
from torch_geometric.utils import to_dense_adj
import numpy as np

from utils.chem import BOND_TYPES_DECODER, BOND_TYPES_ENCODER
from model.utils import load_activation
from model.layers import MultiLayerPerceptron, assemble_atom_pair_feature
from model.utils import load_edge_encoder, load_encoder, get_distance


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class CondenseEncoderEpsNetwork(nn.Module):
    def __init__(self, config):
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
            config.num_atom_feat, config.hidden_dim // 2, bias=False
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

        self.edge_cat = torch.nn.Sequential(
            torch.nn.Linear(
                self.edge_encoder.out_channels * 2,
                self.edge_encoder.out_channels,
            ),
            load_activation(config.edge_cat_act),
            torch.nn.Linear(
                self.edge_encoder.out_channels,
                self.edge_encoder.out_channels,
            ),
        )

    def condensed_edge_embedding(
        self,
        edge_length,
        edge_length_T,
        edge_type_r,
        edge_type_p,
        edge_attr=None,
        emb_type="bond_w_d"
    ):

        assert emb_type in ["bond_w_d", "bond_wo_d", "add_d"]
        _enc = self.edge_encoder
        _cat_fn = self.edge_cat

        if emb_type == "bond_wo_d":
            edge_attr_r = _enc.bond_emb(edge_type_r)
            edge_attr_p = _enc.bond_emb(edge_type_p)
            edge_attr = _cat_fn(torch.cat([edge_attr_r, edge_attr_p], dim=-1))

        elif emb_type == "bond_w_d":
            edge_attr_r = _enc(edge_length, edge_length_T, edge_type_r)  # Embed edges
            edge_attr_p = _enc(edge_length, edge_length_T, edge_type_p)
            edge_attr = _cat_fn(torch.cat([edge_attr_r, edge_attr_p], dim=-1))

        elif emb_type == "add_d":
            edge_attr = _enc.mlp(edge_length, edge_length_T) * edge_attr

        return edge_attr

    def graph_encoding(self, rxn_graph, tt, pos, pos_T, **kwargs):
        """
        Args:
            rxn_graph: rxn graph object (atom_type, r_feat, p_feat, edge_type_r, edge_type_p, edge_index)
            t: time parameter 0 <= t <= 1 (float) (G, )
            pos: structure of noisy structure (N, 3)
            pos_T: structure of initial structure (at time = 1) (N, 3)
        """
        batch = rxn_graph.batch  # batch: batch index (N, )
        tt_node = tt.index_select(0, batch).unsqueeze(-1)  # Convert tt (G, ) to (N, 1)

        # 1) condensed atom embedding
        atom_emb = self.atom_embedding(rxn_graph.atom_type)
        atom_feat_emb_r = self.atom_feat_embedding(rxn_graph.r_feat.float())
        atom_feat_emb_p = self.atom_feat_embedding(rxn_graph.p_feat.float())
        z1 = atom_emb + atom_feat_emb_r
        z2 = atom_feat_emb_p - atom_feat_emb_r
        zz = torch.cat([z1, z2], dim=-1)

        # 2) edge_embedding
        edge_index = rxn_graph.current_edge_index
        edge_length = get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)
        edge_length_T = get_distance(pos_T, edge_index).unsqueeze(-1)  # (E, 1)
        edge_attr = self.condensed_edge_embedding(
            edge_length,
            edge_length_T,
            rxn_graph.current_edge_feat_r,
            rxn_graph.current_edge_feat_p
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

        node = self.graph_encoding(rxn_graph, tt, pos, pos_T, **kwargs)

        edge_index, type_r, type_p = rxn_graph.full_edge()
        length = get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)
        length_T = get_distance(pos_T, edge_index).unsqueeze(-1)  # (E, 1)
        edge = self.condensed_edge_embedding(length, length_T, type_r, type_p)
        h_pair = assemble_atom_pair_feature(node, edge_index, edge)  # (E, 2H)
        pred = self.score_mlp(h_pair)  # (E, 1)
        return pred
