import torch
from torch import nn
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import numpy as np

from utils.chem import ATOM_ENCODER, BOND_TYPES_ENCODER
from model_tsdiff import load_encoder, activation_loader
from model_tsdiff.common import MultiLayerPerceptron, assemble_atom_pair_feature, extend_ts_graph_order_radius
from model_tsdiff.geometry import get_distance, eq_transform
from model_tsdiff.edge import *


class CondenseEncoderEpsNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        self.edge_encoder = get_edge_encoder(config)
        assert config.hidden_dim % 2 == 0
        self.atom_embedding = nn.Embedding(100, config.hidden_dim // 2)
        # feat_dim = len(ATOM_ENCODER)
        assert config.feat_dim >= sum([len(val) for val in ATOM_ENCODER.values()])
        self.atom_feat_embedding = nn.Linear(
            config.feat_dim, config.hidden_dim // 2, bias=False
        )

        """
        The graph neural network that extracts node-wise features.
        """
        self.encoder = load_encoder(config, "encoder")

        """
        `output_mlp` takes a mixture of two nodewise features and edge features as input and outputs
            gradients w.r.t. edge_length (out_dim = 1).
        """
        self.grad_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1],
            activation=activation_loader(config.mlp_act),
        )

        """
        Incorporate parameters together
        """
        self.model_embedding = nn.ModuleList(
            [
                self.atom_embedding,
                self.atom_feat_embedding,
            ]
        )
        self.model = nn.ModuleList(
            [self.edge_encoder, self.encoder, self.grad_dist_mlp]
        )

        # betas = get_beta_schedule(
        #     beta_schedule=config.beta_schedule,
        #     beta_start=config.beta_start,
        #     beta_end=config.beta_end,
        #     num_diffusion_timesteps=config.num_diffusion_timesteps,
        # )
        # betas = torch.from_numpy(betas).float()
        # self.betas = nn.Parameter(betas, requires_grad=False)
        # variances
        # alphas = (1.0 - betas).cumprod(dim=0)
        # self.alphas = nn.Parameter(alphas, requires_grad=False)
        # self.num_timesteps = self.betas.size(0)

        # self.num_bond_types = len(BOND_TYPES)
        self.num_bond_types = len(BOND_TYPES_ENCODER)
        self.edge_cat = torch.nn.Sequential(
            torch.nn.Linear(
                self.edge_encoder.out_channels * 2,
                self.edge_encoder.out_channels,
            ),
            activation_loader(config.edge_cat_act),
            torch.nn.Linear(
                self.edge_encoder.out_channels,
                self.edge_encoder.out_channels,
            ),
        )

        if self.config.type == "dsm":
            sigmas = torch.tensor(
                np.exp(
                    np.linspace(
                        np.log(config.sigma_begin),
                        np.log(config.sigma_end),
                        config.num_noise_level,
                    )
                ), dtype=torch.float32
            )
            print(f"Debug: sigmas={sigmas}")
            self.sigmas = nn.Parameter(sigmas, requires_grad=False) # (num_noise_level)

    def _extend_condensed_graph_edge(self, pos, bond_index, bond_type, batch, **kwargs):
        N = pos.size(0)
        cutoff = kwargs.get("cutoff", self.config.edge_cutoff)
        edge_order = kwargs.get("edge_order", self.config.edge_order)

        _g_ext = extend_ts_graph_order_radius
        out = _g_ext(
            N, pos, bond_index, bond_type, batch, order=edge_order, cutoff=cutoff
        )
        edge_index_global, edge_index_local, edge_type_r, edge_type_p = out
        # local index             : (i, j) pairs which are edge of R or P.
        # edge_type_r/edge_type_p : 0, 1, 2, ... 23, 24, ...
        #                           0 -> no edge (bond)
        #                           1, 2, 3 ..-> bond type
        #                           23, 24 -> meaning no bond, but higher order edge. (2-hop or 3-hop)
        # global index            : atom pairs (i, j) which are closer than cutoff
        #                           are added to local_index.
        #

        edge_type_global = torch.zeros_like(edge_index_global[0]) - 1
        adj_global = to_dense_adj(
            edge_index_global, edge_attr=edge_type_global, max_num_nodes=N
        )
        adj_local_r = to_dense_adj(
            edge_index_local, edge_attr=edge_type_r, max_num_nodes=N
        )
        adj_local_p = to_dense_adj(
            edge_index_local, edge_attr=edge_type_p, max_num_nodes=N
        )
        adj_global_r = torch.where(adj_local_r != 0, adj_local_r, adj_global)
        adj_global_p = torch.where(adj_local_p != 0, adj_local_p, adj_global)
        edge_index_global_r, edge_type_global_r = dense_to_sparse(adj_global_r)
        edge_index_global_p, edge_type_global_p = dense_to_sparse(adj_global_p)
        edge_type_global_r[edge_type_global_r < 0] = 0
        edge_type_global_p[edge_type_global_p < 0] = 0
        edge_index_global = edge_index_global_r

        return edge_index_global, edge_index_local, edge_type_global_r, edge_type_global_p

    def _condensed_edge_embedding(self, edge_length, edge_type_r, edge_type_p,
                                  edge_attr=None, emb_type="bond_w_d"):

        assert emb_type in ["bond_w_d", "bond_wo_d", "add_d"]
        _enc = self.edge_encoder
        _cat_fn = self.edge_cat

        if emb_type == "bond_wo_d":
            edge_attr_r = _enc.bond_emb(edge_type_r)
            edge_attr_p = _enc.bond_emb(edge_type_p)
            edge_attr = _cat_fn(torch.cat([edge_attr_r, edge_attr_p], dim=-1))

        elif emb_type == "bond_w_d":
            edge_attr_r = _enc(edge_length, edge_type_r)  # Embed edges
            edge_attr_p = _enc(edge_length, edge_type_p)
            edge_attr = _cat_fn(torch.cat([edge_attr_r, edge_attr_p], dim=-1))

        elif emb_type == "add_d":
            edge_attr = _enc.mlp(edge_length) * edge_attr

        return edge_attr

    def forward_(self, atom_type, r_feat, p_feat, pos, bond_index, bond_type, batch, **kwargs):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """
        _g_ext = self._extend_condensed_graph_edge
        _e_emb = self._condensed_edge_embedding
        _a_emb = self.atom_embedding
        _af_emb = self.atom_feat_embedding
        _enc = self.encoder

        # condensed atom embedding
        a_emb = _a_emb(atom_type)
        af_emb_r = _af_emb(r_feat.float())
        af_emb_p = _af_emb(p_feat.float())
        z1 = a_emb + af_emb_r
        z2 = af_emb_p - af_emb_r
        z = torch.cat([z1, z2], dim=-1)

        # edge extension
        edge_index, _, edge_type_r, edge_type_p = _g_ext(
            pos,
            bond_index,
            bond_type,
            batch,
        )
        edge_length = get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)

        # edge embedding
        edge_attr = _e_emb(
            edge_length,
            edge_type_r,
            edge_type_p,
        )

        # encoding TS geometric graph and atom-pair
        node_attr = _enc(z, edge_index, edge_length, edge_attr=edge_attr)

        edge_ord4inp = self.config.edge_order
        edge_ord4out = self.config.pred_edge_order
        if edge_ord4inp != edge_ord4out:
            edge_index, _, edge_type_r, edge_type_p = _g_ext(
                pos,
                bond_index,
                bond_type,
                batch,
                edge_order=edge_ord4out,
            )
            edge_length = get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)
            edge_attr = _e_emb(
                edge_length,
                edge_type_r,
                edge_type_p,
            )

        h_pair = assemble_atom_pair_feature(node_attr, edge_index, edge_attr)  # (E, 2H)
        edge_inv = self.grad_dist_mlp(h_pair)  # (E, 1)

        return edge_inv, edge_index, edge_length

    # def forward(self, atom_type, r_feat, p_feat, pos, bond_index, bond_type, batch,
    #             time_step, return_edges=True, **kwargs):
    #     """
    #     Args:
    #         atom_type:  Types of atoms, (N, ).
    #         bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
    #         bond_type:  Bond types, (E, ).
    #         batch:      Node index to graph index, (N, ).
    #     """
    # def forward(self, rxn_graph, return_edges=True, **kwargs):
    def forward(self, rxn_graph, pred_type="edge", **kwargs):

        ## processing input variables
        atom_type = rxn_graph.atom_type
        r_feat = rxn_graph.r_feat
        p_feat = rxn_graph.p_feat
        pos = rxn_graph.pos
        bond_index = rxn_graph.edge_index
        # NOTE: directed edge is used in the original version of TSDiff
        # extend to directed edge
        bond_index = torch.cat([bond_index, bond_index.flip(0)], dim=1)
        edge_feat_r = rxn_graph.edge_feat_r
        edge_feat_p = rxn_graph.edge_feat_p
        edge_feat_r = torch.cat([edge_feat_r, edge_feat_r], dim=0)
        edge_feat_p = torch.cat([edge_feat_p, edge_feat_p], dim=0)

        # bond_type = rxn_graph.edge_feat_r * len(BOND_TYPES_ENCODER) + rxn_graph.edge_feat_p
        bond_type = edge_feat_r * len(BOND_TYPES_ENCODER) + edge_feat_p
        batch = rxn_graph.batch
        time_step = rxn_graph.t

        # convert features to one-hot encoding
        r_feat = torch.nn.functional.one_hot(r_feat, num_classes=10).reshape(len(r_feat), -1)
        p_feat = torch.nn.functional.one_hot(p_feat, num_classes=10).reshape(len(p_feat), -1)

        out = self.forward_(
            atom_type,
            r_feat,
            p_feat,
            pos,
            bond_index,
            bond_type,
            batch,
        )

        edge_inv, edge_index, edge_length = out

        # if self.config.type == "dsm":
        #     node2graph = batch
        #     edge2graph = node2graph.index_select(0, edge_index[0])
        #     noise_levels = self.sigmas.index_select(0, time_step)  # (G, )
        #     sigma_edge = noise_levels.index_select(0, edge2graph).unsqueeze(-1)  # (E, 1)
        #     edge_inv = edge_inv / sigma_edge

        # if return_edges:
        #     return edge_inv, edge_index, edge_length
        # else:
        #     return edge_inv

        if pred_type == "edge":
            mask = edge_index[0] < edge_index[1]
            edge_inv = edge_inv[mask]
            return edge_inv
        elif pred_type == "node":
            node_eq = eq_transform(
                edge_inv, pos, edge_index, edge_length
            )  # chain rule (re-parametrization, distance to position)
            return node_eq
        else:
            raise NotImplementedError
