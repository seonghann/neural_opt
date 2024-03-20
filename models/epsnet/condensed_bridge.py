import torch
from torch import nn
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import numpy as np
import math

from utils.chem import BOND_TYPES
from utils import activation_loader
from ..common import MultiLayerPerceptron, assemble_atom_pair_feature, extend_ts_graph_order_radius
from ..encoder import get_edge_encoder, load_encoder
from ..geometry import get_distance, eq_transform


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
        self.config = config

        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        self.edge_encoder = get_edge_encoder(config)
        assert config.hidden_dim % 2 == 0
        self.atom_embedding = nn.Embedding(100, config.hidden_dim // 2)
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

        betas = get_beta_schedule(
            beta_schedule=config.beta_schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            num_diffusion_timesteps=config.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float()
        self.betas = nn.Parameter(betas, requires_grad=False)
        self.sigmas = nn.Parameter(betas.cumsum(dim=0), requires_grad=False)
        # beta : g^2(t) = beta(t)
        # sigmas : sigma^2(t) = \int_0^t beta(t) dt

        self.num_timesteps = self.betas.size(0)

        self.num_bond_types = len(BOND_TYPES)
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

    def radius_graph_extension(self, rxn_graph, pos, pos_T, batch, **kwargs):
        # NOTE : rxn_graph should be extended by "_extend_ts_graph_order" function
        N = pos.size(0)
        cutoff = kwargs.get("cutoff", self.config.edge_cutoff)
        edge_order = kwargs.get("edge_order", self.config.edge_order)

        index = rxn_graph.edge_index
        edge_r = rxn_graph.edge_type_r
        edge_p = rxn_graph.edge_type_p

        graph_adj_r = torch.sparse.LongTensor(index, edge_r, torch.Size([N, N]))
        graph_adj_p = torch.sparse.LongTensor(index, edge_p, torch.Size([N, N]))

        radius_index_1 = radius_graph(pos, r=cutoff, batch=batch, dtype=torch.long)
        radius_adj_1 = torch.sparse.LongTensor(
            radius_index_1,
            torch.zeros_like(radius_index_1[0], dtype=torch.long).long(),
            torch.Size([N, N])
        )
        radius_index_2 = radius_graph(pos_T, r=cutoff, batch=batch, dtype=torch.long)
        radius_adj_2 = torch.sparse.LongTensor(
            radius_index_2,
            torch.zeros_like(radius_index_2[0], dtype=torch.long).long(),
            torch.Size([N, N])
        )

        g_r = (radius_adj_1 + radius_adj_2 + graph_adj_r).coalesce()
        g_p = (radius_adj_1 + radius_adj_2 + graph_adj_p).coalesce()
        assert torch.all(g_r.indices() == g_p.indices()).item()

        edge_index = g_r.indices()
        edge_type_r = g_r.values()
        edge_type_p = g_p.values()
        return edge_index, edge_type_r, edge_type_p

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

    def _condensed_edge_embedding(self, edge_length, edge_length_T, edge_type_r, edge_type_p,
                                  edge_attr=None, emb_type="bond_w_d"):

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

    def new_forward(self, rxn_graph, t, pos, pos_T, batch, **kwargs):
        """
        Args:
            rxn_graph: rxn graph object (atom_type, r_feat, p_feat, edge_type_r, edge_type_p, edge_index)
            t: time parameter 0 <= t <= 1 (float) (G, )
            pos: structure of noisy structure (N, 3)
            pos_T: structure of initial structure (at time = 1) (N, 3)
            batch: batch index (N, )
        """
        t_node = t.index_select(0, batch).unsqueeze(-1)

        # 1) condensed atom embedding
        atom_emb = self.atom_embedding(rxn_graph.atom_type)
        atom_feat_emb_r = self.atom_feat_embedding(rxn_graph.r_feat.float())
        atom_feat_emb_p = self.atom_feat_embedding(rxn_graph.p_feat.float())
        z1 = atom_emb + atom_feat_emb_r
        z2 = atom_feat_emb_p - atom_feat_emb_r
        z = torch.cat([z1, z2], dim=-1)

        # 2) radius-graph extension
        edge_index, edge_type_r, edge_type_p = self.radius_graph_extension(
            rxn_graph, pos, pos_T, batch
        )
        edge_length = get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)
        edge_length_T = get_distance(pos_T, edge_index).unsqueeze(-1)  # (E, 1)

        # 3) edge_embedding
        edge_attr = self._condensed_edge_embedding(
            edge_length,
            edge_length_T,
            edge_type_r,
            edge_type_p
        )

        # encoding TS geometric graph and atom-pair
        node_attr = self.encoder(
            t_node, z, edge_index, edge_length, edge_length_T, edge_attr=edge_attr
        )

        edge_ord4inp = self.config.edge_order
        edge_ord4out = self.config.pred_edge_order
        if edge_ord4inp != edge_ord4out:
            raise NotImplementedError("Edge-extension order should be same for input and output")

        h_pair = assemble_atom_pair_feature(node_attr, edge_index, edge_attr)  # (E, 2H)
        edge_inv = self.grad_dist_mlp(h_pair)  # (E, 1)

        return edge_inv, edge_index, edge_length, edge_length_T

    def _forward(self, atom_type, r_feat, p_feat, pos, pos_T, bond_index, bond_type, batch, **kwargs):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """
        # TODO: pos_T is not used
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
        edge_length_T = get_distance(pos_T, edge_index).unsqueeze(-1)  # (E, 1)

        # edge embedding
        # TODO : edge embedding should be re-formulated to address pos_T and pos
        edge_attr = _e_emb(
            edge_length,
            edge_length_T,
            edge_type_r,
            edge_type_p,
        )

        # encoding TS geometric graph and atom-pair
        # TODO : address edge_length and edge_length_T together
        node_attr = _enc(z, edge_index, edge_length, edge_length_T, edge_attr=edge_attr)

        edge_ord4inp = self.config.edge_order
        edge_ord4out = self.config.pred_edge_order
        if edge_ord4inp != edge_ord4out:
            raise NotImplementedError(
                "edge_ord4inp != edge_ord4out is not implemented yet."
            )
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

    def forward(self, atom_type, r_feat, p_feat, pos, pos_T, bond_index, bond_type, batch,
                time_step, return_edges=True, **kwargs):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """
        out = self._forward(
            atom_type,
            r_feat,
            p_feat,
            pos,
            pos_T,
            bond_index,
            bond_type,
            batch,
        )

        edge_inv, edge_index, edge_length = out

        if return_edges:
            return edge_inv, edge_index, edge_length
        else:
            return edge_inv

    def new_get_loss(
        self,
        rxn_graph,
        t,
        pos_0,
        pos_T,
    ):
        pass

    def get_loss(
        self,
        atom_type,
        r_feat,
        p_feat,
        pos_0,
        pos_T,
        bond_index,
        bond_type,
        batch,
        num_nodes_per_graph,
        num_graphs,
        anneal_power=2.0,
        extend_order=True,
        extend_radius=True,
        coord_type="morse_like",
        mu=None,  # position of the interpolation point (N, 3)
        t=None,
    ):
        node2graph = batch
        dev = pos_0.device
        # set time step and noise level
        # TODO: change time variable from discrete type to contiuous type
        # old_version : t0 = self.config.get("t0", 0)
        # old_version : t1 = self.config.get("t1", self.num_timesteps)

        if t is None:
            sz = num_graphs // 2 + 1
            # old_version : half_1 = torch.randint(t0, t1, size=(sz,), device=dev)
            # old_version : half_2 = t0 + t1 - 1 - half_1
            half_1 = torch.rand(size=(sz,), device=dev) * 0.5
            half_2 = 1 - torch.rand(size=(sz,), device=dev) * 0.5
            time_step = torch.cat([half_1, half_2], dim=0)[:num_graphs] # (G, )

        else:
            time_step = t

        # TODO: change the diffusion parameters into a function on time.
        # old_version : sigma = self.sigmas.index_select(0, time_step)  # (G, )
        sigma_sq = self.sigma_sq(time_step)
        # TODO: sigmas_1 should be defined (sigma_1 = sigma_1(1.0))
        sigma_sq_1 = self.sigma_sq_1  # (1, )
        SNR_ratio = sigma_sq / sigma_sq_1  # ratio of the signal to noise ratio SNR_T/SNR_t

        # Get perturbed structure
        if coord_type == "cartesian":
            # Cartesian Diffusion Bridge Kernel (CDBK)
            sigma_sq_node = sigma_sq.index_select(0, node2graph).unsqueeze(-1)  # (N, 1)
            SNR_ratio_node = SNR_ratio.index_select(0, node2graph).unsqueeze(-1)  # (N, 1)
            # Perterb pos
            mu_hat = pos_0 * (1 - SNR_ratio_node) + pos_T * SNR_ratio_node  # calc mean
            sigma_sq_hat = sigma_sq_node * (1 - SNR_ratio_node)  # calc variance

            noise = torch.randn(size=mu_hat.size(), device=dev)  # sampling noise
            x_t = mu_hat + sigma_sq_hat.sqrt() * noise  # sampling perturbed

        elif coord_type in ["distance", "morse_like"]:
            # General Coordinate Diffusion Bridge Kernel (GCDBK)
            if mu is None:
                raise ValueError("mu should be given for non-cartesian type.")
            # TODO : require edge_index and edge_length, convert_coord function
            # extended edge index and edge length calculation is prerequisited.
            mu_hat = convert_coord(mu, edge_index, edge_length, coord_type=coord_type)  # (E, 3)

            sigma_sq_edge = sigma_sq.index_select(0, edge2graph).unsqueeze(-1)  # (N, 1)
            SNR_ratio_edge = SNR_ratio.index_select(0, edge2graph).unsqueeze(-1)  # (N, 1)
            siga_sq_hat = sigma_sq_edge * (1 - SNR_ratio_edge)  # (E, 1)

            noise = torch.randn(size=mu_hat.size(), device=dev)
            x_t = eq_transform(noise, mu, edge_index, edge_length, coord_type=coord_type)
            x_t = mu_hat + sigma_sq_hat.sqrt() * noise

        else:
            raise NotImplementedError

        # prediction
        edge_inv, edge_index, edge_length = self(
            atom_type, r_feat, p_feat, pos_perturbed, pos_T, bond_index, bond_type,
            batch, time_step, return_edges=True, extend_order=extend_order,
            extend_radius=extend_radius,
        )  # (E, 1)

        node_eq = eq_transform(
            edge_inv, pos_perturbed, edge_index, edge_length
        )  # chain rule (re-parametrization, distance to position)

        # Equation (9) in the DDBM paper
        # target = score(q(x_t) | x_0, x_T)
        #        = (mu - x_t) / sigma_t * 2

        edge2graph = node2graph.index_select(0, edge_index[0])
        std_edge = (sigma * (1 - SNR_ratio)).index_select(0, edge2graph).unsqueeze(-1)  # (E, 1)

        # compute original and perturbed distances
        d_mu = get_distance(mu, edge_index).unsqueeze(-1)  # (E, 1)
        d_perturbed = edge_length

        # compute target
        d_target = d_mu - d_perturbed  # (E, 1), denoising direction
        d_target = d_target / std_edge * 2
        pos_target = eq_transform(
            d_target, pos_perturbed, edge_index, edge_length
        )  # chain rule (re-parametrization, distance to position)

        # calc loss
        loss = (node_eq - pos_target) ** 2
        loss = torch.sum(loss, dim=-1, keepdim=True)

        return loss
