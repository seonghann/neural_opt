# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, radius
from torch_scatter import scatter_mean, scatter_add, scatter_max
from torch_sparse import coalesce
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from utils.chem import BOND_TYPES_ENCODER
# from utils.chem import BOND_TYPES
# from rdkit.Chem.rdchem import BondType as BT
# BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}
# from utils import activation_loader
#
class swish(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        return x * x.sigmoid()

def activation_loader(name):
    if name == "swish":
        return swish()
    else:
        return getattr(nn, name)()


class MeanReadout(nn.Module):
    """Mean readout operator over graphs with variadic sizes."""

    def forward(self, data, input):
        """
        Perform readout over the graph(s).
        Parameters:
            data (torch_geometric.data.Data): batched graph
            input (Tensor): node representations
        Returns:
            Tensor: graph representations
        """
        output = scatter_mean(input, data.batch, dim=0, dim_size=data.num_graphs)
        return output


class SumReadout(nn.Module):
    """Sum readout operator over graphs with variadic sizes."""

    def forward(self, data, input):
        """
        Perform readout over the graph(s).
        Parameters:
            data (torch_geometric.data.Data): batched graph
            input (Tensor): node representations
        Returns:
            Tensor: graph representations
        """
        output = scatter_add(input, data.batch, dim=0, dim_size=data.num_graphs)
        return output


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no activation or dropout in the last layer.
    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            # self.activation = getattr(F, activation)
            self.activation = activation_loader(activation)
        elif isinstance(activation, nn.Module):
            self.activation = activation
        else:
            self.activation = None

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

    def forward(self, input):
        """"""
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation is None:
                    print("No activation in MultiLayerPerceptron")
                else:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x

def index_set_subtraction(index1:torch.LongTensor, index2:torch.LongTensor, max_num_nodes=None):
    dev = index1.device
    adj1 = to_dense_adj(
            index1, 
            edge_attr=torch.arange(1, index1.shape[1]+1).to(dev), 
            max_num_nodes=max_num_nodes
            )
    adj2 = to_dense_adj(
            index2, 
            edge_attr=torch.ones(index2.shape[1]).to(dev), 
            max_num_nodes=max_num_nodes
            )
    
    adj = adj1 - adj2 * (index1.shape[1] + 1)
    mask = (adj > 0)
    index = adj1[mask] - 1
    #return index

    mask = torch.zeros(index1.size(1), device=dev)
    mask[index] = 1
    mask = mask.bool()
    return mask

def _extend_ts_graph_order(num_nodes, edge_index, edge_type, batch, order=3):
    def binarize(x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(adj, order):
        """
        Args:
            adj:        (N, N)
            type_mat:   (N, N)
        Returns:
            Following attributes will be updated:
              - edge_index
              - edge_type
            Following attributes will be added to the data object:
              - bond_edge_index:  Original edge_index.
        """
        adj_mats = [
            torch.eye(adj.size(0), dtype=torch.long, device=adj.device),
            binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device)),
        ]

        for i in range(2, order + 1):
            adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)

        for i in range(1, order + 1):
            order_mat += (adj_mats[i] - adj_mats[i - 1]) * i

        return order_mat

    # num_types = len(BOND_TYPES)
    num_types = len(BOND_TYPES_ENCODER)#; print(f"Debug: num_types(len of BOND_TYPES) is set to {len(BOND_TYPES_ENCODER)}. (in NeuralOPT)")
    # num_types = 5; print(f"Debug: num_types(len of BOND_TYPES) is set to 5. (in NeuralOPT)")
    N = num_nodes

    bond_type_r = edge_type // num_types
    mask_r = bond_type_r != 0
    bond_index_r = edge_index[:, mask_r]
    bond_type_r = bond_type_r[mask_r]

    bond_type_p = edge_type % num_types
    mask_p = bond_type_p != 0
    bond_index_p = edge_index[:, mask_p]
    bond_type_p = bond_type_p[mask_p]

    adj_r = to_dense_adj(bond_index_r, max_num_nodes=N).squeeze(0)
    adj_order_r = get_higher_order_adj_matrix(adj_r, order)
    type_mat_r = to_dense_adj(
        bond_index_r, edge_attr=bond_type_r, max_num_nodes=N
    ).squeeze(0)
    type_highorder_r = torch.where(
        adj_order_r > 1,
        num_types + adj_order_r - 1,
        torch.zeros_like(adj_order_r),
    )
    assert (type_mat_r * type_highorder_r == 0).all()
    type_new_r = type_mat_r + type_highorder_r
    type_mask_r = -(type_new_r != 0).to(torch.float)

    adj_p = to_dense_adj(bond_index_p, max_num_nodes=N).squeeze(0)
    adj_order_p = get_higher_order_adj_matrix(adj_p, order)
    type_mat_p = to_dense_adj(
        bond_index_p, edge_attr=bond_type_p, max_num_nodes=N
    ).squeeze(0)
    type_highorder_p = torch.where(
        adj_order_p > 1,
        num_types + adj_order_p - 1,
        torch.zeros_like(adj_order_p),
    )
    assert (type_mat_p * type_highorder_p == 0).all()
    type_new_p = type_mat_p + type_highorder_p
    type_mask_p = -(type_new_p != 0).to(torch.float)

    type_r = torch.where(type_new_r != 0, type_new_r, type_mask_p).to(torch.long)
    type_p = torch.where(type_new_p != 0, type_new_p, type_mask_r).to(torch.long)

    edge_index_r, edge_type_r = dense_to_sparse(type_r)
    edge_index_p, edge_type_p = dense_to_sparse(type_p)
    edge_type_r[edge_type_r < 0] = 0
    edge_type_p[edge_type_p < 0] = 0

    assert (edge_index_r == edge_index_p).all()
    edge_index_local = edge_index_r

    edge_index_local, edge_type_r = coalesce(
        edge_index_r, edge_type_r.long(), N, N
    )  # modify data
    _, edge_type_p = coalesce(edge_index_p, edge_type_p.long(), N, N)  # modify data

    return edge_index_local, edge_type_r, edge_type_p


def extend_ts_graph_order_radius(
    num_nodes,
    pos,
    edge_index,
    edge_type,
    batch,
    order=3,
    cutoff=10.0,
):

    edge_index_local, edge_type_r, edge_type_p = _extend_ts_graph_order(
        num_nodes, edge_index, edge_type, batch, order=order
    )

    edge_index_global, _ = _extend_to_radius_graph(
        pos, edge_index_local, edge_type_r, cutoff, batch
    )

    return edge_index_global, edge_index_local, edge_type_r, edge_type_p


def assemble_atom_pair_feature(node_attr, edge_index, edge_attr):
    h_row, h_col = node_attr[edge_index[0]], node_attr[edge_index[1]]
    h_pair = torch.cat([h_row * h_col, edge_attr], dim=-1)  # (E, 2H)
    return h_pair


def generate_symmetric_edge_noise(num_nodes_per_graph, edge_index, edge2graph, device):
    num_cum_nodes = num_nodes_per_graph.cumsum(0)  # (G, )
    node_offset = num_cum_nodes - num_nodes_per_graph  # (G, )
    edge_offset = node_offset[edge2graph]  # (E, )

    num_nodes_square = num_nodes_per_graph**2  # (G, )
    num_nodes_square_cumsum = num_nodes_square.cumsum(-1)  # (G, )
    edge_start = num_nodes_square_cumsum - num_nodes_square  # (G, )
    edge_start = edge_start[edge2graph]

    all_len = num_nodes_square_cumsum[-1]

    node_index = edge_index.t() - edge_offset.unsqueeze(-1)
    node_large = node_index.max(dim=-1)[0]
    node_small = node_index.min(dim=-1)[0]
    undirected_edge_id = node_large * (node_large + 1) + node_small + edge_start

    symm_noise = torch.zeros(size=[all_len.item()], device=device)
    symm_noise.normal_()
    d_noise = symm_noise[undirected_edge_id].unsqueeze(-1)  # (E, 1)
    return d_noise


# def _extend_graph_order(num_nodes, edge_index, edge_type, order=3):
#     """
#     Args:
#         num_nodes:  Number of atoms.
#         edge_index: Bond indices of the original graph.
#         edge_type:  Bond types of the original graph.
#         order:  Extension order.
#     Returns:
#         new_edge_index: Extended edge indices.
#         new_edge_type:  Extended edge types.
#     """
# 
#     def binarize(x):
#         return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
# 
#     def get_higher_order_adj_matrix(adj, order):
#         """
#         Args:
#             adj:        (N, N)
#             type_mat:   (N, N)
#         Returns:
#             Following attributes will be updated:
#               - edge_index
#               - edge_type
#             Following attributes will be added to the data object:
#               - bond_edge_index:  Original edge_index.
#         """
#         adj_mats = [
#             torch.eye(adj.size(0), dtype=torch.long, device=adj.device),
#             binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device)),
#         ]
# 
#         for i in range(2, order + 1):
#             adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
#         order_mat = torch.zeros_like(adj)
# 
#         for i in range(1, order + 1):
#             order_mat += (adj_mats[i] - adj_mats[i - 1]) * i
# 
#         return order_mat
# 
#     # num_types = len(BOND_TYPES)
#     num_types = len(BOND_TYPES_ENCODER); print(f"Debug: num_types(len of BOND_TYPES) is set to {len(BOND_TYPES_ENCODER)}. (in NeuralOPT)")
# 
#     N = num_nodes
#     adj = to_dense_adj(edge_index).squeeze(0)
#     adj_order = get_higher_order_adj_matrix(adj, order)  # (N, N)
# 
#     type_mat = to_dense_adj(edge_index, edge_attr=edge_type).squeeze(0)  # (N, N)
#     type_highorder = torch.where(
#         adj_order > 1, num_types**2 + adj_order - 1, torch.zeros_like(adj_order)
#     )
#     assert (type_mat * type_highorder == 0).all()
#     type_new = type_mat + type_highorder
# 
#     new_edge_index, new_edge_type = dense_to_sparse(type_new)
#     _, edge_order = dense_to_sparse(adj_order)
# 
#     # data.bond_edge_index = data.edge_index  # Save original edges
#     new_edge_index, new_edge_type = coalesce(
#         new_edge_index, new_edge_type.long(), N, N
#     )  # modify data
# 
#     # [Note] This is not necessary
#     # data.is_bond = (data.edge_type < num_types)
# 
#     # [Note] In earlier versions, `edge_order` attribute will be added.
#     #         However, it doesn't seem to be necessary anymore so I removed it.
#     # edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N) # modify data
#     # assert (data.edge_index == edge_index_1).all()
# 
#     return new_edge_index, new_edge_type


def _extend_to_radius_graph(
    pos,
    edge_index,
    edge_type,
    cutoff,
    batch,
    unspecified_type_number=0,
    is_sidechain=None,
):

    assert edge_type.dim() == 1
    N = pos.size(0)

    bgraph_adj = torch.sparse.LongTensor(edge_index, edge_type, torch.Size([N, N]))

    if is_sidechain is None:
        rgraph_edge_index = radius_graph(pos, r=cutoff, batch=batch)  # (2, E_r)
    else:
        # fetch sidechain and its batch index
        is_sidechain = is_sidechain.bool()
        dummy_index = torch.arange(pos.size(0), device=pos.device)
        sidechain_pos = pos[is_sidechain]
        sidechain_index = dummy_index[is_sidechain]
        sidechain_batch = batch[is_sidechain]

        assign_index = radius(
            x=pos, y=sidechain_pos, r=cutoff, batch_x=batch, batch_y=sidechain_batch
        )
        r_edge_index_x = assign_index[1]
        r_edge_index_y = assign_index[0]
        r_edge_index_y = sidechain_index[r_edge_index_y]

        rgraph_edge_index1 = torch.stack((r_edge_index_x, r_edge_index_y))  # (2, E)
        rgraph_edge_index2 = torch.stack((r_edge_index_y, r_edge_index_x))  # (2, E)
        rgraph_edge_index = torch.cat(
            (rgraph_edge_index1, rgraph_edge_index2), dim=-1
        )  # (2, 2E)
        # delete self loop
        rgraph_edge_index = rgraph_edge_index[
            :, (rgraph_edge_index[0] != rgraph_edge_index[1])
        ]

    rgraph_adj = torch.sparse.LongTensor(
        rgraph_edge_index,
        torch.ones(rgraph_edge_index.size(1)).long().to(pos.device)
        * unspecified_type_number,
        torch.Size([N, N]),
    )

    composed_adj = (bgraph_adj + rgraph_adj).coalesce()  # Sparse (N, N, T)
    # edge_index = composed_adj.indices()
    # dist = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)

    new_edge_index = composed_adj.indices()
    new_edge_type = composed_adj.values().long()

    return new_edge_index, new_edge_type


# def extend_graph_order_radius(
#     num_nodes,
#     pos,
#     edge_index,
#     edge_type,
#     batch,
#     order=3,
#     cutoff=10.0,
#     extend_order=True,
#     extend_radius=True,
#     is_sidechain=None,
# ):
# 
#     if extend_order:
#         edge_index, edge_type = _extend_graph_order(
#             num_nodes=num_nodes, edge_index=edge_index, edge_type=edge_type, order=order
#         )
#         # edge_index_order = edge_index
#         # edge_type_order = edge_type
# 
#     if extend_radius:
#         edge_index, edge_type = _extend_to_radius_graph(
#             pos=pos,
#             edge_index=edge_index,
#             edge_type=edge_type,
#             cutoff=cutoff,
#             batch=batch,
#             is_sidechain=is_sidechain,
#         )
# 
#     return edge_index, edge_type


def coarse_grain(pos, node_attr, subgraph_index, batch):
    cluster_pos = scatter_mean(pos, index=subgraph_index, dim=0)  # (num_clusters, 3)
    cluster_attr = scatter_add(
        node_attr, index=subgraph_index, dim=0
    )  # (num_clusters, H)
    cluster_batch, _ = scatter_max(
        batch, index=subgraph_index, dim=0
    )  # (num_clusters, )

    return cluster_pos, cluster_attr, cluster_batch


def batch_to_natoms(batch):
    return scatter_add(torch.ones_like(batch), index=batch, dim=0)


def get_complete_graph(natoms):
    """
    Args:
        natoms: Number of nodes per graph, (B, 1).
    Returns:
        edge_index: (2, N_1 + N_2 + ... + N_{B-1}), where N_i is the number of nodes of the i-th graph.
        num_edges:  (B, ), number of edges per graph.
    """
    natoms_sqr = (natoms**2).long()
    num_atom_pairs = torch.sum(natoms_sqr)
    natoms_expand = torch.repeat_interleave(natoms, natoms_sqr)

    index_offset = torch.cumsum(natoms, dim=0) - natoms
    index_offset_expand = torch.repeat_interleave(index_offset, natoms_sqr)

    index_sqr_offset = torch.cumsum(natoms_sqr, dim=0) - natoms_sqr
    index_sqr_offset = torch.repeat_interleave(index_sqr_offset, natoms_sqr)

    atom_count_sqr = (
        torch.arange(num_atom_pairs, device=num_atom_pairs.device) - index_sqr_offset
    )

    index1 = (atom_count_sqr // natoms_expand).long() + index_offset_expand
    index2 = (atom_count_sqr % natoms_expand).long() + index_offset_expand
    edge_index = torch.cat([index1.view(1, -1), index2.view(1, -1)])
    mask = torch.logical_not(index1 == index2)
    edge_index = edge_index[:, mask]

    num_edges = natoms_sqr - natoms  # Number of edges per graph

    return edge_index, num_edges
