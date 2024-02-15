import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse, coalesce, unbatch
from torch_geometric.nn import radius_graph
from .chem import BOND_TYPES_ENCODER


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


class RxnGraph:
    def __init__(
            self,
            atom_type,
            edge_index,
            edge_feat_r,
            edge_feat_p,
            r_feat,
            p_feat,
            batch,
            smarts="",
            order=3,
            cutoff=10.0,
            init_extend=True,
    ):
        self.atom_type = atom_type
        self.edge_index = edge_index
        self.edge_feat_r = edge_feat_r
        self.edge_feat_p = edge_feat_p

        self.batch = batch
        self.r_feat = r_feat
        self.p_feat = p_feat
        self.smarts = smarts
        self.device = atom_type.device
        self.num_nodes = self.atom_type.size(0)

        self.order = order
        self.cutoff = 10.0

        if init_extend:
            self.raw_edge_index = edge_index
            self.raw_edge_feat_r = edge_feat_r
            self.raw_edge_feat_p = edge_feat_p
            # make it undirected
            self.edge_index = torch.cat([self.edge_index, self.edge_index.flip(0)], dim=1)
            self.edge_feat_r = torch.cat([self.edge_feat_r, self.edge_feat_r], dim=0)
            self.edge_feat_p = torch.cat([self.edge_feat_p, self.edge_feat_p], dim=0)

            edge_index, type_r, type_p = self.extend_graph_order(order=order)
            self.edge_index = edge_index
            self.edge_feat_r = type_r
            self.edge_feat_p = type_p

    def to(self, device):
        # all torch.Tensor attributes are moved to the new device
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)
        return self

    def __repr__(self):
        return f"RxnGraph(smarts={self.smarts})"

    @classmethod
    def from_batch(cls, batch, order=3, cutoff=10.0):
        atom_type = batch.x
        edge_index = batch.edge_index
        edge_feat_r = batch.edge_feat_r
        edge_feat_p = batch.edge_feat_p
        r_feat = batch.r_feat
        p_feat = batch.p_feat
        rxn_smarts = batch.rxn_smarts
        batch = batch.batch  # caution: batch is newly declared (overwritten)
        return cls(atom_type, edge_index, edge_feat_r, edge_feat_p, r_feat, p_feat, batch, rxn_smarts, order, cutoff)

    def extend_graph_order(self, order=3):
        N = self.num_nodes
        NumBondTypes = len(BOND_TYPES_ENCODER) - 1

        # mask out the non-bond edges
        # and split r-edge and p-edge
        mask_r = self.edge_feat_r != 0
        edge_index_r = self.edge_index[:, mask_r]
        edge_feat_r = self.edge_feat_r[mask_r]
        mask_p = self.edge_feat_p != 0
        edge_index_p = self.edge_index[:, mask_p]
        edge_feat_p = self.edge_feat_p[mask_p]

        # get bond-type matrix and higher-order matrix (dense type)
        adj = to_dense_adj(edge_index_r, max_num_nodes=N).squeeze(0)
        ord_r = get_higher_order_adj_matrix(adj, order)
        ord_r = torch.where(ord_r > 1, NumBondTypes + ord_r - 1, torch.zeros_like(ord_r))
        bond_r = to_dense_adj(edge_index_r, edge_attr=edge_feat_r, max_num_nodes=N).squeeze(0)

        adj = to_dense_adj(edge_index_p, max_num_nodes=N).squeeze(0)
        ord_p = get_higher_order_adj_matrix(adj, order)
        ord_p = torch.where(ord_p > 1, NumBondTypes + ord_p - 1, torch.zeros_like(ord_p))
        bond_p = to_dense_adj(edge_index_p, edge_attr=edge_feat_p, max_num_nodes=N).squeeze(0)

        # Check if the bond type and higher-order type are overlapped, and merge them
        assert (bond_r * ord_r == 0).all() and (bond_p * ord_p == 0).all()
        new_r = bond_r + ord_r
        new_p = bond_p + ord_p

        # look up the edges only in one of the r-edges and p-edges
        # in that case, set the edge type to -1
        nonzero_r = - (new_r != 0).to(torch.float)
        nonzero_p = - (new_p != 0).to(torch.float)
        type_r = torch.where(new_r != 0, new_r, nonzero_p).to(torch.long)
        type_p = torch.where(new_p != 0, new_p, nonzero_r).to(torch.long)

        # convert it to sparse
        edge_index_r, type_r = dense_to_sparse(type_r)
        edge_index_p, type_p = dense_to_sparse(type_p)

        # replace the non-edge (-1) to 0
        type_r[type_r < 0] = 0
        type_p[type_p < 0] = 0

        assert (edge_index_r == edge_index_p).all()
        _edge_index = edge_index_r

        edge_index, type_r = coalesce(_edge_index, type_r.long(), N, N)  # modify data
        _, type_p = coalesce(_edge_index, type_p.long(), N, N)  # modify data

        return edge_index, type_r, type_p

    def update(self, pos, pos_=None):
        N = self.num_nodes
        r_adj = torch.sparse.LongTensor(self.edge_index, self.edge_feat_r, torch.Size([N, N]))
        p_adj = torch.sparse.LongTensor(self.edge_index, self.edge_feat_p, torch.Size([N, N]))

        radius_idx = radius_graph(pos, r=self.cutoff, batch=self.batch)
        if pos_ is not None:
            radius_idx_2 = radius_graph(pos_, r=self.cutoff, batch=self.batch)
            radius_idx = torch.cat([radius_idx, radius_idx_2], dim=1)
            radius_idx = torch.unique(radius_idx, dim=1)

        radius_adj = torch.sparse.LongTensor(
            radius_idx,
            torch.ones_like(radius_idx[0]).long().to(pos.device),
            torch.Size([N, N])
        )
        radius_adj = radius_adj * -1000

        g_r = (r_adj + radius_adj).coalesce()
        g_p = (p_adj + radius_adj).coalesce()
        assert (g_r.indices() == g_p.indices()).all()

        edge_index = g_r.indices()
        type_r = g_r.values()
        type_p = g_p.values()

        type_r[type_r == -1000] = 0
        type_p[type_p == -1000] = 0
        mask = type_r < 0
        type_r[mask] = type_r[mask] + 1000
        mask = type_p < 0
        type_p[mask] = type_p[mask] + 1000

        self.current_edge_index = edge_index
        self.current_edge_feat_r = type_r
        self.current_edge_feat_p = type_p

    def full_edge(self, upper_triangle=True):
        # make fully connected graph
        N = self.num_nodes
        r_adj = torch.sparse.LongTensor(self.edge_index, self.edge_feat_r, torch.Size([N, N]))
        p_adj = torch.sparse.LongTensor(self.edge_index, self.edge_feat_p, torch.Size([N, N]))

        edge_index_list = []
        for idx in unbatch(torch.arange(N), self.batch):
            edge_index_list.append(torch.combinations(idx).to(self.device).T)
        edge_index = torch.cat(edge_index_list, dim=1)

        full_adj = torch.sparse.LongTensor(
            edge_index,
            torch.ones_like(edge_index[0]).long().to(self.device),
            torch.Size([N, N])
        )

        g_r = (r_adj + full_adj).coalesce()
        g_p = (p_adj + full_adj).coalesce()

        edge_index = g_r.indices()
        type_r = g_r.values() - 1
        type_p = g_p.values() - 1

        if upper_triangle:
            mask = edge_index[0] < edge_index[1]
            edge_index = edge_index[:, mask]
            type_r = type_r[mask]
            type_p = type_p[mask]

        return edge_index, type_r, type_p


class DynamicRxnGraph(RxnGraph):
    def __init__(
        self,
        pos,
        pos_init,
        t,
        atom_type,
        edge_index,
        edge_feat_r,
        edge_feat_p,
        r_feat,
        p_feat,
        batch,
        smarts="",
        order=3,
        cutoff=10.0,
        init_extend=True,
    ):
        super().__init__(
            atom_type,
            edge_index,
            edge_feat_r,
            edge_feat_p,
            r_feat,
            p_feat,
            batch,
            smarts=smarts,
            order=order,
            cutoff=cutoff,
            init_extend=init_extend,
        )
        self.pos = pos
        self.pos_init = pos_init
        self.pos_traj = []
        self.score_traj = []
        self.time_traj = []
        self.t = t

        self.update(pos, pos_=pos_init)

    @classmethod
    def from_graph(cls, rxn_graph, pos, pos_init, t):
        atom_type = rxn_graph.atom_type
        edge_index = rxn_graph.edge_index
        edge_feat_r = rxn_graph.edge_feat_r
        edge_feat_p = rxn_graph.edge_feat_p
        r_feat = rxn_graph.r_feat
        p_feat = rxn_graph.p_feat
        smarts = rxn_graph.smarts
        order = rxn_graph.order
        cutoff = rxn_graph.cutoff
        batch = rxn_graph.batch

        graph = cls(
            pos,
            pos_init,
            t,
            atom_type,
            edge_index,
            edge_feat_r,
            edge_feat_p,
            r_feat,
            p_feat,
            batch,
            smarts=smarts,
            order=order,
            cutoff=cutoff,
            init_extend=False
        )
        return graph

    def update_graph(self, pos, batch, score=None, t=None):
        self.update(pos, self.pos_init)
        self.pos_traj.append(pos.to("cpu"))
        if score is not None:
            self.score_traj.append(score.to("cpu"))
        if t is not None:
            self.time_traj.append(t.to("cpu"))
            self.t = t
        self.pos = pos


if __name__ == "__main__":
    import sys
    from omegaconf import OmegaConf
    from proc import GrambowDataModule

    config = OmegaConf.load('config.yaml')
    datamodule = GrambowDataModule(config)

    for batch in datamodule.train_dataloader():
        break

    atom_type = batch.x
    edge_index = batch.edge_index
    edge_feat_r = batch.edge_feat_r
    edge_feat_p = batch.edge_feat_p
    r_feat = batch.r_feat
    p_feat = batch.p_feat
    smarts = batch.rxn_smarts
    pos = batch.pos[:, 0, :]

    graph = RxnGraph(atom_type, edge_index, edge_feat_r, edge_feat_p, r_feat, p_feat, smarts)

    graph.update(pos, batch.batch)
    e, r, p = graph.current_edge_index, graph.current_edge_feat_r, graph.current_edge_feat_p
