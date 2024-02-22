import torch
from torch_geometric.utils import unbatch, unbatch_edge_index
from utils.chem import ATOMIC_RADII_LIST
from torch.nn.utils.rnn import pad_sequence
from torch import vmap


def redefine_edge_index(edges, batch, num_nodes):
    """
    Given edge_index are block diagonal.
    Redefine edge_index with batch dimension.
    In other words, we will give the batch index to each edge, and new edge index to each edge.
    Args:
        edges (torch.Tensor): node-pair indices of a corresponding edge (2, E)
        batch (torch.Tensor): graph index of each nodes (N, )
        num_nodes (torch.Tensor): number of nodes for each graph (B, )
    Return:
        index_tensor (torch.Tensor): redefined edge index tensor (4, E)
        e.g)
        edges[:, 0] = (i, j)
        index_tensor[:, 0] = (batch_idx, edge_idx, i', j')
        Here, i', j' is the index of node in each graph (not a global graph).
    """
    device = edges.device
    edge_bundle = unbatch_edge_index(edges, batch)
    num_edges = torch.LongTensor([edge.size(1) for edge in edge_bundle])

    batch_index = torch.repeat_interleave(torch.arange(batch.max() + 1), num_edges, dim=0)  # (E, )
    batch_index = batch_index.to(device)

    # new edge_index is (i * num_nodes + j) for each (i, j) in edge_index_bundle
    edge_index = torch.cat(edge_bundle, dim=-1)
    # multiplier = torch.repeat_interleave(num_nodes, num_edges, dim=0)
    # edge_index = edge_index[0] * multiplier + edge_index[1]  # (E, )
    n = num_nodes.max()
    i, j = edge_index
    edge_index = (j - i - 1) + n * i - i * (i + 1) // 2

    index_tensor = torch.cat([
        batch_index.unsqueeze(0),
        edge_index.unsqueeze(0),
        torch.cat(edge_bundle, dim=-1)],
        dim=0)  # (4, E)

    return index_tensor


def redefine_with_pad(src, batch, padding_value=0):
    src = unbatch(src, batch)
    return pad_sequence(src, padding_value=padding_value, batch_first=True)


def _repeat(x, n):
    return x.unsqueeze(-1).expand(-1, n).flatten()


def _stack(x):
    return torch.stack([3 * x, 3 * x + 1, 3 * x + 2]).T.flatten()


class GeodesicSolver(object):
    def __init__(self, config,):
        self.alpha = config.ode_solver.get("alpha", 1.6)
        self.beta = config.ode_solver.get("beta", 2.3)
        self.svd_tol = config.ode_solver.get("svd_tol", 1e-3)
        self.atomic_radius = torch.Tensor(ATOMIC_RADII_LIST)

    def compute_de(self, edge_index, atom_type):
        """
        Compute equilibrium distance between two atom.
        Args:
            edge_index (torch.Tensor): edge index tensor (2, E)
            atom_type (torch.Tensor): atom type tensor (N, )
            atomic_radius (torch.Tensor): pre-defined atomic radius tensor (100, )
        Returns:
            d_e_ij (torch.Tensor): equilibrium distance tensor (E, )
        """
        self.atomic_radius = self.atomic_radius.to(atom_type.device)
        d_e_ij = self.atomic_radius[atom_type[edge_index]].sum(0)
        return d_e_ij

    def compute_d(self, edge_index, pos):
        """
        Compute distance between two atom.
        Args:
            edge_index (torch.Tensor): edge index tensor (2, E)
            pos (torch.Tensor): position tensor (N, 3)
        Returns:
            d_ij (torch.Tensor): distance tensor (E, )
        """
        i, j = edge_index
        return (pos[i] - pos[j]).norm(dim=1)

    def compute_q(self, edge_index, atom_type, pos):
        """
        Compute 'morse' like coordinate.
        q_ij = exp(-alpha * (d_ij - d_e_ij) / d_e_ij) + beta * (d_e_ij / d_ij)
        Args:
            edge_index (torch.Tensor): edge index tensor (2, E)
            atom_type (torch.Tensor): atom type tensor (N, )
            pos (torch.Tensor): position tensor (N, 3)
        Returns:
            A
            q_ij (torch.Tensor): q tensor (E, )
        """
        d_ij = self.compute_d(edge_index, pos)
        d_e_ij = self.compute_de(edge_index, atom_type)
        q_ij = torch.exp(- self.alpha * (d_ij - d_e_ij) / d_e_ij) + self.beta * (d_e_ij / d_ij)
        return q_ij

    def sparse_jacobian_d(self, edge_index, pos):
        """
        Compute jacobian matrix of d_ij.
        Args:
            edge_index (torch.Tensor): edge index tensor (2, E)
            atom_type (torch.Tensor): atom type tensor (N, )
            pos (torch.Tensor): position tensor (N, 3)
        Returns:
            jacobian (torch.Tensor, ): jacobian matrix tensor, expected size (E, 3N)
        """
        N = pos.size(0)
        E = edge_index.size(1)

        i, j = edge_index
        k = torch.arange(i.size(0)).to(pos.device)

        d_ij = self.compute_d(edge_index, pos)
        dd_dx = (pos[i] - pos[j]) / d_ij[:, None]
        # dd_ij/dx_i = (x_i - x_j) / d_ij
        dd_dx = dd_dx.flatten()

        k = k.unsqueeze(-1).expand(E, 3).flatten()
        i = torch.stack([3 * i, 3 * i + 1, 3 * i + 2]).T.reshape(-1)
        j = torch.stack([3 * j, 3 * j + 1, 3 * j + 2]).T.reshape(-1)

        jacobian = torch.sparse_coo_tensor(
            torch.stack([k, i]), dd_dx, (E, 3 * N)
        )
        jacobian += torch.sparse_coo_tensor(
            torch.stack([k, j]), -dd_dx, (E, 3 * N)
        )
        return jacobian

    def sparse_jacobian_q(self, edge_index, atom_type, pos):
        """
        Compute jacobian matrix of q_ij.
        Args:
            edge_index (torch.Tensor): edge index tensor (2, E), we consider directed graph
            atom_type (torch.Tensor): atom type tensor (N, )
            pos (torch.Tensor): position tensor (N, 3)
        Returns:
            jacobian (torch.coo-Tensor, ): jacobian matrix sparse-coo tensor, expected size (E, 3N)
        """
        N = pos.size(0)
        E = edge_index.size(1)
        # dq/dx = dq/dd * dd/dx

        i, j = edge_index
        k = torch.arange(i.size(0)).to(pos.device)

        d_ij = self.compute_d(edge_index, pos)
        d_e_ij = self.compute_de(edge_index, atom_type)
        dd_dx = (pos[i] - pos[j]) / d_ij[:, None]
        # dd_ij/dx_i = (x_i - x_j) / d_ij
        # dq_ij/dd_ij = - alpha / d_e_ij * exp(- alpha / d_e_ij * (d_ij - d_e_ij)) - beta * d_e_ij / d_ij ** 2

        # debug] print all variables and check their type and shape
        dq_dd = - self.alpha / d_e_ij * torch.exp(- self.alpha / d_e_ij * (d_ij - d_e_ij)) - self.beta * d_e_ij / d_ij ** 2

        dq_dx = dq_dd.unsqueeze(-1) * dd_dx  # (E, 3)

        k = k.unsqueeze(-1).expand(E, 3).flatten()
        i = torch.stack([3 * i, 3 * i + 1, 3 * i + 2]).T.reshape(-1)
        j = torch.stack([3 * j, 3 * j + 1, 3 * j + 2]).T.reshape(-1)

        jacobian = torch.sparse_coo_tensor(
            torch.stack([k, i]), dq_dx.flatten(), (E, 3 * N)
        )
        jacobian += torch.sparse_coo_tensor(
            torch.stack([k, j]), -dq_dx.flatten(), (E, 3 * N)
        )
        return jacobian

    def sparse_batch_jacobian_q(self, index_tensor, atom_type, pos):
        """
        Args:
            index_tensor (torch.LongTensor): redefined edge tensor - (batch_idx, edge_idx, i', j'), shape: (4, E)
            atom_type (torch.Tensor): atom type tensor (B, n)
            pos (torch.Tensor): position tensor (B, n, 3)
        Returns:
            jacobian (torch.sparse Tensor): jacobian matrix tensor, expected size (B, e, 3n)
        """
        assert pos.dim() == 3

        B = pos.size(0)
        n = pos.size(1)
        e = n * (n - 1) // 2

        b_i, b_j = index_tensor[[0, 2]], index_tensor[[0, 3]]

        _b_i = (b_i * torch.LongTensor([[n], [1]]).to(b_i.device)).sum(0)  # (E, )
        _b_j = (b_j * torch.LongTensor([[n], [1]]).to(b_j.device)).sum(0)  # (E, )
        _pos = pos.reshape(-1, 3)  # (B * n, 3)
        _atom_type = atom_type.reshape(-1)  # (B * n, )

        _edge_index = torch.stack([_b_i, _b_j])  # (2, E)

        d_ij = self.compute_d(_edge_index, _pos)  # (E, )
        d_e_ij = self.compute_de(_edge_index, _atom_type)  # (E, )

        dd_dx = (_pos[_b_i] - _pos[_b_j]) / d_ij[:, None]  # (E, 3)
        dq_dd = - self.alpha / d_e_ij * torch.exp(- self.alpha / d_e_ij * (d_ij - d_e_ij)) - self.beta * d_e_ij / d_ij ** 2  # (E, )

        dq_dx = dq_dd.unsqueeze(-1) * dd_dx  # (E, 3)

        index = index_tensor[:2]  # (2, E)
        index = torch.repeat_interleave(index, 3, dim=1)  # (2, 3E)

        _i = _stack(index_tensor[2])  # (3E, )
        _j = _stack(index_tensor[3])  # (3E, )

        jacobian = torch.sparse_coo_tensor(
            torch.cat([index, _i.unsqueeze(0)], dim=0),
            dq_dx.flatten(),
            (B, e, 3 * n),
        )
        jacobian += torch.sparse_coo_tensor(
            torch.cat([index, _j.unsqueeze(0)], dim=0),
            -dq_dx.flatten(),
            (B, e, 3 * n),
        )
        return jacobian

    def sparse_batch_hessian_q(self, index_tensor, atom_type, pos):
        """
        Args:
            index_tensor (torch.LongTensor): redefined edge tensor - (batch_idx, edge_idx, i', j'), shape: (4, E)
            atom_type (torch.Tensor): atom type tensor (B, n)
            pos (torch.Tensor): position tensor (B, n, 3)
        Returns:
            hessian (torch.sparse Tensor): hessian matrix tensor, expected size (B, 9 * n ** 2, e)
        """
        assert pos.dim() == 3

        B = pos.size(0)
        n = pos.size(1)
        e = n * (n - 1) // 2

        b_i, b_j = index_tensor[[0, 2]], index_tensor[[0, 3]]

        _b_i = (b_i * torch.LongTensor([[n], [1]]).to(b_i.device)).sum(0)  # (E, )
        _b_j = (b_j * torch.LongTensor([[n], [1]]).to(b_j.device)).sum(0)  # (E, )
        _pos = pos.reshape(-1, 3)  # (B * n, 3)
        _atom_type = atom_type.reshape(-1)  # (B * n, )
        _edge_index = torch.stack([_b_i, _b_j])  # (2, E)

        d_ij = self.compute_d(_edge_index, _pos)  # (E, )
        d_e_ij = self.compute_de(_edge_index, _atom_type)  # (E, )
        d_pos = (_pos[_b_i] - _pos[_b_j])  # (E, 3)
        hess_d_ij = d_pos.reshape(-1, 1, 3) * d_pos.reshape(-1, 3, 1) / (d_ij.reshape(-1, 1, 1) ** 3)
        eye = torch.eye(3).reshape(1, 3, 3).to(d_ij.device)
        hess_d_ij -= eye / d_ij.reshape(-1, 1, 1)  # (E, 3, 3)
        # hess_d_ii = - hess_d_ij, hess_d_jj = - hess_d_ij, hess_d_ji = hess_d_ij

        jacob_d_i = d_pos / d_ij[:, None]  # (E, 3)
        # jacob_d_j = jacob_d_i

        K1 = - (self.alpha / d_e_ij) * torch.exp(-self.alpha * (d_ij - d_e_ij) / d_e_ij) - self.beta * d_e_ij / d_ij ** 2  # (E, )
        K2 = (self.alpha / d_e_ij) ** 2 * torch.exp(-self.alpha * (d_ij - d_e_ij) / d_e_ij) + 2 * self.beta * d_e_ij / d_ij ** 3  # (E, )

        hess_q_ij = K1.reshape(-1, 1, 1) * hess_d_ij + K2.reshape(-1, 1, 1) * jacob_d_i.unsqueeze(1) * (- jacob_d_i).unsqueeze(2)  # (E, 3, 3)
        # hess_q_ji = hess_q_ij, hess_q_ii = - hess_q_ij, hess_q_jj = - hess_q_ij

        index = index_tensor[:2]  # (2, E)
        index = torch.repeat_interleave(index, 9, dim=1)  # (2, 9E)
        index_1, index_2 = index
        col_i, row_i = _stack(_repeat(index_tensor[2], 3)), _repeat(_stack(index_tensor[2]), 3)  # (3E, )
        col_j, row_j = _stack(_repeat(index_tensor[3], 3)), _repeat(_stack(index_tensor[3]), 3)  # (3E, )

        # make sparse hessian tensor
        hess = torch.sparse_coo_tensor(
            torch.stack([index_1, row_i * 3 * n + col_i, index_2], dim=0),
            - hess_q_ij.flatten(),
            (B, 9 * n ** 2, e),
        )  # i, i
        hess += torch.sparse_coo_tensor(
            torch.stack([index_1, row_j * 3 * n + col_j, index_2], dim=0),
            - hess_q_ij.flatten(),
            (B, 9 * n ** 2, e),
        )  # j, j
        hess += torch.sparse_coo_tensor(
            torch.stack([index_1, row_i * 3 * n + col_j, index_2], dim=0),
            hess_q_ij.flatten(),
            (B, 9 * n ** 2, e),
        )  # i, j
        hess += torch.sparse_coo_tensor(
            torch.stack([index_1, row_j * 3 * n + col_i, index_2], dim=0),
            hess_q_ij.flatten(),
            (B, 9 * n ** 2, e),
        )  # j, i
        return hess

    # def sparse_batch_hessian_q(self, index_tensor, atom_type, pos):
    #     """
    #     Args:
    #         index_tensor (torch.LongTensor): redefined edge tensor - (batch_idx, edge_idx, i', j'), shape: (4, E)
    #         atom_type (torch.Tensor): atom type tensor (B, n)
    #         pos (torch.Tensor): position tensor (B, n, 3)
    #     Returns:
    #         hessian (torch.sparse Tensor): hessian matrix tensor, expected size (B, e, 3n, 3n)
    #     """
    #     assert pos.dim() == 3

    #     B = pos.size(0)
    #     n = pos.size(1)
    #     e = n * (n - 1) // 2

    #     b_i, b_j = index_tensor[[0, 2]], index_tensor[[0, 3]]

    #     _b_i = (b_i * torch.LongTensor([[n], [1]]).to(b_i.device)).sum(0)  # (E, )
    #     _b_j = (b_j * torch.LongTensor([[n], [1]]).to(b_j.device)).sum(0)  # (E, )
    #     _pos = pos.reshape(-1, 3)  # (B * n, 3)
    #     _atom_type = atom_type.reshape(-1)  # (B * n, )
    #     _edge_index = torch.stack([_b_i, _b_j])  # (2, E)

    #     d_ij = self.compute_d(_edge_index, _pos)  # (E, )
    #     d_e_ij = self.compute_de(_edge_index, _atom_type)  # (E, )
    #     d_pos = (_pos[_b_i] - _pos[_b_j])  # (E, 3)
    #     hess_d_ij = d_pos.reshape(-1, 1, 3) * d_pos.reshape(-1, 3, 1) / (d_ij.reshape(-1, 1, 1) ** 3)
    #     eye = torch.eye(3).reshape(1, 3, 3).to(d_ij.device)
    #     hess_d_ij -= eye / d_ij.reshape(-1, 1, 1)  # (E, 3, 3)
    #     # hess_d_ii = - hess_d_ij, hess_d_jj = - hess_d_ij, hess_d_ji = hess_d_ij

    #     jacob_d_i = d_pos / d_ij[:, None]  # (E, 3)
    #     # jacob_d_j = jacob_d_i

    #     K1 = - (self.alpha / d_e_ij) * torch.exp(-self.alpha * (d_ij - d_e_ij) / d_e_ij) - self.beta * d_e_ij / d_ij ** 2  # (E, )
    #     K2 = (self.alpha / d_e_ij) ** 2 * torch.exp(-self.alpha * (d_ij - d_e_ij) / d_e_ij) + 2 * self.beta * d_e_ij / d_ij ** 3  # (E, )

    #     hess_q_ij = K1.reshape(-1, 1, 1) * hess_d_ij + K2.reshape(-1, 1, 1) * jacob_d_i.unsqueeze(1) * (- jacob_d_i).unsqueeze(2)  # (E, 3, 3)
    #     # hess_q_ji = hess_q_ij, hess_q_ii = - hess_q_ij, hess_q_jj = - hess_q_ij

    #     index = index_tensor[:2]  # (2, E)
    #     index = torch.repeat_interleave(index, 9, dim=1)  # (2, 9E)
    #     col_i, row_i = _stack(_repeat(index_tensor[2], 3)), _repeat(_stack(index_tensor[2]), 3)  # (3E, )
    #     col_j, row_j = _stack(_repeat(index_tensor[3], 3)), _repeat(_stack(index_tensor[3]), 3)  # (3E, )

    #     # make sparse hessian tensor
    #     hess = torch.sparse_coo_tensor(
    #         torch.cat([index, row_i.unsqueeze(0), col_i.unsqueeze(0)], dim=0),
    #         - hess_q_ij.flatten(),
    #         (B, e, 3 * n, 3 * n),
    #     )  # i, i
    #     hess += torch.sparse_coo_tensor(
    #         torch.cat([index, row_j.unsqueeze(0), col_j.unsqueeze(0)], dim=0),
    #         - hess_q_ij.flatten(),
    #         (B, e, 3 * n, 3 * n),
    #     )  # j, j
    #     hess += torch.sparse_coo_tensor(
    #         torch.cat([index, row_i.unsqueeze(0), col_j.unsqueeze(0)], dim=0),
    #         hess_q_ij.flatten(),
    #         (B, e, 3 * n, 3 * n),
    #     )  # i, j
    #     hess += torch.sparse_coo_tensor(
    #         torch.cat([index, row_j.unsqueeze(0), col_i.unsqueeze(0)], dim=0),
    #         hess_q_ij.flatten(),
    #         (B, e, 3 * n, 3 * n),
    #     )  # j, i
    #     return hess

    def jacobian_d(self, edge_index, pos):
        """
        Compute jacobian matrix of d_ij.
        Args:
            edge_index (torch.Tensor): edge index tensor (2, E)
            pos (torch.Tensor): position tensor (N, 3)
        Returns:
            jacobian (torch.Tensor, ): jacobian matrix tensor, expected size (E, 3N)
        """
        if pos.dim() == 1:
            pos = pos.reshape(-1, 3)
        jacobian = self.sparse_jacobian_d(edge_index, pos).to_dense()
        return jacobian

    def jacobian_q(self, edge_index, atom_type, pos):
        """
        Compute jacobian matrix of q_ij.
        Args:
            edge_index (torch.Tensor): edge index tensor (2, E)
            atom_type (torch.Tensor): atom type tensor (N, )
            pos (torch.Tensor): position tensor (N, 3)
        Returns:
            jacobian (torch.Tensor, ): jacobian matrix tensor, expected size (E, 3N)
        """
        if pos.dim() == 1:
            pos = pos.reshape(-1, 3)
        jacobian = self.sparse_jacobian_q(edge_index, atom_type, pos).to_dense()
        return jacobian

    def hessian_d(self, edge_index, pos):
        """
        Compute hessian matrix of d_ij.
        Args:
            edge_index (torch.Tensor): edge index tensor (2, E)
            pos (torch.Tensor): position tensor (N, 3)
        Returns:
            hessian (torch.Tensor, ): hessian matrix tensor, expected size (E, 3N, 3N)
        """
        if pos.dim() == 1:
            pos = pos.reshape(-1, 3)
        N = pos.size(0)
        E = edge_index.size(1)
        d_ij = self.compute_d(edge_index, pos)
        i, j = edge_index
        k = torch.arange(i.size(0)).to(pos.device)

        # first calculate hessian of d_ij, which is shape of (E, 3, 3)
        d_pos = pos[i] - pos[j]  # (E, 3)
        hess_d_ij = d_pos.reshape(-1, 1, 3) * d_pos.reshape(-1, 3, 1) / (d_ij.reshape(-1, 1, 1) ** 3)
        eye = torch.eye(3).reshape(1, 3, 3).to(d_ij.device)
        hess_d_ij -= eye / d_ij.reshape(-1, 1, 1)

        hess_d_ii = - hess_d_ij
        hess_d_jj = - hess_d_ij
        hess_d_ji = hess_d_ij

        # hessian of d is shape of (E, 3N, 3N)
        # Firstly, make it sparse tensor
        k = _repeat(k, 9)
        col_i = _stack(_repeat(i, 3))
        row_i = _repeat(_stack(i), 3)
        col_j = _stack(_repeat(j, 3))
        row_j = _repeat(_stack(j), 3)
        hess = torch.sparse_coo_tensor(
            torch.stack([k, row_i, col_i]), hess_d_ii.flatten(), (E, 3 * N, 3 * N)
        )
        hess += torch.sparse_coo_tensor(
            torch.stack([k, row_j, col_j]), hess_d_jj.flatten(), (E, 3 * N, 3 * N)
        )
        hess += torch.sparse_coo_tensor(
            torch.stack([k, row_i, col_j]), hess_d_ij.flatten(), (E, 3 * N, 3 * N)
        )
        hess += torch.sparse_coo_tensor(
            torch.stack([k, row_j, col_i]), hess_d_ji.flatten(), (E, 3 * N, 3 * N)
        )

        hessian = hess.to_dense()
        return hessian

    def hessian_q(self, edge_index, atom_type, pos):
        """
        Compute hessian matrix of q_ij.
        Args:
            edge_index (torch.Tensor): edge index tensor (2, E)
            atom_type (torch.Tensor): atom type tensor (N, )
            pos (torch.Tensor): position tensor (N, 3)
        Returns:
            hessian (torch.Tensor, ): hessian matrix tensor, expected size (E, 3N, 3N)
        """
        if pos.dim() == 1:
            pos = pos.reshape(-1, 3)

        hessian_d = self.hessian_d(edge_index, pos)
        jacobian_d = self.jacobian_d(edge_index, pos)
        # d^2q/dadb = d^2d/dadb * K1(d) + dd/da * dd/db * K2(d)
        # K2(d) = d^2q/dd^2
        # = (alpha / d_e_ij) ** 2 * exp(-alpha * (d_ij - d_e_ij) / d_e_ij) + 2 * beta * d_e_ij / d_ij ** 3
        # K1(d) = dq/dd
        # = -alpha * exp(-alpha * (d_ij - d_e_ij) / d_e_ij) / d_e_ij - beta * d_e_ij / d_ij ** 2

        d_ij = self.compute_d(edge_index, pos)
        d_e_ij = self.compute_de(edge_index, atom_type)
        K1 = - (self.alpha / d_e_ij) * torch.exp(-self.alpha * (d_ij - d_e_ij) / d_e_ij) - self.beta * d_e_ij / d_ij ** 2
        K2 = (self.alpha / d_e_ij) ** 2 * torch.exp(-self.alpha * (d_ij - d_e_ij) / d_e_ij) + 2 * self.beta * d_e_ij / d_ij ** 3

        hessian_q = K1.reshape(-1, 1, 1) * hessian_d + K2.reshape(-1, 1, 1) * jacobian_d.unsqueeze(1) * jacobian_d.unsqueeze(2)
        return hessian_q

    def dq2dx(self, dq, pos, edge_index, atom_type):
        # dx = J^-1 dq
        jacob = self.jacobian_q(edge_index, atom_type, pos)
        jacob_inv = torch.linalg.pinv(jacob, rtol=1e-4, atol=self.svd_tol)
        dx = jacob_inv @ dq
        return dx

    def dx2dq(self, dx, pos, edge_index, atom_type):
        # dq = J dx
        jacob = self.jacobian_q(edge_index, atom_type, pos)
        dq = jacob @ dx.flatten()
        return dq

    def advance(self, x, x_dot, edge_index, atom_type,
                q_type="morse", dt=1e-2, verbose=False
                ):
        if q_type == "morse":
            hess = self.hessian_q(edge_index, atom_type, x)  # (E, 3N, 3N)
            jacob = self.jacobian_q(edge_index, atom_type, x)  # (E, 3N)
        elif q_type == "DM":
            hess = self.hessian_d(edge_index, x)
            jacob = self.jacobian_d(edge_index, x)

        J = jacob
        JG = torch.linalg.pinv(J, rtol=1e-4, atol=self.svd_tol).T

        christoffel = torch.einsum("mij, mk->kij", hess, JG)
        x_ddot = - torch.einsum("j,kij,i->k", x_dot, christoffel, x_dot)

        new_x = x + x_dot * dt
        new_x_dot = x_dot + x_ddot * dt

        # dotproduct
        if verbose:
            print(f"\t\tdebug: x_dot size = {x_dot.norm():0.8f}, x_ddot size = {x_ddot.norm():0.8f}")
            print(f"\t\tdebug: dx norm = {(new_x - x).norm():0.8f}, dx_dot norm = {(new_x_dot - x_dot).norm():0.8f}")
        return new_x, new_x_dot

    def initialize(self, x, q_dot, edge_index, atom_type, q_type="morse"):
        if q_type == "morse":
            jacob = self.jacobian_q(edge_index, atom_type, x)
        elif q_type == "DM":
            jacob = self.jacobian_d(edge_index, x)

        J = jacob
        J_inv = torch.linalg.pinv(J, rtol=1e-4, atol=self.svd_tol)
        x_dot = J_inv @ q_dot
        q_dot = J @ x_dot

        x = x.flatten()
        n = x.size(0) // 3

        total_time = x_dot.norm()
        x_dot = x_dot / x_dot.norm()

        if q_type == "morse":
            q = self.compute_q(edge_index, atom_type, x.reshape(-1, 3))
        elif q_type == "DM":
            q = self.compute_d(edge_index, x.reshape(-1, 3))
        else:
            raise NotImplementedError

        return x, x_dot, total_time, q, q_dot

    def batch_initialize(self, x, q_dot, edge_index, atom_type, batch, num_nodes, q_type="morse"):
        """
        Args:
            x (torch.Tensor): initial position tensor (B, n, 3)
            q_dot (torch.Tensor): initial velocity tensor (B, e)
            index_tensor (torch.Tensor): redefined edge tensor - (batch_idx, edge_idx, i', j'), shape: (4, E)
            atom_type (torch.Tensor): atom type tensor (B, n)

        """
        assert q_type == "morse"

        B = batch.max() + 1
        n = num_nodes.max()
        e = n * (n - 1) // 2

        index_tensor = redefine_edge_index(edge_index, batch, num_nodes)  # (4, E)
        x = redefine_with_pad(x, batch)  # (B, n, 3)
        atom_type = redefine_with_pad(atom_type, batch, padding_value=-1)  # (B, n)
        q_dot = torch.sparse_coo_tensor(
            index_tensor[:2],
            q_dot.flatten(),
            (B, e),
        ).to_dense()  # (B, e)

        J = self.sparse_batch_jacobian_q(index_tensor, atom_type, x).to_dense()  # (B, e, 3n)
        J_inv = vmap(torch.linalg.pinv)(J, rtol=1e-4, atol=self.svd_tol)  # (B, 3n, e)

        x_dot = torch.bmm(J_inv, q_dot.unsqueeze(-1)).reshape(B, n, 3)  # (B, n, 3)
        q_dot = torch.bmm(J, x_dot.reshape(B, -1, 1)).squeeze(-1)  # (B, e)

        total_time = torch.sqrt(x_dot.pow(2).sum(dim=(1, 2)) / num_nodes)  # (B, )
        total_time = x_dot.reshape(B, -1).norm(dim=-1)
        x_dot = x_dot / total_time.reshape(-1, 1, 1)  # (B, n, 3)

        if q_type == "morse":
            q = self.batch_compute_q(index_tensor, atom_type, x)
        elif q_type == "DM":
            q = self.batch_compute_d(index_tensor, x)
        else:
            raise NotImplementedError

        return x, x_dot, total_time, q, q_dot, atom_type, index_tensor

    def batch_compute_q(self, index_tensor, atom_type, x):
        B = index_tensor[0].max() + 1
        n = index_tensor[2:].max() + 1
        e = n * (n - 1) // 2

        x = x.reshape(-1, 3)  # (B * n, 3)
        atom_type = atom_type.reshape(-1)  # (B * n, )
        i, j = index_tensor[2:]
        batch_index = index_tensor[0]
        i, j = i + batch_index * n, j + batch_index * n

        edge_index = torch.stack([i, j])  # (2, E)
        q_value = self.compute_q(edge_index, atom_type, x)

        q = torch.sparse_coo_tensor(
            index_tensor[:2],
            q_value,
            (B, e),
        ).to_dense()
        return q

    def batch_compute_d(self, index_tensor, x):
        B = index_tensor[0].max() + 1
        n = index_tensor[2:].max() + 1
        e = n * (n - 1) // 2

        x = x.reshape(-1, 3)  # (B * n, 3)
        i, j = index_tensor[2:]
        batch_index = index_tensor[0]
        i, j = i + batch_index * n, j + batch_index * n

        edge_index = torch.stack([i, j])  # (2, E)
        d_value = self.compute_d(edge_index, x)

        d = torch.sparse_coo_tensor(
            index_tensor[:2],
            d_value,
            (B, e),
        ).to_dense()
        return d

    def batch_compute_de(self, index_tensor, atom_type):
        n = index_tensor[2:].max() + 1

        atom_type = atom_type.reshape(-1)
        i, j = index_tensor[2:]
        batch_index = index_tensor[0]
        i, j = i + batch_index * n, j + batch_index * n

        edge_index = torch.stack([i, j])  # (2, E)
        return self.compute_de(edge_index, atom_type)

    def batch_geodesic_ode_solve(self, x, q_dot, edge_index, atom_type, batch, num_nodes, q_type="morse",
                                 num_iter=100, ref_dt=1e-2, max_dt=1e-1, verbose=0, max_iter=1000, err_thresh=5,
                                 method="Euler"):
        """
        Args:
            x: torch.Tensor, initial position tensor (N, 3)
            q_dot: torch.Tensor, initial velocity tensor (E, )
            edge_index: torch.Tensor, edge index tensor (2, E)
            atom_type: torch.Tensor, atom type tensor (N, )
            batch: torch.Tensor, batch tensor (N, )
            num_nodes: torch.Tensor, number of nodes tensor (B, )
        """
        x, x_dot, total_time, q, q_dot, atom_type, index_tensor = self.batch_initialize(
            x,
            q_dot,
            edge_index,
            atom_type,
            batch,
            num_nodes,
            q_type=q_type
        )
        init = {"x": x, "x_dot": x_dot / total_time.reshape(-1, 1, 1), "q": q, "q_dot": q_dot}
        # x: (B, n, 3), x_dot: (B, n, 3), total_time: (B, ), q: (B, e), q_dot: (B, e), index_tensor: (4, E)
        init_q_dot_norm = q_dot.norm(dim=1) / total_time  # (B, )
        B = x.size(0)

        ref_dt = torch.where(total_time / num_iter < ref_dt, total_time / num_iter, ref_dt)  # (B, )
        dt = ref_dt

        current_time = torch.zeros_like(total_time)  # (B, )
        iter = 0
        done = total_time <= current_time + 1e-6  # (B, )

        while (~done).any() and iter < max_iter:
            x_new, x_dot_new, q_dot = self.batch_advance(
                x, x_dot,
                index_tensor,
                atom_type,
                done,
                q_type=q_type,
                dt=dt,
                verbose=verbose >= 3,
                return_qdot=True,
                method=method
            )

            var_dt = 1 / x_dot_new.reshape(B, -1).norm(dim=-1) * ref_dt
            cond = ref_dt > var_dt
            var_dt = torch.where(cond, ref_dt, var_dt)
            cond = max_dt > var_dt
            dt_new = torch.where(cond, var_dt, max_dt)

            cond = total_time - current_time < dt_new
            dt_new = torch.where(cond, total_time - current_time, dt_new)

            q_dot_norm = q_dot.norm(dim=1)  # (B, )
            err = (q_dot_norm - init_q_dot_norm[~done]).abs() / init_q_dot_norm[~done] * 100
            if (err > err_thresh).any():
                restart_mask = err > err_thresh
                x_new[~done][restart_mask] = x[~done][restart_mask]
                x_dot_new[~done][restart_mask] = x_dot[~done][restart_mask]
                dt_new[~done][restart_mask] = dt[~done][restart_mask] * 0.5
                ref_dt[~done][restart_mask] = ref_dt[~done][restart_mask] * 0.5
                if verbose >= 0:
                    print("[Warning] veolocity error is too large, restart with smaller time step.")
                    print(f"\t\terr = {err}")

            dt = dt_new
            x, x_dot = x_new, x_dot_new

            current_time += dt
            iter += 1
            done = total_time <= current_time + 1e-6  # (B, )

            if verbose >= 1:
                print(f"iter = {iter}, \n\tcurrent_time = {current_time}, \n\ttotal_time = {total_time}, \n\tdt = {dt}, \n\tdone = {done}")

        if verbose >= 1:
            print(f"iter = {iter}, \n\tcurrent_time = {current_time}, \n\ttotal_time = {total_time}, \n\tdt = {dt}, \n\tdone = {done}")

        if q_type == "morse":
            q_last = self.batch_compute_q(index_tensor, atom_type, x)
        elif q_type == "DM":
            q_last = self.batch_compute_d(index_tensor, x)
        J = self.sparse_batch_jacobian_q(index_tensor, atom_type, x)
        x_dot *= total_time.reshape(-1, 1, 1)
        q_dot = torch.bmm(J, x_dot.reshape(B, -1, 1)).squeeze(-1)  # (B, e)
        last = {"x": x, "x_dot": x_dot, "q": q_last, "q_dot": q_dot}

        return init, last, iter, index_tensor

    def batch_advance(self, x, x_dot, index_tensor, atom_type, done, dt, q_type="morse", verbose=False, return_qdot=False, method="Euler"):
        if method == "Euler":
            return self.batch_advance_euler(x, x_dot, index_tensor, atom_type, done, dt, q_type=q_type, verbose=verbose, return_qdot=return_qdot)

        elif method == "Heun":
            return self.batch_advance_heun(x, x_dot, index_tensor, atom_type, done, dt, q_type=q_type, verbose=verbose, return_qdot=return_qdot)

        else:
            raise NotImplementedError

    def batch_advance_euler(self, x, x_dot, index_tensor, atom_type, done, dt, q_type="morse", verbose=False, return_qdot=False):
        """
        Args:
            x: torch.Tensor, initial position tensor (B, n, 3)
            x_dot: torch.Tensor, initial velocity tensor (B, n, 3)
            index_tensor: torch.Tensor, redefined edge index tensor (4, E)
            atom_type: torch.Tensor, atom type tensor (B, n)
            done: torch.BoolTensor, already finishied flag tensor (B, )
            dt: torch.Tensor, time step tensor (B, )
        """

        dt = dt[~done]
        if q_type == "morse":
            J = self.sparse_batch_jacobian_q(index_tensor, atom_type, x)  # (B, e, 3n)
        elif q_type == "DM":
            J = self.sparse_batch_jacobian_d(index_tensor, x)

        B = (~done).sum()
        not_done_index = torch.where(~done)[0]  # sparse
        J = J.index_select(0, not_done_index)  # sparse (B, e, 3n)

        if return_qdot:
            q_dot = torch.bmm(J, x_dot[~done].reshape(B, -1, 1)).squeeze(-1)
        else:
            q_dot = None

        christoffel = self.batch_christoffel(index_tensor, atom_type, x, J, not_done_index)  # (B, n, n, n)
        x_ddot = - torch.einsum("bj,bkij,bi->bk", x_dot[~done].reshape(B, -1), christoffel, x_dot[~done].reshape(B, -1))
        x_ddot = x_ddot.reshape(B, -1, 3)

        # dt must be non-negative
        cond = dt > 0
        _dt = torch.where(cond, dt, torch.zeros_like(dt))
        update_x = x_dot[~done] * _dt.reshape(-1, 1, 1)
        update_x_dot = x_ddot * _dt.reshape(-1, 1, 1)
        x[~done] += update_x
        x_dot[~done] += update_x_dot

        if verbose:
            print(f"\t\tdebug: dx = {update_x.reshape(B, -1).norm(dim=-1)}, dx_dot norm = {update_x_dot.reshape(B, -1).norm(dim=-1)}")

        return x, x_dot, q_dot

    def batch_advance_heun(self, x, x_dot, index_tensor, atom_type, done, dt, q_type="morse", verbose=False, return_qdot=False):
        """
        Args:
            x: torch.Tensor, initial position tensor (B, n, 3)
            x_dot: torch.Tensor, initial velocity tensor (B, n, 3)
            index_tensor: torch.Tensor, redefined edge index tensor (4, E)
            atom_type: torch.Tensor, atom type tensor (B, n)
            done: torch.BoolTensor, already finishied flag tensor (B, )
            dt: torch.Tensor, time step tensor (B, )
        """
        dt = dt[~done]
        if q_type == "morse":
            J = self.sparse_batch_jacobian_q(index_tensor, atom_type, x)
        elif q_type == "DM":
            J = self.sparse_batch_jacobian_d(index_tensor, x)

        B = (~done).sum()
        not_done_index = torch.where(~done)[0]
        J = J.index_select(0, not_done_index)  # sparse (B, e, 3n)

        if return_qdot:
            q_dot = torch.bmm(J, x_dot[~done].reshape(B, -1, 1)).squeeze(-1)
        else:
            q_dot = None

        christoffel = self.batch_christoffel(index_tensor, atom_type, x, J, not_done_index)  # (B, n, n, n)
        x_ddot1 = - torch.einsum("bj,bkij,bi->bk", x_dot[~done].reshape(B, -1), christoffel, x_dot[~done].reshape(B, -1))
        x_ddot1 = x_ddot1.reshape(B, -1, 3)

        # dt must be non-negative
        cond = dt > 0
        _dt = torch.where(cond, dt, torch.zeros_like(dt))
        new_x, new_x_dot = x.clone(), x_dot.clone()
        new_x[~done] += x_dot[~done] * _dt.reshape(-1, 1, 1)
        new_x_dot[~done] += x_ddot1 * _dt.reshape(-1, 1, 1)

        if q_type == "morse":
            J = self.sparse_batch_jacobian_q(index_tensor, atom_type, new_x)
        elif q_type == "DM":
            J = self.sparse_batch_jacobian_d(index_tensor, new_x)
        J = J.index_select(0, not_done_index)  # sparse (B, e, 3n)

        christoffel = self.batch_christoffel(index_tensor, atom_type, new_x, J, not_done_index)  # (B, n, n, n)
        x_ddot2 = - torch.einsum("bj,bkij,bi->bk", new_x_dot[~done].reshape(B, -1), christoffel, new_x_dot[~done].reshape(B, -1))
        x_ddot2 = x_ddot2.reshape(B, -1, 3)

        update_x = (x_dot[~done] + new_x_dot[~done]) * _dt.reshape(-1, 1, 1) / 2
        update_x_dot = (x_ddot1 + x_ddot2) * _dt.reshape(-1, 1, 1) / 2
        x[~done] += update_x
        x_dot[~done] += update_x_dot

        if verbose:
            print(f"\t\tdebug: dx = {update_x.reshape(B, -1).norm(dim=-1)}, dx_dot norm = {update_x_dot.reshape(B, -1).norm(dim=-1)}")

        return x, x_dot, q_dot

    def batch_christoffel(self, index_tensor, atom_type, x, J, not_done_index):
        B, e, n3 = J.shape
        # TODO : re-implement the hessian function so that the output shape is (B, e, 3n * 3n)
        hess = self.sparse_batch_hessian_q(index_tensor, atom_type, x)  # sparse (B, 3n * 3n, e)
        hess = hess.index_select(0, not_done_index)  # sparse (B, 3n * 3n, e)
        J_inv = vmap(torch.linalg.pinv)(J.to_dense(), rtol=1e-4, atol=self.svd_tol).transpose(-1, -2)  # dense (B, e, 3n)
        christoffel = torch.bmm(hess, J_inv).transpose(-1, -2).reshape(B, n3, n3, n3)
        # Gamma^k_ij = christoffel[:, k, i, j]
        return christoffel

    def geodesic_ode_solve(self, x, q_dot, edge_index, atom_type, q_type="morse",
                           num_iter=100, check_dot_every=10,
                           ref_dt=1e-2, max_dt=1e-1, verbose=0):
        """
        Args:
            x: torch.Tensor, initial position tensor (N, 3)
            q_dot: torch.Tensor, initial velocity tensor (E, )
            edge_index: torch.Tensor, edge index tensor (2, E)
            atom_type: torch.Tensor, atom type tensor (N, )
        """

        x, x_dot, total_time, q, q_dot = self.initialize(x, q_dot, edge_index, atom_type, q_type=q_type)

        ref_dt = min(total_time / num_iter, ref_dt)
        dt = ref_dt
        if verbose >= 1:
            print(f"initial dt = {ref_dt:0.6f}, total_expected_iter = {total_time / ref_dt:1.0f}")

        current_time = 0
        iter = 0
        total_dq = 0
        # debug, check all variables' shape
        while total_time > current_time:
            x_new, x_dot_new = self.advance(x, x_dot, edge_index, atom_type, q_type=q_type, dt=dt, verbose=verbose >= 3)
            current_time += dt

            # calculate dq
            if q_type == "morse":
                q_new = self.compute_q(edge_index, atom_type, x_new.reshape(-1, 3))
            elif q_type == "DM":
                q_new = self.compute_d(edge_index, x_new.reshape(-1, 3))

            dq = (q_new - q).norm()
            total_dq += dq

            q = q_new
            iter += 1

            x = x_new
            x_dot = x_dot_new
            dt = min(max(ref_dt, 1 / x_dot.norm() * ref_dt), max_dt)

            if total_time - current_time < dt:
                dt = total_time - current_time

        return x.reshape(-1, 3), x_dot.reshape(-1, 3) * total_time, iter, total_dq, q_dot.norm()


if __name__ == "__main__":
    import omegaconf
    config = omegaconf.OmegaConf.load("../configs/config.yaml")
