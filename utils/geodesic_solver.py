import torch
from torch_geometric.utils import unbatch, unbatch_edge_index
from utils.chem import ATOMIC_RADII_LIST
from torch.nn.utils.rnn import pad_sequence
from torch import vmap


from time import time


# NOTE: For debugging, will be deprecated
def timer(func):
    """Check time.

    Usage)
    >>> @timer
    >>> def method1(...):
    >>>     ...
    >>>     return
    """

    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"Elapsed time[{func.__name__}]: {end - start} sec", flush=True)
        return result

    return wrapper


def get_pad_mask(num_nodes):
    N = num_nodes.max()
    mask = torch.BoolTensor([True, False]).repeat(len(num_nodes)).to(num_nodes.device)
    num_repeats = torch.stack([num_nodes, N - num_nodes]).T.flatten()
    mask = mask.repeat_interleave(num_repeats)
    return mask


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
    n = num_nodes.max()
    i, j = edge_index
    edge_index = (j - i - 1) + n * i - i * (i + 1) // 2

    index_tensor = torch.cat([
        batch_index.unsqueeze(0),
        edge_index.unsqueeze(0),
        torch.cat(edge_bundle, dim=-1)],
        dim=0)  # (4, E)

    return index_tensor


def sequence_length_mask(num_nodes, max_N=None):
    if max_N is None:
        max_N = torch.max(num_nodes)
    mask = torch.arange(max_N).to(num_nodes.device).unsqueeze(0) < num_nodes.unsqueeze(1)
    return mask


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
        self.gamma = config.ode_solver.get("gamma", 0.0)
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
        q_ij = torch.exp(- self.alpha * (d_ij - d_e_ij) / d_e_ij) + self.beta * (d_e_ij / d_ij) + self.gamma * (d_ij / d_e_ij)
        return q_ij

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
        # hess_d_ii = - hess_d_ij, hess_d_jj = - hess_d_ij, hess_d_ji = hess_d_ij

        # hessian of d is shape of (E, 3N, 3N), make it a sparse tensor
        k = _repeat(k, 9)
        col_i, row_i = _stack(_repeat(i, 3)), _repeat(_stack(i), 3)
        col_j, row_j = _stack(_repeat(j, 3)), _repeat(_stack(j), 3)
        row = torch.cat([row_i, row_j, row_i, row_j])
        col = torch.cat([col_i, col_j, col_j, col_i])
        edge = k.repeat(4)
        index = torch.stack([edge, row, col])
        val = torch.cat([-hess_d_ij, -hess_d_ij, hess_d_ij, hess_d_ij], dim=0).flatten()
        hess = torch.sparse_coo_tensor(index, val, (E, 3 * N, 3 * N))
        hess = hess.to_dense()
        return hess

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
        K1 = - (self.alpha / d_e_ij) * torch.exp(-self.alpha * (d_ij - d_e_ij) / d_e_ij) - self.beta * d_e_ij / d_ij ** 2 + self.gamma / d_e_ij
        K2 = (self.alpha / d_e_ij) ** 2 * torch.exp(-self.alpha * (d_ij - d_e_ij) / d_e_ij) + 2 * self.beta * d_e_ij / d_ij ** 3

        hessian_q = K1.reshape(-1, 1, 1) * hessian_d + K2.reshape(-1, 1, 1) * jacobian_d.unsqueeze(1) * jacobian_d.unsqueeze(2)
        return hessian_q

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

        k = _repeat(k, 3)
        i = _stack(i)
        j = _stack(j)

        index = torch.stack([torch.cat([k, k,], dim=0), torch.cat([i, j], dim=0)])
        val = torch.cat([dd_dx, -dd_dx], dim=0)
        jacobian = torch.sparse_coo_tensor(index, val, (E, 3 * N))
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
        dq_dd = - self.alpha / d_e_ij * torch.exp(- self.alpha / d_e_ij * (d_ij - d_e_ij)) - self.beta * d_e_ij / d_ij ** 2 + self.gamma / d_e_ij

        dq_dx = dq_dd.unsqueeze(-1) * dd_dx  # (E, 3)

        k = _repeat(k, 3)
        i = _stack(i)
        j = _stack(j)

        index = torch.stack([torch.cat([k, k,], dim=0), torch.cat([i, j], dim=0)])
        val = torch.cat([dq_dx, -dq_dx], dim=0).flatten()
        jacobian = torch.sparse_coo_tensor(index, val, (E, 3 * N))
        return jacobian

    def sparse_batch_jacobian_d(self, index_tensor, pos, **kwargs):
        assert pos.dim() == 3
        B, n = pos.size(0), pos.size(1)
        e = n * (n - 1) // 2

        _b = index_tensor[0]
        _i = index_tensor[2] + n * _b
        _j = index_tensor[3] + n * _b
        _pos = pos.reshape(-1, 3)  # (B * n, 3)
        _edge_index = torch.stack([_i, _j])

        d_ij = self.compute_d(_edge_index, _pos)  # (E, )

        dd_dx = (_pos[_i] - _pos[_j]) / d_ij[:, None]  # (E, 3)
        index = index_tensor[:2]  # (2, E)
        index = torch.repeat_interleave(index, 3, dim=1)  # (2, 3E)

        _i = _stack(index_tensor[2])  # (3E, )
        _j = _stack(index_tensor[3])  # (3E, )

        index = torch.cat([index.repeat(1, 2), torch.cat([_i, _j]).unsqueeze(0)], dim=0)
        val = torch.cat([dd_dx, -dd_dx], dim=0).flatten()

        jacobian = torch.sparse_coo_tensor(index, val, (B, e, 3 * n))
        return jacobian

    def sparse_batch_jacobian_q(self, index_tensor, pos, atom_type=None):
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

        _b = index_tensor[0]
        _i = index_tensor[2] + n * _b
        _j = index_tensor[3] + n * _b
        _pos = pos.reshape(-1, 3)  # (B * n, 3)
        _atom_type = atom_type.reshape(-1)  # (B * n, )
        _edge_index = torch.stack([_i, _j])  # (2, E)

        d_ij = self.compute_d(_edge_index, _pos)  # (E, )
        d_e_ij = self.compute_de(_edge_index, _atom_type)  # (E, )

        dd_dx = (_pos[_i] - _pos[_j]) / d_ij[:, None]  # (E, 3)
        dq_dd = - self.alpha / d_e_ij * torch.exp(- self.alpha / d_e_ij * (d_ij - d_e_ij)) - self.beta * d_e_ij / d_ij ** 2 + self.gamma / d_e_ij  # (E, )

        dq_dx = dq_dd.unsqueeze(-1) * dd_dx  # (E, 3)

        index = index_tensor[:2]  # (2, E)
        index = torch.repeat_interleave(index, 3, dim=1)  # (2, 3E)

        _i = _stack(index_tensor[2])  # (3E, )
        _j = _stack(index_tensor[3])  # (3E, )

        index = torch.cat([index.repeat(1, 2), torch.cat([_i, _j]).unsqueeze(0)], dim=0)
        val = torch.cat([dq_dx, -dq_dx], dim=0).flatten()

        jacobian = torch.sparse_coo_tensor(index, val, (B, e, 3 * n))
        return jacobian

    def sparse_batch_hessian_d(self, index_tensor, pos, **kwargs):
        B, n = pos.size(0), pos.size(1)
        e = n * (n - 1) // 2
        _b = index_tensor[0]
        _i = index_tensor[2] + n * _b
        _j = index_tensor[3] + n * _b
        _pos = pos.reshape(-1, 3)
        _edge_index = torch.stack([_i, _j])

        d_ij = self.compute_d(_edge_index, _pos)
        d_pos = (_pos[_i] - _pos[_j])
        hess_d_ij = d_pos.reshape(-1, 1, 3) * d_pos.reshape(-1, 3, 1) / (d_ij.reshape(-1, 1, 1) ** 3)
        eye = torch.eye(3).reshape(1, 3, 3).to(d_ij.device)
        hess_d_ij -= eye / d_ij.reshape(-1, 1, 1)

        b_e_index = index_tensor[:2]
        b_e_index = torch.repeat_interleave(b_e_index, 9, dim=1)
        index_1, index_2 = b_e_index
        col_i, row_i = _stack(_repeat(index_tensor[2], 3)), _repeat(_stack(index_tensor[2]), 3)
        col_j, row_j = _stack(_repeat(index_tensor[3], 3)), _repeat(_stack(index_tensor[3]), 3)

        index = torch.stack([
            index_1.repeat(4),
            torch.cat([row_i, row_j, row_i, row_j]) * 3 * n + torch.cat([col_i, col_j, col_j, col_i]),
            index_2.repeat(4)
        ])
        val = torch.cat([-hess_d_ij, -hess_d_ij, hess_d_ij, hess_d_ij], dim=0).flatten()
        hess = torch.sparse_coo_tensor(index, val, (B, 9 * n ** 2, e))
        return hess

    def sparse_batch_hessian_q(self, index_tensor, pos, atom_type=None):
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
        _b = index_tensor[0]
        _i = index_tensor[2] + n * _b
        _j = index_tensor[3] + n * _b
        _pos = pos.reshape(-1, 3)  # (B * n, 3)
        _pos = _pos.contiguous()
        _atom_type = atom_type.reshape(-1)  # (B * n, )
        _atom_type = _atom_type.contiguous()
        _edge_index = torch.stack([_i, _j]).contiguous()  # (2, E)

        torch.cuda.empty_cache()
        d_ij = self.compute_d(_edge_index, _pos)  # (E, )
        d_e_ij = self.compute_de(_edge_index, _atom_type)  # (E, )
        d_pos = (_pos[_i] - _pos[_j])  # (E, 3)
        hess_d_ij = d_pos.reshape(-1, 1, 3) * d_pos.reshape(-1, 3, 1) / (d_ij.reshape(-1, 1, 1) ** 3)
        eye = torch.eye(3).reshape(1, 3, 3).to(d_ij.device)
        hess_d_ij -= eye / d_ij.reshape(-1, 1, 1)  # (E, 3, 3)
        # hess_d_ii = - hess_d_ij, hess_d_jj = - hess_d_ij, hess_d_ji = hess_d_ij

        jacob_d_i = d_pos / d_ij[:, None]  # (E, 3)
        # jacob_d_j = jacob_d_i

        K1 = - (self.alpha / d_e_ij) * torch.exp(-self.alpha * (d_ij - d_e_ij) / d_e_ij) - self.beta * d_e_ij / d_ij ** 2 + self.gamma / d_e_ij  # (E, )
        K2 = (self.alpha / d_e_ij) ** 2 * torch.exp(-self.alpha * (d_ij - d_e_ij) / d_e_ij) + 2 * self.beta * d_e_ij / d_ij ** 3  # (E, )

        hess_q_ij = K1.reshape(-1, 1, 1) * hess_d_ij + K2.reshape(-1, 1, 1) * jacob_d_i.unsqueeze(1) * (- jacob_d_i).unsqueeze(2)  # (E, 3, 3)
        # hess_q_ji = hess_q_ij, hess_q_ii = - hess_q_ij, hess_q_jj = - hess_q_ij

        b_e_index = index_tensor[:2]  # (2, E)
        b_e_index = torch.repeat_interleave(b_e_index, 9, dim=1)  # (2, 9E)
        index_1, index_2 = b_e_index
        col_i, row_i = _stack(_repeat(index_tensor[2], 3)), _repeat(_stack(index_tensor[2]), 3)  # (3E, )
        col_j, row_j = _stack(_repeat(index_tensor[3], 3)), _repeat(_stack(index_tensor[3]), 3)  # (3E, )

        index = torch.stack([
            index_1.repeat(4),
            torch.cat([row_i, row_j, row_i, row_j]) * 3 * n + torch.cat([col_i, col_j, col_j, col_i]),
            index_2.repeat(4)
        ])
        val = torch.cat([-hess_q_ij, -hess_q_ij, hess_q_ij, hess_q_ij], dim=0).flatten()
        hess = torch.sparse_coo_tensor(index, val, (B, 9 * n ** 2, e))
        return hess

    def dq2dx(self, dq, pos, edge_index, atom_type, q_type="morse"):
        # dx = J^-1 dq
        if q_type == "morse":
            jacob = self.jacobian_q(edge_index, atom_type, pos)
        elif q_type == "DM":
            jacob = self.jacobian_d(edge_index, pos)
        else:
            raise NotImplementedError
        jacob_inv = torch.linalg.pinv(jacob, rtol=1e-4, atol=self.svd_tol)
        dx = jacob_inv @ dq
        return dx

    def dx2dq(self, dx, pos, edge_index, atom_type, q_type="morse"):
        # dq = J dx
        if q_type == "morse":
            jacob = self.jacobian_q(edge_index, atom_type, pos)
        elif q_type == "DM":
            jacob = self.jacobian_d(edge_index, pos)
        else:
            raise NotImplementedError
        dq = jacob @ dx.flatten()
        return dq

    def dq_dd(self, pos, atom_type, edge_index, q_type="morse"):
        i, j = edge_index
        d_ij = self.compute_d(edge_index, pos)
        d_e_ij = self.compute_de(edge_index, atom_type)
        dq_dd = - self.alpha / d_e_ij * torch.exp(- self.alpha / d_e_ij * (d_ij - d_e_ij)) - self.beta * d_e_ij / d_ij ** 2 + self.gamma / d_e_ij
        return dq_dd

    def compute_d_or_q(self, pos, atom_type, edge_index, q_type="morse"):
        if q_type == "morse":
            q = self.compute_q(edge_index, atom_type, pos)
        elif q_type == "DM":
            q = self.compute_d(edge_index, pos)
        else:
            raise NotImplementedError
        return q

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
        q = torch.sparse_coo_tensor(index_tensor[:2], q_value, (B, e),).to_dense()
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
        d = torch.sparse_coo_tensor(index_tensor[:2], d_value, (B, e),).to_dense()
        return d

    def batch_compute_de(self, index_tensor, atom_type):
        n = index_tensor[2:].max() + 1

        atom_type = atom_type.reshape(-1)
        i, j = index_tensor[2:]
        batch_index = index_tensor[0]
        i, j = i + batch_index * n, j + batch_index * n

        edge_index = torch.stack([i, j])  # (2, E)
        de = self.compute_de(edge_index, atom_type)
        return de

    def batch_geodesic_ode_solve(self, x, q_dot, edge_index, atom_type, batch, num_nodes, q_type="morse",
                                 num_iter=10, ref_dt=1e-2, max_dt=1e-1, min_dt=1e-3, verbose=0, max_iter=1000, err_thresh=0.05,
                                 pos_adjust_scaler=0.05, pos_adjust_thresh=1e-3, method="Euler"):
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
            q_type=q_type,
            pos_adjust_scaler=pos_adjust_scaler,
            pos_adjust_thresh=pos_adjust_thresh,
            verbose=verbose,
        )
        init = {
            "x": x.clone(),
            "x_dot": x_dot.clone() * total_time.reshape(-1, 1, 1),
            "q": q.clone(),
            "q_dot": q_dot.clone() * total_time.reshape(-1, 1),
        }
        # x: (B, n, 3), x_dot: (B, n, 3), total_time: (B, ), q: (B, e), q_dot: (B, e), index_tensor: (4, E)
        init_q_dot_norm = q_dot.norm(dim=1)  # (B, )
        B = x.size(0)

        _ = total_time / num_iter
        ref_dt = _.clip(max=ref_dt)
        dt = ref_dt

        current_time = torch.zeros_like(total_time)  # (B, )
        iter = torch.zeros_like(total_time, dtype=torch.long)  # (B, )
        done = total_time <= (current_time + 1e-6)  # (B, )
        ban_index = torch.LongTensor([]).to(device=x.device)

        cnt = 0
        while (~done).any() and (iter[~done] < max_iter).any():

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
            dt_new = (ref_dt.clip(min=var_dt)).clip(min=min_dt, max=max_dt)

            q_dot_norm = q_dot.norm(dim=1)  # (B, )
            err = (q_dot_norm - init_q_dot_norm[~done]).abs() / init_q_dot_norm[~done]
            not_done_index = torch.where(~done)[0]

            if (err > err_thresh).any():
                restart_index = not_done_index[err > err_thresh]
                x_new[restart_index] = init["x"][restart_index]
                x_dot_new[restart_index] = init["x_dot"][restart_index]
                dt_new[restart_index] = dt[restart_index] * 0.5
                ref_dt[restart_index] = ref_dt[restart_index] * 0.5
                current_time[restart_index] = 0
                iter[restart_index] = 0

                new_ban = torch.where(ref_dt < min_dt)[0]
                _ = torch.isin(new_ban, not_done_index[err > err_thresh])
                new_ban = new_ban[_]
                if new_ban.numel() > 0:
                    ban_index = torch.cat([ban_index, new_ban])
                    ban_index = ban_index.unique()
                    done[ban_index] = True
                    ref_dt[ban_index] = min_dt

                    if verbose >= 1:
                        print(f"[Warning] Some samples ({new_ban}) are banned due to numerical unstability.")
                        print("[Warning] veolocity error is too large, restart with smaller time step.")
                        print(f"err = {err[err > err_thresh]}")

                update_index = not_done_index[err <= err_thresh]
                current_time[update_index] += dt[update_index]
                iter[update_index] += 1
            else:
                current_time += dt
                iter += 1

            cnt += 1
            x, x_dot = x_new, x_dot_new
            remain_time = (total_time - current_time).relu()
            dt_new = torch.stack([dt_new, remain_time]).min(dim=0).values
            dt = dt_new

            done[~done] = remain_time[~done] < 1e-6

            if verbose >= 1:
                ban_mask = torch.zeros_like(done).bool()
                ban_mask[ban_index] = True
                print(f"iter = {iter}, \n\tcurrent_time = \n\t{current_time}, \n\ttotal_time = \n\t{total_time}, \n\tdt = \n\t{dt}, \n\tref_dt ={ref_dt}, \n\tdone = \n\t{done}, \n\tban_mask = {ban_mask}\n\n")
            if cnt >= 2 * max_iter:
                break

        if q_type == "morse":
            q_last = self.batch_compute_q(index_tensor, atom_type, x)
            J = self.sparse_batch_jacobian_q(index_tensor, x, atom_type=atom_type).to_dense()
        elif q_type == "DM":
            q_last = self.batch_compute_d(index_tensor, x)
            J = self.sparse_batch_jacobian_d(index_tensor, x, atom_type=atom_type).to_dense()

        x_dot *= total_time.reshape(-1, 1, 1)
        q_dot = torch.bmm(J, x_dot.reshape(B, -1, 1)).squeeze(-1)  # (B, e)
        last = {"x": x, "x_dot": x_dot, "q": q_last, "q_dot": q_dot}
        stats = {"iter": iter, "current_time": current_time, "total_time": total_time, "ban_index": ban_index}

        return init, last, iter, index_tensor, stats

    def batch_projection(self, query_vec, pos, atom_type, edge_index, batch, num_nodes, q_type="morse", proj_type="euclidean"):
        if q_type == "morse":
            calc_jacob = self.sparse_batch_jacobian_q
        elif q_type == "DM":
            calc_jacob = self.sparse_batch_jacobian_d
        else:
            raise NotImplementedError

        assert proj_type in ["euclidean", "manifold"]

        B = batch.max() + 1
        n = num_nodes.max()
        e = n * (n - 1) // 2

        index_tensor = redefine_edge_index(edge_index, batch, num_nodes)  # (4, E)
        atom_type = redefine_with_pad(atom_type, batch, padding_value=-1)  # (B, n)
        pos = redefine_with_pad(pos, batch)  # (B, n, 3)

        J = calc_jacob(index_tensor, pos, atom_type=atom_type).to_dense()  # (B, e, 3n)
        J_inv = batch_pinv1(J, rtol=1e-4, atol=self.svd_tol)  # (B, 3n, e)

        if proj_type == "euclidean":
            x = redefine_with_pad(query_vec, batch)  # (B, n, 3)
            proj_matrix = torch.bmm(J_inv, J)  # (B, 3n, 3n)
            proj = torch.bmm(proj_matrix, x.reshape(B, -1, 1)).reshape(-1, 3)
            pad_mask = get_pad_mask(num_nodes)
            proj = proj[pad_mask]

        elif proj_type == "manifold":
            x = torch.sparse_coo_tensor(index_tensor[:2], query_vec.flatten(), (B, e),).to_dense()
            proj_matrix = torch.bmm(J, J_inv)  # (B, e, e)
            proj = torch.bmm(proj_matrix, x.reshape(B, -1, 1)).squeeze(-1)  # (B, e)
            pad_mask = index_tensor[1] + index_tensor[0] * e
            proj = proj.flatten()[pad_mask]

        return proj

    def batch_dx2dq(self, dx, pos, atom_type, edge_index, batch, num_nodes, q_type="morse"):
        if q_type == "morse":
            calc_jacob = self.sparse_batch_jacobian_q
        elif q_type == "DM":
            calc_jacob = self.sparse_batch_jacobian_d
        else:
            raise NotImplementedError

        B = num_nodes.size(0)
        n = num_nodes.max()
        e = n * (n - 1) // 2

        index_tensor = redefine_edge_index(edge_index, batch, num_nodes)  # (4, E)
        atom_type = redefine_with_pad(atom_type, batch, padding_value=-1)  # (B, n)
        pos = redefine_with_pad(pos, batch)  # (B, n, 3)
        dx = redefine_with_pad(dx, batch)  # (B, n, 3)

        J = calc_jacob(index_tensor, pos, atom_type=atom_type).to_dense()  # (B, e, 3n)

        dq = torch.bmm(J, dx.reshape(B, -1, 1))
        pad_mask = index_tensor[1] + index_tensor[0] * e
        dq = dq.flatten()[pad_mask]

        return dq

    def batch_dq2dx(self, dq, pos, atom_type, edge_index, batch, num_nodes, q_type="morse"):
        if q_type == "morse":
            calc_jacob = self.sparse_batch_jacobian_q
        elif q_type == "DM":
            calc_jacob = self.sparse_batch_jacobian_d
        else:
            raise NotImplementedError

        B = num_nodes.size(0)
        n = num_nodes.max()
        e = n * (n - 1) // 2

        index_tensor = redefine_edge_index(edge_index, batch, num_nodes)  # (4, E)
        atom_type = redefine_with_pad(atom_type, batch, padding_value=-1)  # (B, n)
        pos = redefine_with_pad(pos, batch)  # (B, n, 3)
        dq = torch.sparse_coo_tensor(index_tensor[:2], dq.flatten(), (B, e),).to_dense()  # (B, e)

        J = calc_jacob(index_tensor, pos, atom_type=atom_type).to_dense()  # (B, e, 3n)
        J_inv = batch_pinv1(J, rtol=1e-4, atol=self.svd_tol)  # (B, 3n, e)

        dx = torch.bmm(J_inv, dq.reshape(B, -1, 1)).reshape(-1, 3)
        pad_mask = get_pad_mask(num_nodes)
        dx = dx[pad_mask]

        return dx

    def batch_initialize(self, x, q_dot, edge_index, atom_type, batch, num_nodes, q_type="morse",
                         pos_adjust_scaler=0.05, pos_adjust_thresh=1e-3, verbose=0):
        """
        Args:
            x (torch.Tensor): initial position tensor (B, n, 3)
            q_dot (torch.Tensor): initial velocity tensor (B, e)
            index_tensor (torch.Tensor): redefined edge tensor - (batch_idx, edge_idx, i', j'), shape: (4, E)
            atom_type (torch.Tensor): atom type tensor (B, n)

        """
        if q_type == "morse":
            calc_jacob = self.sparse_batch_jacobian_q
        elif q_type == "DM":
            calc_jacob = self.sparse_batch_jacobian_d
        else:
            raise NotImplementedError

        B = batch.max() + 1
        n = num_nodes.max()
        e = n * (n - 1) // 2

        index_tensor = redefine_edge_index(edge_index, batch, num_nodes)  # (4, E)
        x = redefine_with_pad(x, batch)  # (B, n, 3)
        atom_type = redefine_with_pad(atom_type, batch, padding_value=-1)  # (B, n)
        q_dot = torch.sparse_coo_tensor(index_tensor[:2], q_dot.flatten(), (B, e),).to_dense()  # (B, e)

        J = calc_jacob(index_tensor, x, atom_type=atom_type).to_dense()  # (B, e, 3n)
        x, flag = self.pos_adjust(x, J, num_nodes, scaler=pos_adjust_scaler, thresh=pos_adjust_thresh)
        if flag:
            # recompute jacobian
            if verbose >= 1:
                print("Adjusting positions because of numerical unstability...")
            J = calc_jacob(index_tensor, x, atom_type=atom_type).to_dense()  # (B, e, 3n)

        J_inv = batch_pinv1(J, rtol=1e-4, atol=self.svd_tol)  # (B, 3n, e)

        x_dot = torch.bmm(J_inv, q_dot.unsqueeze(-1)).reshape(B, n, 3)  # (B, n, 3)
        q_dot = torch.bmm(J, x_dot.reshape(B, -1, 1)).squeeze(-1)  # (B, e)

        total_time = x_dot.reshape(B, -1).norm(dim=-1)
        x_dot = x_dot / total_time.reshape(-1, 1, 1)  # (B, n, 3)
        q_dot = q_dot / total_time.reshape(-1, 1)  # (B, e)

        if q_type == "morse":
            q = self.batch_compute_q(index_tensor, atom_type, x)
        elif q_type == "DM":
            q = self.batch_compute_d(index_tensor, x)
        else:
            raise NotImplementedError

        return x, x_dot, total_time, q, q_dot, atom_type, index_tensor

    def pos_adjust(self, x, J, num_nodes, scaler=0.05, thresh=1e-3):
        """
        Adjust position tensor so that singular values less than threshold becomes larger than threshold.
        By pushing the position along the singular vectors corresponding to the small (but non-zero) singular values,
        we can adjust position with minimal change in q-coordinates.
        Args:
            x (torch.Tensor): position tensor (B, n, 3)
            J (torch.Tensor): jacobian tensor (B, e, 3n)
            num_nodes (torch.Tensor): number of nodes for each graph (B, )
        """
        U, S, Vh = batch_svd1(J)  # S : (B, e), U : (B, e, e), Vh : (B, 3n, 3n)
        mask = S < thresh  # select small (but nonzero) singular values
        len_mask = sequence_length_mask(3 * num_nodes - 6, max_N=S.size(1))
        mask = torch.logical_and(mask, len_mask)

        coeff = torch.randn(mask.size()).to(x.device)  # coefficient for singular vectors for adjustment
        coeff = coeff.masked_fill(~mask, 0)  # only selected singular vectors
        coeff = coeff / ((coeff.pow(2).sum(-1, keepdim=True)).sqrt() + 1e-10) * scaler  # normalize and scale

        if coeff.size(1) != Vh.size(1):
            _ = torch.zeros(coeff.size(0), Vh.size(1)).to(coeff.device)
            _[:, :coeff.size(1)] = coeff
            coeff = _
        pos_adjust = (coeff.unsqueeze(-1) * Vh).sum(1).reshape(len(num_nodes), -1, 3)  # (B, n, 3)
        flag = mask.any()
        x += pos_adjust

        return x, flag

    def batch_advance(self, x, x_dot, index_tensor, atom_type, done, dt, q_type="morse", verbose=False, return_qdot=False, method="Euler"):
        if method == "Euler":
            return self.batch_advance_euler(x, x_dot, index_tensor, atom_type, done, dt, q_type=q_type, verbose=verbose, return_qdot=return_qdot)

        elif method == "Heun":
            return self.batch_advance_heun(x, x_dot, index_tensor, atom_type, done, dt, q_type=q_type, verbose=verbose, return_qdot=return_qdot)

        elif method == "RK4":
            return self.batch_advance_rk4(x, x_dot, index_tensor, atom_type, done, dt, q_type=q_type, verbose=verbose, return_qdot=return_qdot)

        else:
            raise NotImplementedError(f"method {method} is not implemented.")

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

        if q_type == "morse":
            calc_jacob = self.sparse_batch_jacobian_q
        elif q_type == "DM":
            calc_jacob = self.sparse_batch_jacobian_d
        else:
            raise NotImplementedError

        dt = dt[~done].clone().detach()
        J = calc_jacob(index_tensor, x, atom_type=atom_type)

        B = (~done).sum()
        not_done_index = torch.where(~done)[0]
        J = J.index_select(0, not_done_index)  # sparse (B, e, 3n)

        if return_qdot:
            q_dot = torch.bmm(J.to_dense(), x_dot[~done].reshape(B, -1, 1)).squeeze(-1)
        else:
            q_dot = None

        _dt = dt.clip(min=0)

        dx, dx_dot = self._advance(done, x, x_dot, index_tensor, atom_type, _dt, q_type=q_type, J=J)

        x[~done] += dx
        x_dot[~done] += dx_dot

        if verbose:
            print(f"\t\tdebug: dx = {dx.reshape(B, -1).norm(dim=-1)}, dx_dot norm = {dx_dot.reshape(B, -1).norm(dim=-1)}")

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

        if q_type == "morse":
            calc_jacob = self.sparse_batch_jacobian_q
        elif q_type == "DM":
            calc_jacob = self.sparse_batch_jacobian_d
        else:
            raise NotImplementedError

        dt = dt[~done].clone().detach()
        J = calc_jacob(index_tensor, x, atom_type=atom_type)

        B = (~done).sum()
        not_done_index = torch.where(~done)[0]
        J = J.index_select(0, not_done_index)  # sparse (B, e, 3n)

        if return_qdot:
            q_dot = torch.bmm(J.to_dense(), x_dot[~done].reshape(B, -1, 1)).squeeze(-1)
        else:
            q_dot = None

        _dt = dt.clip(min=0)

        x_new, x_dot_new = x.clone(), x_dot.clone()

        dx1, dx_dot1 = self._advance(done, x_new, x_dot_new, index_tensor, atom_type, _dt, q_type=q_type, J=J)
        x_new[~done] = x_new[~done] + dx1
        x_dot_new[~done] = x_dot[~done] + dx_dot1

        dx2, dx_dot2 = self._advance(done, x_new, x_dot_new, index_tensor, atom_type, _dt, q_type=q_type)
        update_x = (dx1 + dx2) / 2
        update_x_dot = (dx_dot1 + dx_dot2) / 2

        x[~done] += update_x
        x_dot[~done] += update_x_dot

        if verbose:
            print(f"\t\tdebug: dx = {update_x.reshape(B, -1).norm(dim=-1)}, dx_dot norm = {update_x_dot.reshape(B, -1).norm(dim=-1)}")

        return x, x_dot, q_dot

    def batch_advance_rk4(self, x, x_dot, index_tensor, atom_type, done, dt, q_type="morse", verbose=False, return_qdot=False):
        """
        Args:
            x: torch.Tensor, initial position tensor (B, n, 3)
            x_dot: torch.Tensor, initial velocity tensor (B, n, 3)
            index_tensor: torch.Tensor, redefined edge index tensor (4, E)
            atom_type: torch.Tensor, atom type tensor (B, n)
            done: torch.BoolTensor, already finishied flag tensor (B, )
            dt: torch.Tensor, time step tensor (B, )
        """

        if q_type == "morse":
            calc_jacob = self.sparse_batch_jacobian_q
        elif q_type == "DM":
            calc_jacob = self.sparse_batch_jacobian_d
        else:
            raise NotImplementedError

        dt = dt[~done].clone().detach()
        J = calc_jacob(index_tensor, x, atom_type=atom_type)

        B = (~done).sum()
        not_done_index = torch.where(~done)[0]
        J = J.index_select(0, not_done_index)  # sparse (B, e, 3n)

        if return_qdot:
            q_dot = torch.bmm(J.to_dense(), x_dot[~done].reshape(B, -1, 1)).squeeze(-1)
        else:
            q_dot = None

        _dt = dt.clip(min=0)

        x_new, x_dot_new = x.clone(), x_dot.clone()
        dx1, dx_dot1 = self._advance(done, x_new, x_dot_new, index_tensor, atom_type, _dt, q_type=q_type, J=J)
        x_new[~done] = x_new[~done] + dx1 * 0.5
        x_dot_new[~done] = x_dot[~done] + dx_dot1 * 0.5

        dx2, dx_dot2 = self._advance(done, x_new, x_dot_new, index_tensor, atom_type, _dt, q_type=q_type)
        x_new, x_dot_new = x.clone(), x_dot.clone()
        x_new[~done] = x_new[~done] + dx2 * 0.5
        x_dot_new[~done] = x_dot[~done] + dx_dot2 * 0.5

        dx3, dx_dot3 = self._advance(done, x_new, x_dot_new, index_tensor, atom_type, _dt, q_type=q_type)
        x_new, x_dot_new = x.clone(), x_dot.clone()
        x_new[~done] = x_new[~done] + dx3
        x_dot_new[~done] = x_dot[~done] + dx_dot3

        dx4, dx_dot4 = self._advance(done, x_new, x_dot_new, index_tensor, atom_type, _dt, q_type=q_type)
        update_x = (dx1 + 2 * dx2 + 2 * dx3 + dx4) / 6
        update_x_dot = (dx_dot1 + 2 * dx_dot2 + 2 * dx_dot3 + dx_dot4) / 6

        x[~done] += update_x
        x_dot[~done] += update_x_dot

        if verbose:
            print(f"\t\tdebug: dx = {update_x.reshape(B, -1).norm(dim=-1)}, dx_dot norm = {update_x_dot.reshape(B, -1).norm(dim=-1)}")

        return x, x_dot, q_dot

    def _advance(self, done, x, x_dot, index_tensor, atom_type, dt, q_type="morse", J=None):
        B = (~done).sum()
        not_done_index = torch.where(~done)[0]

        if J is None:  # compute J
            if q_type == "morse":
                J = self.sparse_batch_jacobian_q(index_tensor, x, atom_type=atom_type)
                J = J.index_select(0, not_done_index)
            elif q_type == "DM":
                J = self.sparse_batch_jacobian_d(index_tensor, x, atom_type=atom_type)
                J = J.index_select(0, not_done_index)
            else:
                raise NotImplementedError

        christoffel = self.batch_christoffel(index_tensor, atom_type, x, J, not_done_index, q_type=q_type)  # (B, n, n, n)
        x_ddot = - torch.einsum("bj,bkij,bi->bk", x_dot[~done].reshape(B, -1), christoffel, x_dot[~done].reshape(B, -1))
        x_ddot = x_ddot.reshape(B, -1, 3)

        dx = x_dot[~done] * dt.reshape(-1, 1, 1)
        dx_dot = x_ddot * dt.reshape(-1, 1, 1)
        return dx, dx_dot

    def batch_christoffel(self, index_tensor, atom_type, x, J, not_done_index, q_type="morse"):
        n3 = x.size(1) * 3
        B = not_done_index.numel()

        if q_type == "morse":
            hess = self.sparse_batch_hessian_q(index_tensor, x, atom_type=atom_type)  # sparse (B, 3n * 3n, e)
        elif q_type == "DM":
            hess = self.sparse_batch_hessian_d(index_tensor, x, atom_type=atom_type)  # sparse (B, 3n * 3n, e)
        else:
            raise NotImplementedError

        hess = hess.index_select(0, not_done_index)  # sparse (B, 3n * 3n, e)
        J_inv = batch_pinv1(J, rtol=1e-4, atol=self.svd_tol).transpose(-1, -2)  # dense (B, e, 3n)

        christoffel = torch.bmm(hess, J_inv).transpose(-1, -2).reshape(B, n3, n3, n3)
        # Gamma^k_ij = christoffel[:, k, i, j]
        return christoffel


def batch_pinv1(J, rtol=1e-4, atol=None):
    return vmap(torch.linalg.pinv)(J.to_dense(), rtol=rtol, atol=atol)


def batch_pinv2(J, rtol=1e-4, atol=None):
    def sparse_pinv(x):
        return torch.linalg.pinv(x.to_dense(), rtol=rtol, atol=atol)
    return vmap(sparse_pinv)(J)


def batch_svd1(J):
    return vmap(torch.linalg.svd)(J.to_dense())


def batch_svd2(J):
    def sparse_svd(x):
        return torch.linalg.svd(x.to_dense())
    return vmap(sparse_svd)(J)


if __name__ == "__main__":
    pass
