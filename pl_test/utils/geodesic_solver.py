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
        K1 = - (self.alpha / d_e_ij) * torch.exp(-self.alpha * (d_ij - d_e_ij) / d_e_ij) - self.beta * d_e_ij / d_ij ** 2
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
        dq_dd = - self.alpha / d_e_ij * torch.exp(- self.alpha / d_e_ij * (d_ij - d_e_ij)) - self.beta * d_e_ij / d_ij ** 2

        dq_dx = dq_dd.unsqueeze(-1) * dd_dx  # (E, 3)

        k = _repeat(k, 3)
        i = _stack(i)
        j = _stack(j)

        index = torch.stack([torch.cat([k, k,], dim=0), torch.cat([i, j], dim=0)])
        val = torch.cat([dq_dx, -dq_dx], dim=0).flatten()
        jacobian = torch.sparse_coo_tensor(index, val, (E, 3 * N))
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

        _b = index_tensor[0]
        _i = index_tensor[2] + n * _b
        _j = index_tensor[3] + n * _b
        _pos = pos.reshape(-1, 3)  # (B * n, 3)
        _atom_type = atom_type.reshape(-1)  # (B * n, )
        _edge_index = torch.stack([_i, _j])  # (2, E)

        d_ij = self.compute_d(_edge_index, _pos)  # (E, )
        d_e_ij = self.compute_de(_edge_index, _atom_type)  # (E, )

        dd_dx = (_pos[_i] - _pos[_j]) / d_ij[:, None]  # (E, 3)
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
        # TODO
        # jacobian = jacobian.to_dense()
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

        K1 = - (self.alpha / d_e_ij) * torch.exp(-self.alpha * (d_ij - d_e_ij) / d_e_ij) - self.beta * d_e_ij / d_ij ** 2  # (E, )
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

        total_time = x_dot.norm()
        x_dot = x_dot / x_dot.norm()

        if q_type == "morse":
            q = self.compute_q(edge_index, atom_type, x.reshape(-1, 3))
        elif q_type == "DM":
            q = self.compute_d(edge_index, x.reshape(-1, 3))
        else:
            raise NotImplementedError

        return x, x_dot, total_time, q, q_dot

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
        J = J.to_dense()
        U, S, Vh = torch.vmap(torch.linalg.svd)(J)  # S : (B, e), U : (B, e, e), Vh : (B, 3n, 3n)
        mask = (S > 1e-9) & (S < thresh)  # select small (but nonzero) singular values
        coeff = torch.randn(mask.size()).to(x.device)  # coefficient for singular vectors for adjustment
        coeff = coeff.masked_fill(~mask, 0)  # only selected singular vectors
        coeff = coeff / ((coeff.pow(2).sum(-1, keepdim=True)).sqrt() + 1e-10) * scaler  # normalize and scale

        pos_adjust = (coeff.unsqueeze(-1) * Vh).sum(1).reshape(len(num_nodes), -1, 3)  # (B, n, 3)
        flag = mask.any()
        x += pos_adjust
        return x, flag

    def batch_initialize(self, x, q_dot, edge_index, atom_type, batch, num_nodes, q_type="morse",
                         pos_adjust_scaler=0.05, pos_adjust_thresh=1e-3):
        """
        Args:
            x (torch.Tensor): initial position tensor (B, n, 3)
            q_dot (torch.Tensor): initial velocity tensor (B, e)
            index_tensor (torch.Tensor): redefined edge tensor - (batch_idx, edge_idx, i', j'), shape: (4, E)
            atom_type (torch.Tensor): atom type tensor (B, n)

        """
        # TODO: adjust position with singular values
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
        x, flag = self.pos_adjust(x, J, num_nodes, scaler=pos_adjust_scaler, thresh=pos_adjust_thresh)
        if flag:
            # recompute jacobian
            print("Adjusting positions because of numerical unstability...")
            J = self.sparse_batch_jacobian_q(index_tensor, atom_type, x).to_dense()  # (B, e, 3n)

        J_inv = torch.stack([torch.linalg.pinv(j, rtol=1e-4, atol=self.svd_tol) for j in J])
        # J_inv = vmap(torch.linalg.pinv)(J, rtol=1e-4, atol=self.svd_tol)  # (B, 3n, e)

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
        ban_index = torch.LongTensor([])

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

                new_ban = torch.where(ref_dt < min_dt)[0].cpu()
                _ = torch.isin(new_ban, not_done_index[err > err_thresh])
                new_ban = new_ban[_]
                if new_ban.numel() > 0:
                    ban_index = torch.cat([ban_index, new_ban])
                    ban_index = ban_index.unique()
                    done[ban_index] = True
                    ref_dt[ban_index] = min_dt
                    print(f"[Warning] Some samples ({new_ban}) are banned due to numerical unstability.")

                    if verbose >= 0:
                        print(f"[Warning] (iter={cnt, iter[new_ban]+1})veolocity error is too large, restart with smaller time step.")
                        torch.set_printoptions(precision=4, sci_mode=False)
                        print(f"err = {err[err > err_thresh]}")
                        torch.set_printoptions()

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
                print(f"iter = {iter}, \n\tcurrent_time = \n\t{current_time}, \n\ttotal_time = \n\t{total_time}, \n\tdt = \n\t{dt}, \n\tdone = \n\t{done}")
            if cnt >= 2 * max_iter:
                break

        if q_type == "morse":
            q_last = self.batch_compute_q(index_tensor, atom_type, x)
        elif q_type == "DM":
            q_last = self.batch_compute_d(index_tensor, x)
        J = self.sparse_batch_jacobian_q(index_tensor, atom_type, x)
        x_dot *= total_time.reshape(-1, 1, 1)
        q_dot = torch.bmm(J, x_dot.reshape(B, -1, 1)).squeeze(-1)  # (B, e)
        last = {"x": x, "x_dot": x_dot, "q": q_last, "q_dot": q_dot}
        stats = {"iter": iter, "current_time": current_time, "total_time": total_time, "ban_index": ban_index}

        return init, last, iter, index_tensor, stats

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

        _dt = torch.where(dt > 0, dt, torch.zeros_like(dt))
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
        dt = torch.tensor(dt[~done])
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
        dt = dt[~done]
        if q_type == "morse":
            J = self.sparse_batch_jacobian_q(index_tensor, atom_type, x)
        elif q_type == "DM":
            J = self.sparse_batch_jacobian_d(index_tensor, x)

        B = (~done).sum()
        not_done_index = torch.where(~done)[0]
        J = J.index_select(0, not_done_index)

        if return_qdot:
            q_dot = torch.bmm(J, x_dot[~done].reshape(B, -1, 1)).squeeze(-1)
        else:
            q_dot = None

        _dt = torch.where(dt > 0, dt, torch.zeros_like(dt))
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
        if J is None:
            if q_type == "morse":
                J = self.sparse_batch_jacobian_q(index_tensor, atom_type, x)
                J = J.index_select(0, not_done_index)
            elif q_type == "DM":
                J = self.sparse_batch_jacobian_d(index_tensor, x)
                J = J.index_select(0, not_done_index)
            else:
                raise NotImplementedError
        christoffel = self.batch_christoffel(index_tensor, atom_type, x, J, not_done_index)  # (B, n, n, n)
        x_ddot = - torch.einsum("bj,bkij,bi->bk", x_dot[~done].reshape(B, -1), christoffel, x_dot[~done].reshape(B, -1))
        x_ddot = x_ddot.reshape(B, -1, 3)

        dx = x_dot[~done] * dt.reshape(-1, 1, 1)
        dx_dot = x_ddot * dt.reshape(-1, 1, 1)
        return dx, dx_dot

    def batch_christoffel(self, index_tensor, atom_type, x, J, not_done_index):
        J = J.to_dense()
        n3 = x.size(1) * 3
        B = not_done_index.numel()
        hess = self.sparse_batch_hessian_q(index_tensor, atom_type, x)  # sparse (B, 3n * 3n, e)
        hess = hess.index_select(0, not_done_index)  # sparse (B, 3n * 3n, e)
        J_inv = vmap(torch.linalg.pinv)(J, rtol=1e-4, atol=self.svd_tol).transpose(-1, -2)  # dense (B, e, 3n)

        christoffel = torch.bmm(hess, J_inv).transpose(-1, -2).reshape(B, n3, n3, n3)
        # Gamma^k_ij = christoffel[:, k, i, j]
        return christoffel


if __name__ == "__main__":
    import omegaconf
    config = omegaconf.OmegaConf.load("../configs/config.yaml")
