import torch


def _repeat(x, n):
    return x.unsqueeze(-1).expand(-1, n).flatten()


def _stack(x):
    return torch.stack([3 * x, 3 * x + 1, 3 * x + 2]).T.flatten()


class GeodesicSolver(object):
    def __init__(self, alpha=1.6, beta=2.3, atomic_radius=None, svd_tol=1e-3,):
        self.alpha = alpha
        self.beta = beta
        self.atomic_radius = atomic_radius
        self.svd_tol = svd_tol

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

        dq_dd = - self.alpha / d_e_ij * torch.exp(- self.alpha / d_e_ij * (d_ij - d_e_ij)) - self.beta * d_e_ij / d_ij ** 2

        dq_dx = dq_dd.unsqueeze(-1) * dd_dx # (E, 3)

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
        d_pos = pos[i] - pos[j] # (E, 3)
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

    def advance(self, x, x_dot, edge_index, atom_type,
                q_type="morse", dt=1e-2, verbose=False
                ):
        if q_type == "morse":
            hess = self.hessian_q(edge_index, atom_type, x) # (E, 3N, 3N)
            jacob = self.jacobian_q(edge_index, atom_type, x) # (E, 3N)
        elif q_type == "DM":
            hess = self.hessian_d(edge_index, x)
            jacob = self.jacobian_d(edge_index, x)

        J = jacob
        JG = torch.linalg.pinv(J, rtol=1e-4, atol=1e-2).T
        # if refine_xdot:
        #     x_dot = JG.T @ J @ x_dot
        #     if verbose:
        #         print(f"\t\t\tdebug: x_dot norm = {x_dot.norm():0.6f}")

        christoffel = torch.einsum("mij, mk->kij", hess, JG)
        x_ddot = - torch.einsum("j,kij,i->k", x_dot, christoffel, x_dot)

        # x_ddot and x_dot should be perpendicular
        q_ddot = J @ x_ddot
        q_dot = J @ x_dot

        new_x = x + x_dot * dt
        new_x_dot = x_dot + x_ddot * dt

        # dotproduct
        if verbose:
            print(f"\t\tdebug: <x_ddot, x_dot> = {(q_ddot * q_dot).sum()}")
            print(f"\t\tdebug: <x_ddot, x_dot> = {((jacob.T @ jacob) @ x_dot.reshape(-1, 1) * x_ddot).sum()}")
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

    def geodesic_ode_solve(self, x, q_dot, edge_index, atom_type, q_type="morse",
                               num_iter=100, check_dot_every=10,
                               ref_dt=1e-2, max_dt=1e-1, verbose=0
                               ):

        x, x_dot, total_time, q, q_dot = self.initialize(x, q_dot, edge_index, atom_type, q_type=q_type)

        ref_dt = min(total_time / num_iter, ref_dt)
        dt = ref_dt
        if verbose >= 1:
            print(f"initial dt = {ref_dt:0.6f}, total_expected_iter = {total_time / ref_dt:1.0f}")

        current_time = 0
        iter = 0
        total_dq = 0
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

        # check numerical stability
        # q_dot.norm() should be same as total_dq

        return x.reshape(-1, 3), iter, total_dq, q_dot.norm()
