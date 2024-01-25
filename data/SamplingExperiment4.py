## import libararies
import numpy as np
import torch

from scipy.spatial.distance import cdist, pdist
import scipy

import os
import ase.io
import ase
import sys
import copy

from torch_scatter import scatter_add

from lst_interp import *
import tqdm

import matplotlib.pyplot as plt
# import seaborn as sns

import sys
sys.path.append("/home/share/DATA/NeuralOpt/Interpolations/Geodesic_interp")
# from get_geodesic_energy import get_rijlist_and_re, compute_wij, morse_scaler
from get_geodesic_energy import morse_scaler, ATOMIC_RADIUS


## Experiment setting : random seed, precision, device
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    return

set_seed(0)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)


## wrapper class to save atoms object
class Wrapper:
    def __init__(self, atoms_0, atoms_T, q_type="DM", alpha=1.7, beta=0.01, gamma=0.01, using_jacobian=True, svd_tol=1e-4):
        self.atoms_0 = atoms_0
        self.atoms_T = atoms_T
        # assert q_type in ["DM", "morse"]
        self.q_type = q_type
        self.svd_tol = svd_tol
        self.re = torch.Tensor(self.get_re(atoms_T))
        self.alpha, self.beta = alpha, beta
        self.gamma = gamma
        self.morse_scaler = morse_scaler(self.re, self.alpha, self.beta)
        self.scaler_factor = 1.0
        self.using_jacobian = using_jacobian
        return

    def get_re(self, atoms, threshold=np.inf):
        from scipy.spatial import KDTree

        rijset = set()
        tree = KDTree(atoms.positions)
        pairs = tree.query_pairs(threshold)
        rijset.update(pairs)
        rijlist = sorted(rijset)

        radius = np.array([ATOMIC_RADIUS.get(atom.capitalize(), 1.5) for atom in atoms.get_chemical_symbols()])
        re = np.array([radius[i] + radius[j] for i, j in rijlist])
        return re

    def calc_inverse_jacobian(self, pos, q_type):
        edge_index, edg_length = self.pos_to_dist(pos)
        distance = pdist(pos)
        distance_e = self.get_re(self.atoms_T)
        inverse_jacobain = []

        for ij, d, de in zip(edge_index.T, distance, distance_e):
            jacob = torch.zeros(size=pos.size())
            i, j = ij
            pos_i, pos_j = pos[i], pos[j]
            d_pos = pos_i - pos_j
            if q_type == "DM":
                dr_dd = d / d_pos
                jacob[i] = dr_dd
                jacob[j] = - dr_dd

            elif q_type == "morse":
                dr_dq = d ** 3 * de  / - (self.alpha * np.exp(- self.alpha * (d / de - 1)) + self.beta * de ** 2) / d_pos
                jacob[i] = dr_dq
                jacob[j] = - dr_dq
            inverse_jacobain.append(jacob.flatten())
        return torch.stack(inverse_jacobain, dim=0)

    def calc_jacobian(self, pos, q_type):
        # pos = Tensor, (N, 3)
        edge_index, edge_length = self.pos_to_dist(pos)
        distance = pdist(pos)
        distance_e = self.get_re(self.atoms_T)

        jacobian = []
        for i_idx in range(len(pos)):
            j_idx = list(range(len(pos)))
            j_idx.remove(i_idx)
            j_idx = torch.LongTensor(j_idx)

            j_mask = torch.any(edge_index == i_idx, axis=0)
            dd_dx = torch.zeros(size=(len(edge_length), 3))
            dq_dx = torch.zeros(size=(len(edge_length), 3))
            pos_i = pos[i_idx].reshape(1, -1)
            pos_j = pos[j_idx]

            dist = distance[j_mask].reshape(-1, 1)
            dd_dx[j_mask] += (pos_i - pos_j) / dist

            if q_type == "DM":
                jacobian.append(dd_dx.T)

            elif q_type == "morse":
                dq_dd = - (self.alpha / distance_e[j_mask]) * np.exp(-self.alpha * (distance[j_mask] - distance_e[j_mask]) / distance_e[j_mask])
                dq_dd -= self.beta * distance_e[j_mask] / (distance[j_mask] ** 2)
                dq_dx[j_mask] += dd_dx[j_mask] * dq_dd.reshape(-1, 1)
                jacobian.append(dq_dx.T)

            elif q_type == "morese+DM":
                raise NotImplementedError

        return torch.cat(jacobian, dim=0)

    def calc_distance_hessian(self, pos, edge_index, distance):
        N = len(pos)
        K = len(edge_index)
        hessian = torch.zeros(size=(K, 3 * N, 3 * N))
        for k, (ij, d_ij) in enumerate(zip(edge_index, distance)):
            i, j = ij
            pos_i, pos_j = pos[i], pos[j]

            # calculate hessian related to i, j atoms
            d_pos = pos_i - pos_j
            hess_ij = d_pos.reshape(1, -1) * d_pos.reshape(-1, 1)
            hess_ij /= d_ij ** 3
            hess_ij -= torch.eye(3) / d_ij

            # calculate hessian related to i, i atoms
            hess_ii = d_pos.reshape(1, -1) * d_pos.reshape(-1, 1)
            hess_ii /= - d_ij ** 3
            hess_ii += torch.eye(3) / d_ij

            # hess_ii = hess_jj
            hess_jj = hess_ii

            hessian[k, 3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] += hess_ii
            hessian[k, 3 * j:3 * (j + 1), 3 * j:3 * (j + 1)] += hess_jj
            hessian[k, 3 * i:3 * (i + 1), 3 * j:3 * (j + 1)] += hess_ij
            hessian[k, 3 * j:3 * (j + 1), 3 * i:3 * (i + 1)] += hess_ij

        return hessian

    def calc_hessian(self, pos, q_type=None):
        if q_type is None:
            q_type = self.q_type

        edge_index, edge_length = self.pos_to_dist(pos)
        edge_index = edge_index.T
        distance = pdist(pos)
        distance_e = self.get_re(self.atoms_T)

        hessian = self.calc_distance_hessian(pos, edge_index, distance)

        if q_type == "DM":
            return hessian

        elif q_type == "morse":
            dq_dd = - self.alpha / distance_e * np.exp(-self.alpha * (distance - distance_e) / distance_e)
            dq_dd -= self.beta * distance_e / (distance ** 2)
            hessian_q = hessian * dq_dd.reshape(-1, 1, 1)

            for k, (ij, d_ij, de_ij) in enumerate(zip(edge_index, distance, distance_e)):
                i, j = ij
                pos_i, pos_j = pos[i], pos[j]
                # calculate hessian related to i, j atoms
                d_pos = pos_i - pos_j
                hess_ij = d_pos.reshape(1, -1) * d_pos.reshape(-1, 1)
                hess_ij /= - d_ij ** 2
                coeff = self.alpha ** 2 / de_ij ** 2 * np.exp(-self.alpha * (d_ij - de_ij) / de_ij)  + 2 * self.beta * de_ij / (d_ij ** 3)
                hess_ij *= coeff

                # calculate hessian related to i, i atoms
                hess_ii = - hess_ij

                hessian_q[k, 3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] += hess_ii
                hessian_q[k, 3 * j:3 * (j + 1), 3 * j:3 * (j + 1)] += hess_ii
                hessian_q[k, 3 * i:3 * (i + 1), 3 * j:3 * (j + 1)] += hess_ij
                hessian_q[k, 3 * j:3 * (j + 1), 3 * i:3 * (i + 1)] += hess_ij

            return hessian_q

        elif q_type == "morese+DM":
            raise NotImplementedError
        return

    def eq_transform(self, score_d, pos, edge_index, edge_length):
        if self.using_jacobian:
            jacobian = self.calc_jacobian(pos, q_type=self.q_type)

            score_pos = jacobian @ score_d.reshape(-1, 1)
            return score_pos.reshape(-1, 3)

        if self.q_type == "morse":
            edge_length = torch.Tensor(pdist(pos))

            N = pos.size(0)
            dd_dr = - (self.alpha / self.re) * torch.exp(-self.alpha * (edge_length - self.re) / self.re) / edge_length
            dd_dr -= self.beta * self.re / (edge_length ** 3)
            dd_dr = dd_dr.reshape(-1, 1)
            dd_dr = dd_dr * (pos[edge_index[0]] - pos[edge_index[1]])
            score_d = score_d.reshape(-1, 1)
            score_d *= self.scaler_factor
            score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N)
            score_pos += scatter_add(-dd_dr * score_d, edge_index[1], dim=0, dim_size=N)

        elif self.q_type == "DM":
            N = pos.size(0)
            dd_dr = (1.0 / edge_length).reshape(-1, 1) * (pos[edge_index[0]] - pos[edge_index[1]])
            score_d = score_d.reshape(-1, 1)
            score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N)
            score_pos += scatter_add(-dd_dr * score_d, edge_index[1], dim=0, dim_size=N)

        elif self.q_type == "morse+DM":
            edge_length = torch.Tensor(pdist(pos))
            N = pos.size(0)
            dd_dr = - (self.alpha / self.re) * torch.exp(-self.alpha * (edge_length - self.re) / self.re) / edge_length
            dd_dr -= self.beta * self.re / (edge_length ** 3)
            dd_dr += self.gamma / edge_length
            dd_dr = dd_dr.reshape(-1, 1)
            score_d *= self.scaler_factor
            dd_dr = dd_dr * (pos[edge_index[0]] - pos[edge_index[1]])
            score_d = score_d.reshape(-1, 1)
            score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N)
            score_pos += scatter_add(-dd_dr * score_d, edge_index[1], dim=0, dim_size=N)
        else:
            raise NotImplementedError
        return score_pos

    def pos_to_dist(self, pos, q_type=None):
        if q_type is None:
            q_type = self.q_type
        if q_type == "morse":
            rij = pdist(pos)
            wij = self.morse_scaler(rij)[0] * self.scaler_factor
            # print(wij, type(wij))
            # length = torch.Tensor(wij)
            length = wij
            index = torch.LongTensor(np.stack(np.triu_indices(len(pos), 1)))
        elif q_type == "DM":
            length = torch.Tensor(pdist(pos))
            index = torch.LongTensor(np.stack(np.triu_indices(len(pos), 1)))
        elif q_type == "morse+DM":
            rij = pdist(pos)
            wij = self.morse_scaler(rij)[0] * self.scaler_factor
            wij += self.gamma * rij
            length = torch.Tensor(wij)
            index = torch.LongTensor(np.stack(np.triu_indices(len(pos), 1)))
        else:
            raise NotImplementedError
        return index, length

    def reverse_diffusion_process(self, x_t, t, dt, params, x_0, x_T, coord="Cartesian", h_coeff=0.0, verbose=True,
                                  using_jacobian=True, sampling_test=1, inner_iteration=5):
        beta_t = params.beta(t)

        if coord == "Cartesian":
            diff, coeff, v1, v2, v3, v4 = self.reverse_score(x_t, t, params, x_0, x_T, verbose=verbose)
            reverse_score_ = diff * coeff
            dw = torch.sqrt(beta_t * dt) * torch.randn_like(diff)
            dx = - 1.0 * reverse_score_ * dt + dw
        else:
            diff, coeff, v1, v2, v3, v4 = self.reverse_score2(x_t, t, params, x_0, x_T, verbose=verbose)
            index, d_t = self.pos_to_dist(x_t)

            if sampling_test == 0:
                # every displacement is first calculated on the q-space, and then transformed to the Cartesian space
                reverse_score_ = diff * coeff
                dw = torch.sqrt(beta_t * dt) * torch.randn_like(diff)
                dd = - 1.0 * reverse_score_ * dt + dw

                dx = self.eq_transform(dd, x_t, index, d_t)

            elif sampling_test == 1:
                reverse_score_ = diff * coeff
                dw = torch.sqrt(beta_t * dt) * torch.randn_like(diff)
                dd = - 1.0 * reverse_score_ * dt + dw   
                x_tm1 = self.exponential_ode_solver(x_t, -dd, q_type=self.q_type, num_iter=inner_iteration, check_dot_every=3)
                dx = - x_tm1 + x_t

        x_tm1 = x_t - dx
        return x_tm1, v1, v2, v3, v4

    def reverse_ode_process(self, x_t, t, dt, params, x_0, x_T, coord="Cartesian", h_coeff=0.0, verbose=True,
                            using_jacobian=True, sampling_test=1, inner_iteration=5):
        beta_t = params.beta(t)

        if coord == "Cartesian":
            diff, coeff, v1, v2, v3, v4 = self.reverse_score(x_t, t, params, x_0, x_T, verbose=verbose)
            reverse_score_ = diff * coeff
            dx = - 0.5 * reverse_score_ * dt
            print(f"Debug ({t:0.3f}): \n\t1) diff norm and dx norm {diff.norm():0.4f}, {dx.norm():0.6f}")
        else:
            diff, coeff, v1, v2, v3, v4 = self.reverse_score2(x_t, t, params, x_0, x_T, verbose=verbose)
            index ,d_t = self.pos_to_dist(x_t)

            if sampling_test == 0:
                reverse_score_ = diff * coeff
                dd = - 0.5 * reverse_score_ * dt
                dx = self.eq_transform(dd, x_t, index, d_t)
                # Want to check why eq-transform does not work well
                diff_d = diff
                diff_x = self.eq_transform(diff_d, x_t, index, d_t)
                print(f"Debug ({t:0.3f}): \n\t1) diff-d norm and diff-x norm {diff_d.norm():0.4f}, {diff_x.norm():0.4f} \n\t2) dd-norm and dx-norm {dd.norm():0.6f}, {dx.norm():0.6f}")
                print(f"\t3) dx-norm/dd-norm {dx.norm()/dd.norm():0.6f}")

            elif sampling_test == 1:
                reverse_score_ = diff * coeff
                dd = - 0.5 * reverse_score_ * dt
                print(f"debug ] time : {t:0.3f}")
                print(f"debug ] diff.norm() : {diff.norm()}")
                print(f"debug ] dd.norm() : {dd.norm()}")
                x_tm1 = self.exponential_ode_solver(x_t, -dd, q_type=self.q_type, num_iter=inner_iteration, check_dot_every=3)
                dx = - x_tm1 + x_t
                print(f"debug ] dx.norm() : {dx.norm()}")

        x_tm1 = x_t - dx
        return x_tm1, v1, v2, v3, v4

    def reverse_score(self, x_t, t, params, x_0, x_T, verbose=True):
        # calculate parameters
        beta_t = params.beta(t)
        sigma_t_square = params.sigma_square(t)
        sigma_T_square = params.sigma_1

        SNRTt = params.SNR(t)
        sigma_t_hat_square = sigma_t_square * (1 - SNRTt)

        # calc mu_hat
        mu_hat = x_T * SNRTt + x_0 * (1 - SNRTt)

        # calc difference
        diff = mu_hat - x_t

        # calc_score
        coeff =  1 / (sigma_t_hat_square) * beta_t
        score = diff * coeff

        # for debug
        if self.q_type == "DM":
        # if self.q_type in ["DM", "morse"]: # debugging # calculate err corresponding the metric
            _, d_T = self.pos_to_dist(x_T)
            _, d_t = self.pos_to_dist(x_t)
            _, d_0 = self.pos_to_dist(x_0)
            _, d_mu_hat = self.pos_to_dist(mu_hat)
            v1 = (d_mu_hat - d_t).abs().mean()
            # v2 = (d_mu_hat - d_T).abs().mean()
            v2 = (d_0 - d_t).abs().mean()
            v3 = (mu_hat - x_t.numpy()).abs().mean()
            v4 = (mu_hat - x_T.numpy()).abs().mean()
        # elif self.q_type == "morse":
        elif self.q_type in ["morse", "morse+DM", "Cartesian"]:
            version = "DMAE"
            # version = "Morse-RMSD"
            if version == "DMAE":
                d_T = torch.Tensor(pdist(x_T))
                d_mu_hat = torch.Tensor(pdist(mu_hat))  # typo=2의 경우, 이렇게 하면 안될 듯.
                d_t = torch.Tensor(pdist(x_t))
                d_0 = torch.Tensor(pdist(x_0))
                v1 = (d_mu_hat - d_t).abs().mean()
                # v2 = (d_mu_hat - d_T).abs().mean()
                v2 = (d_0 - d_t).abs().mean()
                v3 = abs(mu_hat - x_t.numpy()).mean()
                v4 = abs(mu_hat - x_T.numpy()).mean()
            else:
                _, d_T = self.pos_to_dist(x_T)
                _, d_mu_hat = self.pos_to_dist(mu_hat)  # typo=2의 경우, 이렇게 하면 안될 듯.
                _, d_t = self.pos_to_dist(x_t)
                _, d_0 = self.pos_to_dist(x_0)
                v1 = (d_mu_hat - d_t).norm()
                # v2 = (d_mu_hat - d_T).abs().mean()
                v2 = (d_0 - d_t).norm()
                v3 = abs(mu_hat - x_t.numpy()).mean()
                v4 = abs(mu_hat - x_T.numpy()).mean()
        else:
            raise NotImplementedError
        if verbose:
            print(f"{t:0.3f}\t{v1:0.4f}\t\t{v2:0.4f}\t\t{v3:0.4f}\t\t{v4:0.4f}\t\t{torch.linalg.norm(score, dim=-1).max():0.4f}")
        return diff, coeff, v1, v2, v3, v4

    def reverse_score2(self, x_t, t, params, x_0, x_T, verbose=True):
        # calculate parameters
        beta_t = params.beta(t)
        sigma_t_square = params.sigma_square(t)
        sigma_T_square = params.sigma_1

        SNRTt = params.SNR(t)
        sigma_t_hat_square = sigma_t_square * (1 - SNRTt)

        # calc mu_hat
        typo = 2

        if typo == 1:
            mu_hat = x_T * SNRTt + x_0 * (1 - SNRTt)
            _, d_mu_hat = self.pos_to_dist(mu_hat)
        if typo == 2:
            _, d_0 = self.pos_to_dist(x_0)
            _, d_T = self.pos_to_dist(x_T)
            d_mu_hat = d_T * SNRTt + d_0 * (1 - SNRTt)
            mu_hat = x_T * SNRTt + x_0 * (1 - SNRTt)  # for debugging
        if typo == 3:
            mu_hat = interpolate_LST(x_0.numpy(), x_T.numpy(), SNRTt.item())
            _, d_mu_hat = self.pos_to_dist(mu_hat)

        # calc difference
        index, d_t = self.pos_to_dist(x_t)
        diff_d = d_mu_hat - d_t
        diff = diff_d
        coeff =  1 / (sigma_t_hat_square) * beta_t

        # for debugging
        d_T = torch.Tensor(pdist(x_T))
        d_mu_hat = torch.Tensor(pdist(mu_hat))  # typo=2의 경우, 이렇게 하면 안될 듯.
        d_t = torch.Tensor(pdist(x_t))
        d_0 = torch.Tensor(pdist(x_0))
        v_loss_mae = (d_mu_hat - d_t).abs().mean()  # DMAE
        v_acc_mae = (d_0 - d_t).abs().mean()  # DMAE

        original_q_type = copy.deepcopy(self.q_type)
        self.q_type = "morse"
        _, d_T = self.pos_to_dist(x_T)
        _, d_mu_hat = self.pos_to_dist(mu_hat)  # typo=2의 경우, 이렇게 하면 안될 듯.
        _, d_t = self.pos_to_dist(x_t)
        _, d_0 = self.pos_to_dist(x_0)
        v_loss_norm = (d_mu_hat - d_t).norm()  # q-norm
        v_acc_norm = (d_0 - d_t).norm()  # q-norm
        self.q_type = original_q_type

        if verbose:
            print(f"{t:0.3f}\t{v_loss_mae:0.4f}\t\t{v_acc_mae:0.4f}\t\t{v_loss_norm:0.4f}\t\t{v_acc_norm:0.4f}\t")
        return diff, coeff, v_loss_mae, v_acc_mae, v_loss_norm, v_acc_norm

    def exponential_ode_solver(self, x0, q_dot0, q_type="morse", num_iter=100, check_dot_every=10, init_dt=1e-2, max_dt=1e-1, verbose=False):

        def one_step(x, x_dot, q_type=q_type, dt=1e-2, wrapper=self, refine_xdot=False, verbose=False):
            hess = wrapper.calc_hessian(x.reshape(-1, 3), q_type=q_type)
            jacob = wrapper.calc_jacobian(x.reshape(-1, 3), q_type=q_type).T

            # J, J_inv = wrapper.refine_jacobian(jacob)
            J = jacob
            J_inv = torch.linalg.pinv(J, rtol=1e-4, atol=1e-2)

            JG = J_inv.T
            if refine_xdot:
                x_dot = J_inv @ J @ x_dot
                if verbose:
                    print(f"\t\t\tdebug: x_dot norm = {x_dot.norm():0.6f}")
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

        jacob = self.calc_jacobian(x0, q_type=q_type).T
        # J, J_inv = self.refine_jacobian(jacob)
        J = jacob
        J_inv = torch.linalg.pinv(J, rtol=1e-4, atol=self.svd_tol)

        # debugging
        proj_q_dot = J @ J_inv @ q_dot0
        if verbose >= 1:
            print(f"\tdebug: proj_q_dot norm = {proj_q_dot.norm():0.4f}")
            print(f"\tdebug: proj_q_dot norm-ratio = {(proj_q_dot - q_dot0).norm()/ q_dot0.norm():0.4f}")

        # initialization
        x_dot0 = J_inv @ q_dot0
        x = x0.flatten()
        x_dot = x_dot0

        total_time = x_dot.norm()
        x_dot = x_dot / x_dot.norm()

        q = self.pos_to_dist(x.reshape(-1, 3))[1]
        # make time grid, 0 ~ total_time.
        # time spacing should be smaller than 1e-2
        # init_dt = 5e-2
        # init_dt = 1e-1
        # if total_time > num_iter * init_dt:
        #     num_iter = int(total_time / init_dt)

        # t = torch.linspace(0, total_time, num_iter + 1)[:-1]
        # # t = torch.linspace(0, 1, num_iter + 1)[:-1]
        # dt_ = t[1] - t[0]
        # dt = dt_
        # print(f"\tdebug: x_dot0.norm() = {x_dot0.norm()}")
        # print(f"\tdebug: x_dot0.norm() = {total_time.item():0.6f}")
        # solve the geodesic ODE iteratively
        # for i, t_i in enumerate(t):

        dt_ = min(total_time / num_iter, init_dt)
        dt = dt_
        if verbose >= 1:
            print(f"initial dt = {dt_:0.6f}, total_expected_iter = {total_time / dt_:1.0f}")

        if verbose == 1:
            print("Progress-bar\n0%[--------------------]100%")
            print("0%[", end="")
        current_time = 0
        iter = 0
        cnt = 0
        total_dq = 0
        while total_time > current_time:
            # do_refine = i % check_dot_every == 0
            do_refine = False
            x_new, x_dot_new = one_step(x, x_dot, q_type=q_type, dt=dt, wrapper=self, refine_xdot=do_refine, verbose=verbose >= 3)
            current_time += dt

            # calculate dq
            q_new = self.pos_to_dist(x_new.reshape(-1, 3))[1]
            dq = (q_new - q).norm()
            total_dq += dq
            q = q_new
            if verbose >= 2:
                if iter % 25 == 0:
                    print(f"\tdebug: time = ({(current_time / total_time) * 100:0.4f}%), iter = {iter}, dt = {dt:0.6f}, dq = {dq:0.6f}")
            iter += 1

            x = x_new
            x_dot = x_dot_new
            # dt = x_dot.norm() * dt_
            dt = min(max(dt_, 1 / x_dot.norm() * dt_), max_dt)
            if current_time / total_time > cnt / 10 and verbose == 1:
                print("--", end="")
                cnt += 1
            if total_time - current_time < dt:
                dt = total_time - current_time
        # for i, t_i in enumerate(t):
        #     print(f"\tdebug: time = {t_i:0.4f} ({i/len(t) * 100:0.2f}%)")
        #     # do_refine = i % check_dot_every == 0
        #     do_refine = False
        #     x, x_dot = one_step(x, x_dot, q_type=q_type, dt=dt, wrapper=self, refine_xdot=do_refine)
        if verbose == 1:
            print("]100%")
        return x.reshape(-1, 3), iter, total_dq

    def svd(self, jacob, verbose=False):
        U, S, Vh = torch.linalg.svd(jacob)
        num_zeros = (S < self.svd_tol).sum()
        dim = len(S) - num_zeros
        S = S[:dim]
        U = U[:, :dim]
        Vh = Vh[:dim, :]
        if verbose:
            print(f"\t\t\tdebug: dim = {dim}, num_zeros = {num_zeros}, singular values = {S[-1].item():0.6f} ~ {S[0].item():0.6f}")
        return U, S, Vh

    def refine_jacobian(self, jacob):
        # find non-zero singular values
        U, S, Vh = self.svd(jacob)
        J = U @ torch.diag(S) @ Vh
        J_inv = Vh.T @ torch.diag(1 / S) @ U.T
        return J, J_inv


def rmsd(x):
    return (x ** 2).sum(-1).mean().sqrt()


def solve_ode(atoms_T, atoms_0, idx, initial="random", sigma=1e-1, alpha=1.4, beta=0.6, check_singular_values=False, sv_cutoff=1e-2, init_dt=5e-3, max_dt=5e-2, verbose=1):
    # load molecules
    q_type = "morse"
    xT = atoms_T[idx]
    file_name = list(xT.info.keys())[0].split("/")[-1].split(".")[0]
    gt_idx = int(file_name[2:])
    x0 = atoms_0[gt_idx]

    # load wrapper
    gamma = 0.0

    print(f"Debug: alpha, beta = {alpha}, {beta}")
    wrapper = Wrapper(x0, xT, q_type=q_type, alpha=alpha, beta=beta, gamma=gamma, using_jacobian=True, svd_tol=1e-10)

    x0 = torch.Tensor(x0.positions)
    xT = torch.Tensor(xT.positions)

    jacob = wrapper.calc_jacobian(xT, q_type=q_type).T
    hess = wrapper.calc_hessian(xT, q_type=q_type)
    U, S, Vh = torch.linalg.svd(jacob)
    U_list = [U[:, i] for i in range(len(S) - 6)]
    V_list = [Vh[i] for i in range(len(S) - 6)]

    J_inv = torch.linalg.pinv(jacob, rtol=1e-4, atol=1e-4)

    if check_singular_values:
        print("index\t\t x_ddot\t\t x_dot\t\t s\t\t u\t\t v")
        for idx in range(len(S) - 6):
            coeff = 1e-1
            v = V_list[idx] * coeff
            u = U_list[idx] * coeff

            x_dot = J_inv @ u
            s = S[idx]

            # x_dot = v
            christoffel = torch.einsum("mij, mk->kij", hess, J_inv.T)
            x_ddot = - torch.einsum("j,kij,i->k", v, christoffel, v)
            print(f"idx = {idx} :\t {x_ddot.norm():0.4f},\t {x_dot.norm():0.4f},\t {s:0.4f},\t {u.norm():0.4f},\t {v.norm():0.4f}")

    q0 = wrapper.pos_to_dist(x0)[1]
    qT = wrapper.pos_to_dist(xT)[1]

    if initial == "random":
        q_dot0 = jacob @ J_inv @ torch.randn(q0.size()) * sigma
    elif initial == "LSV":  # largest singular value
        q_dot0 = U_list[0] * sigma
        print(f"associated singular value : {S[0]:0.4f}")
    elif initial == "SSV":  # smallest singular value
        mask = S > sv_cutoff
        U = U[:, :len(S)]
        S = S[mask]
        U = U[:, mask]
        u = U[:, -1]
        q_dot0 = u * sigma
        print(f"associated singular value : {S[-1]:0.4f}")
    elif initial == "toward_zero":
        q_dot0 = jacob @ J_inv @ (q0 - qT)
    else:
        raise NotImplementedError

    print(f"initial q_dot0 norm = {q_dot0.norm():0.4f}")
    x, total_iter, total_dq = wrapper.exponential_ode_solver(xT, q_dot0, num_iter=10, check_dot_every=100000, init_dt=init_dt, max_dt=max_dt, verbose=verbose)
    q = wrapper.pos_to_dist(x)[1]
    print(f"initial q_dot0 norm = {q_dot0.norm():0.4f}")
    print(f"Total iteration : {total_iter}")
    print("\t\t\tx vs xT, \tx vs x0, \tx0 vs xT")
    print(f"RMSD : \t\t{rmsd(x - xT):0.6f}, \t{rmsd(x0 - x):0.6f}, \t{rmsd(x0 - xT):0.6f}")
    print(f"delta Q-norm : \t{(q - qT).norm():0.6f}, \t{(q0 - q).norm():0.6f}, \t{(q0 - qT).norm():0.6f}")

    return q_dot0.norm(), total_dq


if __name__ == "__main__":
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True, nargs="+")
    parser.add_argument("--initial", type=str, default="SSV")
    parser.add_argument("--sigma", type=float, default=1e-1)
    parser.add_argument("--alpha", type=float, default=1.4)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--check_singular_values", action="store_true")
    parser.add_argument("--sv_cutoff", type=float, default=1e-2)
    parser.add_argument("--init_dt", type=float, default=1e-3)
    parser.add_argument("--max_dt", type=float, default=5e-2)
    parser.add_argument("--verbose", type=int, default=2)
    args = parser.parse_args()

    # prety print args
    print("--------------------------------------------------")
    for arg in vars(args):
        print(f"{arg} : {getattr(args, arg)}")

    # load molecules
    atoms_0 = list(ase.io.iread("wb97xd3/wb97xd3_rxn_ts.xyz"))
    atoms_T = list(ase.io.iread("pm7/pm7_rxn_ts.xyz"))

    for idx in range(args.idx[0], args.idx[1]):
        print("--------------------------------------------------")
        st = time.time()
        print(f"index : {idx}")
        dq_thm, dq_num = solve_ode(
            atoms_T, atoms_0, idx, initial="SSV", sigma=args.sigma, alpha=args.alpha, beta=args.beta,
            check_singular_values=args.check_singular_values, sv_cutoff=args.sv_cutoff, init_dt=args.init_dt, max_dt=args.max_dt,
            verbose=args.verbose)
        length_error = abs(dq_thm - dq_num) / dq_thm
        print(f"percent error : {length_error * 100:0.2f}%")
        et = time.time()
        print(f"Elapsed time : {et - st:0.4f}")
        print("--------------------------------------------------")
