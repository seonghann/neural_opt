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

class SamplingParams:
    def __init__(
        self,
        sampling_type,
        c=0.1,
        beta_std=0.125,
        sigma_max=1e-1,
        sigma_min=1e-7,
        sigma1_correction=1e-7,
    ):
        assert sampling_type in ["bell-shaped", "linear", "constant"]
        self.sampling_type = sampling_type
        print(f"Debug: sampling_type = {sampling_type}")

        if sampling_type in ["linear", "constant"]:
            # recommend c=0.1 for "linear", c=np.sqrt(0.1) for "constant"
            self.c = c
            self.sigma_min = sigma_min
            self.sigma_1 = self.sigma_square(1.0)
            self.sigma_0 = self.sigma_square(0.0)
            print(f"Debug: self.c = {self.c}")
            print("Debug: sigma_1, sigma_0 = ", self.sigma_1, self.sigma_0)
        elif sampling_type == "bell-shaped":
            self.sigma_linear_coef = 0.0
            self.beta_std = beta_std
            self.sigma_max = sigma_max
            if sigma_min is None:
                self.sigma_min = sigma_max * 5e-3
            else:
                self.sigma_min = sigma_min
            self.normalizer = 1 / (beta_std * np.sqrt(2 * np.pi))

            self.sigma_1 = self.sigma_square(1.0) + sigma1_correction
            self.sigma_0 = self.sigma_square(0.0)
            # print(self.sigma_1, self.sigma_0)
            print("Debug: std_beta, sigma_max, sigma_min = ", std_beta, sigma_max, sigma_min)
            print("Debug: sigma_1, sigma_0 = ", self.sigma_1, self.sigma_0)
        else:
            raise NotImplementedError
        return

    def beta(self, t):
        if self.sampling_type == "linear":
            b = self.c * 2 * t
        elif self.sampling_type == "constant":
            b = self.c**2 * torch.ones_like(t)
        elif self.sampling_type == "bell-shaped":
            if isinstance(t, torch.Tensor):
                b = torch.exp(-((t - 0.5) / self.beta_std) ** 2 / 2)
            else:
                b = np.exp(-((t - 0.5) / self.beta_std) ** 2 / 2)
            b = b * self.normalizer * self.sigma_max
            b += self.sigma_linear_coef
        else:
            raise NotImplementedError
        return b

    def sigma_square(self, t):
        if self.sampling_type == "linear":
            # s_sq = self.c * t**2 + 1e-4
            s_sq = self.c * t**2 + self.sigma_min
        elif self.sampling_type == "constant":
            # s_sq = self.c**2 * t + 1e-3
            s_sq = self.c**2 * t + self.sigma_min
        elif self.sampling_type == "bell-shaped":
            erf_scaler = self.sigma_max / 2 # / (self.beta_std * np.sqrt(8) * 2)
            if isinstance(t, torch.Tensor):
                s_sq = erf_scaler * (1 + torch.special.erf( (t - 0.5) / (np.sqrt(2) * self.beta_std) )) + self.sigma_min
                s_sq = s_sq - erf_scaler * (1 + scipy.special.erf( (0 - 0.5) / (np.sqrt(2) * self.beta_std) )) + self.sigma_min
                s_sq += self.sigma_linear_coef * t
            else:
                s_sq = erf_scaler * (1 + scipy.special.erf( (t - 0.5) / (np.sqrt(2) * self.beta_std) )) + self.sigma_min
                s_sq = s_sq - erf_scaler * (1 + scipy.special.erf( (0 - 0.5) / (np.sqrt(2) * self.beta_std) )) + self.sigma_min
                s_sq += self.sigma_linear_coef * t
        else:
            raise NotImplementedError
        return s_sq

    def SNR(self, t):
        return self.sigma_square(t) / self.sigma_1


## wrapper class to save atoms object

class Wrapper:
    def __init__(self, atoms_0, atoms_T, q_type="DM", alpha=1.7, beta=0.01, gamma=0.01, using_jacobian=True, svd_tol=1e-4):
        self.atoms_0 = atoms_0
        self.atoms_T = atoms_T
        # assert q_type in ["DM", "morse"]
        self.q_type = q_type

        self.re = torch.Tensor(self.get_re(atoms_T))
        self.svd_tol = svd_tol
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
                x_tm1 = self.exponential_ode_solver(x_t, -dd, q_type=self.q_type, num_iter=inner_iteration, check_dot_every=5)
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

    def exponential_ode_solver(self, x0, q_dot0, q_type="morse", num_iter=100, check_dot_every=10):

        t = torch.linspace(0, 1, num_iter + 1)[:-1]
        dt = t[1] - t[0]

        def one_step(x, x_dot, q_type=q_type, dt=dt, wrapper=self, refine_xdot=False):
            hess = wrapper.calc_hessian(x.reshape(-1, 3), q_type=q_type)
            jacob = wrapper.calc_jacobian(x.reshape(-1, 3), q_type=q_type).T

            J, J_inv = wrapper.refine_jacobian(jacob)
            JG = J_inv.T

            if refine_xdot:
                x_dot = J_inv @ J @ x_dot

            christoffel = torch.einsum("mij, mk->kij", hess, JG)
            x_ddot = - torch.einsum("j,kij,i->k", x_dot, christoffel, x_dot)

            # x_ddot and x_dot should be perpendicular (for verbose)
            q_ddot = J @ x_ddot
            q_dot = J @ x_dot

            # remove q_dot component from q_ddot
            # q_ddot -= (q_ddot * q_dot).sum() * q_dot / (q_dot * q_dot).sum()
            # x_ddot = J_inv @ q_ddot

            new_x = x + x_dot * dt
            new_x_dot = x_dot + x_ddot * dt

            # dotproduct
            print(f"\t\tdebug: <x_ddot, x_dot> = {(q_ddot * q_dot).sum()}")
            print(f"\t\tdebug: dx norm = {(new_x - x).norm():0.4f}, dx_dot norm = {(new_x_dot - x_dot).norm():0.4f}")
            return new_x, new_x_dot

        jacob = self.calc_jacobian(x0, q_type=q_type)
        J, J_inv = self.refine_jacobian(jacob)

        # debugging
        proj_q_dot = J.T @ J_inv.T @ q_dot0
        print(f"\tdebug: proj_q_dot norm = {(proj_q_dot - q_dot0).norm()/ q_dot0.norm():0.4f}")

        x_dot0 = J_inv.T @ q_dot0
        x = x0.flatten()
        x_dot = x_dot0
        for i, t_i in enumerate(t):
            do_refine = i % check_dot_every == 0
            x, x_dot = one_step(x, x_dot, q_type=q_type, dt=dt, wrapper=self, refine_xdot=do_refine)
        return x.reshape(-1, 3)

    def svd(self, jacob):
        U, S, Vh = torch.linalg.svd(jacob)
        num_zeros = (S < self.svd_tol).sum()
        dim = len(S) - num_zeros
        S = S[:dim]
        U = U[:, :dim]
        Vh = Vh[:dim, :]
        print(f"\t\t\tdebug: dim = {dim}, num_zeros = {num_zeros}")
        return U, S, Vh

    def refine_jacobian(self, jacob):
        # find non-zero singular values
        U, S, Vh = self.svd(jacob)    
        J = U @ torch.diag(S) @ Vh
        J_inv = Vh.T @ torch.diag(1 / S) @ U.T
        return J, J_inv


if __name__ == "__main__":

    def experiment1(
        idx,
        sampling_type,
        coord_type,
        h_coeff=0.0,
        num_time_steps=200,
        verbose=False,
        plot=True,
        q_type="morse",
        sampling_test=1,
        atoms_0=atoms_0,
        atoms_T=atoms_T,
        inner_iteration=3,
        outer_iteration=None,
    ):
        set_seed(0)

        # find matched index
        xT = atoms_T[idx]
        file_name = list(xT.info.keys())[0].split("/")[-1].split(".")[0]
        gt_idx = int(file_name[2:])
        x0 = atoms_0[gt_idx]

        print(f"idx, gt_idx: {idx}, {gt_idx}")
        gamma = 0.0
        alpha, beta = 1.7, 0.01

        print(f"Debug: alpha, beta = {alpha}, {beta}")
        wrapper = Wrapper(x0, xT, q_type=q_type, alpha=alpha, beta=beta, gamma=gamma, using_jacobian=True, svd_tol=svd_tol)

        pos0 = torch.Tensor(x0.get_positions())
        posT = torch.Tensor(xT.get_positions())

        margin = 0.001
        t = torch.linspace(0 + margin, 0.9, num_time_steps + 1)[:-1]
        dt = (t[1:] - t[:-1]).mean()

        torch.set_printoptions(precision=6, sci_mode=False)
        reverse_traj = [posT]
        x = posT
        v1s = []
        v2s = []
        v3s = []
        v4s = []

        if verbose:
            print(f"(Debug) mu_d - d_t\t mu_d - d_T\t mu_x - x_t\t mu_x - x_T\t score")
        for idx, i in enumerate(torch.flip(t, dims=(0,))):
            if sampling_type == "ode":
                x, v1, v2, v3, v4 = wrapper.reverse_ode_process(x, i, dt, params, pos0, posT, coord=coord_type, h_coeff=h_coeff, verbose=verbose, sampling_test=sampling_test, inner_iteration=inner_iteration)
            else: # sde
                x, v1, v2, v3, v4 = wrapper.reverse_diffusion_process(x, i, dt, params, pos0, posT, coord=coord_type, h_coeff=h_coeff, verbose=verbose, sampling_test=sampling_test, inner_iteration=inner_iteration)

            v1s.append(v1); v2s.append(v2); v3s.append(v3); v4s.append(v4)
            reverse_traj.append(x)

            if idx == outer_iteration: break

    return reverse_traj, v1s, v2s, v3s, v4s

    def plot_traj(v, label="", ylabel=""): # r"||$\hat{d}_t - d_t$||"
        plt.plot(v, label=label)
        plt.xlabel(r"time step", fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.legend()
        return

    sampling_type = ["bell-shaped", "linear", "constant"][0]

    if sampling_type in ["linear", "constant"]:
        # for linear and square sampling
        c = 0.1
        sigma_min = 1e-7
        sigma_max = 0.1
        std_beta = None
        sigma1_correction = None
    else:
        # bell-shaped sampling
        c = 0.1
        # std_beta = 0.125
        std_beta = 0.125
        # sigma_max = 0.10
        sigma_max = 0.01
        sigma_min = 1e-6
        sigma1_correction = 1e-8


    params = SamplingParams(
        sampling_type=sampling_type,
        c=c,
        beta_std=std_beta,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        sigma1_correction=sigma1_correction,
        )


    """
    sampling_type: "sde", "ode", "hybrid"
    coord_type: "Cartesian", "Distance"
    """

    num_time_steps = [100, 250, 500, 1000][1]
    num_time_steps = 40
    h_coeff = 0.0
    verbose = [True, False][1]
    q_type = ["DM", "morse"][1]
    sampling_test = [0, 1][1]
    print(f"q_type = {q_type}")

    for idx in range(50):
        err_at_zero_t = []
        reverse_traj, v1, v2, v3, v4 = experiment1(idx, "sde", "Distance", h_coeff, num_time_steps, verbose=verbose, q_type=q_type, sampling_test=sampling_test)
        err_at_zero_t.append(v2[-1].item())
        plot_traj(v2[::-1], "sde")

        reverse_traj, v1, v2, v3, v4 = experiment1(idx, "ode", "Distance", h_coeff, num_time_steps, 
                                                verbose=verbose, q_type=q_type, sampling_test=sampling_test, 
                                                inner_iteration=200, outer_iteration=250)
        err_at_zero_t.append(v2[-1].item())
        plot_traj(v2[::-1], "ode-acc")
        plot_traj(v1[::-1], "ode-loss")

        plt.ylim(0, v2[num_time_steps // 2] * 1.2)
        plt.title(f"rxn idx: {idx}")
        plt.legend()
        plt.savefig(f"figs/fig-{idx}.png")
        print("err_at_zero_t = ", err_at_zero_t)
    print("Done!")
