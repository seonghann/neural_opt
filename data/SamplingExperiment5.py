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
from ase.io import write

from torch_scatter import scatter_add

from lst_interp import *
import tqdm

import matplotlib.pyplot as plt
# import seaborn as sns

import sys
sys.path.append("/home/share/DATA/NeuralOpt/Interpolations/Geodesic_interp")
from get_geodesic_energy import morse_scaler, ATOMIC_RADIUS

sys.path.append("../sampling")
from sampling import SamplingParams

sys.path.append("../utils")
from utils import set_seed, timer

## Experiment setting : random seed, precision, device
set_seed(0)
# torch.set_default_dtype(torch.float32)  # TODO: now this doesn't work.
torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)
torch.set_printoptions(linewidth=300)


## wrapper class to save atoms object
class Wrapper:
    def __init__(
        self,
        atoms_0,
        atoms_T,
        q_type="DM",
        alpha=1.7,
        beta=0.01,
        gamma=0.01,
        using_jacobian=True,
        svd_tol=1e-4,
        noise_svd_tol=1e-2,
        verbose=False,
    ):
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
        self.verbose = verbose
        self.noise_svd_tol = noise_svd_tol
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

    # @timer
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
    
    # @timer
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
    
    # @timer
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

            elif sampling_test == 2:
                reverse_score_ = diff * coeff
                # dw = torch.sqrt(beta_t * dt) * torch.randn_like(diff)
                # dd = - 1.0 * reverse_score_ * dt + dw
                dd = - 1.0 * reverse_score_ * dt
                x_tm1 = self.exponential_ode_solver(x_t, -dd, q_type=self.q_type, num_iter=inner_iteration, check_dot_every=3)
                dx = - x_tm1 + x_t

                ## Add noise
#                 # dw = torch.sqrt(beta_t * dt) * torch.randn_like(diff)
#                 dw = torch.sqrt(beta_t * dt) * torch.randn_like(dx)  # random noise in Cartensian coordinates
#                 print(f"Debug: Cartensian noise={dw}")

                jacob = self.calc_jacobian(x_t, q_type=self.q_type).T
                # U, S, Vh = self.svd(jacob)
                
                U, S, Vh = torch.svd(jacob)
                # num_zeros = (S < 1e-1).sum()
                num_zeros = (S < self.noise_svd_tol).sum()
                dim = len(S) - num_zeros
                S = S[:dim]
                U = U[:, :dim]
                Vh = Vh[:dim, :]
        
                # J = U @ torch.diag(S) @ Vh
                J_inv = Vh.T @ torch.diag(1 / S) @ U.T
                dw_q = U @ torch.randn(size=(U.shape[1],))
                dw_q *= torch.sqrt(beta_t * dt)
                dw_x = J_inv @ dw_q
                dw_x = dw_x.reshape(-1, 3)
                dw = dw_x
                # print(f"Debug: U.T @ U \n={U.T @ U}")
#                 print(f"Debug: x_t.shape={x_t.shape}")
#                 print(f"Debug: dw_x=\n{dw_x}")
#                 raise ValueError
                
                dx += dw

            elif sampling_test == 3:
                reverse_score_ = diff * coeff
#                 dw = torch.sqrt(beta_t * dt) * torch.randn_like(diff)
                
                jacob = self.calc_jacobian(x_t, q_type=self.q_type).T
                U, S, Vh = torch.svd(jacob)
                num_zeros = (S < self.noise_svd_tol).sum()
                # num_zeros = (S < 1e-10).sum()
                dim = len(S) - num_zeros
                S = S[:dim]
                U = U[:, :dim]
                Vh = Vh[:dim, :]
                J_inv = Vh.T @ torch.diag(1 / S) @ U.T
                dw = U @ torch.randn(size=(U.shape[1],))
                dw *= torch.sqrt(beta_t * dt) 

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
                if self.verbose:
                    print(f"Debug ({t:0.3f}): \n\t1) diff-d norm and diff-x norm {diff_d.norm():0.4f}, {diff_x.norm():0.4f} \n\t2) dd-norm and dx-norm {dd.norm():0.6f}, {dx.norm():0.6f}")
                    print(f"\t3) dx-norm/dd-norm {dx.norm()/dd.norm():0.6f}")
                
            # elif sampling_test == 1:
            # elif sampling_test == 1 or sampling_test == 2:
            else:
                reverse_score_ = diff * coeff
                dd = - 0.5 * reverse_score_ * dt
                if self.verbose:
                    print(f"debug ] time : {t:0.3f}")
                    print(f"debug ] diff.norm() : {diff.norm()}")
                    print(f"debug ] dd.norm() : {dd.norm()}")
                x_tm1 = self.exponential_ode_solver(x_t, -dd, q_type=self.q_type, num_iter=inner_iteration, check_dot_every=3)
                dx = - x_tm1 + x_t
                if self.verbose:
                    print(f"debug ] dx.norm() : {dx.norm()}")
                
        x_tm1 = x_t - dx
        return x_tm1, v1, v2, v3, v4

    def reverse_score(self, x_t, t, params, x_0, x_T, verbose=True):
        # calculate parameters
        beta_t = params.beta(t)
        sigma_t_square = params.sigma_square(t)
        # sigma_T_square = params.sigma_1

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

    # @timer
    def reverse_score2(self, x_t, t, params, x_0, x_T, verbose=True):
        # calculate parameters
        beta_t = params.beta(t)
        sigma_t_square = params.sigma_square(t)
        # sigma_T_square = params.sigma_1

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
    
    # @timer
    def exponential_ode_solver(self, x0, q_dot0, q_type="morse", num_iter=100, check_dot_every=10):
    # def _exponential_ode_solver(self, x0, q_dot0, q_type="morse", num_iter=100, check_dot_every=10):
        
        t = torch.linspace(0, 1, num_iter + 1)[:-1]
        dt = t[1] - t[0]
#         t = torch.linspace(0, 1, num_iter)
#         dt = 1 / num_iter
        
        def one_step(x, x_dot, q_type=q_type, dt=dt, wrapper=self, refine_xdot=False):
            hess = wrapper.calc_hessian(x.reshape(-1, 3), q_type=q_type)
            jacob = wrapper.calc_jacobian(x.reshape(-1, 3), q_type=q_type).T
            
            J, J_inv = wrapper.refine_jacobian(jacob)
            JG = J_inv.T
            
            if refine_xdot:
                x_dot = J_inv @ J @ x_dot
            
            christoffel = torch.einsum("mij, mk->kij", hess, JG)
            x_ddot = - torch.einsum("j,kij,i->k", x_dot, christoffel, x_dot)
            # x_ddot and x_dot should be perpendicular 
            q_ddot = J @ x_ddot
            q_dot = J @ x_dot
            # remove q_dot component from q_ddot
            # q_ddot -= (q_ddot * q_dot).sum() * q_dot / (q_dot * q_dot).sum()
            # x_ddot = J_inv @ q_ddot
            
            new_x = x + x_dot * dt
            new_x_dot = x_dot + x_ddot * dt
            
            # dotproduct
            if self.verbose:
                print(f"\t\tdebug: <x_ddot, x_dot> = {(q_ddot * q_dot).sum()}")
                print(f"\t\tdebug: dx norm = {(new_x - x).norm():0.4f}, dx_dot norm = {(new_x_dot - x_dot).norm():0.4f}")
            return new_x, new_x_dot
        
        jacob = self.calc_jacobian(x0, q_type=q_type).T
        J, J_inv = self.refine_jacobian(jacob)
        
        # debugging
        if self.verbose:
            proj_q_dot = J @ J_inv @ q_dot0
            print(f"\tdebug: proj_q_dot norm = {(proj_q_dot - q_dot0).norm()/ q_dot0.norm():0.4f}")
        
        # initialization
        x_dot0 = J_inv @ q_dot0
        if self.verbose:
        # if True:
            # Debugging: q_dot0, x_dot0 크기 비교.
            print(f"Debug: x_dot0/q_dot0={x_dot0.norm()/q_dot0.norm()}, x_dot0.norm()={x_dot0.norm()}, q_dot0.norm()={q_dot0.norm()}")
        x = x0.flatten()
        x_dot = x_dot0
        
        # solve the geodesic ODE iteratively
        for i, t_i in enumerate(t):
            do_refine = i % check_dot_every == 0
            x, x_dot = one_step(x, x_dot, q_type=q_type, dt=dt, wrapper=self, refine_xdot=do_refine)
        return x.reshape(-1, 3)
    
    # def exponential_ode_solver(self, x0, q_dot0, q_type="morse", num_iter=100, check_dot_every=10, thresh=1e-2, max_dt=1e-1, verbose=False):
    def _exponential_ode_solver(self, x0, q_dot0, q_type="morse", num_iter=100, check_dot_every=10, thresh=1e-2, max_dt=1e-1, verbose=False):
        
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
        # thresh = 5e-2
        # thresh = 1e-1
        # if total_time > num_iter * thresh:
        #     num_iter = int(total_time / thresh)
        
        # t = torch.linspace(0, total_time, num_iter + 1)[:-1]
        # # t = torch.linspace(0, 1, num_iter + 1)[:-1]
        # dt_ = t[1] - t[0]
        # dt = dt_
        # print(f"\tdebug: x_dot0.norm() = {x_dot0.norm()}")
        # print(f"\tdebug: x_dot0.norm() = {total_time.item():0.6f}")
        # solve the geodesic ODE iteratively
        # for i, t_i in enumerate(t):
        
        dt_ = min(total_time / num_iter, thresh)
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
                print(f"--", end="")
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
#         return x.reshape(-1, 3), iter, total_dq
        return x.reshape(-1, 3)
    
    def svd(self, jacob):
        U, S, Vh = torch.linalg.svd(jacob)
        if self.verbose:
        # if True:
            print(f"Debug: Singular values = {S}")
        num_zeros = (S < self.svd_tol).sum()
        dim = len(S) - num_zeros
        S = S[:dim]
        U = U[:, :dim]
        Vh = Vh[:dim, :]
        if self.verbose:
        # if True:
            print(f"\t\t\tdebug: dim = {dim}, num_zeros = {num_zeros}")
        return U, S, Vh

    def refine_jacobian(self, jacob):
        # find non-zero singular values
        U, S, Vh = self.svd(jacob)
        J = U @ torch.diag(S) @ Vh
        J_inv = Vh.T @ torch.diag(1 / S) @ U.T
        return J, J_inv



def find_matched_index(rxn_indices, json_file="/home/share/DATA/NeuralOpt/SQM_data/pm7/final.20240103.json"):
    """
    Find matched indices of PM7 data matched input rxn indices. 
    Return matched PM7 indices and reaction indices.
    """
    import pandas as pd

    df = pd.read_json(path_or_buf=json_file, orient="records")
    df_indices = np.array(df.index.tolist())
    df_rxn_indices = np.array(df["rxn index"].values)

    is_matched = [idx in rxn_indices for idx in df_rxn_indices]
    matched_rxn_indices = df_rxn_indices[is_matched]
    matched_pm7_indices = df_indices[is_matched]
    return matched_pm7_indices, matched_rxn_indices


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
    inner_iteration=3,
    outer_iteration=None,
    svd_tol=1e-4,
    noise_svd_tol=1e-2,
    alpha=1.7,
    beta=0.01,
    params=None,
    atoms_0=None,
    atoms_T=None,
    save_traj=False,
):
    set_seed(0)

    # find matched index
    xT = atoms_T[idx]
    file_name = list(xT.info.keys())[0].split("/")[-1].split(".")[0]
    gt_idx = int(file_name[2:])
    x0 = atoms_0[gt_idx]
    
    print(f"idx, gt_idx: {idx}, {gt_idx}")
    gamma = 0.0
    
    print(f"Debug: alpha, beta = {alpha}, {beta}")
    wrapper = Wrapper(x0, xT, q_type=q_type, alpha=alpha, beta=beta, gamma=gamma, using_jacobian=True, svd_tol=svd_tol, verbose=verbose, noise_svd_tol=noise_svd_tol)

    pos0 = torch.Tensor(x0.get_positions())
    posT = torch.Tensor(xT.get_positions())

    ## TEST
    noise = torch.randn_like(posT) * 0.03
    posT += noise; print(f"TEST: add noise to posT!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"noise=\n{noise}")

    margin = 0.001
    # t = torch.linspace(0 + margin, 1 - 10 * margin, num_time_steps + 1)[:-1]
    t = torch.linspace(0 + margin, 0.9, num_time_steps + 1)[:-1]
    dt = (t[1:] - t[:-1]).mean()
    print(f"t-sampilng range: [{t[0], t[-1]}], dt={dt}")

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
    # for idx, i in tqdm.tqdm(enumerate(torch.flip(t, dims=(0,))), total=len(t)):
        if sampling_type == "ode":
            x, v1, v2, v3, v4 = wrapper.reverse_ode_process(x, i, dt, params, pos0, posT, coord=coord_type, h_coeff=h_coeff, verbose=verbose, sampling_test=sampling_test, inner_iteration=inner_iteration)
        else: # sde
            x, v1, v2, v3, v4 = wrapper.reverse_diffusion_process(x, i, dt, params, pos0, posT, coord=coord_type, h_coeff=h_coeff, verbose=verbose, sampling_test=sampling_test, inner_iteration=inner_iteration)

        v1s.append(v1); v2s.append(v2); v3s.append(v3); v4s.append(v4)
        reverse_traj.append(x)
        
        if idx == outer_iteration: break


    if save_traj:
        save_filename = f"{sampling_type}_{gt_idx}.xyz"
        print(f"Save {sampling_type} traj: {save_filename}")
        atoms_traj = [x0.copy() for i in range(len(reverse_traj))]
        for i in range(len(reverse_traj)):
            atoms_traj[i].set_positions(reverse_traj[i])
        atoms_traj[0].write(save_filename, append=False)
        for atoms in atoms_traj[1:]:
            atoms.write(save_filename, append=True)


    return reverse_traj, v1s, v2s, v3s, v4s


def plot_traj(v, label="", ylabel=""): # r"||$\hat{d}_t - d_t$||"
    plt.plot(v, label=label)
    plt.xlabel(r"time step", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend()
    return


if __name__ == "__main__":
    show_figure = False
    show_figure = True
    print(f"* show_figure: {show_figure}")
    save_traj = False
    save_traj = True
    print(f"* save_traj: {save_traj}")

    X_T_type = ["PM7", "geodesic"][0]


    ## Load TS geometries
    atoms_0 = list(ase.io.iread("wb97xd3/wb97xd3_rxn_ts.xyz")); print("Load TS(wb97xd3) as X_0")
    if X_T_type == "PM7":
        atoms_T = list(ase.io.iread("pm7/pm7_rxn_ts.xyz")); print("Load TS(PM7) as X_T")
    elif X_T_type == "geodesic":
        # atoms_T = list(ase.io.iread("wb97xd3_geodesic/wb97xd3_geodesic_rxn_ts.xyz")); print("Load TS(geodesic) as X_T")  ## (alpha, beta)=(1.7, 0.01)
        atoms_T = list(ase.io.iread("/home/share/DATA/NeuralOpt/SQM_data/wb97xd3_geodesic/wb97xd3_geodesic_rxn_ts.xyz")); print("Load TS(geodesic) as X_T")  ## (alpha, beta)=(1.6, 2.3)
    else:
        raise NotImplementedError


    ## Set sampling parameters
    # sampling_type = ["bell-shaped", "monomial"][1]
    sampling_type = ["bell-shaped", "monomial"][0]
    print(f"* sampling_type : {sampling_type}")

    if sampling_type == "monomial":
        std_beta = None
        sigma_max = 0.01
        sigma_min = 5 * 1e-5
        sigma_min = 1e-6
        order = 2
    elif sampling_type == "bell-shaped":
        std_beta = 0.125
        sigma_max = 0.001
        sigma_min = 1e-6 * 2
        order = None
    else:
        raise NotImplementedError

    params = SamplingParams(
        sampling_type=sampling_type,
        beta_std=std_beta,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        order=order,
        )


    """
    sampling_type: "sde", "ode"
    coord_type: "Distance", "Morse"
    """
    num_time_steps = 20
    num_time_steps = 40
    inner_iteration = 3
    h_coeff = 0.0
    verbose = [True, False][1]
    q_type = ["DM", "morse"][1]
    sampling_test = [0, 1][1]
    sampling_test = 3
    print(f"sampling_test={sampling_test}")
    print(f"q_type = {q_type}")

    # svd_tol = 1e-4
    # noise_svd_tol = 1e-4
    # svd_tol = 1e-3
    # noise_svd_tol = 1e-3
    # svd_tol = 1e-2
    # noise_svd_tol = 1e-2
    svd_tol = 1e-10
    noise_svd_tol = 1e-10
    print(f"num_time_steps, inner_iteration, svd_tol, noise_svd_tol: {num_time_steps}, {inner_iteration}, {svd_tol}, {noise_svd_tol}")
    # alpha, beta = 1.7, 0.01
    alpha, beta = 1.6, 2.3


    err_dct = dict()


    import pickle
    # valid_indices = pickle.load(open("/home/share/DATA/NeuralOpt/data/data_split.pkl", "rb"))["valid_index"]; print("Load valid_index")
    valid_indices = pickle.load(open("/home/share/DATA/NeuralOpt/data/data_split.pkl", "rb"))["test_index"]; print("Load test_index")
    # valid_indices = [43]
    # valid_indices = [1169,2471,4061,4649,5147,5590,5764,8701,8702]
    # valid_indices = [5764, 5804]
    valid_indices = [1169,2471,4061 ,4649 ,5147 ,5590 ,5764 ,8701 ,8702]
    len(f"len(valid_indices): {valid_indices}")
    print(f"Debug: valid_indices={valid_indices}")

    if X_T_type == "PM7":
        valid_indices = find_matched_index(valid_indices)[0]

    # for idx in range(50):
    # for idx in range(3):
    # for idx in [13,18,19,40,41,42,44]:
    # for idx in [13,18]:
    for idx in valid_indices:
        err_at_zero_t = []
        reverse_traj, v1, v2, v3, v4 = experiment1(idx, "sde", "Distance", h_coeff, num_time_steps, 
                                                   verbose=verbose, q_type=q_type, sampling_test=sampling_test,
                                                   inner_iteration=inner_iteration, outer_iteration=250, svd_tol=svd_tol, noise_svd_tol=noise_svd_tol,
                                                   alpha=alpha, beta=beta, params=params, atoms_0=atoms_0, atoms_T=atoms_T, save_traj=save_traj,)
        err_at_zero_t.append(v2[-1].item())

        if show_figure:
            plot_traj(v2[::-1], "sde-acc")
            plot_traj(v1[::-1], "sde-loss")

        reverse_traj, v1, v2, v3, v4 = experiment1(idx, "ode", "Distance", h_coeff, num_time_steps, 
                                                   verbose=verbose, q_type=q_type, sampling_test=sampling_test, 
                                                   inner_iteration=inner_iteration, outer_iteration=250, svd_tol=svd_tol, noise_svd_tol=noise_svd_tol,
                                                   alpha=alpha, beta=beta, params=params, atoms_0=atoms_0, atoms_T=atoms_T, save_traj=save_traj,)
        err_at_zero_t.append(v2[-1].item())

        if show_figure:
            plot_traj(v2[::-1], "ode-acc")
            plot_traj(v1[::-1], "ode-loss")

        print("err_at_zero_t = ", err_at_zero_t, flush=True)
        err_dct[idx] = err_at_zero_t
        
        if show_figure:
            plt.ylim(0,)
            plt.title(f"rxn idx: {idx}")
            plt.legend()
            plt.show()
    print("Done!")
    print(err_dct)


    ## Save err_dct
    import json
    # 딕셔너리를 JSON 형식으로 파일로 저장
    # save_filename = 'err_dict_20240119.json'
    # save_filename = 'err_dict_20240121.json'
    save_filename = 'tmp.json'
    with open(save_filename, 'w') as file:
        json.dump(err_dct, file)
        print(f"Save '{save_filename}'.")
