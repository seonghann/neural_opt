import ase.io
import ase
import sys
import numpy as np
import os
import torch
from scipy.spatial.distance import pdist
import copy
from torch_scatter import scatter_add


def eq_transform(score_d, pos, edge_index, edge_length):
    N = pos.size(0)
    dd_dr = (1.0 / edge_length).reshape(-1, 1) * (pos[edge_index[0]] - pos[edge_index[1]])
    score_d = score_d.reshape(-1, 1)
    score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N)
    score_pos += scatter_add(-dd_dr * score_d, edge_index[1], dim=0, dim_size=N)
    return score_pos


def pos_to_dist(pos):
    length = torch.Tensor(pdist(pos))
    index = torch.LongTensor(np.stack(np.triu_indices(len(pos), 1)))
    return index, length


def pinned_score(x_t, t, betas, x_T):
    sigma_t_square = betas[:t].sum(dim=0)
    sigma_T_squre = betas.sum(dim=0)
    # return (x_T - x_t) / (sigma_T_squre - sigma_t_square)
    return (x_T - x_t) * 2 / (sigma_T_squre - sigma_t_square)
    # return (Proj_to_TPM(q_T, q_t)) * 2 / (sigma_T_squre - sigma_t_square)


def pinned_score2(x_t, t, betas, x_T):
    index, d_t = pos_to_dist(x_t)
    _, d_T = pos_to_dist(x_T)
    sigma_t_square = betas[:t].sum(dim=0)
    sigma_T_square = betas.sum(dim=0)
    score_d = (d_T - d_t) * 2 / (sigma_T_square - sigma_t_square)
    score_pos = eq_transform(score_d, x_t, index, d_t)
    return score_pos


def pinned_diffusion_proccess(x_t, t, betas, x_T, coord="Cartesian"):
    beta_t = betas[t]
    if coord == "Cartesian":
        dx = beta_t * pinned_score(x_t, t, betas, x_T) + torch.sqrt(beta_t) * torch.randn_like(x_t)
    else:
        dx = beta_t * pinned_score2(x_t, t, betas, x_T) + torch.sqrt(beta_t) * torch.randn_like(x_t)
    x_tp1 = x_t + dx
    return x_tp1


def diffusion_proccess(x_t, beta_t):
    dx = x_t + torch.sqrt(beta_t) * torch.randn_like(x_t)
    x_tp1 = x_t + dx
    return x_tp1


def diffusion_kernel(x0, t, beta_t):
    # t = 1, 2, 3, ... T
    x_t = x0 + torch.sqrt(beta_t[:t].sum()) * torch.randn_like(x0)
    return x_t


def sigmoidal(x):
    return 1.0 / (1.0 + torch.exp(-x))


def get_beta_schedule(min_t, max_t, num_steps, min_beta, max_beta):
    # num_steps = T
    # sigmoidal schedule
    t = torch.linspace(min_t, max_t, num_steps)
    beta_t = sigmoidal(t) * (max_beta - min_beta) + min_beta
    # beta_t = t * (max_beta - min_beta) + min_beta
    return beta_t


def reverse_diffusion_process(x_t, t, betas, x_0, x_T, coord="Cartesian", h_coeff=0.0):
    beta_t = betas[t]
    if coord == "Cartesian":
        dx = beta_t * (h_coeff * pinned_score(x_t, t, betas, x_T) - reverse_score(x_t, t, betas, x_0, x_T)) # + torch.sqrt(beta_t) * torch.randn_like(x_t)
    else:
        dx = beta_t * (h_coeff * pinned_score2(x_t, t, betas, x_T) - reverse_score2(x_t, t, betas, x_0, x_T)) # + torch.sqrt(beta_t) * torch.randn_like(x_t)
    print(f"Debug dx at time {t} : {dx.abs().mean()}")
    x_tm1 = x_t - dx
    return x_tm1


def reverse_ode_process(x_t, t, betas, x_0, x_T, coord="Cartesian", h_coeff=0.0):
    beta_t = betas[t]
    if coord == "Cartesian":
        dx = beta_t * (h_coeff * pinned_score(x_t, t, betas, x_T) - 0.5 * reverse_score(x_t, t, betas, x_0, x_T))
    else:
        dx = beta_t * (h_coeff * pinned_score2(x_t, t, betas, x_T) - 0.5 * reverse_score2(x_t, t, betas, x_0, x_T))
    print(f"(Debug) dx at time {t} : \n{dx.abs().mean()}")
    x_tm1 = x_t - dx
    return x_tm1


def reverse_score(x_t, t, betas, x_0, x_T):
    # sigma_t_square = betas[:t].sum(dim=0)
    # simga_T_square = betas.sum(dim=0)
    # SNR_t = 1/sigma_t_square
    # SNR_T = 1/sigma_T_square
    # mu_hat = (SNR_T/SNR_t) * (alpha_t/alpha_T) * x_T + alpha_t * x_0 * (1 - (SNR_T/SNR_t))
    # sigma_t_hat = (1 - (SNR_T/SNR_t)) * sigma_t_square
    sigma_t_square = betas[:t].sum(dim=0)
    sigma_T_square = betas.sum(dim=0)
    SNRTt = sigma_t_square / sigma_T_square
    mu_hat = SNRTt * x_T + x_0 * (1 - SNRTt)
    sigma_t_hat_square = sigma_t_square * (1 - SNRTt)
    score_pos = (mu_hat - x_t) * 2 / (sigma_t_hat_square)

    index, d_t = pos_to_dist(x_t)
    _, d_mu_hat = pos_to_dist(mu_hat)
    print(f"(Debug) dd : \n{d_mu_hat - d_t}")
    print(f"(Debug) dx : \n{mu_hat - x_t}")
    print(f"(Debug) sigma_t_hat_square at time {t} : \n{sigma_t_hat_square}")
    print(f"(Debug) score size (cart) : \n{score_pos}")
    return score_pos


def reverse_score2(x_t, t, betas, x_0, x_T):
    sigma_t_square = betas[:t].sum(dim=0)
    sigma_T_square = betas.sum(dim=0)
    SNRTt = sigma_t_square / sigma_T_square
    mu_hat = SNRTt * x_T + x_0 * (1 - SNRTt)
    # _, d_0 = pos_to_dist(x_0)
    # _, d_T = pos_to_dist(x_T)
    # d_mu_hat = d_T * SNRTt + d_0 * (1 - SNRTt)

    sigma_t_hat_square = sigma_t_square * (1 - SNRTt)

    index, d_t = pos_to_dist(x_t)
    _, d_mu_hat = pos_to_dist(mu_hat)
    score_d = (d_mu_hat - d_t) * 2 / (sigma_t_hat_square)
    score_pos = eq_transform(score_d, x_t, index, d_t)
    print(f"(Debug) dd : \n{d_mu_hat - d_t}")
    print(f"(Debug) dx : \n{mu_hat - x_t}")
    print(f"(Debug) sigma_t_hat_square at time {t} : \n{sigma_t_hat_square}")
    print(f"(Debug) score_d : \n{score_d}")
    print(f"(Debug) score_pos : \n{score_pos}")
    return score_pos


if __name__ == "__main__":
    atoms_0 = list(ase.io.iread("wb97xd3/wb97xd3_rxn_ts.xyz"))
    # atoms_T = list(ase.io.iread("pm7/pm7_rxn_ts.xyz"))
    atoms_T = list(ase.io.iread("pm7/pm7_rxn_r.xyz"))

    # define index of noisy TS and find associated index of ground truth TS
    idx = 0

    xT = atoms_T[idx]
    file_name = list(xT.info.keys())[0].split("/")[-1].split(".")[0]
    gt_idx = int(file_name[2:])
    x0 = atoms_0[gt_idx]

    pos0 = torch.Tensor(x0.get_positions())
    posT = torch.Tensor(xT.get_positions())

    betas = get_beta_schedule(-5, 5, 20, 1e-5, 1e-4)
    # betas = get_beta_schedule(0, 10, 100, 1e-5, 1e-2)
    alphas = (1.0 - betas).cumprod(dim=0)
    print(betas)
    sigma_square = betas.cumsum(dim=0)
    print(sigma_square)
    print(betas / sigma_square)
    SNRTt = sigma_square / sigma_square[-1]
    print(betas / sigma_square / (1 - SNRTt))

    x = pos0
    # forward traj with cartesian
    # forward_traj = [pos0]
    # for i, beta in enumerate(betas):
    #     # x = diffusion_proccess(x, beta)
    #     x = pinned_diffusion_proccess(x, i, betas, posT, coord="Cartesian")
    #     forward_traj.append(x)

    # if os.path.exists("forward_traj.xyz"):
    #     os.remove("forward_traj.xyz")
    # for trj, alpha in zip(forward_traj, alphas):
    #     # x0.set_positions(trj * alpha)
    #     x0.set_positions(trj)
    #     ase.io.write("forward_traj.xyz", x0, append=True)

    # # forwrd traj with distance
    # for i, beta in enumerate(betas):
    #     # x = diffusion_proccess(x, beta)
    #     x = pinned_diffusion_proccess(x, i, betas, posT, coord="Distance")
    #     forward_traj.append(x)

    # # if the xyz file exist, first delete it and rewrite it
    # if os.path.exists("forward_traj_distance.xyz"):
    #     os.remove("forward_traj_distance.xyz")
    # for trj, alpha in zip(forward_traj, alphas):
    #     # x0.set_positions(trj * alpha)
    #     x0.set_positions(trj)
    #     ase.io.write("forward_traj_distance.xyz", x0, append=True)

    h_coeff = 0.0
    if sys.argv[1] == "cartesian":
        print("with carteisian")
        reverse_traj = [posT]
        x = posT
        for i in range(len(betas))[::-1][:int(sys.argv[2])]:
            x = reverse_diffusion_process(x, i, betas, pos0, posT, coord="Cartesian", h_coeff=h_coeff)
            reverse_traj.append(x)

        if os.path.exists("reverse_traj.xyz"):
            os.remove("reverse_traj.xyz")
        for trj, alpha in zip(reverse_traj, alphas):
            # x1.set_positions(trj * alpha)
            xT.set_positions(trj)
            ase.io.write("reverse_traj.xyz", xT, append=True)

    elif sys.argv[1] == "distance":
        print("with distance")
        reverse_traj = [posT]
        x = posT
        for i in range(len(betas))[::-1][:int(sys.argv[2])]:
            x = reverse_diffusion_process(x, i, betas, pos0, posT, coord="Distance", h_coeff=h_coeff)
            reverse_traj.append(x)

        if os.path.exists("reverse_traj_distance.xyz"):
            os.remove("reverse_traj_distance.xyz")
        for trj, alpha in zip(reverse_traj, alphas):
            # x1.set_positions(trj * alpha)
            xT.set_positions(trj)
            ase.io.write("reverse_traj_distance.xyz", xT, append=True)


    # reverse_traj = [posT]
    # x = posT
    # for i in range(len(betas))[::-1]:
    #     x = reverse_ode_process(x, i, betas, pos0, posT, coord="Cartesian", h_coeff=h_coeff)
    #     reverse_traj.append(x)

    # if os.path.exists("reverse_ode_traj.xyz"):
    #     os.remove("reverse_ode_traj.xyz")
    # for trj, alpha in zip(reverse_traj, alphas):
    #     # x1.set_positions(trj * alpha)
    #     xT.set_positions(trj)
    #     ase.io.write("reverse_ode_traj.xyz", xT, append=True)

    # reverse_traj = [posT]
    # x = posT
    # for i in range(len(betas))[::-1]:
    #     x = reverse_ode_process(x, i, betas, pos0, posT, coord="Distance", h_coeff=h_coeff)
    #     reverse_traj.append(x)

    # if os.path.exists("reverse_ode_traj_distance.xyz"):
    #     os.remove("reverse_ode_traj_distance.xyz")
    # for trj, alpha in zip(reverse_traj, alphas):
    #     # x1.set_positions(trj * alpha)
    #     xT.set_positions(trj)
    #     ase.io.write("reverse_ode_traj_distance.xyz", xT, append=True)
