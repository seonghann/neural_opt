import omegaconf
import sys
sys.path.append("../")
import tqdm
from diffusion.noise_scheduler import load_noise_scheduler
from utils.geodesic_solver import GeodesicSolver
from utils.geodesic_solver_backup import GeodesicSolver as GeodesicSolver_backup
from dataset.data_module import GrambowDataModule

from torch_geometric.data import Batch
from utils.rxn_graph import RxnGraph, DynamicRxnGraph
import torch
import numpy as np
import random


def _masking(num_nodes):
    N = num_nodes.max()
    mask = torch.BoolTensor([True, False]).repeat(len(num_nodes)).to(num_nodes.device)
    num_repeats = torch.stack([num_nodes, N - num_nodes]).T.flatten()
    mask = mask.repeat_interleave(num_repeats)
    return mask


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def noise_level_sampling(data, noise_schedule=""):
    g_length = data.geodesic_length[:, 1:]  # (G, T-1)
    g_last_length = data.geodesic_length[:, -1:]
    SNR_ratio = g_length / g_last_length  # (G, T-1) SNR_ratio = SNR(1)/SNR(t)
    t = noise_schedule.get_time_from_SNRratio(SNR_ratio)  # (G, T-1)

    random_t = torch.rand(size=(t.size(0), 1), device=SNR_ratio.device)
    diff = abs(t - random_t)
    index = torch.argmin(diff, dim=1)

    t = t[(torch.arange(t.size(0)).to(index), index)]
    sampled_SNR_ratio = SNR_ratio[(torch.arange(SNR_ratio.size(0)).to(index), index)]
    return index, t, sampled_SNR_ratio


def round_trip_init(data, config="", noise_schedule="", geodesic_solver="", verbose=0, seed=0, dtype=torch.float32, q_type="morse"):
    fix_seed(seed)
    graph = RxnGraph.from_batch(data)
    full_edge, _, _ = graph.full_edge(upper_triangle=True)

    node2graph = graph.batch
    edge2graph = node2graph.index_select(0, full_edge[0])
    num_nodes = data.ptr[1:] - data.ptr[:-1]

    # sampling time step
    t_index, tt, SNR_ratio = noise_level_sampling(data, noise_schedule=noise_schedule)  # (G, ), (G, ), (G, )

    t_index_node = t_index.index_select(0, node2graph)  # (N, )
    mean = data.pos[(torch.arange(len(t_index_node)), t_index_node)].to(dtype)  # (N, 3)
    pos_init = data.pos[:, -1].to(dtype)

    sigma_hat = noise_schedule.get_sigma_hat(tt).to(dtype)  # (G, )
    sigma_hat_edge = sigma_hat.index_select(0, edge2graph)  # (E, )
    noise = torch.randn(size=(full_edge.size(1),), device=full_edge.device) * sigma_hat_edge  # dq, (E, )
    noise = noise.to(dtype) * 100
    return mean, noise, full_edge, graph.atom_type, node2graph, num_nodes


def solve(pos, q_dot, edge, atom_type, node2graph, num_nodes, config="", solver="", verbose=0, dtype=torch.float32, q_type="morse"):
    init, last, iter, index_tensor, stats = solver.batch_geodesic_ode_solve(
        pos,
        q_dot,
        edge,
        atom_type,
        node2graph,
        num_nodes,
        num_iter=config.manifold.ode_solver.iter,
        max_iter=config.manifold.ode_solver.max_iter,
        ref_dt=config.manifold.ode_solver.ref_dt,
        min_dt=config.manifold.ode_solver.min_dt,
        max_dt=config.manifold.ode_solver.max_dt,
        err_thresh=config.manifold.ode_solver.vpae_thresh,
        verbose=verbose,
        method="Heun",
        pos_adjust_scaler=config.manifold.ode_solver.pos_adjust_scaler,
        pos_adjust_thresh=config.manifold.ode_solver.pos_adjust_thresh,
        q_type=q_type,
    )

    batch_x_out = last["x"]
    batch_x_dot_out = last["x_dot"]
    unbatch_node_mask = _masking(num_nodes)
    x_out = batch_x_out.reshape(-1, 3)[unbatch_node_mask]
    x_dot_out = batch_x_dot_out.reshape(-1, 3)[unbatch_node_mask]

    batch_q_dot_out = last["q_dot"]
    e = batch_q_dot_out.size(1)
    unbatch_edge_index = index_tensor[1] + index_tensor[0] * e
    q_dot_out = batch_q_dot_out.reshape(-1)[unbatch_edge_index]  # (E, )
    q_out = last["q"].reshape(-1)[unbatch_edge_index]  # (E, )

    edge2graph = node2graph.index_select(0, edge[0])

    retry_index = stats["ban_index"].sort().values
    if len(retry_index) > 0:
        node_select = torch.isin(node2graph, retry_index)
        edge_select = torch.isin(edge2graph, retry_index)
        _batch = torch.arange(len(retry_index), device=pos.device).repeat_interleave(num_nodes[retry_index])
        _num_nodes = num_nodes[retry_index]
        _num_edges = _num_nodes * (_num_nodes - 1) // 2
        _ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=_num_nodes.device), _num_nodes.cumsum(0)])
        _edge = index_tensor[2:][:, edge_select] + _ptr[:-1].repeat_interleave(_num_edges)
        print(f"[Re-solve] geodesic solver failed at {len(retry_index)}/{len(num_nodes)}, Retry...")
        _init, _last, _iter, _index_tensor, _stats = solver.batch_geodesic_ode_solve(
            pos[node_select],
            q_dot[edge_select],
            _edge,
            atom_type[node_select],
            _batch,
            _num_nodes,
            num_iter=config.manifold.ode_solver.iter,
            max_iter=config.manifold.ode_solver.max_iter,
            ref_dt=config.manifold.ode_solver._ref_dt,
            min_dt=config.manifold.ode_solver._min_dt,
            max_dt=config.manifold.ode_solver._max_dt,
            err_thresh=config.manifold.ode_solver.vpae_thresh,
            verbose=verbose,
            method="RK4",
            pos_adjust_scaler=config.manifold.ode_solver.pos_adjust_scaler,
            pos_adjust_thresh=config.manifold.ode_solver.pos_adjust_thresh,
            q_type=q_type,
        )

        _batch_x_out = _last["x"]
        _batch_x_dot_out = _last["x_dot"]
        _unbatch_node_mask = _masking(_num_nodes)
        _x_out = _batch_x_out.reshape(-1, 3)[_unbatch_node_mask]
        _x_dot_out = _batch_x_dot_out.reshape(-1, 3)[_unbatch_node_mask]

        _batch_q_dot = _last["q_dot"]
        _batch_q = _last["q"]
        _e = _batch_q_dot.size(1)
        _unbatch_edge_index = _index_tensor[1] + _index_tensor[0] * _e
        _q_dot_out = _batch_q_dot.reshape(-1)[_unbatch_edge_index]  # (E', )
        _q_out = _batch_q.reshape(-1)[_unbatch_edge_index]  # (E', )

        x_out[node_select] = _x_out
        x_dot_out[node_select] = _x_dot_out
        q_dot_out[edge_select] = _q_dot_out
        q_out[edge_select] = _q_out

        ban_index = _stats["ban_index"].sort().values
        ban_index = retry_index[ban_index]
        for k, v in last.items():
            _v = _last[k]
            print(k, v.shape, _v.shape)
            _ = torch.zeros_like(v[retry_index])
            b = len(retry_index)
            _ = _.reshape(b, -1)
            numel = _v.numel() // b
            _[:, :numel] += _v.reshape(b, -1)
            _ = _.reshape(v[retry_index].shape)
            v[retry_index] = _
    else:
        ban_index = torch.LongTensor([])

    out = {"x": x_out, "x_dot": x_dot_out, "q": q_out, "q_dot": q_dot_out}
    return init, last, out, index_tensor, ban_index


def round_trip(data, config="", noise_schedule="", geodesic_solver="", verbose=0, seed=0, dtype=torch.float32, q_type="morse"):

    x, q_dot, edge, atom_type, node2graph, num_nodes = round_trip_init(
        data,
        config=config,
        noise_schedule=noise_schedule,
        geodesic_solver=geodesic_solver,
        verbose=verbose,
        seed=seed,
        dtype=dtype,
        q_type=q_type,
    )
    init, last, out, _, ban_index = solve(
        x,
        q_dot,
        edge,
        atom_type,
        node2graph,
        num_nodes,
        solver=geodesic_solver,
        config=config,
        verbose=verbose,
        dtype=dtype,
        q_type=q_type,
    )

    x_out = out["x"]
    x_dot_out = out["x_dot"]
    q_dot_out = out["q_dot"]
    _, round_last, round_out, _, round_ban_index = solve(
        x_out,
        - q_dot_out,
        edge,
        atom_type,
        node2graph,
        num_nodes,
        solver=geodesic_solver,
        config=config,
        verbose=verbose,
        dtype=dtype,
        q_type=q_type,
    )

    B = data.num_graphs
    round_x_err = (round_last["x"] - init["x"]).reshape(B, -1).norm(dim=-1)
    round_vpae = (round_last["q_dot"].norm(dim=-1) - init["q_dot"].norm(dim=-1)) / init["q_dot"].norm(dim=-1)
    round_q_dot_err = (round_last["q_dot"] + init["q_dot"]).norm(dim=-1)
    round_q_err = (round_last["q"] - init["q"]).norm(dim=-1)

    forward_vpae = (last["q_dot"].norm(dim=-1) - init["q_dot"].norm(dim=-1)) / init["q_dot"].norm(dim=-1)
    forward_q_err = (last["q_dot"] - init["q_dot"]).norm(dim=-1)
    forward_x_err = (last["x"] - init["x"]).reshape(B, -1).norm(dim=-1)

    print(f"\n\nban_index : \n{ban_index.abs()}")
    print(f"round_ban_index : \n{round_ban_index.abs()}\n\n")

    print(f"forward_vpae : \n{forward_vpae.abs()}")
    print(f"forward_q_err : \n{forward_q_err.abs()}")
    print(f"forward_x_err : \n{forward_x_err.abs()}\n\n")

    print(f"round_vpae : \n{round_vpae.abs()}")
    print(f"round_q_err : \n{round_q_err.abs()}")
    print(f"round_x_err : \n{round_x_err.abs()}")
    print(f"round_q_dot_err : \n{round_q_dot_err.abs() / init['q_dot'].norm(dim=-1)}\n\n")

    print(f"round_q/forward_q : \n{round_q_err / forward_q_err.abs()}")
    print(f"round_x/forward_x : \n{round_x_err / forward_x_err.abs()}\n\n")

    forward_rmsd = torch.tensor([RMSD(last["x"][i], init["x"][i]) for i in range(B)])
    forward_dmae = torch.tensor([DMAE(last["x"][i], init["x"][i]) for i in range(B)])
    round_rmsd = torch.tensor([RMSD(round_last["x"][i], init["x"][i]) for i in range(B)])
    round_dmae = torch.tensor([DMAE(round_last["x"][i], init["x"][i]) for i in range(B)])
    print(f"forward_rmsd : \n{forward_rmsd.abs()}")
    print(f"forward_dmae : \n{forward_dmae.abs()}\n\n")
    print(f"round_rmsd : \n{round_rmsd.abs()}")
    print(f"round_dmae : \n{round_dmae.abs()}\n\n")


def DMAE(x, y):
    """x : (N, 3)"""
    mask = (x == 0).all(dim=-1)
    x = x[~mask]
    y = y[~mask]
    i, j = np.triu_indices_from(x, k=1)
    dist_x = (x[i] - x[j]).pow(2).sum(dim=-1).sqrt()
    dist_y = (y[i] - y[j]).pow(2).sum(dim=-1).sqrt()
    dmae = (dist_x - dist_y).abs().mean()
    return dmae


def RMSD(x, y):
    """x : (N, 3)"""
    # x contains padding
    mask = (x == 0).all(dim=-1)
    x = x[~mask]
    y = y[~mask]
    N = x.size(0)
    rmsd = (x - y).pow(2).sum(dim=-1).mean().sqrt()
    return rmsd


def apply_noise(data, config="", noise_schedule="", geodesic_solver="", verbose=0, seed=0, dtype=torch.float32):
    fix_seed(seed)
    graph = RxnGraph.from_batch(data)
    full_edge, _, _ = graph.full_edge(upper_triangle=True)

    node2graph = graph.batch
    edge2graph = node2graph.index_select(0, full_edge[0])
    num_nodes = data.ptr[1:] - data.ptr[:-1]

    # sampling time step
    t_index, tt, SNR_ratio = noise_level_sampling(data, noise_schedule=noise_schedule)  # (G, ), (G, ), (G, )

    t_index_node = t_index.index_select(0, node2graph)  # (N, )
    mean = data.pos[(torch.arange(len(t_index_node)), t_index_node)]  # (N, 3)
    pos_init = data.pos[:, -1]

    mean = mean.to(dtype)
    pos_init = pos_init.to(dtype)

    # sigma = SNR_ratio * self.noise_schedule.get_sigma(torch.ones_like(SNR_ratio))
    # sigma_hat = sigma * (1 - SNR_ratio)  # (G, )
    sigma_hat = noise_schedule.get_sigma_hat(tt)  # (G, )
    sigma_hat = sigma_hat.to(dtype)

    sigma_hat_edge = sigma_hat.index_select(0, edge2graph)  # (E, )
    sigma_hat_edge = sigma_hat_edge.to(dtype)
    noise = torch.randn(size=(full_edge.size(1),), device=full_edge.device) * sigma_hat_edge  # dq, (E, )
    noise = noise.to(dtype)

    # apply noise
    print(mean.pow(2).sum(), noise.pow(2).sum())
    init, last, iter, index_tensor, stats = geodesic_solver.batch_geodesic_ode_solve(
        mean,
        noise,
        full_edge,
        graph.atom_type,
        node2graph,
        num_nodes,
        num_iter=config.manifold.ode_solver.iter,
        max_iter=config.manifold.ode_solver.max_iter,
        ref_dt=config.manifold.ode_solver.ref_dt,
        min_dt=config.manifold.ode_solver.min_dt,
        max_dt=config.manifold.ode_solver.max_dt,
        err_thresh=config.manifold.ode_solver.vpae_thresh,
        verbose=verbose,
        method="Heun",
        pos_adjust_scaler=config.manifold.ode_solver.pos_adjust_scaler,
        pos_adjust_thresh=config.manifold.ode_solver.pos_adjust_thresh,
    )
    for k, v in init.items():
        print(k, v.sum())

    batch_pos_noise = last["x"]  # (B, n, 3)
    batch_x_dot = last["x_dot"]  # (B, n, 3)
    unbatch_node_mask = _masking(num_nodes)
    pos_noise = batch_pos_noise.reshape(-1, 3)[unbatch_node_mask]  # (N, 3)
    x_dot = batch_x_dot.reshape(-1, 3)[unbatch_node_mask]  # (N, 3)

    batch_q_dot = last["q_dot"]  # (B, e)
    e = batch_q_dot.size(1)
    unbatch_edge_index = index_tensor[1] + index_tensor[0] * e
    q_dot = batch_q_dot.reshape(-1)[unbatch_edge_index]  # (E, )

    # Check stability, percent error > threshold, then re-solve
    retry_index = stats["ban_index"].sort().values
    if len(retry_index) > 0:
        node_select = torch.isin(node2graph, retry_index)
        edge_select = torch.isin(edge2graph, retry_index)
        _batch = torch.arange(len(retry_index), device=mean.device).repeat_interleave(num_nodes[retry_index])
        _num_nodes = num_nodes[retry_index]
        _num_edges = _num_nodes * (_num_nodes - 1) // 2
        _ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=_num_nodes.device), _num_nodes.cumsum(0)])
        _full_edge = index_tensor[2:][:, edge_select] + _ptr[:-1].repeat_interleave(_num_edges)

        print(f"[Re-solve] geodesic solver failed at {len(retry_index)}/{len(data)}, Retry...")
        _init, _last, _iter, _index_tensor, _stats = geodesic_solver.batch_geodesic_ode_solve(
            mean[node_select],
            noise[edge_select],
            _full_edge,
            graph.atom_type[node_select],
            _batch,
            _num_nodes,
            num_iter=config.manifold.ode_solver.iter,
            max_iter=config.manifold.ode_solver.max_iter,
            ref_dt=config.manifold.ode_solver._ref_dt,
            min_dt=config.manifold.ode_solver._min_dt,
            max_dt=config.manifold.ode_solver._max_dt,
            err_thresh=config.manifold.ode_solver.vpae_thresh,
            verbose=0,
            method="RK4",
            pos_adjust_scaler=config.manifold.ode_solver.pos_adjust_scaler,
            pos_adjust_thresh=config.manifold.ode_solver.pos_adjust_thresh,
        )

        _batch_pos_noise = _last["x"]  # (b, n', 3)
        _batch_x_dot = _last["x_dot"]  # (b, n', 3)
        _unbatch_node_mask = _masking(_num_nodes)
        _pos_noise = _batch_pos_noise.reshape(-1, 3)[_unbatch_node_mask]  # (N', 3)
        _x_dot = _batch_x_dot.reshape(-1, 3)[_unbatch_node_mask]  # (N', 3)

        _batch_q_dot = _last["q_dot"]  # (b, e')
        _e = _batch_q_dot.size(1)
        _unbatch_edge_index = _index_tensor[1] + _index_tensor[0] * _e
        _q_dot = _batch_q_dot.reshape(-1)[_unbatch_edge_index]  # (E', )

        pos_noise[node_select] = _pos_noise
        x_dot[node_select] = _x_dot
        q_dot[edge_select] = _q_dot

        ban_index = _stats["ban_index"].sort().values
        ban_index = retry_index[ban_index]
    else:
        ban_index = torch.LongTensor([])

    beta = noise_schedule.get_beta(tt)
    coeff = beta / sigma_hat
    coeff_node = coeff.index_select(0, node2graph)  # (N, )
    coeff_edge = coeff.index_select(0, edge2graph)  # (E, )
    # target is not exactly the score function.
    # target = beta * score
    target_x = - x_dot * coeff_node.unsqueeze(-1)
    target_q = - q_dot * coeff_edge.unsqueeze(-1)

    if len(ban_index) > 0:
        rxn_idx = data.rxn_idx[ban_index]
        print(f"[Warning] geodesic solver failed at {len(ban_index)}/{len(data)}\n"
              f"rxn_idx: {rxn_idx}\n time index: {t_index[ban_index]}")
        ban_node_mask = torch.isin(node2graph, ban_index)
        ban_edge_mask = torch.isin(edge2graph, ban_index)
        ban_batch_mask = torch.isin(torch.arange(len(data), device=mean.device), ban_index)

        data = Batch.from_data_list(data[~ban_batch_mask])
        graph = RxnGraph.from_batch(data)

        pos_noise = pos_noise[~ban_node_mask]
        x_dot = x_dot[~ban_node_mask]
        pos_init = pos_init[~ban_batch_mask]
        tt = tt[~ban_batch_mask]
        target_x = target_x[~ban_node_mask]
        target_q = target_q[~ban_edge_mask]

    return graph, pos_noise, pos_init, tt, target_x, target_q


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--solver", type=str, default="test")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float")
    parser.add_argument("--q_type", type=str, default="morse", choices=["morse", "DM"])
    parser.add_argument("--test_type", type=str, choices=["round_trip", "apply_noise"], default="round_trip")
    args = parser.parse_args()

    torch.set_printoptions(precision=5, sci_mode=False, edgeitems=10, threshold=1000, profile="full")
    dtype = torch.float32 if args.dtype == "float" else torch.float64
    torch.set_num_threads(4)

    config = omegaconf.OmegaConf.load('configs/config.yaml')
    datamodule = GrambowDataModule(config)
    solver = GeodesicSolver(config.manifold)
    noise_schedule = load_noise_scheduler(config.diffusion)

    if args.test_type == "apply_noise":
        for i, batch in tqdm.tqdm(enumerate(datamodule.test_dataloader())):
            batch = batch.to(args.device)
            graph, pos_noise, pos_init, tt, target_x, target_q = apply_noise(batch, config=config, noise_schedule=noise_schedule, geodesic_solver=solver, verbose=args.verbose, dtype=dtype, q_type=args.q_type)

    if args.test_type == "round_trip":
        for i, batch in tqdm.tqdm(enumerate(datamodule.test_dataloader())):
            batch = batch.to(args.device)
            round_trip(batch, config=config, noise_schedule=noise_schedule, geodesic_solver=solver, verbose=args.verbose, dtype=dtype, q_type=args.q_type)
            break
