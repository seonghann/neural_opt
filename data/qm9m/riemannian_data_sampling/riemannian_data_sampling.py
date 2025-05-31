"""
Riemannian data sampling for fine-tuning
"""

import numpy as np
import torch
from utils.rxn_graph import MolGraph, RxnGraph
from torch_geometric.data import Batch
import pandas as pd


def _masking(num_nodes):
    N = num_nodes.max()
    mask = torch.BoolTensor([True, False]).repeat(len(num_nodes)).to(num_nodes.device)
    num_repeats = torch.stack([num_nodes, N - num_nodes]).T.flatten()
    mask = mask.repeat_interleave(num_repeats)
    return mask


def ode_noise_sampling(
    config,
    geodesic_solver,
    # self,
    pos,
    q_noise,
    edge_index,
    atom_type,
    node2graph,
    edge2graph,
    num_nodes,
    data,
    graph,
    time_step,
    retry=False,
):
    q_type = "morse"

    init, last, iter, index_tensor, stats = geodesic_solver.batch_geodesic_ode_solve(
        pos,
        q_noise,
        edge_index,
        atom_type,
        node2graph,
        num_nodes,
        q_type=q_type,
        num_iter=config.manifold.ode_solver.iter,
        max_iter=config.manifold.ode_solver.max_iter,
        ref_dt=config.manifold.ode_solver.ref_dt,
        min_dt=config.manifold.ode_solver.min_dt,
        max_dt=config.manifold.ode_solver.max_dt,
        err_thresh=config.manifold.ode_solver.vpae_thresh,
        verbose=0,
        method="Heun",
        pos_adjust_scaler=config.manifold.ode_solver.pos_adjust_scaler,
        pos_adjust_thresh=config.manifold.ode_solver.pos_adjust_thresh,
    )

    batch_pos_noise = last["x"]  # (B, n, 3)
    batch_x_dot = last["x_dot"]  # (B, n, 3)
    unbatch_node_mask = _masking(num_nodes)
    pos_noise = batch_pos_noise.reshape(-1, 3)[unbatch_node_mask]  # (N, 3)
    x_dot = batch_x_dot.reshape(-1, 3)[unbatch_node_mask]  # (N, 3)

    batch_q_dot = last["q_dot"]  # (B, e)
    batch_q_init = init["q"]  # (B, e)
    e = batch_q_dot.size(1)
    unbatch_edge_index = index_tensor[1] + index_tensor[0] * e
    q_dot = batch_q_dot.reshape(-1)[unbatch_edge_index]  # (E, )
    q_init = batch_q_init.reshape(-1)[unbatch_edge_index]  # (E, )

    # Check stability, percent error > threshold, then re-solve
    if retry:
        retry_index = stats["ban_index"].sort().values
        if len(retry_index) > 0:
            node_select = torch.isin(node2graph, retry_index)
            edge_select = torch.isin(edge2graph, retry_index)
            # _batch = torch.arange(len(retry_index), device=mean.device).repeat_interleave(num_nodes[retry_index])
            _batch = torch.arange(
                len(retry_index), device=pos.device
            ).repeat_interleave(num_nodes[retry_index])
            _num_nodes = num_nodes[retry_index]
            _num_edges = _num_nodes * (_num_nodes - 1) // 2
            _ptr = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=_num_nodes.device),
                    _num_nodes.cumsum(0),
                ]
            )
            _full_edge = index_tensor[2:][:, edge_select] + _ptr[:-1].repeat_interleave(
                _num_edges
            )

            print(
                f"[Resolve] geodesic solver failed at {len(retry_index)}/{len(data)}, Retry..."
            )
            _init, _last, _iter, _index_tensor, _stats = (
                geodesic_solver.batch_geodesic_ode_solve(
                    pos[node_select],
                    q_noise[edge_select],
                    _full_edge,
                    atom_type[node_select],
                    _batch,
                    _num_nodes,
                    q_type=q_type,
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
    else:
        ban_index = stats["ban_index"].sort().values
        print(f"Debug: ban_index={ban_index}")

    # Deprecated (for debugging, 24.07.29)
    # coeff = torch.ones_like(tt)
    # coeff_node = coeff.index_select(0, node2graph)  # (N, )
    # coeff_edge = coeff.index_select(0, edge2graph)  # (E, )
    # target_x = - x_dot * coeff_node.unsqueeze(-1)
    # target_q = - q_dot * coeff_edge
    target_x = -x_dot
    target_q = -q_dot

    if len(ban_index) > 0:
        ban_index = ban_index.to(torch.long)
        # rxn_idx = [data.rxn_idx[i] for i in ban_index]
        idx = [data.idx[i] for i in ban_index]
        print(
            f"\n[Warning] geodesic solver failed at {len(ban_index)}/{len(data)}\n\tidx: {idx}"
        )
        ban_node_mask = torch.isin(node2graph, ban_index)
        ban_edge_mask = torch.isin(edge2graph, ban_index)
        # ban_batch_mask = torch.isin(torch.arange(len(data), device=mean.device), ban_index)
        ban_batch_mask = torch.isin(
            torch.arange(len(data), device=pos.device), ban_index
        )

        data = Batch.from_data_list(data[~ban_batch_mask])
        # graph = self.graph.from_batch(data)
        # graph = MolGraph.from_batch(data)
        graph = Graph.from_batch(data)

        pos_noise = pos_noise[~ban_node_mask]
        # x_dot = x_dot[~ban_node_mask]
        # pos_init = pos_init[~ban_node_mask]
        # tt = tt[~ban_batch_mask]
        time_step = time_step[~ban_batch_mask]
        target_x = target_x[~ban_node_mask]
        target_q = target_q[~ban_edge_mask]

    # pos_noise = center_pos(pos_noise, data.batch)
    # return pos_noise, target_x, target_q
    # return pos_noise, target_x, target_q, graph, time_step
    return pos_noise, target_x, target_q, graph, time_step, ban_index


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./riemannian_data_sampling.yaml",
        help="config yaml file path",
    )
    parser.add_argument(
        "--save_xyz",
        type=str,
        default=None,
        help="save path of samples' xyz files (default: None)",
    )
    parser.add_argument("--save_csv", type=str, required=True, help="save as csv file")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--sampling_type",
        type=str,
        default="riemannian",
        help="sampling type (default: riemannian)",
        choices=["cartesian", "riemannian"],
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="coefficient alpha for the Morse scaler",
        default=None,
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="coefficient beta for the Morse scaler",
        default=None,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="coefficient gamma for the Morse scaler",
        default=0.0,
    )
    parser.add_argument(
        "--svd_tol",
        type=float,
        help="svd_tol for Jacobian inverse",
    )
    parser.add_argument("--t0", type=int, default=None, required=False)
    parser.add_argument("--t1", type=int, default=None, required=False)
    parser.add_argument(
        "--retry",
        action="store_true",
        help="set retry=True in ode_noise_sampling",
    )
    parser.add_argument(
        "--dataloader", type=str, choices=["train", "val", "test"], default="test"
    )
    parser.add_argument("--graph", type=str, default="mol", choices=["mol", "rxn"])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    args = parser.parse_args()
    print(args)
    if args.graph == "mol":
        Graph = MolGraph
    elif args.graph == "rxn":
        Graph = RxnGraph
    device = torch.device(args.device)

    import os
    from torch_scatter import scatter_sum
    from ase import Atoms

    from dataset.data_module import load_datamodule
    from omegaconf import OmegaConf
    from utils.geodesic_solver import GeodesicSolver
    from diffusion.noise_scheduler import load_noise_scheduler
    from eval_accuracy2 import remap2atomic_numbers

    if not os.path.exists(args.save_xyz):
        os.makedirs(args.save_xyz)
        print(f"Create the directory: {args.save_xyz}")
    else:
        print(f"The directory {args.save_xyz} already exists")
        exit(1)

    torch.manual_seed(args.seed)

    config = OmegaConf.load(args.config_yaml)
    config.manifold.ode_solver.alpha = args.alpha
    config.manifold.ode_solver.beta = args.beta
    config.manifold.ode_solver.gamma = args.gamma
    config.manifold.ode_solver.svd_tol = args.svd_tol
    config.diffusion.scheduler.t0 = args.t0
    config.diffusion.scheduler.t1 = args.t1
    print("config=\n", config)

    datamodule = load_datamodule(config)
    geodesic_solver = GeodesicSolver(config.manifold)
    noise_schedule = load_noise_scheduler(config.diffusion)
    q_type = "morse"

    results = {
        "idx": [],
        "smarts": [],
        "time_step": [],
        "perr_straight": [],
        "perr_projected": [],
    }

    if args.dataloader == "train":
        dataloader = datamodule.train_dataloader()
    elif args.dataloader == "val":
        dataloader = datamodule.val_dataloader()
    elif args.dataloader == "test":
        dataloader = datamodule.test_dataloader()
    else:
        ValueError()
    print(f"Load {args.dataloader} dataloader")

    for i_batch, data in enumerate(dataloader):
        print(f"i_batch={i_batch}", flush=True)

        data = data.to(device)
        graph = Graph.from_batch(data)
        edge_index = graph.full_edge(upper_triangle=True)[0]
        batch_size = len(data.ptr) - 1

        node2graph = graph.batch
        edge2graph = node2graph.index_select(0, edge_index[0])
        num_nodes = data.ptr[1:] - data.ptr[:-1]

        natoms = node2graph.bincount().tolist()
        nedges = edge2graph.bincount().tolist()
        _atom_type = data.x.split(natoms)

        pos_0 = data.pos[:, 0, :]
        # pos_T = data.pos[:, -1, :]  # MMFF structures

        # Time sampling
        t0 = config.diffusion.scheduler.t0
        t1 = config.diffusion.scheduler.t1
        time_step = torch.randint(
            max(t0, 1), t1, size=(batch_size,)
        , device=device)
        print(f"Debug: time_step in [{min(time_step)}, {max(time_step)}]")
        a = noise_schedule.get_alpha(time_step, device=device)

        if args.sampling_type == "cartesian":
            # Perterb pos in Euclidean
            a_pos = a.index_select(0, node2graph).unsqueeze(-1)
            pos_noise = torch.randn(size=pos_0.size(), device=device)
            pos_t = pos_0 + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()

            q_0 = geodesic_solver.compute_d_or_q(
                pos_0, graph.atom_type, edge_index, q_type=q_type
            )
            q_t = geodesic_solver.compute_d_or_q(
                pos_t, graph.atom_type, edge_index, q_type=q_type
            )

            q_straight = q_0 - q_t
            q_target = geodesic_solver.batch_projection(
                q_straight,
                pos_t,
                graph.atom_type,
                edge_index,
                graph.batch,
                num_nodes,
                q_type=q_type,
                proj_type="manifold",
            )
            pos_target = geodesic_solver.batch_dq2dx(
                q_target,
                pos_t,
                graph.atom_type,
                edge_index,
                graph.batch,
                num_nodes,
                q_type=q_type,
            ).reshape(-1, 3)

            norm = lambda x: scatter_sum(x.square(), edge2graph).sqrt()
            norm_ref = norm(q_target)
            perr_straight = norm(q_target - q_straight) / norm_ref * 100
            perr_projected = torch.zeros_like(perr_straight)
            results["perr_straight"].extend(perr_straight.tolist())
            results["perr_projected"].extend(perr_projected.tolist())
            results["time_step"].extend(time_step.tolist())
            data_idx = data.idx
            results["idx"].extend(data_idx)
            smarts = data.smarts
            results["smarts"].extend(smarts)
        elif args.sampling_type == "riemannian":
            ## Perterb pos in Riemannian
            a_edge = a.index_select(0, edge2graph)
            q_noise = torch.randn(size=(edge_index.size(1),), device=device)
            q_noise *= (1.0 - a_edge).sqrt() / a_edge.sqrt()
            pos_t, pos_target, q_target, graph, time_step, ban_index = (
                ode_noise_sampling(
                    config,
                    geodesic_solver,
                    pos_0,
                    q_noise,
                    edge_index,
                    graph.atom_type,
                    node2graph,
                    edge2graph,
                    num_nodes,
                    data,
                    graph,
                    time_step,
                    retry=args.retry,
                )
            )

            ## Masking failed cases
            ban_node_mask = torch.isin(node2graph, ban_index)
            ban_batch_mask = torch.isin(torch.arange(len(data)), ban_index)
            edge_index = graph.full_edge(upper_triangle=True)[0]
            node2graph = graph.batch
            edge2graph = node2graph.index_select(0, edge_index[0])
            tmp = np.array(data.x.split(natoms), dtype=object)
            if len(data) == 1:
                _atom_type = data.x.split(natoms)
            else:
                _atom_type = np.array(data.x.split(natoms), dtype=object)[~ban_batch_mask]
            data_idx = torch.tensor(data.idx)[~ban_batch_mask].tolist()
            natoms = node2graph.bincount().tolist()
            nedges = edge2graph.bincount().tolist()
            pos_0 = pos_0[~ban_node_mask]
            smarts = np.array(data.smarts)[~ban_batch_mask.numpy()]

            q_target = geodesic_solver.batch_projection(
                q_target,
                pos_t,
                graph.atom_type,
                edge_index,
                graph.batch,
                # num_nodes,
                graph.batch.bincount(),
                q_type=q_type,
                proj_type="manifold",
            )

            ## Check perr
            q_0 = geodesic_solver.compute_d_or_q(
                pos_0, graph.atom_type, edge_index, q_type=q_type
            )
            q_t = geodesic_solver.compute_d_or_q(
                pos_t, graph.atom_type, edge_index, q_type=q_type
            )
            q_straight = q_0 - q_t
            q_projected = geodesic_solver.batch_projection(
                q_straight,
                pos_t,
                graph.atom_type,
                edge_index,
                graph.batch,
                # num_nodes,
                graph.batch.bincount(),
                q_type=q_type,
                proj_type="manifold",
            )
            norm = lambda x: scatter_sum(x.square(), edge2graph).sqrt()
            norm_ref = norm(q_target)
            perr_straight = norm(q_target - q_straight) / norm_ref * 100
            perr_projected = norm(q_target - q_projected) / norm_ref * 100

            results["perr_straight"].extend(perr_straight.tolist())
            results["perr_projected"].extend(perr_projected.tolist())
            results["time_step"].extend(time_step.tolist())
            results["idx"].extend(data_idx)
            results["smarts"].extend(smarts)
        else:
            raise ValueError()

        # ######################################
        # import pandas as pd

        # df = pd.DataFrame(results)
        # pd.options.display.max_rows = None
        # pd.options.display.max_columns = None
        # print(df)
        # print("mean     : ", df["perr_straight"].mean(), df["perr_projected"].mean())
        # print("median   : ", df["perr_straight"].median(), df["perr_projected"].median())
        # ######################################

        if args.save_xyz:
            ## Write as xyz files
            pos_0 = pos_0.split(natoms)
            pos_t = pos_t.split(natoms)
            pos_target = pos_target.split(natoms)
            q_target = q_target.split(nedges)
            # data_idx = data.idx
            # data_idx = results["idx"]
            # smarts = data.smarts

            for i in range(len(pos_0)):
                filename = (
                    f"./{args.save_xyz}/idx{data_idx[i]}-{time_step[i].item()}.xyz"
                )

                atom_type = remap2atomic_numbers(_atom_type[i])

                atoms_pos_0 = Atoms(symbols=atom_type, positions=pos_0[i])
                atoms_pos_t = Atoms(symbols=atom_type, positions=pos_t[i])
                atoms_pos_target = Atoms(symbols=atom_type, positions=pos_target[i])

                comment = f'pos_0 idx={data_idx[i]} time_step={time_step[i].item()} smarts="{smarts[i]}" q_target={q_target[i].tolist()}'
                atoms_pos_0.write(filename, comment=comment, append=False)
                comment = "pos_t"
                atoms_pos_t.write(filename, comment=comment, append=True)
                comment = "pos_target"
                atoms_pos_target.write(filename, comment=comment, append=True)

                print(f"Save {filename}")

    ## Save perr info as csv file
    import pandas as pd

    df = pd.DataFrame(results)
    df.to_csv(args.save_csv)
    print(f"Save {args.save_csv}")
