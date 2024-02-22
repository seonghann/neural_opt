# Unit test for the solver module
import torch
import omegaconf
import tqdm
from utils.geodesic_solver import GeodesicSolver, redefine_edge_index, redefine_with_pad
from dataset.data_module import GrambowDataModule
from utils.rxn_graph import RxnGraph
from torch_geometric.data import Batch
import argparse


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(seed)


def batch_index_generator(index_set, batch_size=100):
    for i in range(len(index_set) // batch_size):
        yield index_set[i * batch_size: (i + 1) * batch_size]
    if len(index_set) % batch_size != 0:
        if len(index_set) // batch_size == 0:
            yield index_set[:]
        else:
            yield index_set[(i + 1) * batch_size:]

            config = omegaconf.OmegaConf.load('configs/config.yaml')
            datamodule = GrambowDataModule(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_set", type=str, nargs="+", default=None)
    parser.add_argument("--ref_dt", type=float, default=5e-2)
    parser.add_argument("--svd_tol", type=float, default=1e-2)
    parser.add_argument("--save", type=str, default="geodesic_ode_test_2.pt")
    parser.add_argument("--load", type=str, default="geodesic_ode_test.pt")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="double")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--method", type=str, default="Euler")
    args = parser.parse_args()

    if args.dtype in ["float32", "float", "single"]:
        args.dtype = torch.float32
    if args.dtype in ["float64", "double"]:
        args.dtype = torch.float64
    torch.set_default_dtype(args.dtype)

    # load data and solver
    config = omegaconf.OmegaConf.load('configs/config.yaml')
    datamodule = GrambowDataModule(config)

    batch_size = args.batch_size
    solver = GeodesicSolver(config.manifold)
    solver.svd_tol = args.svd_tol

    if args.index_set is not None:
        index_set = args.index_set
        # index may be string of "int," or "int"
        index_set = [int(i.split(",")[0]) for i in index_set]
    else:
        index_set = list(range(1000))
    test_dataset = datamodule.test_dataset
    save = []

    load_data = torch.load(args.load)["batch_data"]
    if args.seed is not None:
        set_seed(args.seed)

    for _idx in tqdm.tqdm(batch_index_generator(index_set, batch_size=batch_size), total=len(index_set) // batch_size + 1):
        batch = Batch.from_data_list(test_dataset[_idx]).to(args.device)

        # make input
        atom_type = batch.x
        num_nodes = batch.ptr[1:] - batch.ptr[:-1]
        pos = batch.pos[:, 0].to(args.dtype)
        rxn_graph = RxnGraph.from_batch(batch)
        edge_index, _, _ = rxn_graph.full_edge()

        q_dot = torch.randn((edge_index.size(1), 1)) * 1e-2
        if args.seed is None:
            q_dot_load = torch.cat([load_data[_i]["q_init"] for _i in _idx]).reshape(-1, 1)
            assert q_dot.shape == q_dot_load.shape
            q_dot = q_dot_load
        q_dot = q_dot.to(args.device).to(args.dtype)

        # solve geodesic ode with batched jacobian and hessian
        init, last, iter, index_tensor = solver.batch_geodesic_ode_solve(
            pos,
            q_dot,
            edge_index,
            atom_type,
            batch.batch,
            num_nodes,
            verbose=0,
            ref_dt=args.ref_dt,
            method=args.method
        )

        x_init = init["x"]
        q_init = init["q"]
        q_dot_init = init["q_dot"]
        x_last = last["x"]
        q_last = last["q"]
        q_dot_last = last["q_dot"]

        for j, datamodule_idx in enumerate(_idx):
            x_init_ = x_init[j][:num_nodes[j]]
            x_last_ = x_last[j][:num_nodes[j]]
            q_init_ = q_init[j][index_tensor[1][index_tensor[0] == j]]
            q_last_ = q_last[j][index_tensor[1][index_tensor[0] == j]]
            q_dot_init_ = q_dot_init[j][index_tensor[1][index_tensor[0] == j]]
            q_dot_last_ = q_dot_last[j][index_tensor[1][index_tensor[0] == j]]
            _save = {
                "rxn_idx": batch[j].rxn_idx,
                "idx": datamodule_idx,
                "x_init": x_init_,
                "q_init": q_init_,
                "q_dot_init": q_dot_init_,
                "x_last": x_last_,
                "q_last": q_last_,
                "q_dot_last": q_dot_last_,
            }
            save.append(_save)

    torch.save(save, args.save)
