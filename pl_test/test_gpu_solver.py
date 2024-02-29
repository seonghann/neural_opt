import torch
torch.set_printoptions(precision=3, sci_mode=False,)
import ase.io
import torch
import omegaconf
from utils.geodesic_solver import GeodesicSolver, redefine_edge_index, redefine_with_pad
from dataset.data_module import GrambowDataModule
from utils.rxn_graph import RxnGraph, DynamicRxnGraph
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch, unbatch_edge_index, subgraph
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
dtype = torch.float32
torch.set_default_dtype(torch.float32)

import pandas as pd

import ase
from utils.chem import ATOMIC_NUMBERS
ATOMIC_NUMBERS = dict([(v, k) for k, v in ATOMIC_NUMBERS.items()])


class PlaceHolder(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        msg = "PlaceHolder("
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                if v.dim() == 0:
                    if v.dtype == torch.float32 or v.dtype == torch.float64:
                        msg += f"{k}={v.item():0.4f}, "
                    else:
                        msg += f"{k}={v}, "
                else:
                    msg += f"{k}={v.shape}, "
            else:
                if isinstance(v, float):
                    msg += f"{k}={v:0.4f}, "
                else:
                    msg += f"{k}={v}, "
        msg += ")"
        return msg

    # show key of the object
    def keys(self):
        return self.__dict__.keys()

    def return_dict(self):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.cpu().detach().numpy()
        return self.__dict__


# load batch-size data using yield
def batching(data, batch_size=10):
    num_batches = len(data) // batch_size
    for i in range(num_batches):
        yield data[i * batch_size: (i + 1) * batch_size]
    if len(data) % batch_size != 0:
        yield data[num_batches * batch_size: ]


def compute_jacobian(batch, solver, noise=0.0, dtype=torch.float64):
    rxn_graph = RxnGraph.from_batch(batch)
    edge_index = rxn_graph.full_edge()[0]
    num_nodes = batch.ptr[1:] - batch.ptr[:-1]
    pos = (batch.pos[:, 0] + noise * torch.randn_like(batch.pos[:, 0])).to(dtype)
    batch_pos = redefine_with_pad(pos, batch.batch).to(dtype)
    batch_atom_type = redefine_with_pad(batch.x, batch.batch)
    index_tensor = redefine_edge_index(edge_index, batch.batch, num_nodes)
    J = solver.sparse_batch_jacobian_q(
        index_tensor,
        batch_atom_type,
        batch_pos,
    ).to_dense()
    return J


def compute_last_singular_values(batch, solver, noise=0.0, dtype=torch.float64):
    N = batch.ptr[1:] - batch.ptr[:-1]
    J = compute_jacobian(batch, solver=solver, noise=noise, dtype=dtype)
    svd_res = torch.vmap(torch.linalg.svd)(J)
    singular_values = svd_res.S
    max_N = N.max()
    sv_idx = 3 * N - 7 + torch.arange(len(N)) * (max_N * 3)
    min_sv = singular_values.flatten()[sv_idx]
    return min_sv


def compute_all_singular_values(batch, solver, noise=0.0, dtype=torch.float64):
    N = batch.ptr[1:] - batch.ptr[:-1]
    J = compute_jacobian(batch, solver=solver, noise=noise, dtype=dtype)
    svd_res = torch.vmap(torch.linalg.svd)(J)
    singular_values = svd_res.S
    non_zero_singular_values = []
    for n, sv in zip(N, singular_values):
        non_zero_singular_values.append(sv[:(3 * n - 6)])
    return non_zero_singular_values


def solve_gode(
    batch,
    solver,
    method='Heun',
    noise=0.0,
    seed=0,
    dtype=torch.float64,
    verbose=0,
    q_dot=None,
    pos=None,
    q_dot_scale=0.01,
    num_iter=100,
    ref_dt=1e-2,
    max_dt=1e-1,
    min_dt=1e-3,
    max_iter=1000,
    err_thresh=0.03,
    pos_adjust_scaler=0.05,
    pos_adjust_thresh=1e-3,
    dev="cpu"
):
    """
    Args:
        batch (torch_geometric.data.Batch) : batch of data
        solver (GeodesicSolver) : solver
        method (str) : choice of method, 'Euler', 'Heun', 'RK4'.
        noise (float) : noise to add to the initial position (deprecated)
        seed (int) : seed for random number generator. Related to initial q_dot and position noise.
        dtype (torch.dtype) : data type of the inputs
        verbose (int) : verbose level
        q_dot (torch.Tensor) : initial q_dot. If None, random q_dot is generated.
        pos (torch.Tensor) : initial position. If None, batch.pos[:, 0] is used.
        q_dot_scale (float) : scale of random q_dot
        -- parameters for ode solver --
    """
    torch.manual_seed(seed)
    # set inputs
    batch = batch.to(dev)
    rxn_graph = RxnGraph.from_batch(batch).to(dev)

    if pos is None:
        x = batch.pos[:, 0].to(dtype)
    else:
        x = pos

    if noise != 0:
        # warning: noise is deprecated
        x += noise * torch.randn_like(x).to(dtype)
    edge_index = rxn_graph.full_edge()[0]
    if q_dot is None:
        q_dot = torch.rand(edge_index.shape[1]).to(dtype) * q_dot_scale
    q_dot = q_dot.to(dev)
    atom_type = batch.x
    num_nodes = batch.ptr[1:] - batch.ptr[:-1]

    init, last, iter, index_tensor, ban_index = solver.batch_geodesic_ode_solve(
        x,
        q_dot,
        edge_index,
        atom_type,
        batch.batch,
        num_nodes,
        method=method,
        verbose=verbose,
        num_iter=num_iter, ref_dt=ref_dt, max_dt=max_dt, min_dt=min_dt, max_iter=max_iter, err_thresh=err_thresh,
        pos_adjust_scaler=pos_adjust_scaler, pos_adjust_thresh=pos_adjust_thresh,

    )
    return init, last, iter, index_tensor, ban_index


if __name__ == "__main__":
    # load data and solver
    config = omegaconf.OmegaConf.load('configs/config.yaml')
    datamodule = GrambowDataModule(config)
    batch = Batch.from_data_list(datamodule.test_dataset[: 10])
    solver = GeodesicSolver(config.manifold)

    init, last, iter, index_tensor, ban_index = solve_gode(
        batch,
        solver,
        method="Heun",
        noise=0.0,
        seed=0,
        verbose=1,
        q_dot=None,
        pos=None,
        q_dot_scale=0.01,
        ref_dt=5e-2,
        num_iter=10,
        dev="cuda"
    )

    # init, last has keys of ['x', 'x_dot', 'q', 'q_dot']
    # x, x_dot : (num_graphs, num_nodes.max(), 3)
    # q, q_dot : (num_graphs, num_edges.max())
    # index_tensor contains all information about the edges of graph
    # index_tensor : (4, num_edges.sum())
    # ban_index contains the index of graph that are not solved
    num_graphs = batch.num_graphs  # scalar
    num_nodes = batch.ptr[1:] - batch.ptr[:-1]  # (num_graphs, )
    num_edges = num_nodes * (num_nodes - 1) // 2  # (num_graphs, )
