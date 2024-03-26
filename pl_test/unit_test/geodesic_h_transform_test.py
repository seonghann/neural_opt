"""
Geodesic interpolation으로 계산된 q_dot을 geodesic solver로 update했을 때, x_T로 잘 도착하는 지 확인하기 위한 코드.
"""

def DMAE(x, y):
    """x : (N, 3)"""
    mask = (x == 0).all(dim=-1)
    x = x[~mask]
    y = y[~mask]
    assert x.shape == y.shape, "x and y should have the same size."

    dmae = (torch.pdist(x) - torch.pdist(y)).abs().mean()
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--solver", type=str, default="test")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float")
    parser.add_argument("--test_type", type=str, choices=["round_trip", "apply_noise"], default="round_trip")
    parser.add_argument("--seed", type=int, default=0, help="random seed number")
    args = parser.parse_args()
    print(f"args: {args}")
    ############################################################################

    import sys
    sys.path.append("../")

    import torch
    torch.set_printoptions(precision=3, sci_mode=False)
    dtype = torch.float32 if args.dtype == "float" else torch.float64

    import tqdm
    import omegaconf
    from utils.geodesic_solver import GeodesicSolver
    from dataset.data_module import GrambowDataModule
    from diffusion.noise_scheduler import load_noise_scheduler


    ## Make objects
    config = omegaconf.OmegaConf.load('../configs/config.yaml')
    datamodule = GrambowDataModule(config)
    noise_schedule = load_noise_scheduler(config.diffusion)
    solver = GeodesicSolver(config.manifold)


    ## Prepare inputs
    # refer to 'round_trip_init()'
    def make_inputs(data, config="", seed=0, dtype=torch.float32):
        from solver_test import fix_seed
        from utils.rxn_graph import RxnGraph, DynamicRxnGraph

        fix_seed(seed)
        graph = RxnGraph.from_batch(data)
        full_edge, _, _ = graph.full_edge(upper_triangle=True)

        node2graph = graph.batch
        num_nodes = data.ptr[1:] - data.ptr[:-1]

        pos = data.pos.to(dtype)  # shape == (natoms, nimages, 3)
        return pos, full_edge, graph.atom_type, node2graph, num_nodes

    # from solver_test import DMAE, RMSD


    for i, batch in tqdm.tqdm(enumerate(datamodule.test_dataloader())):
        print(batch)
        print(f"iter = {i}"); print(f"Debug: rxn_idx={batch.rxn_idx}"); print(f"Debug: geodesic_length={batch.geodesic_length}")

        batch = batch.to(args.device)
        x, edge, atom_type, node2graph, num_nodes = make_inputs(
            batch,
            config=config,
            seed=args.seed,
            dtype=dtype
        )

        # natoms = torch.unique_consecutive(batch.batch, return_counts=True)[1]
        natoms = batch.ptr[1:] - batch.ptr[:-1]

        # x.shape == (natoms, nimages, 3)
        x_start = x[:, 0, :].contiguous()
        x_end = x[:, -1, :].contiguous()


        print(batch.x)
        from make_q_dot import make_q_dot, linear_q_dot
        def remap2symbol(atomic_numbers):
            MAPPING = {0: "H", 1: "C", 2: "N", 3: "O"}
            symbols = [MAPPING[num.item()] for num in atomic_numbers]
            return symbols

        x_start_list = x_start.split(natoms.tolist())
        x_end_list = x_end.split(natoms.tolist())
        # (nbatch,)---(natoms, 3)
        x_inp = [torch.stack((x_T, x_0)) for x_T, x_0 in zip(x_start_list, x_end_list)]
        # x_inp.shape == (nbatch,)---(2, natoms, 3)

        symbols = batch.x.split(natoms.tolist())
        symbols = [remap2symbol(sym) for sym in symbols]

        # q_dot = make_q_dot(x_inp, symbols, nimages=30, tol=1e-4, maxiter=1000)
        q_dot = make_q_dot(x_inp, symbols)
        # q_dot = linear_q_dot(x_inp, symbols)


        ## solve function
        from solver_test import solve
        res = solve(
            x_start,
            q_dot,
            edge,
            atom_type,
            node2graph,
            num_nodes,
            solver=solver,
            config=config,
            verbose=args.verbose,
            dtype=dtype,
        )

        out = res[2]
        x_out = out["x"]
        q_out = out["q"]

        rmsd = [RMSD(x1, x2).item() for x1, x2 in zip(x_out.split(natoms.tolist()), x_end_list)]
        dmae = [DMAE(x1, x2).item() for x1, x2 in zip(x_out.split(natoms.tolist()), x_end_list)]
        dmae2 = [DMAE(x1, x2).item() for x1, x2 in zip(x_start_list, x_end_list)]
        rmsd = torch.tensor(rmsd); dmae = torch.tensor(dmae); dmae2 = torch.tensor(dmae2)
        print(f"Debug: [i={i}] RMSD(end, out): \n{rmsd}")
        print(F"DEbug: [i={i}] DMAE(end, out): \n{dmae}")
        print(F"DEbug: [i={i}] DMAE(start, end): \n{dmae2}")

        # if i > 10:
        #     break
        # break

    """
    TODO:
    dq를 geodesic interpolation으로 계산하고,
    solver_test.solve()함수를 이용해서 update시키기.

    edge, atom_Type, node2graph등 solve()에 필요한 inputs을 만드는 코드는, round_trip_init()을 참고.

    padding-version을 geometric-sparse-version으로 복원하는 코드는 solver_test나 apply_noise함수의 뒷부분에서 return하기 전에 수앻된다. 

    SDE sampling은 diffusion/diffusion_model.py의 sample_batch 함수를 참고.
    """
