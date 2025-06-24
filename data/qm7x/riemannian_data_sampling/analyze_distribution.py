"""
Calculate RMSD, DMAE, q_norm(geodesic length) and save it as csv file.
"""

import sys
import os
import torch
from utils.chem import ATOMIC_NUMBERS


# Add the project root directory to sys.path for upper-level imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from evaluate_accuracy import GeometryMetrics


def map_atomic_numbers(atom_numbers):
    atom_type = [ATOMIC_NUMBERS[atom_number] for atom_number in atom_numbers]
    atom_type = torch.tensor(atom_type, dtype=torch.long)
    return atom_type


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml", type=str, required=True, help="config yaml file path"
    )
    parser.add_argument("--save_csv", type=str, required=True, help="save as csv file")
    # TODO: Change arg name
    parser.add_argument(
        "--mmff_xyz_path", type=str, required=True, help="path of mmff xyz files"
    )
    parser.add_argument("--xyz_path", type=str, required=True, help="path of xyz files")
    parser.add_argument(
        "--t0_x", type=int, default=0, help="t0 of time schedule (default: 0)"
    )
    parser.add_argument(
        "--t1_x", type=int, default=1500, help="t1 of time schedule (default: 1500)"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
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
    args = parser.parse_args()
    print(args)
    ##########################################################3

    # TODO: Check os.path.exists of xyz path

    import pandas as pd
    from diffusion.noise_scheduler import load_noise_scheduler
    from omegaconf import OmegaConf
    from utils.geodesic_solver import GeodesicSolver

    config = OmegaConf.load(args.config_yaml)
    config.manifold.ode_solver.alpha = args.alpha
    config.manifold.ode_solver.beta = args.beta
    config.manifold.ode_solver.gamma = args.gamma
    print("config=\n", config)

    geodesic_solver = GeodesicSolver(config.manifold)
    noise_schedule = load_noise_scheduler(config.diffusion)
    # q_type = "morse"

    metrics_calculator = GeometryMetrics(geodesic_solver)

    ## Read xyz
    ## Compare noise distributions
    from math import sqrt
    import numpy as np
    import torch
    from glob import glob
    from ase.io import iread
    # from eval_accuracy2 import calc_RMSD2, calc_DMAE, calc_q_norm

    torch.manual_seed(args.seed)

    filenames = glob(f"{args.xyz_path}/*.xyz")

    results = {
        "time_step_x": [],
        "time_step_q": [],
        "idx": [],
        #
        "rmsd": [],
        "dmae": [],
        "q_norm": [],
        #
        "_rmsd": [],
        "_dmae": [],
        "_q_norm": [],
        #
        "__rmsd": [],
        "__dmae": [],
        "__q_norm": [],
    }

    for i, filename in enumerate(filenames):
        print(f"Debug: filename={filename}")
        atoms = list(iread(filename))
        assert len(atoms) == 3

        info = atoms[0].info
        idx = info["idx"]
        time_step = info["time_step"]
        q_target = info["q_target"]
        results["idx"].append(idx)
        results["time_step_q"].append(time_step)

        atoms_0 = atoms[0]  # DFT structure
        atoms_t2 = atoms[1]  # riemannian sampled structure

        ## Cartesian noised structure
        atoms_t = atoms_0.copy()
        time_step = torch.randint(args.t0_x, args.t1_x, size=(1,))  # , device=device)
        results["time_step_x"].append(time_step.item())
        a = noise_schedule.get_alpha(time_step).item()
        pos_noise = np.random.randn(*atoms_0.positions.shape)
        pos_noise *= sqrt(1.0 - a) / sqrt(a)
        atoms_t.positions += pos_noise

        ## MMFF structure
        # raw_datadir: /home/share/DATA/QM9M/MMFFtoDFT_input
        mmff_filepath = f"{args.mmff_xyz_path}/idx{idx}.xyz"
        atoms_T = list(iread(mmff_filepath))[1]  # MMFF structure

        pos_0 = torch.from_numpy(atoms_0.positions)
        pos_t = torch.from_numpy(atoms_t.positions)
        pos_t2 = torch.from_numpy(atoms_t2.positions)
        pos_T = torch.from_numpy(atoms_T.positions)

        _atom_type = map_atomic_numbers(atoms_0.get_atomic_numbers())

        # 1) noise distribution in Euclidean
        rmsd = GeometryMetrics.calc_rmsd_aligned(atoms_0, atoms_t)
        dmae = GeometryMetrics.calc_dmae(pos_0, pos_t).item()
        q_norm = metrics_calculator.calc_q_norm(pos_0, pos_t, _atom_type).item()
        results["rmsd"].append(rmsd)
        results["dmae"].append(dmae)
        results["q_norm"].append(q_norm)

        # 2) noise distribution in Riemannian
        rmsd = GeometryMetrics.calc_rmsd_aligned(atoms_0, atoms_t2)
        dmae = GeometryMetrics.calc_dmae(pos_0, pos_t2).item()
        q_norm = metrics_calculator.calc_q_norm(pos_0, pos_t2, _atom_type).item()
        results["_rmsd"].append(rmsd)
        results["_dmae"].append(dmae)
        results["_q_norm"].append(q_norm)

        # 3) distribution of MMFF structures
        rmsd = GeometryMetrics.calc_rmsd_aligned(atoms_0, atoms_T)
        dmae = GeometryMetrics.calc_dmae(pos_0, pos_T).item()
        q_norm = metrics_calculator.calc_q_norm(pos_0, pos_T, _atom_type).item()
        results["__rmsd"].append(rmsd)
        results["__dmae"].append(dmae)
        results["__q_norm"].append(q_norm)

    df = pd.DataFrame(results)
    df = df.sort_values(by="time_step_q")
    pd.options.display.max_rows = None
    print(df)
    if args.save_csv:
        df.to_csv(args.save_csv)
        print(f"Save {args.save_csv}")

    print("Mean error: Cartesian sampling, Riemannian sampling, MMFF")
    print("RMSD", df["rmsd"].mean(), df["_rmsd"].mean(), df["__rmsd"].mean())
    print("DMAE", df["dmae"].mean(), df["_dmae"].mean(), df["__dmae"].mean())
    print("q_norm", df["q_norm"].mean(), df["_q_norm"].mean(), df["__q_norm"].mean())

    print("Median error: Cartesian sampling, Riemannian sampling, MMFF")
    print("RMSD", df["rmsd"].median(), df["_rmsd"].median(), df["__rmsd"].median())
    print("DMAE", df["dmae"].median(), df["_dmae"].median(), df["__dmae"].median())
    print(
        "q_norm", df["q_norm"].median(), df["_q_norm"].median(), df["__q_norm"].median()
    )
