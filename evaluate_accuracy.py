"""
Molecular geometry evaluation script for calculating RMSD, DMAE, and q_norm metrics.
Refactored for improved readability and maintainability while preserving functionality.
This is the base module that contains shared functionality.
"""

import argparse
import os
import pickle
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from ase import Atoms
from ase.build.rotate import minimize_rotation_and_translation

from utils.chem import ATOMIC_NUMBERS
from dataset.data_module import load_datamodule
from utils.geodesic_solver import GeodesicSolver


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    rmsd: float
    dmae: float
    q_norm: float
    rmsd_xT: float
    dmae_xT: float
    q_norm_xT: float


class GeometryMetrics:
    """Static methods for calculating molecular geometry metrics."""

    def __init__(self, geodesic_solver: Optional[GeodesicSolver] = None):
        self.geodesic_solver = geodesic_solver

    @staticmethod
    def calc_rmsd(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate RMSD between two position arrays."""
        return np.sqrt(np.mean(np.sum((pos1 - pos2) ** 2, axis=1)))

    @staticmethod
    def calc_rmsd_aligned(
        mol1: Atoms, mol2: Atoms, mass_weighted: bool = False
    ) -> float:
        """Calculate RMSD after optimal alignment."""
        minimize_rotation_and_translation(mol1, mol2)
        p1, p2 = mol1.positions, mol2.positions

        if mass_weighted:
            mass = mol1.get_masses()
            rmsd = np.sqrt(
                np.mean(np.sum(mass.reshape(-1, 1) * (p1 - p2) ** 2, axis=1))
            )
        else:
            rmsd = np.sqrt(np.mean(np.sum((p1 - p2) ** 2, axis=1)))

        return rmsd

    @staticmethod
    def calc_dmae(pos_ref: torch.Tensor, pos_prb: torch.Tensor) -> torch.Tensor:
        """Calculate Distance Matrix Absolute Error."""
        dm_ref = torch.cdist(pos_ref, pos_ref)
        dm_prb = torch.cdist(pos_prb, pos_prb)
        dmae = torch.triu(abs(dm_ref - dm_prb), diagonal=1).sum()
        dmae /= len(pos_ref) * (len(pos_ref) - 1) / 2
        return dmae

    def calc_q_norm(
        self, pos_ref: torch.Tensor, pos_prb: torch.Tensor, atom_type: torch.Tensor
    ) -> torch.Tensor:
        """Calculate q-norm using geodesic solver."""
        if self.geodesic_solver is None:
            raise ValueError("GeodesicSolver is required for q_norm calculation")

        natoms = len(pos_ref)
        edge_index = torch.triu_indices(natoms, natoms, offset=1)

        q_prb = self.geodesic_solver.compute_q(edge_index, atom_type, pos_prb)
        q_ref = self.geodesic_solver.compute_q(edge_index, atom_type, pos_ref)

        norm_err = (q_prb - q_ref).square().sum().sqrt()
        return norm_err

    @staticmethod
    def atomic_number_to_index(atom_numbers: torch.Tensor) -> torch.Tensor:
        """Convert atomic numbers to indices using ATOMIC_NUMBERS mapping."""
        indices = [ATOMIC_NUMBERS[atom_number.item()] for atom_number in atom_numbers]
        return torch.tensor(
            indices, dtype=atom_numbers.dtype, device=atom_numbers.device
        )

    def calculate_metrics(
        self,
        pos_ref: torch.Tensor,
        pos_gen: torch.Tensor,
        pos_xT: torch.Tensor,
        atom_type_tensor: torch.Tensor,
        atomic_numbers: Optional[torch.Tensor] = None,
    ) -> EvaluationMetrics:
        """Calculate all metrics for a molecule."""
        # Convert atom types if needed
        if atomic_numbers is not None:
            atom_symbols = [atom_number for atom_number in atomic_numbers]
            atom_type_indices = self.atomic_number_to_index(atomic_numbers)
        else:
            index_to_atomic = {v: k for k, v in ATOMIC_NUMBERS.items()}
            atom_symbols = [index_to_atomic[num.item()] for num in atom_type_tensor]
            atom_type_indices = atom_type_tensor

        # Create ASE Atoms objects
        atoms_ref = Atoms(symbols=atom_symbols, positions=pos_ref)
        atoms_gen = Atoms(symbols=atom_symbols, positions=pos_gen)
        atoms_xT = Atoms(symbols=atom_symbols, positions=pos_xT)

        # Calculate metrics for generated vs reference
        rmsd = self.calc_rmsd_aligned(atoms_ref, atoms_gen).item()
        dmae = self.calc_dmae(pos_ref, pos_gen).item()
        q_norm = self.calc_q_norm(pos_ref, pos_gen, atom_type_indices).item()

        # Calculate metrics for xT vs reference
        rmsd_xT = self.calc_rmsd_aligned(atoms_ref, atoms_xT).item()
        dmae_xT = self.calc_dmae(pos_ref, pos_xT).item()
        q_norm_xT = self.calc_q_norm(pos_ref, pos_xT, atom_type_indices).item()

        return EvaluationMetrics(
            rmsd=rmsd,
            dmae=dmae,
            q_norm=q_norm,
            rmsd_xT=rmsd_xT,
            dmae_xT=dmae_xT,
            q_norm_xT=q_norm_xT,
        )


class MolecularAlignment:
    """Handles molecular alignment operations."""

    @staticmethod
    def get_substruct_matches(smarts: str) -> List[Tuple]:
        """Get substructure matches from SMARTS pattern."""
        from rdkit import Chem

        def _get_substruct_matches(smarts_pattern: str) -> List[Tuple]:
            mol = Chem.MolFromSmarts(smarts_pattern)
            matches = list(mol.GetSubstructMatches(mol, uniquify=False))
            atom_map = np.array([atom.GetAtomMapNum() for atom in mol.GetAtoms()]) - 1
            map_inv = np.argsort(atom_map)

            for i in range(len(matches)):
                matches[i] = tuple(atom_map[np.array(matches[i])[map_inv]])

            return matches

        smarts_list = smarts.split(">>")

        if len(smarts_list) == 2:
            smarts_r, smarts_p = smarts_list
            matches_r = _get_substruct_matches(smarts_r)
            matches_p = _get_substruct_matches(smarts_p)
            matches = set(matches_r) & set(matches_p)
        elif len(smarts_list) == 1:
            matches = _get_substruct_matches(smarts_list[0])
            matches = set(matches)
        else:
            raise ValueError("Invalid SMARTS pattern format")

        matches = list(matches)
        matches.sort()
        return matches

    @staticmethod
    def get_min_rmsd_match(
        matches: List[Tuple], ref_atoms: Atoms, prb_atoms: Atoms
    ) -> Tuple[List, float, float]:
        """Find the match with minimum RMSD."""
        rmsds = []
        ref_atoms = ref_atoms.copy()
        prb_atoms = prb_atoms.copy()

        ref_pos = ref_atoms.positions.copy()
        prb_pos = prb_atoms.positions.copy()

        for match in matches:
            prb_atoms.positions = prb_pos[list(match)].copy()
            minimize_rotation_and_translation(ref_atoms, prb_atoms)
            match_pos = prb_atoms.positions
            rmsd = GeometryMetrics.calc_rmsd(ref_pos, match_pos)
            rmsds.append(rmsd)

        best_idx = rmsds.index(min(rmsds))
        return list(matches[best_idx]), rmsds[0], min(rmsds)

    @staticmethod
    def align_jhwoo(
        ref_atoms: Atoms,
        prb_atoms: Atoms,
        smarts: str,
        silent: bool = False,
        target: str = "RMSD",
    ) -> torch.Tensor:
        """Align molecules using jhwoo method."""
        matches = MolecularAlignment.get_substruct_matches(smarts)

        ref_pos = ref_atoms.positions
        prb_pos = prb_atoms.positions

        if target == "DMAE":
            # TODO: Fix this line (copy the code here)
            # Refer to: /home/share/DATA/NeuralOpt/utils/alignXYZ.py
            import sys

            sys.path.append("/home/share/DATA/NeuralOpt/utils")
            from alignXYZ import get_min_dmae_match

            match, init_target, final_target = get_min_dmae_match(
                matches, ref_pos, prb_pos
            )
        elif target == "RMSD":
            match, init_target, final_target = MolecularAlignment.get_min_rmsd_match(
                matches, ref_atoms, prb_atoms
            )
        else:
            raise NotImplementedError(
                f"target={target}, should be one of ['DMAE', 'RMSD']"
            )

        if not silent and abs(init_target - final_target) > 1e-5:
            print(f"Alignment: {init_target} -> {final_target}")

        ret_pos = prb_pos[match]
        return torch.from_numpy(ret_pos)


class ResultsHandler:
    """Handles saving and displaying results."""

    @staticmethod
    def configure_pandas_display():
        """Configure pandas display options for better output formatting."""
        pd.set_option("display.float_format", "{:.3f}".format)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_seq_items", None)
        pd.set_option("display.max_colwidth", 500)
        pd.set_option("expand_frame_repr", True)

    @staticmethod
    def create_dataframe_from_metrics(
        metrics_list: List[EvaluationMetrics], data_idx_list: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Create a pandas DataFrame from list of metrics."""
        data = {
            "rmsd_xT": [m.rmsd_xT for m in metrics_list],
            "rmsd": [m.rmsd for m in metrics_list],
            "dmae_xT": [m.dmae_xT for m in metrics_list],
            "dmae": [m.dmae for m in metrics_list],
            "q_norm_xT": [m.q_norm_xT for m in metrics_list],
            "q_norm": [m.q_norm for m in metrics_list],
        }

        if data_idx_list is not None:
            data["data_idx"] = data_idx_list

        df = pd.DataFrame(data)
        return df.sort_values(by="rmsd")

    @staticmethod
    def print_statistics(df: pd.DataFrame):
        """Print comprehensive statistics of the evaluation results."""
        print("=" * 100)
        print("Statistics")
        print("# of data: ", len(df))

        # Convert tensor columns to numpy for statistics
        metrics = ["rmsd", "dmae", "q_norm"]
        stats_data = {}

        for metric in metrics:
            stats_data[metric] = torch.tensor(df[metric].tolist())
            stats_data[f"{metric}_xT"] = torch.tensor(df[f"{metric}_xT"].tolist())

        print(f"[x0 vs predicted]")
        for metric in metrics:
            data = stats_data[metric]
            print(
                f"{metric.upper():6} (mean, median): {data.mean():.3g}, {data.median():.3g}"
            )

        print(f"[x0 vs xT]")
        for metric in metrics:
            data = stats_data[f"{metric}_xT"]
            print(
                f"{metric.upper():6} (mean, median): {data.mean():.3g}, {data.median():.3g}"
            )

        print(f"x0 vs xT")
        for metric in metrics:
            data = stats_data[f"{metric}_xT"]
            q1, q3 = np.percentile(data, 25), np.percentile(data, 75)
            print(f"{metric.upper():6} (Q1, Q3): {q1:.3g}, {q3:.3g}")

        print("=" * 100)

    @staticmethod
    def save_xyz_files(
        save_dir: str,
        pos_ref_list: List[torch.Tensor],
        predicted_data: Dict[str, List],
        data_idx_list: List[int],
        df: pd.DataFrame,
        index_to_atomic: Dict,
    ):
        """Save molecular structures as XYZ files."""
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory {save_dir}")

        pos_list = predicted_data["pos_list"]
        atom_type_list = predicted_data["atom_type_list"]
        smarts_list = predicted_data["smarts_list"]

        # Create lookup dictionary for metrics
        metrics_lookup = df.set_index("data_idx").to_dict("index")

        for i, (pos_ref, pos_gen) in enumerate(zip(pos_ref_list, pos_list)):
            idx = data_idx_list[i]

            if idx not in metrics_lookup:
                continue  # Skip if not in results (e.g., banned)

            atom_type_tensor = atom_type_list[i]
            smarts = smarts_list[i]

            atom_symbols = [index_to_atomic[num.item()] for num in atom_type_tensor]
            atoms_gen = Atoms(symbols=atom_symbols, positions=pos_gen)

            metrics = metrics_lookup[idx]
            comment = (
                f'idx={idx} smarts="{smarts}" '
                f"rmsd={metrics['rmsd']:.6f} dmae={metrics['dmae']:.6f} "
                f"q_norm={metrics['q_norm']:.6f}"
            )

            filename = os.path.join(save_dir, f"idx{idx}.xyz")
            atoms_gen.write(filename, comment=comment)
            print(f"Saved {filename}")


class MolecularEvaluator:
    """Main class for evaluating molecular geometry predictions."""

    def __init__(
        self, config_path: str, geodesic_solver: Optional[GeodesicSolver] = None
    ):
        self.config = OmegaConf.load(config_path)
        if geodesic_solver is None:
            self.geodesic_solver = GeodesicSolver(self.config.manifold)
        else:
            self.geodesic_solver = geodesic_solver
        self.datamodule = load_datamodule(self.config)
        self.index_to_atomic = {v: k for k, v in ATOMIC_NUMBERS.items()}
        self.metrics_calculator = GeometryMetrics(self.geodesic_solver)

    def load_reference_data(self) -> Tuple[List[torch.Tensor], List[int]]:
        """Load reference positions and data indices from test dataloader."""
        pos_ref_list = []
        data_idx_list = []

        for batch in self.datamodule.test_dataloader():
            natoms = batch.batch.bincount()
            pos_ref = batch.pos[:, 0, :].split(natoms.tolist())
            pos_ref_list.extend(pos_ref)
            data_idx_list.extend(batch.idx)

        return pos_ref_list, data_idx_list

    def load_predicted_data(self, prb_pt_path: str) -> Dict[str, List]:
        """Load predicted positions and related data from file."""
        data = torch.load(prb_pt_path, map_location="cpu", weights_only=False)
        print(f"Load {prb_pt_path}")

        pos_list = []
        xT_list = []  # starting geometries (e.g., MMFF of MMFFtoDFT)
        atom_type_list = []
        smarts_list = []

        for batch in data:
            natoms = batch.batch.bincount()

            pos = batch.pos.split(natoms.tolist())
            pos_list.extend(pos)

            xT = batch.pos_traj[0].split(natoms.tolist())
            xT_list.extend(xT)

            atom_type = batch.atom_type.split(natoms.tolist())
            atom_type_list.extend(atom_type)

            smarts_list.extend(batch.smarts)

        return {
            "pos_list": pos_list,
            "xT_list": xT_list,
            "atom_type_list": atom_type_list,
            "smarts_list": smarts_list,
        }

    def evaluate_all_molecules(
        self,
        pos_ref_list: List[torch.Tensor],
        predicted_data: Dict[str, List],
        data_idx_list: List[int],
        align_target: str = "none",
        ban_index: List[int] = None,
    ) -> pd.DataFrame:
        """Evaluate all molecules and return DataFrame with results."""
        if ban_index is None:
            ban_index = []

        metrics_list = []
        valid_data_idx_list = []

        pos_list = predicted_data["pos_list"]
        xT_list = predicted_data["xT_list"]
        atom_type_list = predicted_data["atom_type_list"]
        smarts_list = predicted_data["smarts_list"]

        for i, (pos_ref, pos_gen, xT) in enumerate(
            zip(pos_ref_list, pos_list, xT_list)
        ):
            data_idx = data_idx_list[i]

            if data_idx in ban_index:
                print(f"Pass data_idx={data_idx} because it's in ban_index.")
                continue

            # Prepare atomic structures
            atom_type_tensor = atom_type_list[i]
            smarts = smarts_list[i]
            atom_symbols = self._remap_to_atomic_symbols(atom_type_tensor)

            atoms_ref = Atoms(symbols=atom_symbols, positions=pos_ref)
            atoms_gen = Atoms(symbols=atom_symbols, positions=pos_gen)
            atoms_xT = Atoms(symbols=atom_symbols, positions=xT)

            # Apply alignment if requested
            if align_target.lower() != "none":
                pos_gen = self._align_molecules(
                    atoms_ref, atoms_gen, smarts, align_target
                )
                xT = self._align_molecules(atoms_ref, atoms_xT, smarts, align_target)

            # Calculate metrics
            molecule_metrics = self.metrics_calculator.calculate_metrics(
                pos_ref, pos_gen, xT, atom_type_tensor
            )

            # Store results
            metrics_list.append(molecule_metrics)
            valid_data_idx_list.append(data_idx)

        # Convert to DataFrame and sort
        df = ResultsHandler.create_dataframe_from_metrics(
            metrics_list, valid_data_idx_list
        )
        print("Sorted with rmsd")

        return df

    def _remap_to_atomic_symbols(self, atomic_numbers: torch.Tensor) -> List[str]:
        """Convert atomic numbers to atomic symbols."""
        return [self.index_to_atomic[num.item()] for num in atomic_numbers]

    def _align_molecules(
        self, ref_atoms: Atoms, prb_atoms: Atoms, smarts: str, target: str
    ) -> torch.Tensor:
        """Align molecules using the specified target metric."""
        return MolecularAlignment.align_jhwoo(
            ref_atoms, prb_atoms, smarts, target=target
        )


def load_ban_index(ban_index_path: Optional[str]) -> List[int]:
    """Load banned indices from pickle file."""
    if ban_index_path is None:
        return []

    with open(ban_index_path, "rb") as f:
        return pickle.load(f)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate molecular geometry predictions using RMSD, DMAE, and q_norm metrics"
    )
    parser.add_argument(
        "--config_yaml", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--prb_pt", type=str, required=True, help="Path to predicted results .pt file"
    )
    parser.add_argument(
        "--align_target",
        type=str,
        default="none",
        choices=["DMAE", "RMSD", "none"],
        help="Alignment target metric",
    )
    parser.add_argument(
        "--save_dir", type=str, default="", help="Directory to save XYZ files"
    )
    parser.add_argument(
        "--ban_index",
        type=str,
        default=None,
        help="Path to pickle file containing banned indices",
    )
    parser.add_argument(
        "--save_csv", type=str, default=None, help="Path to save results CSV file"
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    print(args)

    # Initialize components
    geodesic_solver = GeodesicSolver(OmegaConf.load(args.config_yaml).manifold)
    evaluator = MolecularEvaluator(args.config_yaml, geodesic_solver)

    # Load data
    ban_index = load_ban_index(args.ban_index)
    pos_ref_list, data_idx_list = evaluator.load_reference_data()
    predicted_data = evaluator.load_predicted_data(args.prb_pt)

    # Evaluate molecules
    results_df = evaluator.evaluate_all_molecules(
        pos_ref_list, predicted_data, data_idx_list, args.align_target, ban_index
    )

    # Configure display and show results
    ResultsHandler.configure_pandas_display()
    print(results_df)

    # Save CSV if requested
    if args.save_csv:
        results_df.to_csv(args.save_csv, index=False)
        print(f"Saved {args.save_csv}")

    # Print statistics
    ResultsHandler.print_statistics(results_df)

    # Save XYZ files if requested
    if args.save_dir:
        ResultsHandler.save_xyz_files(
            args.save_dir,
            pos_ref_list,
            predicted_data,
            data_idx_list,
            results_df,
            evaluator.index_to_atomic,
        )


if __name__ == "__main__":
    main()
