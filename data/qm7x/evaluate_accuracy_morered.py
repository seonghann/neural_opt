import sys
import os
import argparse
from typing import List, Dict, Any

import torch
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

# Local module import from the same directory
from pt_to_xyz import split_batch_by_idx_m

# Add the project root directory to sys.path for upper-level imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from evaluate_accuracy import (
    EvaluationMetrics,
    GeometryMetrics,
    ResultsHandler,
    GeodesicSolver,
)


class SimplifiedMolecularEvaluator:
    """Simplified evaluator that reuses base components for alternative data format."""

    def __init__(self, config_path: str):
        self.config = OmegaConf.load(config_path)
        self.geodesic_solver = GeodesicSolver(self.config.manifold)
        self.metrics_calculator = GeometryMetrics(self.geodesic_solver)
        self.results_handler = ResultsHandler()

    def load_test_data(self, data_path: str) -> List[Any]:
        """Load test data from the specified path."""
        test_dataloader = torch.load(data_path, map_location="cpu", weights_only=False)
        print(f"Loaded {len(test_dataloader)} batches from {data_path}")
        return test_dataloader

    def evaluate_sample(self, sample: Dict[str, torch.Tensor]) -> EvaluationMetrics:
        """Evaluate a single molecular sample using the base calculator."""
        atomic_numbers = sample["_atomic_numbers"]
        pos_ref = sample["_positions_eq"]
        pos_xT = sample["_positions_noneq"]
        pos_gen = sample["_positions"]

        # Use the shared metrics calculator
        return self.metrics_calculator.calculate_metrics(
            pos_ref, pos_gen, pos_xT, None, atomic_numbers
        )

    def evaluate_all_samples(
        self, test_dataloader: List[Any]
    ) -> List[EvaluationMetrics]:
        """Evaluate all samples in the test dataloader."""
        all_metrics = []

        for batch in tqdm(
            test_dataloader, desc="Processing batches", total=len(test_dataloader)
        ):
            split_batch = split_batch_by_idx_m(batch)

            for sample in split_batch:
                metrics = self.evaluate_sample(sample)
                all_metrics.append(metrics)

        return all_metrics

    def run_evaluation(self, data_path: str, save_csv: str = None) -> pd.DataFrame:
        """Run the complete evaluation pipeline using shared components."""
        # Load data
        test_dataloader = self.load_test_data(data_path)

        # Evaluate all samples
        all_metrics = self.evaluate_all_samples(test_dataloader)

        # Process results using shared handler
        self.results_handler.configure_pandas_display()
        df = self.results_handler.create_dataframe_from_metrics(all_metrics)
        print("Sorted with rmsd")
        print(df)

        # Save results if requested
        if save_csv:
            df.to_csv(save_csv, index=False)
            print(f"Saved results to {save_csv}")

        # Print statistics using shared handler
        self.results_handler.print_statistics(df)

        return df


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate molecular generation accuracy using shared components"
    )
    parser.add_argument(
        "--config_yaml", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--prb_pt", type=str, required=True, help="Path to test data (.pt file)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Directory to save xyz files (currently unused)",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="Path to save accuracy results as CSV file",
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    print(f"Arguments: {args}")

    # Create simplified evaluator and run evaluation
    evaluator = SimplifiedMolecularEvaluator(args.config_yaml)
    results_df = evaluator.run_evaluation(args.prb_pt, args.save_csv)

    return results_df


if __name__ == "__main__":
    main()
