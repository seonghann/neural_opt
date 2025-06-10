import os
import argparse
import glob
import pickle
import re


def find_wrong_smarts_samples(prb_dir):
    """
    Find XYZ files with smarts="None" in comment line

    Parameters:
    -----------
    prb_dir : str
        Directory containing XYZ files

    Returns:
    --------
    list : List of indices with problematic SMARTS
    """

    wrong_indices = []
    xyz_files = glob.glob(os.path.join(prb_dir, "*.xyz"))

    print(f"Scanning {len(xyz_files)} XYZ files in {prb_dir}")
    print("-" * 50)

    for xyz_file in sorted(xyz_files):
        try:
            with open(xyz_file, "r") as f:
                lines = f.readlines()

                # Check if file has at least 2 lines (atom count and comment)
                if len(lines) < 2:
                    continue

                # Second line is the comment line
                comment_line = lines[1].strip()

                # Check if smarts="None" exists in comment
                if 'smarts="None"' in comment_line:
                    # Extract filename without extension
                    basename = os.path.basename(xyz_file)
                    filename = os.path.splitext(basename)[0]

                    # Extract number from filename (e.g., 'idx123' -> 123)
                    match = re.search(r"idx(\d+)", filename)
                    if match:
                        idx_number = int(match.group(1))
                        wrong_indices.append(idx_number)
                        print(
                            f"Found problematic sample: {filename} (index: {idx_number})"
                        )
                    else:
                        print(
                            f"Warning: Could not extract number from filename: {filename}"
                        )

        except Exception as e:
            print(f"Error reading {xyz_file}: {e}")

    return wrong_indices


def main():
    parser = argparse.ArgumentParser(
        description="Find XYZ samples with problematic SMARTS"
    )
    parser.add_argument(
        "--prb_dir",
        type=str,
        required=True,
        help="Directory containing XYZ files to process",
    )
    parser.add_argument(
        "--save_index",
        type=str,
        default=None,
        help="Save problematic indices as pickle file (e.g., wrong_samples.pkl)",
    )

    args = parser.parse_args()

    # Check if directory exists
    if not os.path.exists(args.prb_dir):
        print(f"Error: Directory {args.prb_dir} does not exist")
        return

    # Find wrong samples
    wrong_indices = find_wrong_smarts_samples(args.prb_dir)

    # Sort indices in ascending order
    wrong_indices.sort()

    # Print summary
    print("\n" + "=" * 50)
    print(f"SUMMARY: Found {len(wrong_indices)} problematic samples")
    print("=" * 50)

    if wrong_indices:
        print("Problematic sample indices (sorted):")
        print("Indices:", wrong_indices)

        # Save to pickle file if requested
        if args.save_index:
            try:
                with open(args.save_index, "wb") as f:
                    pickle.dump(wrong_indices, f)
                print(f"Problematic indices saved as pickle to: {args.save_index}")
                print(f"Data type: List[int] with {len(wrong_indices)} elements")
            except Exception as e:
                print(f"Error saving pickle file: {e}")
    else:
        print("No problematic samples found!")

        # Save empty list if requested
        if args.save_index:
            try:
                with open(args.save_index, "wb") as f:
                    pickle.dump([], f)
                print(f"Empty list saved as pickle to: {args.save_index}")
            except Exception as e:
                print(f"Error saving pickle file: {e}")


if __name__ == "__main__":
    main()
