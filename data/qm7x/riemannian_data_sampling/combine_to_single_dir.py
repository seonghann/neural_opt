import shutil
from pathlib import Path


t1 = 150
times = 30

seed_list = list(range(1, times))
source_dirs = [
    f"xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t{t1}_seed{seed}_train"
    for seed in seed_list
]
source_dirs += [
    f"xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t{t1}_seed42_train",
    f"xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t{t1}_seed42_val",
    f"xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t{t1}_seed42",  # test set
]

target_dir = Path(f"xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t{t1}_{times}times")
target_dir.mkdir(exist_ok=True, parents=True)

# Copying and counting
total_copied = 0

for src in source_dirs:
    count = 0
    for xyz_file in Path(src).glob("*.xyz"):
        shutil.copy(xyz_file, target_dir / xyz_file.name)
        count += 1
    total_copied += count
    print(f"[{src}] → {count} files copied.")

# Check the final count of .xyz files in the target directory
final_count = len(list(target_dir.glob("*.xyz")))
print(f"\n✅ Total .xyz files in '{target_dir}': {final_count}")
if final_count < total_copied:
    print("⚠️ Warning: Some files may have been overwritten due to name collisions.")
