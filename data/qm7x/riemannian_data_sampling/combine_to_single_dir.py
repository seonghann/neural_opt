import shutil
from pathlib import Path

# 9개 원본 폴더 목록
source_dirs = [
    # "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed1",
    "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed1_train",
    # "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed1_val",
    # "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed2",
    "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed2_train",
    # "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed2_val",
    "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed42",
    "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed42_train",
    "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed42_val",
]
target_dir = Path("xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed42_1_2")

source_dirs = [
    "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed1_train",
    "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed2_train",
    "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed3_train",
    "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed4_train",
    "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed5_train",
    "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed6_train",
    "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed7_train",
    "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed8_train",
    "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed9_train",

    "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed42",
    "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed42_train",
    "xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed42_val",
]
target_dir = Path("xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_10times")


target_dir.mkdir(exist_ok=True, parents=True)

# 복사 및 카운팅
total_copied = 0

for src in source_dirs:
    count = 0
    for xyz_file in Path(src).glob("*.xyz"):
        shutil.copy(xyz_file, target_dir / xyz_file.name)
        count += 1
    total_copied += count
    print(f"[{src}] → {count} files copied.")

# 최종 파일 수 확인 (실제로 target에 있는 .xyz 파일 수)
final_count = len(list(target_dir.glob("*.xyz")))
print(f"\n✅ Total .xyz files in '{target_dir}': {final_count}")
if final_count < total_copied:
    print("⚠️ Warning: Some files may have been overwritten due to name collisions.")
