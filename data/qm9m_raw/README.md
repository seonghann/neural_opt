# QM9M Dataset Processing Directory

This directory is dedicated to processing the QM9M dataset.
It includes scripts and data splits for converting raw `.sdf` files into usable `.xyz` files containing both DFT and MMFF geometries.

- **Source**: https://yzhang.hpc.nyu.edu/IMA/
- **Reference**:  
  Lu, J., Wang, C., & Zhang, Y. (2019).
  *Predicting molecular energy using force-field optimized geometries and atomic vector representations learned from an improved deep tensor neural network*.  
  *Journal of Chemical Theory and Computation, 15(7), 4113–4121.*

---

## Directory Structure

- `./qm9_mmff.tar.bz2`: Compressed archive of raw SDF data (downloaded from source)
- `./qm9_mmff/`: Extracted `.sdf` files
  Contains `{i}.sdf` and `{i}.mmff.sdf` for `i=1` to `133885`
- `./data_split.py`, `./data_split.pkl`:
  Script and file for random data splitting into train/validation/test sets
- `./preprocessing.py`:
  Converts `.sdf` files into `.xyz` files containing both DFT and MMFF geometries; also adds SMARTS identifiers
- `./MMFFtoDFT_input/`:
  Contains 133,885 processed `.xyz` files, each including both DFT and MMFF geometries
- `./process_wrong_samples.py`:
  Identifies problematic samples in the test set (e.g., exact matches between MMFF & DFT, or multi-molecule structures)
- `./riemannian_data_sampling/`:
  Directory for data sampling on a Riemmanian manifold

---

## How to Use

### 1. Download and extract the QM9M dataset

```bash
mkdir qm9_mmff
tar -xvjf qm9_mmff.tar.bz2 -C qm9_mmff
```

### 2. Generate processed `.xyz` files from `.sdf`
```bash
python preprocessing.py --sdf_path ./qm9_mmff --save_dir ./MMFFtoDFT_input
```

### 3. Identify problematic samples in the test set
```bash
python process_wrong_samples.py
```

This will save a list of erroneous indices in `wrong_samples.pkl`.


### 4. Perform data splitting (train/validation/test)
```bash
python data_split.py
```

This will generate data_split.pkl containing:

- `train_index`: 100,000 samples
- `valid_index`: 23,885 samples
- `test_index`: 10,000 samples

(Consistent with QM9 conventions: 100k training, 10% test, remainder validation after excluding 3,054 invalid molecules)
