# Riemannian Data Sampling

This directory provides scripts and configuration files for sampling molecular structures on a Riemannian manifold.
The sampled structures and associated vector fields (e.g., `dq`) can be used to train score-based generative models in the context of molecular geometry generation.

---

## Files

- `riemannian_data_sampling.py`:
  Main script to perform Riemannian data sampling
- `riemannian_data_sampling.yaml`:
  YAML configuration file specifying dataset, split, and atom selection
- `analyze_distribution.py`:
  Computes statistics (e.g., RMSD, DMAE, vector norm) between sampled structures and DFT references
- `plot_distribution.py`:
  Plots the distribution of computed metrics (RMSD, DMAE, etc.)

---

## Sampling Example

The following commands generate noisy structures on the Riemannian manifold and the corresponding `dq` target vectors.

Using multiple random seeds introduces structural diversity in training data:

```bash
python riemannian_data_sampling.py --config_yaml ./riemannian_data_sampling.yaml --sampling_type riemannian --alpha 1.7 --beta 0.01 --svd_tol 1e-2 --t0 0 --t1 150 --save_xyz xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150_seed42 --save_csv ./alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150.sampling.csv --seed 42
python riemannian_data_sampling.py --config_yaml ./riemannian_data_sampling.yaml --sampling_type riemannian --alpha 1.7 --beta 0.01 --svd_tol 1e-2 --t0 0 --t1 150 --save_xyz xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150_seed1 --save_csv test.csv --seed 1
python riemannian_data_sampling.py --config_yaml ./riemannian_data_sampling.yaml --sampling_type riemannian --alpha 1.7 --beta 0.01 --svd_tol 1e-2 --t0 0 --t1 150 --save_xyz xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150_seed2 --save_csv test.csv --seed 2
```


## Combine Multiple Samples

To merge results from the above sampling runs into a single directory for unified analysis:

```bash
mkdir xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150_seed42_1_2

cp xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150_seed42/* \
   xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150_seed42_1_2/

cp xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150_seed1/* \
   xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150_seed42_1_2/

cp xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150_seed2/* \
   xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150_seed42_1_2/
```


## Evaluate Sampled Structures

To analyze and compare sampled structures against DFT references (as well as MMFF and Cartesian sampling):

```bash
python analyze_distribution.py \
  --config_yaml riemannian_data_sampling.yaml \
  --xyz_path xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150_seed42 \
  --t0_x 1 \
  --t1_x 1500 \
  --mmff_xyz_path ../MMFFtoDFT_input \
  --alpha 1.7 \
  --beta 0.01 \
  --save_csv ./alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150.analyzing.csv
```


## Plot Distributions

To visualize distributions of RMSD, DMAE, and q-norms:
```bash
python plot_distribution.py \
  --sampling_csv ./alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150.sampling.csv \
  --analyzing_csv ./alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150.analyzing.csv \
  --visualize
```
