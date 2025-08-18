# Riemannian Data Sampling

This directory provides scripts and configuration files for sampling molecular structures on a Riemannian manifold.
The sampled structures and associated vector fields (e.g., `dq`) can be used to train score-based generative models in the context of molecular geometry generation.


## Sampling Example

The following commands generate noisy structures on the Riemannian manifold and the corresponding `dq` target vectors.

Using multiple random seeds introduces structural diversity in training data:

```bash
python riemannian_data_sampling.py --config_yaml ./riemannian_data_sampling.yaml --sampling_type riemannian --alpha 1.7 --beta 0.01 --svd_tol 1e-2 --t0 0 --t1 150 --save_xyz xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150_seed42 --save_csv ./alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150.sampling.csv --seed 42
python riemannian_data_sampling.py --config_yaml ./riemannian_data_sampling.yaml --sampling_type riemannian --alpha 1.7 --beta 0.01 --svd_tol 1e-2 --t0 0 --t1 150 --save_xyz xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150_seed42_val --seed 42 --dataloader val
python riemannian_data_sampling.py --config_yaml ./riemannian_data_sampling.yaml --sampling_type riemannian --alpha 1.7 --beta 0.01 --svd_tol 1e-2 --t0 0 --t1 150 --save_xyz xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150_seed42_train --seed 42 --dataloader train
for seed in {1..29}
do
  python riemannian_data_sampling.py --config_yaml ./riemannian_data_sampling.yaml --sampling_type riemannian --alpha 1.7 --beta 0.01 \
    --svd_tol 1e-2 --t0 0 --t1 300 --save_xyz xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150_seed${seed}_train --seed ${seed} --dataloader train
done
```

To merge results from the above sampling runs into a single directory for unified analysis:
```bash
python combine_to_single_dir.py
```


## Plot Distributions


```bash
python analyze_distribution.py --config_yaml riemannian_data_sampling.yaml --xyz_path xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150_seed42_smallest --t0_x 1 --t1_x 1500 --mmff_xyz_path ../qm7x_xyz_smallest --alpha 1.7 --beta 0.01 --save_csv alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t150_t1_x1500_smallest.analyzing.with_energy.csv --gpu --calculate_energy

python analyze_distribution.py --config_yaml riemannian_data_sampling.yaml --xyz_path xyz_alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_seed42 --t0_x 1 --t1_x 2000 --mmff_xyz_path ../qm7x_xyz --alpha 1.7 --beta 0.01 --save_csv alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_t1_x2000.analyzing.with_energy.csv --gpu --calculate_energy
```

To visualize distributions of RMSD, DMAE, and q-norms:
```bash
python plot_distribution.py --sampling_csv ./alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300.sampling.csv --analyzing_csv alpha1.7_beta0.01_gamma0.0_svdtol_1e-2_t300_t1_x2000.analyzing.csv --visualize
```
