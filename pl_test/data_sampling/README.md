Riemannian manifold 위에서의 score 학습을 위한, 데이터 처리.

Files:
  - ./riemannian_data_sampling.py: riemannian data sampling
  - ./riemannian_data_sampling.yaml: riemannian data sampling에 필요한 config 파일.


        # riemannian data sampling
        $ python riemannian_data_sampling.py --config_yaml ./riemannian_data_sampling.yaml --sampling_type riemannian --alpha 3.0 --beta 0.5 --svd_tol 1e-6 --t0 0 --t1 150 --save_xyz xyz_alpha3.0_beta0.5_gamma0.0_svdtol_1e-6_t150 --save_csv test.csv


        # Calculate RMSD, DMAE, q_norm of structures (riemannian sampled, cartesian sampled, MMFF structures with respect to DFT structures)
        $ python analyze_distribution.py --config_yaml riemannian_data_sampling.yaml --save_csv test.csv --xyz_path xyz_alpha3.5_beta1.0_gamma0.0_svdtol_1e-6_t150 --t0_x 1 --t1_x 1500 --mmff_xyz_path /home/share/DATA/QM9M/MMFFtoDFT_input --alpha 3.5 --beta 1.0

        # Visualize RMSD, DMAE, q_norm distribution
        $ python plot_distribution.py --sampling_csv ./alpha3.0_beta0.5_gamma0.0_svdtol_1e-6_t150.sampling.csv --analyzing_csv ./alpha3.0_beta0.5_gamma0.0_svdtol_1e-6_t150.analyzing.csv
