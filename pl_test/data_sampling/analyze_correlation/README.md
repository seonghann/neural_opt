Analyse the correlation b/w structural error and energy error.

- ./QM9M_SP_CALC: QM9M dataset의 MMFF 및 DFT 구조의 Gaussian 계산 결과.
- ./QM9M_SP_CALC/QM9M_SP.csv: 계산된 energies를 csv 파일로 정리.
- ./qm9m.csv: index, energies, structural erros등을 저장한 csv 파일.

      # Make csv file containing dE, smarts information
      $ python add_properties_to_csv.py --input_csv /home/share/DATA/QM9M/Geodesic_processing/optimize_geodesic_coeffs/QM9M_SP_CALC/QM9M_SP.csv --sdf_path /home/share/DATA/QM9M/sdf_files --output_csv ./qm9m.csv


      # Add DMAE and RMSD information
      $ python write_error_to_csv.py --error_type DMAE --input_csv ./qm9m.csv --output_csv ./qm9m.csv --dft_results_path1 /home/share/DATA/QM9M/Geodesic_processing/optimize_geodesic_coeffs/QM9M_SP_CALC/results/MMFF --dft_results_path2 /home/share/DATA/QM9M/Geodesic_processing/optimize_geodesic_coeffs/QM9M_SP_CALC/results/DFT
      $ python write_error_to_csv.py --error_type RMSD --input_csv ./qm9m.csv --output_csv ./qm9m.csv --dft_results_path1 /home/share/DATA/QM9M/Geodesic_processing/optimize_geodesic_coeffs/QM9M_SP_CALC/results/MMFF --dft_results_path2 /home/share/DATA/QM9M/Geodesic_processing/optimize_geodesic_coeffs/QM9M_SP_CALC/results/DFT


      # Remove wrong samples
      $ python process_wrong_samples_from_csv.py --input_csv ./qm9m.csv --output_csv ./qm9m.csv


      # Add q_norm information corresponding alpha, beta, gamma values
      $ python write_error_to_csv.py --error_type q_norm --input_csv ./qm9m.csv --output_csv ./qm9m.csv --dft_results_path1 /home/share/DATA/QM9M/Geodesic_processing/optimize_geodesic_coeffs/QM9M_SP_CALC/results/MMFF --dft_results_path2 /home/share/DATA/QM9M/Geodesic_processing/optimize_geodesic_coeffs/QM9M_SP_CALC/results/DFT --alpha 1.7 --beta 0.01
      $ python write_error_to_csv.py --error_type q_norm --input_csv ./qm9m.csv --output_csv ./qm9m.csv --dft_results_path1 /home/share/DATA/QM9M/Geodesic_processing/optimize_geodesic_coeffs/QM9M_SP_CALC/results/MMFF --dft_results_path2 /home/share/DATA/QM9M/Geodesic_processing/optimize_geodesic_coeffs/QM9M_SP_CALC/results/DFT --alpha 3.0 --beta 0.5
      $ ...


      # Calculate correlation and visualization scatter plot
      $ python calc_correlation.py --error_type q_norm --alpha 1.7 --beta 0.01 --input_csv ./qm9m.csv --visualize


      # Visualize correlation scatter graph
      $ python write_geodesic_length_to_csv.py --error_type geodesic_length --input_csv ./qm9m.csv --output_csv ./qm9m.csv --dft_results_path1 /home/share/DATA/QM9M/Geodesic_processing/optimize_geodesic_coeffs/QM9M_SP_CALC/results/MMFF --dft_results_path2 /home/share/DATA/QM9M/Geodesic_processing/optimize_geodesic_coeffs/QM9M_SP_CALC/results/DFT --alpha 1.7 --beta 0.01
      python calc_correlation.py --error_type geodesic_length --alpha 1.7 --beta 0.01 --input_csv ./qm9m.csv --visualize
      python calc_correlation.py --error_type RMSD --input_csv ./qm9m.csv --visualize
      python calc_correlation.py --error_type DMAE --input_csv ./qm9m.csv --visualize
