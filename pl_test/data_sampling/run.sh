#!/bin/bash

# 파라미터 쌍 설정
# pair: alpha, beta, t1
param_pairs=(
  "3.0 0.5 150"
  "3.5 1.0 150"
  "4.5 1.5 300"
  "5.0 2.0 800"
  "6.0 2.5 800"
  "6.5 3.0 1000"
)

# 루프를 통해 각 쌍에 대해 스크립트 실행
for pair in "${param_pairs[@]}"; do
  alpha=$(echo $pair | cut -d ' ' -f 1)
  beta=$(echo $pair | cut -d ' ' -f 2)
  t1=$(echo $pair | cut -d ' ' -f 3)
  echo "Running with alpha=${alpha}, beta=${beta}, t1=${t1}"

  # riemannian_data_sampling.py 명령어 구성 및 출력
  cmd_sampling="python riemannian_data_sampling.py \
    --config_yaml ./riemannian_data_sampling.yaml \
    --sampling_type riemannian \
    --alpha ${alpha} \
    --beta ${beta} \
    --svd_tol 1e-6 \
    --t0 0 \
    --t1 ${t1} \
    --save_xyz xyz_alpha${alpha}_beta${beta}_gamma0.0_svdtol_1e-6_t${t1} \
    --save_csv alpha${alpha}_beta${beta}_gamma0.0_svdtol_1e-6_t${t1}.sampling.csv"
  
  echo $cmd_sampling
  eval $cmd_sampling

  # analyze_distribution.py 명령어 구성 및 출력
  cmd_analyze="python analyze_distribution.py \
    --config_yaml riemannian_data_sampling.yaml \
    --save_csv alpha${alpha}_beta${beta}_gamma0.0_svdtol_1e-6_t${t1}.analyzing.csv \
    --xyz_path xyz_alpha${alpha}_beta${beta}_gamma0.0_svdtol_1e-6_t${t1} \
    --t0_x 1 \
    --t1_x 1500 \
    --mmff_xyz_path /home/share/DATA/QM9M/MMFFtoDFT_input \
    --alpha ${alpha} \
    --beta ${beta}"
  
  echo $cmd_analyze
  eval $cmd_analyze
done

