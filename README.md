# Riemannian Denoising Score Matching (R-DSM) for Molecular Optimization with Chemical Accuracy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/seonghann/neural_opt/tree/refactoring/LICENSE)

Refer the paper, **"Riemannian Denoising Score Matching for Molecular Optimization with Chemical Accuracy"**.
[[arXiv]](https://arxiv.org/abs/2411.19769)

![Cover Image](assets/Schematic.png)

---

## Environments

### Install via Conda (Recommended)

To set up the environment, use the following commands:

```bash
conda create -n neural_opt_cu118 python=3.9 -y
conda activate neural_opt_cu118
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
pip install torch-geometric==2.6.1
pip install pytorch-lightning==2.2.4 matplotlib pandas==2.2.2
pip install rdkit==2025.3.2
pip install ase==3.25.0
pip install scipy==1.13.1
pip install omegaconf==2.3.0
pip install wandb==0.20.1
```

---

## Dataset

Data processing details are described in the `data/qm9m` directory.

Processed data can be downloaded from our Zenodo repository: [https://zenodo.org/records/15561806](https://zenodo.org/records/15561806)

The Zenodo repository contains both the processed datasets and the data processing codes used in this work.


## Training

Hyperparameters and training configurations are specified in the YAML files located in `./configs/`.

```bash
# Pretraining R-DSM with Euclidean noise-sampling
python main.py configs/training.qm9.rdsm.yaml
# Fine-tuning R-DSM with Riemannian noise-sampling
python main.py configs/finetuning.qm9.rdsm.yaml
```

---

## Sampling

Checkpoints for models trained on the QM9 and GEOM-QM9 datasets are available in `./checkpoints/*.ckpt`.

To reproduce the results from the paper:


```bash
# Sampling (generates 'save_dynamic.qm9.rdsm.finetuned.pt')
python main.py configs/sampling.qm9.rdsm.yaml
# Performance evaluation (calculates RMSD, D-MAE, etc.)
python evaluate_accuracy.py \
  --config_yaml configs/sampling.qm9.rdsm.yaml \
  --prb_pt save_dynamic.qm9.rdsm.finetuned.pt \
  --ban_index /home/share/DATA/QM9M/wrong_samples.pkl
```

---

## Citation

Please consider citing the our paper if you find it helpful. Thank you!

```
@article{woo2024riemannian,
  title={Riemannian Denoising Score Matching for Molecular Structure Optimization with Accurate Energy},
  author={Woo, Jeheon and Kim, Seonghwan and Kim, Jun Hyeong and Kim, Woo Youn},
  journal={arXiv preprint arXiv:2411.19769},
  year={2024}
}
```

