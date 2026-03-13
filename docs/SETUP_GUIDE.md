# neural_opt Setup Guide (Server Migration)

Date: 2026-03-13

## 1. Clone Repos

```bash
git clone https://github.com/seonghann/RxnExpPipe.git
cd RxnExpPipe
git clone -b refactoring https://github.com/seonghann/neural_opt.git
```

## 2. Transfer Data

From the source server:
```bash
# tarball location: /workspace/RxnExpPipe/neural_opt_data.tar.gz (484MB)
scp user@source:/workspace/RxnExpPipe/neural_opt_data.tar.gz .
```

On the target server:
```bash
cd neural_opt
tar xzf ../neural_opt_data.tar.gz
```

This extracts:
```
neural_opt/
├── data/qm9m/data_split.pkl                       ← train/val/test split indices
├── data_qm9m_MMFFtoDFT/processed/                 ← E-DM training data (PyG cached)
│   ├── train_proc.pt, valid_proc.pt, test_proc.pt
├── data_qm9m_riemannian_3seeds/processed/          ← R-DM finetuning data (PyG cached)
│   ├── train_proc.pt (918MB), valid_proc.pt (73MB), test_proc.pt (empty)
├── checkpoints/
│   ├── qm9.edsm/best/                             ← E-DM checkpoint (epoch 350, val_loss=125.3)
│   ├── qm9.edsm/last/                             ← E-DM last checkpoint (epoch 351)
│   ├── edsm.qm9.ckpt                              ← E-DM published checkpoint (논문 저자 제공)
│   ├── rdsm.qm9.pretrained.ckpt                    ← R-DM pretrained (논문 저자 제공)
│   └── rdsm.qm9.finetuned.ckpt                    ← R-DM finetuned (논문 저자 제공)
```

## 3. Python Environment

Required packages:
```
torch >= 2.1
torch_geometric
torch_scatter, torch_sparse
rdkit
ase
omegaconf
accelerate
wandb
safetensors
tqdm
```

Recommended: create a virtualenv at `/venv/neural_opt/`.

## 4. Verify Setup

```bash
cd neural_opt

# Import check
python -c "from diffusion.model import DiffusionModel; print('OK')"

# Quick sampling test (uses published checkpoint)
python sample.py configs/sampling.qm9.rdsm.yaml --batch_idx_end 2
# Expected: RMSD ~0.034 Å
```

## 5. Training Commands

All commands run from `neural_opt/` directory.

### E-DM Training (from scratch)

```bash
python train.py configs/training.qm9.edsm.yaml
```
- Data: `data_qm9m_MMFFtoDFT/processed/` (Euclidean noise, MMFF→DFT)
- Loss: `lambda_x=1, lambda_q=0` (Cartesian prediction)
- Epochs: 3000, batch_size: 300

### E-DM Training (resume from checkpoint)

```bash
python train.py configs/training.qm9.edsm.yaml --resume checkpoints/qm9.edsm/last
```
- Resumes from epoch 351, best val_loss=125.3

### R-DM Training from Scratch

```bash
python train.py configs/finetuning.qm9.rdsm.yaml
```
- Data: `data_qm9m_riemannian_3seeds/processed/` (Riemannian projected noise)
- Loss: `lambda_x=0, lambda_q=1` (internal coordinate prediction)
- noise_type: `diffusion_custom`, transform: `projection_dq2dx`
- Epochs: 6000, batch_size: 100

### R-DM Training (initialize from E-DM checkpoint)

```bash
python train.py configs/finetuning.qm9.rdsm.yaml --resume_pl checkpoints/qm9.edsm/best
```
- Loads E-DM weights (76/152 keys), R-DM extra layers randomly initialized
- `--resume_pl` = model weights only, no optimizer state

## 6. Key Config Differences

| Config | noise_type | transform | lambda_x | lambda_q | Data |
|--------|-----------|-----------|----------|----------|------|
| training.qm9.edsm | diffusion | eq_transform | 1 | 0 | MMFFtoDFT |
| training.qm9.rdsm | diffusion | projection_dq2dx | 0 | 1 | MMFFtoDFT |
| finetuning.qm9.rdsm | diffusion_custom | projection_dq2dx | 0 | 1 | riemannian_3seeds |

- **E-DM**: Euclidean noise → predict Cartesian displacements
- **R-DM pretraining** (`training.qm9.rdsm`): Euclidean noise projected to tangent space → predict q
- **R-DM finetuning** (`finetuning.qm9.rdsm`): Custom Riemannian noise → predict q

## 7. Data Pipeline

If processed `.pt` files are missing, PyG will regenerate from raw data.
This takes hours — always transfer processed files.

```
Raw data locations (NOT in tarball — only needed if reprocessing):
  data/qm9m/MMFFtoDFT_input/          ← E-DM raw xyz (133K files)
  data/qm9m/riemannian_xyz/combined/   ← R-DM raw xyz (311K symlinks)
```

## 8. GPU Requirements

- E-DM training (batch=300): ~20-30GB VRAM
- R-DM training (batch=100): ~10-15GB VRAM
- Sampling: ~5-10GB VRAM

Tested on H200 (144GB). Should work on A100/V100 with batch size adjustment.
