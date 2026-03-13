# 서버 이전 및 학습 재개 준비
- Date: 2026-03-13
- Author: Seonghwan Kim + Claude
- Repo / Module: `neural_opt/`
- Branch / Commit: `refactoring` @ `3652b35`

---

## 1. Context & Motivation (WHY)

### 1.1 Project Context

R-DM (Riemannian Diffusion Model) 논문 재현 프로젝트.
현재 서버(H200)의 GPU가 다른 프로젝트(DFTREPA)에 의해 점유되어 학습 진행이 불가능하여,
다른 서버로 코드+데이터를 이전하고 학습을 재개하기 위한 준비 작업.

### 1.2 Previous Work Summary

**완료된 작업 (260303 ~ 260312):**

| Date | Task | Progress Log | Status |
|------|------|-------------|--------|
| 03-03 | R-DM QM9 sampling reproduction | [[260303_rdm_reproduction_sampling.md]] | 완료 (RMSD 0.034 Å) |
| 03-04 | Codebase audit & dead code backup | [[260304_codebase_audit.md]] | 완료 |
| 03-05 | PL → Accelerate 리팩터링 (5단계) | [[260305_neural_opt_refactoring.md]] | 완료 |
| 03-11 | 전체 파이프라인 검증 | [[260311_reproduction_pipeline.md]] | 완료 |
| 03-12 | ODE solver 최적화 (11.5x speedup) | [[260312_riemannian_noise_optimization.md]] | 완료 |
| 03-12 | Riemannian data 생성 (3 seeds) | (위와 동일) | 완료 (311K files) |
| 03-13 | E-DM pretraining | - | epoch 351까지, val_loss=125.3 |
| 03-13 | R-DM training (scratch) | - | 미완료 (GPU 부족) |

**핵심 성과:**
1. PL 제거 + Accelerate 전환, 코드 19.6% 축소 (7921 → 6367 LOC)
2. ODE solver CPU offload로 11.5x speedup (수치 정합성 유지)
3. Riemannian noisy data 3 seeds 생성 완료 (train 288K + valid 23K = 311K xyz files)
4. E-DM pretraining epoch 351 (best val_loss=125.3)
5. 전체 R-DM pipeline end-to-end 검증 완료 (sampling RMSD 0.034 Å 재현)

### 1.3 Hypothesis Status

- [채택] PL 제거 후 Accelerate 전환해도 결과 동일 → RMSD 0.034 Å 재현
- [채택] CPU offload pinv로 수치 정합성 유지하면서 11.5x 가속 가능
- [채택] Riemannian ODE solver가 논문과 수치적으로 정합하는 noise 생성
- [검증중] R-DM from scratch 학습이 논문 성능 재현 가능 ← **다음 서버에서 검증**
- [검증중] E-DM pretraining → R-DM finetuning 파이프라인이 논문 성능 재현 가능

---

## 2. Current State (이전 시점의 상태)

### 2.1 Code

```
Branch: refactoring (3 commits ahead of origin, now pushed)
Latest commit: 3652b35

핵심 변경:
1. sample.py     — Accelerate checkpoint 지원, partial weight loading 허용
2. data_module.py — empty dataset 처리 (test split)
3. finetuning config — batch_size 300→100, datadir/raw_datadir 경로 수정
4. riemannian_data_sampling.py — GPU 호환성 수정 (5건 device mismatch)
5. manifold/solver.py — CPU offload pinv/svd (11.5x speedup)
```

### 2.2 Data

```
data_qm9m_MMFFtoDFT/processed/          ← E-DM training data
  train_proc.pt  (309MB)

data_qm9m_riemannian_3seeds/processed/  ← R-DM finetuning data
  train_proc.pt  (918MB)   ← 288,547 molecules (seed42+43+44 train + seed42 valid symlinks)
  valid_proc.pt  (73MB)    ← 22,913 molecules (valid set, seed42)
  test_proc.pt   (1.3KB)   ← empty (no riemannian test data)

data/qm9m/data_split.pkl               ← train/val/test split indices
```

### 2.3 Checkpoints

```
checkpoints/qm9.edsm/best/    ← E-DM best (epoch 350, val_loss=125.3)
checkpoints/qm9.edsm/last/    ← E-DM last (epoch 351)
checkpoints/edsm.qm9.ckpt     ← E-DM published (논문 저자)
checkpoints/rdsm.qm9.*.ckpt   ← R-DM published (논문 저자)
```

### 2.4 Git Repos

| Repo | URL | Branch | Last Commit |
|------|-----|--------|-------------|
| RxnExpPipe | github.com/seonghann/RxnExpPipe (private) | main | 6e669aa |
| neural_opt | github.com/seonghann/neural_opt | refactoring | 3652b35 |

---

## 3. Pending Tasks (다음 서버에서 수행)

### 3.1 E-DM Pretraining 재개 (Priority: Medium)

E-DM은 epoch 351까지 학습됨 (best val_loss=125.3).
논문에서는 3000 epochs 학습. 재개하려면:

```bash
python train.py configs/training.qm9.edsm.yaml --resume checkpoints/qm9.edsm/last
```

- Data: `data_qm9m_MMFFtoDFT/processed/` (tarball에 포함)
- 목표: val_loss 수렴 확인 (논문 수치 미공개, best effort)

### 3.2 R-DM Training from Scratch (Priority: High)

Riemannian projected noise로 처음부터 R-DM 학습:

```bash
python train.py configs/finetuning.qm9.rdsm.yaml
```

- Data: `data_qm9m_riemannian_3seeds/processed/` (tarball에 포함)
- Config 핵심: `noise_type: diffusion_custom`, `transform: projection_dq2dx`, `lambda_q=1`
- 목표: sampling RMSD ≤ 0.034 Å (published finetuned checkpoint 수준)

### 3.3 R-DM Training from E-DM (Priority: Medium)

E-DM checkpoint으로 초기화 후 R-DM 학습:

```bash
python train.py configs/finetuning.qm9.rdsm.yaml --resume_pl checkpoints/qm9.edsm/best
```

- E-DM의 76 keys 로드, R-DM 추가 76 keys는 random init
- 3.2 vs 3.3 결과 비교하여 E-DM pretrain의 효과 확인

### 3.4 Sampling & Evaluation

학습 완료 후 sampling으로 최종 검증:

```bash
python sample.py configs/sampling.qm9.rdsm.yaml
```

성공 기준: RMSD median ≤ 0.034 Å, D-MAE median ≤ 0.011 Å

---

## 4. Migration Checklist

### Source Server → Target Server

- [x] `neural_opt` code pushed to GitHub (refactoring branch)
- [x] `RxnExpPipe` scaffold pushed to GitHub (main branch)
- [x] Data tarball created: `neural_opt_data.tar.gz` (484MB)
  - [x] E-DM processed data
  - [x] R-DM processed data (riemannian_3seeds)
  - [x] data_split.pkl
  - [x] E-DM checkpoints (best/last)
  - [x] Published checkpoints (edsm, rdsm pretrained/finetuned)
- [x] Setup guide: `docs/SETUP_GUIDE.md`

### Target Server Setup

- [ ] Clone repos (RxnExpPipe + neural_opt)
- [ ] Transfer and extract tarball
- [ ] Create Python environment
- [ ] Verify import: `python -c "from diffusion.model import DiffusionModel"`
- [ ] Verify sampling: `python sample.py configs/sampling.qm9.rdsm.yaml --batch_idx_end 2`
- [ ] Start R-DM training (Task 3.2)

---

## 5. Traceability

### Git
- Branch: `refactoring`
- Commits: `886683c` (리팩터링) → `ee9aa07` (ODE solver) → `3652b35` (R-DM setup)

### Related Documents
- [[docs/progress/260311_reproduction_pipeline.md]]: Pipeline 전체 검증
- [[docs/progress/260312_riemannian_noise_optimization.md]]: ODE solver 최적화 + data 생성
- [[docs/SETUP_GUIDE.md]]: 서버 설정 가이드

### Analysis Scripts
- [[analyze/260312_solver_profiling.py]]: pinv = 98% of time
- [[analyze/260312_pinv_cpu_benchmark.py]]: CPU 83.9x speedup
- [[analyze/260312_solver_optimization_verify.py]]: End-to-end 11.5x, RMSD 3.88e-06
- [[analyze/260312_riemannian_noise_comparison.py]]: Noise 통계 논문 정합 확인

### Tarball Contents
```
neural_opt_data.tar.gz (484MB):
  data_qm9m_riemannian_3seeds/processed/  ← R-DM data (991MB uncompressed)
  data_qm9m_MMFFtoDFT/processed/          ← E-DM data (309MB uncompressed)
  data/qm9m/data_split.pkl                ← split indices
  checkpoints/qm9.edsm/{best,last}/       ← E-DM training checkpoints
  checkpoints/edsm.qm9.ckpt               ← published E-DM
  checkpoints/rdsm.qm9.pretrained.ckpt    ← published R-DM pretrained
  checkpoints/rdsm.qm9.finetuned.ckpt     ← published R-DM finetuned
```
