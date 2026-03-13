# 논문 재현 파이프라인 검증 및 Deprecation 정리
- Date: 2026-03-11
- Author: Seonghwan Kim + Claude
- Repo / Module: `neural_opt/`
- Branch / Commit: `refactoring` @ `886683c`

---

## 1. Context & Motivation (WHY)

### 1.1 Project Context

이 태스크는 [[docs/obsidian/70_projects/PROJ-001 RXN Exploration Pipeline.md]]의 인프라 정비 단계 마무리에 해당한다.
5단계 리팩터링이 완료된 코드베이스 ([[docs/progress/260305_neural_opt_refactoring.md]])에서,
논문의 **전체 재현 파이프라인을 end-to-end로 검증**하고,
재현에 필요하지 않은 파일들에 **deprecation 표시**를 추가하여
Phase 2 (reaction system 확장) 진입 전 코드베이스를 정리하는 것이 목표이다.

리팩터링 검증은 지금까지 subset sampling (4 batch, ~1,200 molecules)으로만 수행되었고,
**training, finetuning, riemannian data sampling** 등 전체 pipeline의 동작은 아직 검증되지 않았다.

### 1.2 Previous Attempts

- [[docs/progress/260303_rdm_reproduction_sampling.md]]: R-DM QM9 sampling reproduction 성공
  - 환경: Python 3.12, PyTorch 2.8.0+cu128, H200
  - 결과: RMSD median 0.034 Å, D-MAE median 0.011 Å (baseline 확보)
  - PL 기반 원본 코드로 실행

- [[docs/progress/260304_codebase_audit.md]]: Module 사용 분석 + dead code backup 완료
  - USED: 22 files, 7,921 LOC → UNUSED: 35 files, 7,424 LOC (`_backup/`으로 이동)
  - 구조 문제 4건 식별: model 이중 구조, SchNet 이중 구현, diffusion_model.py god class, 하드코딩 경로

- [[docs/progress/260305_neural_opt_refactoring.md]]: 5단계 전면 리팩터링 완료
  - Step 1: Dead code 제거 (diffusion_model.py 2,058 → 984 LOC)
  - Step 2: model/ + model_tsdiff/ 통합 → 단일 model/ 패키지
  - Step 3+4: diffusion_model.py 분리 + PL → Accelerate 전환
  - Step 5: manifold/ 디렉토리 생성
  - 최종 검증: 22개 모듈 import 전부 성공, subset sampling RMSD 0.034 Å (baseline 일치)
  - LOC: 7,921 → 6,367 (-19.6%)

### 1.3 Hypothesis Status

- [채택] PL을 제거하고 Accelerate로 전환해도 sampling 결과가 동일하다
  → [[docs/progress/260305_neural_opt_refactoring.md]]: subset sampling으로 확인 완료
- [검증중] Accelerate 기반 training loop (`train.py`)가 정상 동작하여 학습이 진행된다
  → 이 태스크에서 검증 예정 (RDSM pretraining + finetuning + EDSM training)
- [검증중] 리팩터링 후 `riemannian_data_sampling.py`가 정상 동작한다
  → 이 태스크에서 검증 예정 (bug fix 포함)
- [검증중] 전체 R-DM 파이프라인 (pretrain → data sampling → finetune → sample)이 end-to-end 동작한다
  → 이 태스크에서 검증 예정

### 1.4 Current Problem

**현재 문제 상황:**

리팩터링 후 검증이 **sampling만** 수행되었다. 논문의 전체 재현 파이프라인은 4단계인데,
training과 data sampling은 아직 검증되지 않았다:

```
논문 R-DM 재현 파이프라인:
  Step 1: RDSM Pretraining ──────── 미검증 ← train.py (Accelerate, 신규)
  Step 1.5: Riemannian Data Sampling ── 미검증 ← riemannian_data_sampling.py (bug 3건)
  Step 2: RDSM Finetuning ─────── 미검증 ← train.py --resume_pl (checkpoint resume)
  Step 3: RDSM Sampling ─────────── 검증 완료 ← sample.py (RMSD 0.034 Å)

논문 E-DM 재현 파이프라인:
  Training: EDSM Training ──────── 미검증 ← train.py (다른 config)
  Sampling: EDSM Sampling ─────── 미검증 ← sample.py (다른 config)
```

추가로, 리팩터링 중 발견된 **3건의 버그**가 수정되었으나 아직 검증되지 않았다:

| # | 위치 | 원래 코드 | 수정 코드 | 원인 |
|---|------|----------|----------|------|
| 1 | `riemannian_data_sampling.py:175` | `Graph` (undefined) | `MolGraph.from_batch(data)` | 리팩터링 시 import 경로 변경에서 누락 |
| 2 | `model/__init__.py:16` | `self.optim = torch.optim.AdamW(...)` | `return torch.optim.AdamW(...)` | standalone 함수에 `self` 잔존 (PL method였던 흔적) |
| 3 | `model/__init__.py:23` | `optim_type` (undefined) | `cfg.type` | 변수명 미갱신 |

그리고 코드베이스에 **재현에 불필요한 파일들**이 아직 정리되지 않았다:

```
재현 불필요 파일 (deprecation 대상):
  _backup/                              ~30 files   이미 deprecated된 dead code
  verify_subset.py                      1 file      quick sanity check, sample.py로 대체 가능
  data/qm9m/data_split.py              1 file      one-time 데이터 전처리 (완료)
  data/qm9m/preprocessing.py           1 file      one-time 데이터 전처리 (완료)
  data/qm9m/process_wrong_samples.py   1 file      one-time 데이터 전처리 (완료)
  data/qm9m/MMFFtoDFT_input/results/   ~8 files    Gaussian/DFT 처리 스크립트 (one-time)
  data/qm9m/riemannian_data_sampling/
    analyze_distribution.py             1 file      분석 유틸리티
    plot_distribution.py                1 file      시각화 유틸리티
```

**이 작업이 필요한 이유:**
- Training pipeline이 검증되지 않으면, 모델을 처음부터 재학습하거나 새 데이터로 finetuning할 수 없다
- Bug fix가 검증되지 않으면 riemannian data sampling이 동작하지 않아 finetuning 데이터를 생성할 수 없다
- Deprecation 미정리 상태에서 Phase 2에 진입하면, 어떤 파일이 필수이고 어떤 것이 불필요한지 혼란 발생
- 재현 확인 후 clean removal pass를 위해서는 먼저 deprecation 표시가 선행되어야 함

**해결되었을 때 기대되는 효과:**
- 전체 R-DM + E-DM 파이프라인이 end-to-end로 동작함을 확인 → 코드 신뢰성 확보
- 불필요 파일에 명확한 deprecation notice → 재현 후 clean removal 가능
- Phase 2 (reaction system 확장) 진입 준비 완료

---

## 2. Plan (HOW)

### 2.1 Planned Approach

2-part로 나누어 수행한다.

**Part A: Deprecation 표시 + Bug fix 검증**

1. **Bug fix 검증** — 이미 수정된 3건의 import 동작 확인
   ```bash
   python -c "from model import get_optimizer, get_scheduler; print('model/__init__.py OK')"
   python -c "from data.qm9m.riemannian_data_sampling.riemannian_data_sampling import ode_noise_sampling; print('riemannian_data_sampling OK')"
   ```

2. **Deprecation notice 추가** — 재현 불필요 파일 상단에 deprecation 표시
   - 대상 파일 목록:
     - `verify_subset.py` — `# DEPRECATED: Use `python sample.py ... --batch_idx_end 4` instead`
     - `data/qm9m/data_split.py` — `# DEPRECATED: One-time data prep, already done. data_split.pkl exists.`
     - `data/qm9m/preprocessing.py` — `# DEPRECATED: One-time data prep, already done.`
     - `data/qm9m/process_wrong_samples.py` — `# DEPRECATED: One-time data prep, already done. wrong_samples.pkl exists.`
     - `data/qm9m/riemannian_data_sampling/analyze_distribution.py` — `# DEPRECATED: Analysis utility, not required for reproduction.`
     - `data/qm9m/riemannian_data_sampling/plot_distribution.py` — `# DEPRECATED: Visualization utility, not required for reproduction.`
     - `data/qm9m/MMFFtoDFT_input/results/*.py` — `# DEPRECATED: Gaussian/DFT processing, one-time use.`
   - `_backup/` 디렉토리에 `README.md` 추가: deprecated 파일들의 출처와 이유 설명
   - 삭제하지 않음 — deprecation 표시만 추가

3. **전체 import test** — 리팩터링 후 모든 핵심 모듈 import 확인
   ```
   diffusion.model, diffusion.sampling, diffusion.noise_scheduler
   model (get_optimizer, get_scheduler), model.encoder, model.schnet, model.layers, model.edge, model.geometry
   manifold.solver, manifold.graph
   dataset.data_module
   metrics.metrics
   utils.chem, utils.wandb_utils
   ```

**Part B: 파이프라인 검증 (1-2 epochs each)**

각 단계를 1-2 epoch만 실행하여 crash 없이 동작하는지, loss가 finite한지 확인한다.
기존 pretrained/finetuned 체크포인트를 활용하여 품질 검증(sampling)까지 수행한다.

4. **RDSM Pretraining 검증** (1-2 epochs)
   ```bash
   python train.py configs/training.qm9.rdsm.yaml
   ```
   - Config 경로 수정 필요: `raw_datadir`, `data_split` → 현재 환경 경로
   - 확인: loss finite, gradient flow 정상, crash 없음
   - `use_wandb: False`로 변경하여 wandb 없이 실행

5. **Riemannian Data Sampling 검증** (small subset)
   ```bash
   python data/qm9m/riemannian_data_sampling/riemannian_data_sampling.py
   ```
   - Config: `data/qm9m/riemannian_data_sampling/riemannian_data_sampling.yaml`
   - 소규모 subset으로 실행 (전체 133K 대신 ~100 molecules)
   - 확인: .xyz 파일 생성, valid molecular geometry

6. **RDSM Finetuning 검증** (1-2 epochs)
   ```bash
   python train.py configs/finetuning.qm9.rdsm.yaml --resume_pl checkpoints/rdsm.qm9.pretrained.ckpt
   ```
   - Config 경로 수정 필요: `raw_datadir` → riemannian sampling 결과 경로 또는 기존 데이터
   - 확인: pretrained checkpoint resume 정상, `diffusion_custom` noise type 동작, loss finite

7. **RDSM Sampling 검증** (full)
   ```bash
   python sample.py configs/sampling.qm9.rdsm.yaml
   ```
   - 이미 baseline 확인됨 (RMSD 0.034 Å) — subset은 Step 3+4에서 검증 완료
   - 이번에는 full dataset (10K molecules) 또는 적어도 재확인 차원에서 subset 재실행
   - 확인: RMSD ~0.034 Å 일치

8. **EDSM Training 검증** (1-2 epochs)
   ```bash
   python train.py configs/training.qm9.edsm.yaml
   ```
   - Config 경로 수정 필요
   - 확인: E-DM 독립 파이프라인 동작, `eq_transform` 사용 확인, loss finite

### 2.2 Expected Changes

**수정되는 파일:**
- Config 파일 3개의 데이터 경로 수정 (현재 환경에 맞게):
  - `configs/training.qm9.rdsm.yaml`: `raw_datadir`, `data_split`
  - `configs/finetuning.qm9.rdsm.yaml`: `raw_datadir`, `data_split`
  - `configs/training.qm9.edsm.yaml`: `raw_datadir`, `data_split`

**Deprecation notice 추가되는 파일:**
- `verify_subset.py` (상단 deprecation 주석)
- `data/qm9m/data_split.py` (상단 deprecation 주석)
- `data/qm9m/preprocessing.py` (상단 deprecation 주석)
- `data/qm9m/process_wrong_samples.py` (상단 deprecation 주석)
- `data/qm9m/riemannian_data_sampling/analyze_distribution.py` (상단 deprecation 주석)
- `data/qm9m/riemannian_data_sampling/plot_distribution.py` (상단 deprecation 주석)
- `data/qm9m/MMFFtoDFT_input/results/` 내 Python files (상단 deprecation 주석)

**수정하지 않는 것:**
- Core pipeline 코드 (`train.py`, `sample.py`, `diffusion/`, `model/`, `manifold/` 등)
- Bug fix 3건은 이미 수정 완료 — 추가 수정 없음, 검증만 수행

### 2.3 Verification Plan

**성공 조건 (7항목):**

| # | 조건 | 검증 방법 |
|---|------|----------|
| 1 | Bug fix 3건 검증 (import 통과) | `python -c "..."` import test |
| 2 | 비필수 파일에 deprecation notice 추가 완료 | 파일 확인 |
| 3 | RDSM pretraining 1-2 epoch crash 없음, loss finite | `train.py` 실행, stdout 확인 |
| 4 | Riemannian data sampling이 valid xyz 생성 | `.xyz` 파일 생성 + 내용 검증 |
| 5 | RDSM finetuning 1-2 epoch crash 없음, checkpoint resume 정상 | `train.py --resume_pl` 실행 |
| 6 | RDSM sampling RMSD ~0.034 Å (baseline 일치, ±0.002) | `sample.py` + evaluate |
| 7 | EDSM training 1-2 epoch crash 없음, loss finite | `train.py` 실행 |

**판단 기준:**
- "loss finite" = loss 값이 NaN/Inf가 아닌 유한한 실수값
- "crash 없음" = Python process가 정상 종료 (exit code 0) 또는 epoch 완료 후 수동 종료
- "RMSD ~0.034 Å" = 0.032 ~ 0.036 Å 범위 (기존 baseline 0.034 ± 0.002)

### 2.4 Assumptions & Risks

**Assumptions**
- 기존 데이터 (`data/qm9m/MMFFtoDFT_input/`, 133,886 xyz files)가 training에도 그대로 사용 가능하다
- Accelerate 기반 `train.py`가 단일 GPU에서 정상 동작한다 (multi-GPU 검증은 이 태스크 범위 밖)
- PL checkpoint (`NeuralNet.` prefix)를 `--resume_pl` 옵션으로 Accelerate 모델에 load할 수 있다
  → [[analyze/260305_ckpt_compat.py]]에서 확인 완료: 152 keys, strip 후 100% load
- Riemannian data sampling은 모델 없이 geodesic ODE만 사용하므로, 모델 관련 변경에 무관하다
- `configs/sampling.qm9.edsm.yaml`의 `num_cycles` key가 누락되어 있을 수 있다
  → sampling config에서는 필요하지만, EDSM sampling config에 없으면 추가 필요

**Risks**

1. **Training config의 `use_wandb: True`**
   - 훈련 config 3개 모두 `use_wandb: True`로 설정되어 있음
   - wandb 인증이 안 되어 있으면 실행 실패
   - 완화: `use_wandb: False`로 오버라이드하거나 CLI에서 환경변수 설정

2. **Finetuning data 경로 문제**
   - `finetuning.qm9.rdsm.yaml`의 `raw_datadir`가 riemannian data sampling 결과를 가리킴
   - 아직 해당 데이터가 없을 수 있음 (Step 1.5에서 생성해야 함)
   - 완화: Step 5 → Step 6 순서로 수행 (data sampling → finetuning)
     또는 기존 pretrained data를 사용하여 finetuning pipeline만 검증

3. **EDSM sampling의 `num_cycles` 누락**
   - `sampling.qm9.edsm.yaml`에 `num_cycles` key가 없음
   - `sample.py`에서 `config.sampling.num_cycles` 접근 시 AttributeError 가능
   - 완화: `getattr(config.sampling, 'num_cycles', 1)` 이미 사용중인지 확인,
     필요시 config에 추가

4. **Training에서 wandb 의존성**
   - `train.py`가 `import wandb`를 top-level에서 수행
   - wandb가 설치되어 있지 않으면 import 실패
   - 완화: conda env `neural_opt`에 wandb 설치 확인

5. **Riemannian data sampling 실행 경로**
   - `riemannian_data_sampling.py`가 `manifold.graph`, `utils.chem` 등을 import
   - 실행 시 working directory가 `neural_opt/`여야 함
   - 완화: `python -m data.qm9m.riemannian_data_sampling.riemannian_data_sampling` 또는
     PYTHONPATH 설정

**Mitigation**
1. 모든 training 실행 전 config의 `use_wandb` 확인 및 필요시 override
2. Finetuning은 기존 데이터로 pipeline 검증 (data sampling 결과 사용은 선택적)
3. `num_cycles` 관련 코드를 사전 확인하여 필요시 config 또는 코드 수정
4. conda env `neural_opt`에서 wandb import 사전 확인
5. 모든 스크립트 실행 시 working directory를 `neural_opt/`로 설정

---

<!-- ↑ PRE-REPORT: 여기까지 구현 시작 전에 작성 -->
<!-- ↓ POST-REPORT: 구현 완료 후 아래를 추가 작성 -->

---

## 3. Execution (WHAT HAPPENED)

### 3.1 Iteration History

| # | Date | Analysis | 결론 | Decision |
|---|------|----------|------|----------|
| | | | | |

### 3.2 Actual Actions Taken
- (구현 후 기록)

### 3.3 Differences from Initial Plan
- (구현 후 기록)

### 3.4 Verification Results
- (구현 후 기록)

---

## 4. Conclusion

### 4.1 Summary
- (구현 후 기록)

### 4.2 Hypothesis Update
- (구현 후 기록)

### 4.3 Lessons Learned
- (구현 후 기록)

### 4.4 Next Steps
- (구현 후 기록)

---

## 6. Traceability

### Git
- Branch: `refactoring`
- Commit(s): `886683c` (리팩터링 완료 기준)

### Related Documents
- [[docs/progress/260303_rdm_reproduction_sampling.md]]: Phase 1 reproduction baseline (RMSD 0.034 Å)
- [[docs/progress/260304_codebase_audit.md]]: Module 분석 + dead code backup
- [[docs/progress/260305_neural_opt_refactoring.md]]: 5단계 전면 리팩터링
- [[docs/obsidian/40_experiments/EXP-20260303-01 R-DM Reproduction and Reaction Data Extension.md]]
- [[docs/obsidian/70_projects/PROJ-001 RXN Exploration Pipeline.md]]

### Analysis Scripts
- [[analyze/260305_ckpt_compat.py]]: PL checkpoint 호환성 분석 (이전 태스크)
- [[analyze/260305_torchmetrics_standalone.py]]: torchmetrics 독립 동작 확인 (이전 태스크)

### Key Files (이 태스크에서 다루는 파일)
```
Core pipeline (검증 대상):
  train.py                              ← Accelerate training loop (검증: RDSM pretrain/finetune, EDSM train)
  sample.py                             ← Sampling script (검증: RDSM sampling baseline 재확인)
  evaluate_accuracy.py                  ← Evaluation script
  main.py                               ← Dispatcher (train.py / sample.py로 위임)

Data sampling (검증 대상):
  data/qm9m/riemannian_data_sampling/riemannian_data_sampling.py  ← Bug fix 검증
  data/qm9m/riemannian_data_sampling/riemannian_data_sampling.yaml

Configs (경로 수정 대상):
  configs/training.qm9.rdsm.yaml        ← raw_datadir, data_split 경로 수정
  configs/finetuning.qm9.rdsm.yaml      ← raw_datadir, data_split 경로 수정
  configs/training.qm9.edsm.yaml        ← raw_datadir, data_split 경로 수정
  configs/sampling.qm9.rdsm.yaml        ← 이미 수정 완료 (Phase 1에서)
  configs/sampling.qm9.edsm.yaml        ← 경로 확인 필요

Bug fix 완료 (검증 대상):
  model/__init__.py                      ← self.optim → return, optim_type → cfg.type
  data/qm9m/riemannian_data_sampling/riemannian_data_sampling.py  ← Graph → MolGraph.from_batch(data)

Deprecation 대상:
  verify_subset.py
  data/qm9m/data_split.py
  data/qm9m/preprocessing.py
  data/qm9m/process_wrong_samples.py
  data/qm9m/riemannian_data_sampling/analyze_distribution.py
  data/qm9m/riemannian_data_sampling/plot_distribution.py
  data/qm9m/MMFFtoDFT_input/results/*.py (make_input.py, make_input_batch.py, read_log.py)
  data/qm9m/MMFFtoDFT_input/results/DFT_reopt/*.py (make_traj.py, gather_traj.py, check_reopt_result.py)

Checkpoints (사용):
  checkpoints/rdsm.qm9.pretrained.ckpt  ← finetuning 검증 시 resume
  checkpoints/rdsm.qm9.finetuned.ckpt   ← sampling 검증 시 load
  checkpoints/edsm.qm9.ckpt             ← EDSM sampling 검증 시 load (선택)
```
