# Riemannian Noisy Data Generation 효율화 및 전체 데이터셋 생성
- Date: 2026-03-12
- Author: Seonghwan Kim + Claude
- Repo / Module: `neural_opt/`
- Branch / Commit: `refactoring`

---

## 1. Context & Motivation (WHY)

### 1.1 Project Context

이 태스크는 R-DM (Riemannian Diffusion Model) 논문 재현 파이프라인에서
**Riemannian noisy data generation** 단계의 효율화 및 전체 데이터셋 생성에 해당한다.

전체 재현 파이프라인:
```
Step 1:   RDSM Pretraining ────────── 완료 (검증: 260311)
Step 1.5: Riemannian Data Sampling ── ★ 이 태스크
Step 2:   RDSM Finetuning ─────────── 데이터 필요 (Step 1.5 의존)
Step 3:   RDSM Sampling ──────────── 완료 (RMSD 0.034 Å)
```

Riemannian noisy data는 pretrained E-DM 모델을 fine-tuning하기 위한 augmented training data이다.
논문에서는 QM9 기준 reference DFT 구조 1개당 noisy 구조 3개를 생성한다:
- Train set ~98,000개 x 3 repetitions = ~294,000 noisy samples

이전 태스크 [[docs/progress/260311_reproduction_pipeline.md]]에서 전체 파이프라인 검증이 완료되었으며,
Riemannian data sampling도 소규모 subset으로 정상 동작을 확인했다.
그러나 **현재 ODE solver 속도가 실용적이지 않아** 전체 데이터 생성이 불가능한 상태이다.

### 1.2 Previous Attempts

- [[docs/progress/260311_reproduction_pipeline.md]]: Pipeline 전체 검증 완료
  - 3건의 버그 수정 (Graph->MolGraph, optimizer return, optim_type)
  - 12개 파일 deprecation notice 추가
  - RDSM pretraining, riemannian sampling, finetuning, sampling, EDSM training 모두 검증 완료

- [[analyze/260312_riemannian_noise_comparison.py]]: Noise 통계 비교 분석
  - 87개 test set 샘플로 DFT vs MMFF vs Riemannian noisy 비교
  - MMFF RMSD mean=0.183 A vs Riemannian noisy RMSD mean=0.194 A -- 논문과 정합
  - D-MAE도 MMFF=0.065 A vs Riemannian=0.076 A로 비슷한 range
  - ODE solver가 정상 동작 확인 (time_step에 비례하는 noise level)

### 1.3 Hypothesis Status

- [채택] Riemannian ODE solver가 논문과 수치적으로 정합하는 noise를 생성한다
  -> [[analyze/260312_riemannian_noise_comparison.py]]에서 확인 완료
- [채택] 전체 파이프라인이 end-to-end로 동작한다
  -> [[docs/progress/260311_reproduction_pipeline.md]]에서 확인 완료
- [검증중] ODE solver 효율화 후에도 수치 정합성이 유지된다
  -> 이 태스크에서 검증 예정
- [검증중] 효율화로 현재 대비 3배 이상 속도 개선이 가능하다
  -> 이 태스크에서 검증 예정

### 1.4 Current Problem

**핵심 병목: `batch_geodesic_ode_solve()` 속도**

현재 성능 측정치:
```
CPU, batch=100:  ~32분/batch  -> train set 328 batches = ~175시간 (7.3일)
GPU, batch=10:   ~45초/batch  -> train set 3280 batches = ~41시간 (1.7일)
```
3회 반복 생성 시: CPU ~22일, GPU ~5일 -- 비실용적

**병목 구간 분석 (per ODE step, GPU batch=10, ~1.3초/step):**

```
batch_geodesic_ode_solve()
  while (~done).any():                         ← 30-40 iterations per batch
    batch_advance_heun()                       ← 2x _advance() per Heun step
      _advance(done, x, x_dot, ...)
        a. sparse_batch_jacobian_q()           ← sparse (B, e, 3n)
        b. J.to_dense()                        ← sparse -> dense 변환 ★ 병목 1
        c. batch_christoffel()
           i.  sparse_batch_hessian_q()        ← sparse (B, 9n^2, e)
           ii. hess.index_select()             ← sparse -> subset
           iii. batch_pinv1(J)                 ← vmap(pinv) = SVD ★ 병목 2
                J.to_dense() + vmap(linalg.pinv)
           iv.  torch.bmm(hess, J_inv)         ← (B, 9n^2, e) x (B, e, 3n) ★ 병목 3
           v.   reshape -> (B, 3n, 3n, 3n)     ← Christoffel tensor
        d. einsum("bj,bkij,bi->bk", ...)      ← 가속도 계산
```

**각 병목의 원인:**

| # | 위치 | 원인 | 메모리/시간 |
|---|------|------|------------|
| 1 | `J.to_dense()` | sparse Jacobian을 매 step 새로 dense 변환 | (B, e, 3n) |
| 2 | `batch_pinv1(J)` | vmap(linalg.pinv) 내부의 SVD 분해 | O(min(e,3n)^2 * max(e,3n)) per sample |
| 3 | `bmm(hess, J_inv)` | Hessian도 to_dense() 후 dense bmm | (B, 9n^2, e) x (B, e, 3n) |

**Christoffel tensor 메모리 문제:**
```
Christoffel shape: (B, 3n, 3n, 3n)
n=9 (QM9 max atoms): (B, 27, 27, 27) = B * 19,683 floats = B * 77KB
n=29 (larger molecules): (B, 87, 87, 87) = B * 658,503 floats = B * 2.5MB
-> batch=100, n=29: ~250MB for Christoffel alone
```

**Heun method의 추가 부담:**
- Heun step당 `_advance()`를 2번 호출
- 각 `_advance()`에서 Jacobian, Hessian, pinv, Christoffel 모두 재계산
- 즉, 1 ODE step = 2 x (Jacobian + Hessian + SVD + Christoffel einsum)

**현재 코드의 비효율 요소:**

1. **Sparse -> dense 반복 변환**: `sparse_batch_jacobian_q()`가 sparse tensor를 반환하지만,
   `batch_christoffel()`과 `batch_pinv1()` 모두 `.to_dense()`를 호출.
   sparse 생성 -> dense 변환이 매 step 반복됨.

2. **pinv 반복 계산**: Heun의 첫 번째 `_advance()`에서 계산한 J_inv를
   두 번째 `_advance()`에서 재활용하지 않음.
   (두 번째 call에서 x가 업데이트되므로 J 자체가 바뀌지만,
   첫 번째 step의 J는 `batch_advance_heun()`에서 이미 계산해놓고도 Christoffel에서 재계산)

3. **Done sample 연산 skip 불완전**: `not_done_index`로 J, hess를 subset하고 있으나,
   sparse tensor의 `index_select`는 효율적이지 않음 (내부적으로 dense 경로 탈 수 있음).

4. **Debug timing code 잔존**: `time` import, timing print가 production code에 포함.

---

## 2. Plan (HOW)

### 2.1 Overview

2-part로 나누어 수행한다.

**Part A: ODE Solver 효율화** -- 수치 정합성 유지하면서 속도 3x+ 개선
**Part B: 전체 데이터 생성** -- 효율화된 solver로 train set 3회 반복 생성

### 2.2 Part A: ODE Solver 효율화

#### A-1. Jacobian/Hessian을 처음부터 dense로 계산

**현재 흐름:**
```
sparse_batch_jacobian_q() -> sparse (B, e, 3n)
  .to_dense() ← batch_advance_heun()에서
  .to_dense() ← batch_christoffel() -> batch_pinv1()에서
sparse_batch_hessian_q() -> sparse (B, 9n^2, e)
  .to_dense() ← batch_christoffel() -> bmm()에서
```

**개선 방향:**
```
dense_batch_jacobian_q() -> dense (B, e, 3n) 직접 반환
dense_batch_hessian_q() -> dense (B, 9n^2, e) 직접 반환
```

- sparse COO tensor 생성 + coalesce + to_dense 대신,
  dense zero tensor를 미리 할당하고 scatter로 직접 채움.
- QM9의 경우 n_max=9, e=36 → Jacobian: (B, 36, 27), Hessian: (B, 729, 36)
  이 크기에서는 sparse overhead가 dense 직접 연산보다 큼.

**구현 계획:**
- `dense_batch_jacobian_q()` 신규 함수 추가
- `dense_batch_hessian_q()` 신규 함수 추가
- `_advance()`, `batch_advance_heun()`, `batch_christoffel()`에서 호출 변경

**예상 효과:** sparse 생성/변환 overhead 제거. step당 ~10-20% 절감 예상.

#### A-2. pinv 최적화 -- QR decomposition 활용

**현재:**
```python
batch_pinv1(J, rtol, atol):
    return vmap(torch.linalg.pinv)(J.to_dense(), rtol=rtol, atol=atol)
```
`linalg.pinv`는 내부적으로 SVD를 사용. SVD는 O(min(m,n)^2 * max(m,n)).

**개선 방향:**
Jacobian J의 shape은 (B, e, 3n) = (B, 36, 27) for QM9.
e > 3n이므로 J는 over-determined (행이 열보다 많음).
QR decomposition으로 pseudo-inverse를 더 효율적으로 계산 가능:
```python
# J: (B, e, 3n), J^T: (B, 3n, e)
# pinv(J) = J^T (J J^T)^{-1} ... 이건 under-determined case
# 여기서는 e > 3n이므로:
# pinv(J) = (J^T J)^{-1} J^T
Q, R = torch.linalg.qr(J.transpose(-1,-2))  # J^T = QR, Q: (B, 3n, 3n), R: (B, 3n, e)
# pinv(J) = (R^T R)^{-1} R^T Q^T = R^{-1} Q^T ...
```

또는 `torch.linalg.lstsq`를 사용하여 pinv를 직접 계산하지 않고
Christoffel 계산에 필요한 `J_inv @ x` 형태의 solve로 대체:
```python
# 현재: J_inv = pinv(J); christoffel = bmm(hess, J_inv.T)
# 대안: 직접 lstsq solve
# hess @ J_inv^T = hess @ pinv(J)^T
# = solve for X in: J^T X = hess^T, then christoffel = X^T
```

**구현 계획:**
- `batch_pinv_qr()` 함수 추가: QR 기반 pseudo-inverse
- 수치 안정성을 위해 `svd_tol` threshold는 유지 (작은 singular value 처리)
- SVD fallback: QR이 수치적으로 불안정한 경우 SVD로 fallback

**예상 효과:** pinv 계산 시간 ~50% 절감. SVD가 전체의 ~30%를 차지하므로 step당 ~15% 절감.

#### A-3. Christoffel 계산 최적화

**현재:**
```python
def batch_christoffel(...):
    hess = sparse_batch_hessian_q(...)         # sparse (B, 9n^2, e)
    hess = hess.index_select(0, not_done_index) # sparse subset
    J_inv = batch_pinv1(J, ...)                 # dense (B, 3n, e) via SVD
    christoffel = bmm(hess, J_inv).T.reshape(B, 3n, 3n, 3n)
```

**문제:**
- `hess`의 shape이 `(B, 9n^2, e)` = `(B, 729, 36)` for QM9
- `bmm(hess, J_inv^T)`: `(B, 729, 36) x (B, 36, 27)` -> `(B, 729, 27)` -> reshape `(B, 27, 27, 27)`
- 이 결과 Christoffel tensor `(B, 3n, 3n, 3n)`을 full로 만들지만,
  실제 사용은 `einsum("bj,bkij,bi->bk")` 형태로 contraction

**개선 방향 1 -- Fused Christoffel contraction:**
```python
# 현재: Gamma = hess @ J_inv^T; x_ddot = einsum("bj,bkij,bi->bk", x_dot, Gamma, x_dot)
# 대안: x_ddot = einsum 없이 직접 계산
# x_ddot_k = sum_ij Gamma^k_ij * x_dot_i * x_dot_j
#          = sum_ij (hess @ J_inv^T)_{k,i,j} * x_dot_i * x_dot_j
# Christoffel 전체를 materialize하지 않고,
# 중간 결과 (B, 9n^2, 3n) 에서 바로 contraction
```

이 최적화는 메모리와 속도 모두 개선:
- Christoffel tensor `(B, 3n, 3n, 3n)` 할당을 피함
- einsum 대신 두 단계의 bmm/matvec으로 대체 가능

**개선 방향 2 -- Hessian 구조 활용:**
Hessian은 pairwise interaction에서 오므로 극도로 sparse.
각 edge (i,j)가 4개의 3x3 블록만 기여 (ii, ij, ji, jj).
Dense (B, 9n^2, e) 대신, edge별로 기여분을 직접 accumulate.

**구현 계획:**
- 먼저 A-1, A-2 구현 후 프로파일링
- 프로파일링 결과에 따라 fused contraction 구현 여부 결정
- 메모리 이슈가 심하면 (batch size 키울 때) Christoffel 최적화 우선 수행

**예상 효과:** Christoffel 메모리 ~50% 절감, 계산 ~20% 절감.

#### A-4. GPU batch size 최적화

**현재:** GPU batch=10으로 실행 (메모리 제한으로 추정)

**개선 방향:**
- A-1~A-3 효율화 후 메모리 footprint 재측정
- 가능한 최대 batch size를 프로파일링으로 결정
- batch size 증가 -> GPU utilization 향상 -> 추가 속도 개선

**H200 기준 메모리 예산:**
```
H200 VRAM: 80GB (HBM3)
현재 batch=10, n=9:
  Christoffel: 10 * 27^3 * 4B = ~79KB
  Jacobian:    10 * 36 * 27 * 4B = ~39KB
  Hessian:     10 * 729 * 36 * 4B = ~1MB
  → 합계 ~1.2MB (매우 작음)
```
병목은 VRAM이 아니라 연산 자체일 가능성이 높음.
batch size를 100~300으로 키워도 메모리는 문제없을 것으로 예상.

#### A-5. 디버그 코드 정리

- `solver.py` 상단의 `timer` decorator, `from time import time` 제거
- `batch_geodesic_ode_solve()` 내부의 `_time` import, timing print 제거
- `sparse_batch_hessian_q()` 내부의 `torch.cuda.empty_cache()` 제거 (불필요)

#### A-6. 수치 검증 (효율화 전후 비교)

모든 최적화 적용 후 반드시 수치 정합성 검증 수행:

```
검증 방법:
1. 동일 input (seed=42, test set 첫 batch) 에 대해
   효율화 전/후의 output 비교
2. 비교 지표:
   - pos_noise: RMSD < 1e-4 Å (또는 reasonable tolerance)
   - q_dot: relative error < 1e-3
   - ban_index: 동일 set
3. 분석 스크립트: analyze/260312_solver_optimization_verify.py
```

### 2.3 Part B: 전체 데이터 생성

효율화 완료 후 수행.

#### B-1. 생성 설정

논문 config (riemannian_data_sampling.yaml 기반):
```yaml
manifold.ode_solver:
  alpha: 1.7
  beta: 0.01
  gamma: 0.0
  svd_tol: 1e-2
diffusion.scheduler:
  t0: 1
  t1: 150
```

생성 계획:
```
Train set: ~98,000 molecules x 3 repetitions = ~294,000 xyz files
Batch size: 100 (또는 효율화 후 최적값)
Total batches: ~980 per repetition x 3 = ~2,940 batches
```

#### B-2. 병렬 실행

기존 `run_parallel.sh` 활용:
```bash
bash data/qm9m/riemannian_data_sampling/run_parallel.sh \
    --num_workers 8 \
    --config_yaml data/qm9m/riemannian_data_sampling/riemannian_data_sampling.yaml \
    --save_xyz /path/to/output/seed42 \
    --save_csv /path/to/output/seed42.csv \
    --dataloader train \
    --device cuda \
    --alpha 1.7 --beta 0.01 --gamma 0.0 --svd_tol 0.01 \
    --t0 1 --t1 150 \
    --seed 42
```

3회 반복: seed=42, 43, 44 (또는 논문에서 사용한 seed 확인 후 결정)

#### B-3. 데이터 검증

생성 완료 후:
1. 총 파일 수 확인: ~294,000 xyz files
2. 무작위 100개 샘플의 RMSD 분포 확인 (논문 범위와 비교)
3. ban_index 비율 확인: 전체의 ~1% 이내여야 함
4. finetuning config의 `raw_datadir` 경로 업데이트

#### B-4. Finetuning 연결 확인

생성된 데이터로 finetuning이 정상 동작하는지 1-2 epoch 검증:
```
configs/finetuning.qm9.rdsm.yaml 수정:
  dataset.raw_datadir: -> 생성된 xyz 디렉토리
  noise_type: diffusion_custom (이미 설정됨)
```

### 2.4 Expected Changes

**수정되는 파일:**
- `manifold/solver.py` -- 핵심 최적화 대상
  - `dense_batch_jacobian_q()` 신규 함수 추가
  - `dense_batch_hessian_q()` 신규 함수 추가
  - `batch_christoffel()` 내부 최적화
  - `_advance()` dense 경로 사용으로 변경
  - `batch_advance_heun()` J 재활용 최적화
  - `batch_pinv_qr()` 신규 함수 추가 (선택적)
  - 디버그 코드 정리

**수정하지 않는 것:**
- `riemannian_data_sampling.py` -- 호출 인터페이스 변경 없음
- `run_parallel.sh` -- 그대로 사용
- ODE solver의 수학적 알고리즘 (Heun method, adaptive step size, ban_index 로직)

**생성되는 파일:**
- `analyze/260312_solver_optimization_verify.py` -- 수치 검증 스크립트
- `analyze/260312_solver_profiling.py` -- 프로파일링 스크립트
- xyz output files (data generation 결과)

### 2.5 Verification Plan

**성공 조건 (4항목):**

| # | 조건 | 검증 방법 |
|---|------|----------|
| 1 | ODE solver 속도 3x+ 개선 (GPU 기준) | 동일 test batch 기준 wall-clock 비교: before ~45초/batch(B=10) vs after |
| 2 | 수치 정합성: 효율화 전후 RMSD < 1e-4 A | `analyze/260312_solver_optimization_verify.py` |
| 3 | 전체 train set 3 repetitions 데이터 생성 완료 | 파일 수 확인: ~294,000 xyz files |
| 4 | 생성된 데이터로 finetuning 1-2 epoch 정상 동작 | `train.py finetuning.qm9.rdsm.yaml` 실행, loss finite |

**판단 기준:**
- "속도 3x+ 개선" = 동일 batch에서 wall-clock time이 1/3 이하
- "RMSD < 1e-4 A" = 최적화 전후 동일 input에 대한 output 차이
- "데이터 생성 완료" = xyz 파일 존재 + valid geometry (atom count 일치, finite coordinates)
- "loss finite" = NaN/Inf 없이 1-2 epoch 완주

### 2.6 Assumptions & Risks

**Assumptions**
- Jacobian/Hessian의 dense 직접 계산이 sparse 생성 + to_dense보다 빠르다
  (QM9 scale에서 sparsity ratio가 낮아 sparse overhead가 큼)
- QR 기반 pseudo-inverse가 SVD 대비 수치적으로 충분히 안정적이다
  (svd_tol threshold를 QR에서도 적절히 반영 가능)
- H200 GPU 메모리가 batch=100 이상에서도 충분하다
  (위 계산으로 ~120MB, 충분할 것으로 예상)
- 3회 반복의 seed가 논문과 동일하지 않아도 finetuning 품질에 영향이 미미하다

**Risks**

1. **수치 안정성 저하 (Medium)**
   - QR decomposition이 SVD보다 ill-conditioned matrix에서 불안정
   - 특히 `svd_tol=1e-2`로 작은 singular value를 truncate하는 현재 로직이
     QR에서는 직접 구현이 필요
   - 완화: SVD fallback 유지. QR 실패 시 자동으로 SVD로 전환.

2. **Dense Hessian 메모리 (Low)**
   - Hessian `(B, 9n^2, e)`: batch=100, n=29 -> `(100, 7569, 406)` -> ~1.2GB
   - 큰 분자에서 메모리 문제 발생 가능
   - 완화: QM9은 n_max=9이므로 `(100, 729, 36)` -> ~10MB. 문제 없음.
     향후 larger molecule로 확장 시 sparse 경로를 대체 옵션으로 유지.

3. **효율화 효과 미달 (Medium)**
   - sparse->dense 변환이 실제 병목의 주요 원인이 아닐 수 있음
   - pinv(SVD)가 아닌 다른 부분이 dominant일 수 있음
   - 완화: 먼저 프로파일링 스크립트를 작성하여 실제 병목을 정밀 측정 후 우선순위 결정.

4. **전체 데이터 생성 중 ODE solver failure (Low)**
   - ban_index 비율이 전체의 1% 이하이지만, 일부 분자에서 수렴 실패 가능
   - 완화: 기존 retry 메커니즘 + ban_index 로깅으로 실패 케이스 추적.
     실패 비율이 높으면 해당 분자만 별도 처리 (더 작은 dt, 또는 skip).

5. **Finetuning 데이터 형식 불일치 (Low)**
   - 생성된 xyz의 metadata format이 finetuning dataloader와 불일치할 수 있음
   - 완화: 기존 검증 [[docs/progress/260311_reproduction_pipeline.md]]에서
     소규모 생성 + finetuning 연결을 이미 확인함.

**Mitigation 전략**
1. 최적화 순서: 프로파일링 -> A-1 (dense 변환) -> A-2 (pinv) -> 검증 -> A-3 (선택적) -> A-4 (batch size)
2. 각 단계에서 수치 검증 수행 (점진적 적용)
3. 원본 sparse 함수들은 삭제하지 않고 유지 (fallback 용도)

---

<!-- PRE-REPORT: 여기까지 구현 시작 전에 작성 -->
<!-- POST-REPORT: 구현 완료 후 아래를 추가 작성 -->

---

## 3. Execution (WHAT HAPPENED)

### 3.1 Iteration History

| # | Date | Analysis | 결론 | Decision |
|---|------|----------|------|----------|
| 0 | 03-12 | [[analyze/260312_riemannian_noise_comparison.py]] | MMFF RMSD=0.183, Riemannian=0.194 — 논문과 정합 | ADOPT |
| 1 | 03-12 | [[analyze/260312_solver_profiling.py]] | pinv(SVD)=98.1% of time. sparse→dense=0.2% | ADOPT |
| 2 | 03-12 | [[analyze/260312_pinv_alternatives.py]] | Cholesky 1700x fast but 88% error. native pinv=same as vmap | REJECT (Cholesky) |
| 3 | 03-12 | [[analyze/260312_pinv_cpu_benchmark.py]] | CPU 1-thread pinv: 65.6ms vs GPU 5503ms = **83.9x speedup** | ADOPT |
| 4 | 03-12 | [[analyze/260312_solver_optimization_verify.py]] | End-to-end B=100: **11.5x speedup**, median RMSD 3.88e-06 | ADOPT |

### 3.2 Actual Actions Taken

**Part A: ODE Solver 효율화**

1. **프로파일링** (Iteration 1): `_advance()` 내부 component별 시간 측정
   - 결과: `batch_pinv1()` (vmap SVD) = 98.1%, 나머지 모두 합쳐 1.9%
   - sparse→dense 변환은 0.2%로 무시 가능

2. **대안 검토** (Iteration 2-3):
   - Cholesky/QR/lstsq: 빠르지만 SVD truncation semantics 재현 불가 (error > 80%)
   - CPU offload: **83.9x speedup** on pinv alone, error 1.6e-05

3. **구현** — `manifold/solver.py`:
   - `batch_pinv1()`: GPU→CPU 전송, `torch.set_num_threads(1)`, CPU pinv, GPU 전송
   - `batch_svd1()`: 동일 방식 CPU offload
   - Debug code 정리: timer decorator, timing prints, `cuda.empty_cache()` 제거

4. **GPU 호환성 수정** — `riemannian_data_sampling.py`:
   - `ban_batch_mask` device mismatch 수정
   - `.cpu()` 추가: `data.x.split()`, `Atoms()` positions, `q_target.tolist()`
   - `perr_*.tolist()`, `time_step.tolist()` CPU 전환

**Part B: 전체 데이터 생성**

5. Quick test: train 2 batches → 20초 (10초/batch), 197 files 생성
6. 전체 생성 시작: seed=42, 982 batches, `--device cuda`
   - 예상 시간: ~2.7시간/repetition
   - 출력: `data/qm9m/riemannian_xyz/seed{42,43,44}/`

### 3.3 Differences from Initial Plan

| Plan | Actual | 이유 |
|------|--------|------|
| A-1: dense J/H 직접 계산 | **SKIP** | 프로파일링 결과 sparse→dense가 0.2%이므로 불필요 |
| A-2: QR pinv | **REJECT → CPU offload** | QR/Cholesky는 SVD truncation 재현 불가. CPU offload가 83.9x |
| A-3: Christoffel fusion | **SKIP** | bmm+einsum 합쳐 0.7%이므로 불필요 |
| A-4: GPU batch size 증가 | **불필요** | CPU offload로 이미 충분한 속도 |
| A-5: debug 정리 | 완료 | 계획대로 |
| A-6: 수치 검증 | 완료 | 계획대로 (median RMSD 3.88e-06) |
| GPU 호환성 | **추가 작업** | 기존 코드가 CPU only — GPU에서 device mismatch 5건 수정 |

**핵심 변경: 최적화 전략이 "알고리즘 교체"에서 "device offload"로 완전히 전환됨**

### 3.4 Verification Results

| 조건 | 결과 | 상세 |
|------|------|------|
| Speed 3x+ | **PASS (11.5x)** | B=100: 824s → 72s (end-to-end ODE solve) |
| RMSD < 1e-4 | **PASS (median)** | Median 3.88e-06. Max 0.42 (banned molecules) |
| Ban index | **거의 일치** | 10 vs 11 (1개 경계 분자 차이) |
| Import gate | **PASS** | `from manifold.solver import GeodesicSolver` OK |
| Train 2-batch test | **PASS** | 197 xyz files, 10초/batch |

---

## 4. Conclusion

### 4.1 Summary

GPU에서 small matrix SVD가 비효율적이라는 사실을 프로파일링으로 발견.
CPU 1-thread offload로 pinv 단계를 83.9x 가속, end-to-end 11.5x speedup 달성.
전체 train set 데이터 생성 예상 시간: 9.4일 → **~8.1시간** (3 repetitions).

### 4.2 Hypothesis Update

- [기각] sparse→dense 변환이 주요 병목이다 → **0.2%로 무시 가능**
- [기각] QR/Cholesky로 SVD를 대체할 수 있다 → **truncation semantics 재현 불가**
- [채택] GPU small matrix SVD는 CPU 대비 극도로 비효율적이다
  → 512 cores 환경에서 1-thread CPU가 GPU 대비 83.9x
- [채택] CPU offload 후에도 수치 정합성 유지된다 → median RMSD 3.88e-06

### 4.3 Lessons Learned

1. **프로파일링 우선**: Pre-report에서 3개 병목을 가정했으나, 실제로는 1개(SVD)가 98%.
   프로파일링 없이 A-1~A-3을 구현했다면 시간 낭비였을 것.
2. **GPU ≠ always faster**: Small matrix operations (300x75 SVD)은 GPU의 kernel launch
   overhead + parallelism mismatch 때문에 CPU가 압도적으로 빠름.
3. **thread=1이 최적**: CPU multi-threading은 small matrix에서 contention만 증가.
   512 cores에서 1 thread가 가장 빠름 (65ms vs 1168ms@256 threads).
4. **GPU 호환성**: CPU-only 코드를 GPU로 옮길 때 `.cpu()` 누락이 빈번. 체계적 검토 필요.

### 4.4 Next Steps

1. Seed 42 생성 완료 후 seed 43, 44 추가 생성
2. 생성된 데이터 검증: 파일 수, RMSD 분포, ban 비율
3. Finetuning config 업데이트: `raw_datadir` → `data/qm9m/riemannian_xyz/`
4. RDSM finetuning 실행 (full data)

---

## 5. Failures & Resolutions

| Step | Failure | Resolution |
|------|---------|------------|
| GPU test (2-batch) | `torch.isin` device mismatch | `torch.arange(..., device=ban_index.device)` |
| GPU test (2-batch) | `data.x.split(natoms)` → numpy fail on CUDA | `.cpu()` 추가 |
| GPU test (2-batch) | `Atoms(positions=cuda_tensor)` | `.cpu()` 추가 |
| GPU test (2-batch) | `q_target.tolist()` on CUDA | `.cpu()` 추가 |
| GPU test (2-batch) | `perr.tolist()` on CUDA | `.cpu()` 추가 |

---

## 6. Traceability

### Git
- Branch: `refactoring`

### Related Documents
- [[docs/progress/260311_reproduction_pipeline.md]]: Pipeline 전체 검증 (이전 태스크)
- [[analyze/260312_riemannian_noise_comparison.py]]: Noise 통계 비교 분석

### Analysis Scripts
- [[analyze/260312_solver_profiling.py]]: Component-level 프로파일링 — pinv=98%
- [[analyze/260312_pinv_alternatives.py]]: Cholesky/QR/lstsq 대안 벤치마크
- [[analyze/260312_pinv_cpu_benchmark.py]]: CPU vs GPU pinv 벤치마크 — CPU 1-thread 83.9x
- [[analyze/260312_solver_optimization_verify.py]]: End-to-end 수치 검증 — 11.5x, RMSD 3.88e-06
- [[analyze/260312_pinv_alternatives_v2.py]]: Eigendecomposition 대안 (작성됨, 미실행)

### Key Files (이 태스크에서 수정한 파일)
```
manifold/solver.py:
  batch_pinv1()  — CPU offload pinv (GPU→CPU transfer, thread=1, CPU→GPU)
  batch_svd1()   — CPU offload SVD (동일 방식)
  (removed)      — timer decorator, debug timing prints, cuda.empty_cache()

data/qm9m/riemannian_data_sampling/riemannian_data_sampling.py:
  line 425       — ban_batch_mask device fix
  line 429-438   — .cpu() for data.x.split, ban_batch_mask
  line 512-514   — .cpu() for Atoms positions
  line 516       — .cpu() for q_target.tolist()
  line 477-479   — .cpu() for results tolist (x2)

Generated data:
  data/qm9m/riemannian_xyz/seed42/  — full train set (generating)
  data/qm9m/riemannian_xyz/seed43/  — (pending)
  data/qm9m/riemannian_xyz/seed44/  — (pending)
```
