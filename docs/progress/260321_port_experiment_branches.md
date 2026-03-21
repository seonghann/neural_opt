# Experiment Branch Porting: time_embedding_ablation + conditioning_exp → refactored codebase
- Date: 2026-03-21
- Author: Seonghwan Kim + Claude
- Repo / Module: `neural_opt/`
- Branch / Commit: main (target), `origin/time_embedding_ablation`, `origin/conditioning_exp` (source)

---

## 1. Context & Motivation (WHY)

### 1.1 Project Context
이 태스크는 Nature Computational Science 2026 논문의 supplementary experiments 코드를
리팩터링된 코드베이스에 통합하는 작업이다.

논문 출판 후 두 개의 실험 브랜치(`time_embedding_ablation`, `conditioning_exp`)가
리팩터링 이전 코드 (PyTorch Lightning 기반)에 남아있다.
[[docs/progress/260305_neural_opt_refactoring.md]]에서 PL → Accelerate 전면 리팩터링이
완료되었으므로, 이 브랜치들의 실험 코드를 새 구조에 맞게 porting해야 한다.

상위 연구 목표: [[docs/obsidian/70_projects/PROJ-001 RXN Exploration Pipeline.md]]

### 1.2 Previous Attempts
- [[docs/progress/260305_neural_opt_refactoring.md]]: PL → Accelerate 리팩터링 완료
  - 7,921 LOC → 6,367 LOC (-19.6%)
  - `BridgeDiffusion(pl.LightningModule)` → `DiffusionModel(nn.Module)` + standalone sampling functions
  - `diffusion_model.py` (2,058 LOC god class) → `diffusion/model.py` (443) + `diffusion/sampling.py` (343)
  - `model/` + `model_tsdiff/` → 단일 `model/` 패키지
  - `utils/geodesic_solver.py` → `manifold/solver.py`, `utils/rxn_graph.py` → `manifold/graph.py`
- [[docs/progress/260313_server_migration.md]]: 새 서버 환경 구축 완료
  - RTX 3090, PyTorch 2.5.1+cu121, Accelerate 1.13.0
  - Sampling baseline 재현: RMSD median 0.0336 Å

### 1.3 Hypothesis Status
- [검증중] 리팩터링 전 실험 브랜치의 코드를 리팩터링 후 구조에 clean하게 포팅할 수 있다
  → 이 태스크에서 검증. 파일 매핑이 1:1이 아닌 부분(god class 분리)에서 난이도 존재.
- [채택] `conditioning_exp`는 `time_embedding_ablation`의 subset이다 (4/7 commits 공유)
  → 따라서 `time_embedding_ablation` 기준으로 porting하면 `conditioning_exp` 코드도 포함됨.

### 1.4 Current Problem

**현재 문제 상황:**

두 실험 브랜치에 논문 supplementary 실험에 필요한 6개 feature가 있으나,
모두 리팩터링 이전의 PL 기반 코드 구조(`BridgeDiffusion` god class, `model_tsdiff/`, `utils/rxn_graph.py`)에
작성되어 있어 현재 코드베이스(`DiffusionModel` + standalone functions, `model/`, `manifold/graph.py`)와 호환되지 않는다.

**6개 feature 요약:**

| # | Feature | 용도 | 원본 위치 (old) | 대상 위치 (new) |
|---|---------|------|----------------|----------------|
| 1 | `MolGraph.reset_to_dummy()` | Graph unconditioning (ablation S4) | `utils/rxn_graph.py` | `src/manifold/graph.py` |
| 2 | Graph conditioning in training | Classifier-free guidance training | `diffusion/diffusion_model.py` | `src/diffusion/model.py` + `src/diffusion/sampling.py` |
| 3 | Time embedding in GeoDiffEncoder | Ablation: time → encoder (no improvement) | `model/geodiff_encoder.py` | `src/model/encoder.py` + `src/diffusion/model.py` |
| 4 | Continuous diffusion scheduler | VP-type sigmoid beta schedule | `diffusion/continuous_scheduler.py` | `src/diffusion/continuous_scheduler.py` (new) |
| 5 | New score functions | `predict_vector`, `predict_score` | `diffusion/diffusion_model.py` | `src/diffusion/sampling.py` |
| 6 | New sampling methods | Langevin SDE, gradient descent | `diffusion/diffusion_model.py` | `src/diffusion/sampling.py` |

**핵심 난이도:**

리팩터링에서 가장 큰 변경은 `BridgeDiffusion` god class (2,058 LOC)의 분리였다.
Feature 2, 3, 5, 6은 모두 이 god class의 메서드로 구현되어 있으므로,
단순 copy-paste가 불가능하고 새 구조의 패턴에 맞게 재구성해야 한다:

**코드 품질 주의사항:**

Old branch의 코드는 실험 당시의 prototype 수준으로, 상당히 지저분하다.
현재 refactoring branch는 깔끔하게 정리된 상태이므로, porting 시 반드시 현재의 notation에 맞춰야 한다:

- **Debug print 전부 제거**: `print(f"Debug: ...")`, `print(f"t={t[0]}")` 등
- **Dead code 제거**: 주석 처리된 old code, `TYPE == 1/2/3/4` 같은 hacky 분기
- **하드코딩 제거**: magic number를 config로 빼거나, 의미 있는 변수명 사용
- **현재 naming convention 준수**: 변수명, 함수 signature, docstring 스타일
- **`self.xxx` → `model.xxx` 변환**: standalone function 패턴 (model을 첫 번째 인자로)

```
Old (god class):                          New (분리된 구조):
  BridgeDiffusion.noise_sampling()          DiffusionModel.noise_sampling()
  BridgeDiffusion.sample_batch_*()    →     sample_batch_*() in sampling.py (standalone)
  BridgeDiffusion.predict_*()               predict_*() in sampling.py (standalone)
  BridgeDiffusion.forward()                 DiffusionModel.forward()
```

- Old: `self.noise_schedule`, `self.geodesic_solver` 등을 `self`로 접근
- New: `model`을 첫 번째 인자로 받고, `model.noise_schedule`, `model.geodesic_solver`로 접근

**이 작업이 필요한 이유:**
- 논문 supplementary 실험 (S4 ablation 등)을 재현하려면 이 코드가 필요
- 리팩터링된 codebase에서만 향후 개발이 진행되므로, 실험 코드도 이관해야 함
- 브랜치가 방치되면 코드가 분산되어 재현성 저하

**해결되었을 때 기대 효과:**
- 모든 논문 실험 코드가 단일 코드베이스에 통합
- Supplementary S4 ablation 등 재현 가능
- 향후 확장 시 CFG, continuous scheduler 등을 바로 활용 가능

---

## 2. Plan (HOW)

### 2.1 Planned Approach

의존성 순서에 따라 8단계로 순차 구현. 각 단계마다 import gate를 통과해야 다음으로 진행.

**Step 1: `MolGraph.reset_to_dummy()` (no dependencies)**

대상: `src/manifold/graph.py`

- `MolGraph` 클래스에 `reset_to_dummy()` 메서드 추가
- 동작: edge를 zeroing, node features를 `ATOM_ENCODER`의 dummy 값으로 대체
- `ATOM_ENCODER`는 `src/utils/chem.py:21`에 정의됨
- 검증: import gate + unit test (reset 전후 shape 동일, 값 변경 확인)

**Step 2: Continuous scheduler (new file, no dependencies)**

대상: `src/diffusion/continuous_scheduler.py` (신규)

- `SigmoidDiffusionScheduler(nn.Module)` 클래스 구현
  - VP-type continuous-time scheduler with sigmoid beta schedule
  - Methods: `get_alpha(t)`, `get_sigma(t)`, `get_beta(t)`, `get_SNR(t)`,
    `get_f(t)`, `get_g(t)`, `_log_alpha(t)`, `_log_sigma(t)`
- `src/diffusion/noise_scheduler.py`의 `load_noise_scheduler()`에 `continuous_tsdiff` 등록
- 검증: import gate + smoke test (t 범위 [0, 1]에서 alpha/sigma 값 확인, NaN 없음)

**Step 3: Time embedding in GeoDiffEncoder (touches encoder.py + model.py)**

대상: `src/model/encoder.py`, `src/diffusion/model.py`

encoder.py 변경:
- `use_time_embedding` config 옵션 추가 (default False)
- 활성 시: `emb_dim = hidden_dim - 1`, atom/feat embedding이 `emb_dim` 사용
- `graph_encoding()`: normalized time (1-dim)을 atom embedding에 concat → total = hidden_dim
- `normalized_time` 파라미터를 `graph_encoding()`, `forward()`에 추가

model.py 변경:
- `get_normalized_time()` 메서드 추가: `graph.t / t1` 반환
- `forward()`에서 `normalized_time` 생성 → `NeuralNet(graph, normalized_time=...)` 전달

- 검증: import gate + forward pass (use_time_embedding=False → 기존과 동일 출력)

**Step 4: Graph conditioning (touches model.py + sampling.py)**

대상: `src/diffusion/model.py`, `src/diffusion/sampling.py`

model.py 변경:
- `DiffusionModel.__init__()`: `use_graph_prob` 추가 (from `config.train.graph_condition_prob`, default 1.0)
- `noise_sampling()`: 확률 `1 - use_graph_prob`로 `graph.reset_to_dummy()` 호출 후 noisy graph 생성

sampling.py 변경:
- 모든 sampling 함수에서: `config.sampling.graph_condition` flag 확인, False이면 `graph.reset_to_dummy()` 호출

- 검증: import gate + `use_graph_prob=1.0`에서 기존 동작과 동일

**Step 5: New score functions (sampling.py)**

대상: `src/diffusion/sampling.py`

- `predict_vector(model, graph, dt, batch)` 추가
  - Vector field prediction: `dx = v(x, theta) * dt`
- `predict_score(model, graph, t, dt, batch, stochastic)` 추가
  - Diffusion score with VP → VE conversion
  - Langevin noise term
- 검증: import gate (함수 signature 확인)

**Step 6: New sampling methods (sampling.py, depends on Step 1, 4, 5)**

대상: `src/diffusion/sampling.py`

- `sample_batch_langevin(model, batch, config, ...)` 추가
  - Reverse-time SDE with annealed Langevin dynamics
  - Start from `start_time` (e.g., 0.03), step backward to 0
  - `predict_score` 사용, VP → VE converted coefficients
  - Optional exponential map via geodesic ODE solve
- `sample_batch_gradient_descent(model, batch, config, ...)` 추가
  - Fixed-point iteration using `predict_vector`
  - Steps forward from t=0 to T with fixed dt
- 검증: import gate + shape check (output 구조가 기존 sampling 함수와 일관)

**Step 7: Config files**

대상: `configs/` 디렉토리

신규 config 파일 생성:
- `configs/sampling.qm9.rdsm.langevin.yaml` — Langevin sampling
- `configs/sampling.qm9.rdsm.fixed_point.yaml` — Fixed-point sampling
- `configs/sampling.qm9.rdsm.gradient_descent.yaml` — Gradient descent
- `configs/training.qm9.edsm.cfg.yaml` — CFG training (EDSM)
- `configs/training.qm9.rdsm.cfg.yaml` — CFG training (RDSM)
- `configs/finetuning.qm9.rdsm.cfg.yaml` — CFG finetuning
- `configs/ablation_time_embed_*.yaml` — Time embedding ablation configs

검증: YAML syntax + key consistency (필수 key 존재 여부)

**Step 8: `scripts/sample.py` dispatch update**

대상: `scripts/sample.py`

- `langevin_exp`, `fixed_exp` score type dispatch 추가
- 새 sampling 함수로의 routing
- 검증: import gate + dispatch table 확인

### 2.2 Expected Changes

**신규 파일:**
- `src/diffusion/continuous_scheduler.py` — SigmoidDiffusionScheduler
- `configs/sampling.qm9.rdsm.langevin.yaml`
- `configs/sampling.qm9.rdsm.fixed_point.yaml`
- `configs/sampling.qm9.rdsm.gradient_descent.yaml`
- `configs/training.qm9.edsm.cfg.yaml`
- `configs/training.qm9.rdsm.cfg.yaml`
- `configs/finetuning.qm9.rdsm.cfg.yaml`
- `configs/ablation_time_embed_*.yaml` (복수)

**수정 파일:**
- `src/manifold/graph.py` — `reset_to_dummy()` 메서드 추가
- `src/diffusion/noise_scheduler.py` — `continuous_tsdiff` 등록
- `src/model/encoder.py` — time embedding 옵션 추가
- `src/diffusion/model.py` — `use_graph_prob`, `get_normalized_time()`, forward 수정
- `src/diffusion/sampling.py` — score functions + sampling methods 추가, graph conditioning
- `scripts/sample.py` — dispatch 추가

**파일 매핑 (old → new):**

```
Old (pre-refactoring)                      New (refactored)
─────────────────────                      ────────────────
utils/rxn_graph.py MolGraph           →    src/manifold/graph.py MolGraph
diffusion/diffusion_model.py              src/diffusion/model.py DiffusionModel
  BridgeDiffusion.noise_sampling()    →      DiffusionModel.noise_sampling()
  BridgeDiffusion.forward()           →      DiffusionModel.forward()
  BridgeDiffusion.sample_batch_*()    →    src/diffusion/sampling.py (standalone functions)
  BridgeDiffusion.predict_*()         →    src/diffusion/sampling.py (standalone functions)
diffusion/continuous_scheduler.py     →    src/diffusion/continuous_scheduler.py (new file)
model/geodiff_encoder.py             →    src/model/encoder.py GeoDiffEncoder
configs/*.yaml                        →    configs/*.yaml
```

### 2.3 Verification Plan

**각 Step 공통 (Import gate):**
```bash
python -c "from src.manifold.graph import MolGraph"
python -c "from src.diffusion.model import DiffusionModel"
python -c "from src.diffusion.sampling import sample_batch_langevin"
python -c "from src.model.encoder import GeoDiffEncoder"
```

**Feature-specific 검증:**

| Step | 검증 내용 | 방법 |
|------|-----------|------|
| 1 | `reset_to_dummy()` 동작 | reset 전후 shape 동일, 값 변경 확인 |
| 2 | Continuous scheduler 수치 안정성 | t ∈ [0, 1]에서 alpha/sigma NaN/Inf 없음 |
| 3 | Time embedding backward compat | `use_time_embedding=False` → 기존과 동일 output |
| 4 | Graph conditioning backward compat | `use_graph_prob=1.0` → 기존과 동일 동작 |
| 5 | Score function signatures | argument types, return shape 확인 |
| 6 | Sampling method output structure | 기존 sampling 함수와 동일한 return 구조 |
| 7 | Config validity | YAML parse + 필수 key 존재 |
| 8 | Dispatch coverage | 새 score_type이 routing에 포함 |

**성공 기준:**

1. 6개 feature 전부 구현 완료 (코드 존재 + import 성공)
2. Import gate 통과: 모든 수정 모듈이 clean하게 import됨
3. 기존 sampling 동작 유지: CFM sampling with published checkpoint가 동일 결과
   (RMSD median ~0.034 Å, ±0.002)
4. 새 sampling config가 valid하고 self-consistent
5. 리팩터링 패턴 준수: standalone functions in `sampling.py`, `model`을 first arg로 받음

### 2.4 Assumptions & Risks

**Assumptions**
- `origin/time_embedding_ablation`이 `origin/conditioning_exp`의 superset이므로,
  `time_embedding_ablation` 기준으로 porting하면 모든 코드가 포함됨
- 리팩터링 후 `DiffusionModel`의 attribute 이름(`NeuralNet`, `noise_schedule`,
  `geodesic_solver` 등)이 보존되어 있으므로, old code의 `self.xxx`를 `model.xxx`로
  기계적으로 변환할 수 있다
- `ATOM_ENCODER` (chem.py:21)가 현재 코드에 존재하며 dummy value 생성에 사용 가능
- Continuous scheduler는 self-contained (외부 의존성 없이 `nn.Module` 상속만)

**Risks**

| # | Risk | 확률 | 영향 | Mitigation |
|---|------|------|------|------------|
| 1 | God class 분리로 인해 `self` 참조 변환 시 누락 | 중간 | 높음 | Old code의 `self.xxx` 사용을 전수 조사, 각각 `model.xxx`로 매핑 |
| 2 | Geodesic solver 호출 패턴이 리팩터링 전후로 달라졌을 수 있음 | 낮음 | 중간 | `manifold/solver.py`의 현재 API를 먼저 확인 후 porting |
| 3 | `reset_to_dummy()`가 in-place mutation → sampling 시 원본 graph 오염 | 중간 | 높음 | Clone/deepcopy 여부 확인, 필요 시 copy 후 reset |
| 4 | Time embedding 추가 시 기존 checkpoint의 embedding dimension 불일치 | 낮음 | 낮음 | Default `use_time_embedding=False`이므로 기존 checkpoint에 영향 없음 |
| 5 | Config 간 key 불일치 (old config schema vs new config schema) | 중간 | 낮음 | 기존 config 파일을 base로 하여 새 key만 추가 |

**Mitigation**
- Step별 import gate로 regression 즉시 감지
- 기존 sampling (CFM, R-DM) 결과를 매 step 후 확인하여 backward compatibility 보장
- Old branch 코드를 직접 읽어가며 porting (추측 불가 — 반드시 원본 참조)

---

<!-- ↑ PRE-REPORT: 여기까지 구현 시작 전에 작성 -->
<!-- ↓ POST-REPORT: 구현 완료 후 아래를 추가 작성 -->
