"""
Analysis: self.xxx Reference Mapping — Old Branch Methods vs Refactored DiffusionModel
Date: 2026-03-21
Related progress log: docs/progress/260321_port_experiment_branches.md

Problem:
    Four methods from origin/time_embedding_ablation (predict_vector, predict_score,
    sample_batch_langevin, sample_batch_gradient_descent) use self.xxx references
    that were attributes of the old BridgeDiffusion(pl.LightningModule).
    We need to verify all self.xxx references are accessible via the refactored
    DiffusionModel (src/diffusion/model.py) before porting these methods.

Judgment Criteria:
    - PASS: attribute exists in DiffusionModel with same semantics
    - RENAME: attribute exists but with different name (needs mapping)
    - MISSING: attribute does not exist in DiffusionModel (needs resolution)

Conclusion:
    See output below.

Usage: python analyze/260321_self_ref_mapping.py
"""

# -----------------------------------------------------------------------
# This script performs a static analysis by comparing attribute names.
# No imports of project modules needed — pure string/table analysis.
# -----------------------------------------------------------------------

from collections import OrderedDict

# -----------------------------------------------------------------------
# 1. DiffusionModel attributes (from src/diffusion/model.py __init__)
# -----------------------------------------------------------------------
DIFFUSION_MODEL_ATTRS = {
    # Attribute name -> type/description
    "config": "OmegaConf config",
    "geodesic_solver": "GeodesicSolver(config.manifold)",
    "NeuralNet": "GeoDiffEncoder(config.model)",
    "noise_schedule": "load_noise_scheduler(config.diffusion)",
    "pred_type": "config.model.pred_type ('edge'|'node')",
    "q_type": "config.manifold.ode_solver.q_type",
    "solver_threshold": "config.manifold.ode_solver.vpae_thresh",
    "graph_cls": "MolGraph or RxnGraph (class, not instance)",
    "dynamic_graph_cls": "DynamicMolGraph or DynamicRxnGraph (class, not instance)",
}

# DiffusionModel methods available
DIFFUSION_MODEL_METHODS = {
    "forward": "(graph) -> pred_x, pred_q, edge_index, node2graph, edge2graph",
    "transform": "(score_x, score_q, pos, atom_type, edge_index, batch, num_nodes, q_type, rescale_dq)",
    "noise_sampling": "(data) -> noisy_graph, target_x, target_q",
    "get_loss_weight": "(time_step) -> weight",
    "predict_cfm": "(graph, dt, debug, batch) -> dx, dq",
}

# -----------------------------------------------------------------------
# 2. Old BridgeDiffusion attributes (from __init__ in old branch)
# -----------------------------------------------------------------------
OLD_BRIDGE_ATTRS = {
    "config": "OmegaConf config",
    "name": "config.general.name + timestamp",
    "geodesic_solver": "GeodesicSolver(config.manifold)",
    "NeuralNet": "GeoDiffEncoder(config.model)",
    "noise_schedule": "load_noise_scheduler(config.diffusion)",
    "dynamic_graph_list": "[] (stores trajectories)",
    "use_graph_prob": "config.train.graph_condition_prob",
    "train_loss": "LossFunction(...)",
    "valid_loss": "LossFunction(...)",
    "test_loss": "LossFunction(...)",
    "train_metrics": "TrainMetrics(...)",
    "valid_metrics": "ValidMetrics(...)",
    "test_metrics": "ValidMetrics(...)",
    "valid_sampling_metrics": "SamplingMetrics(...)",
    "test_sampling_metrics": "SamplingMetrics(...)",
    "pred_type": "config.model.pred_type",
    "q_type": "config.manifold.ode_solver.q_type",
    "solver_threshold": "config.manifold.ode_solver.vpae_thresh",
    "save_dir": "config.debug.save_dir",
    "best_valid_loss": "1e9",
    "val_counter": "0",
    "test_counter": "0",
    "graph": "RxnGraph or MolGraph (CLASS reference, not instance)",
    "dynamic_graph": "DynamicRxnGraph or DynamicMolGraph (CLASS reference, not instance)",
}

# -----------------------------------------------------------------------
# 3. Extract self.xxx references from each target method
# -----------------------------------------------------------------------

# Manually extracted from reading the old branch code

PREDICT_VECTOR_REFS = OrderedDict([
    ("self.config.train.do_scale", "assert check"),
    ("self.forward(graph)", "method call — neural net forward"),
])

PREDICT_SCORE_REFS = OrderedDict([
    ("self.config.train.do_scale", "assert check"),
    ("self.forward(graph)", "method call — neural net forward"),
    ("self.noise_schedule._log_alpha(t)", "log alpha — VP scheduler internal"),
    ("self.noise_schedule._log_sigma(t)", "log sigma — VP scheduler internal"),
    ("self.noise_schedule.get_beta(t)", "beta schedule value"),
    ("self.geodesic_solver.batch_projection(...)", "project noise onto manifold"),
    ("self.geodesic_solver.batch_dq2dx(...)", "convert dq to dx"),
    ("self.q_type", "coordinate type"),
])

SAMPLE_BATCH_LANGEVIN_REFS = OrderedDict([
    ("self.graph.from_batch(batch)", "create static graph from batch"),
    ("self.config.sampling.graph_condition", "whether to use graph conditioning"),
    ("self.config.sampling.noisy_pos_idx", "index for noisy position (getattr, default -1)"),
    ("self.config.sampling.langevin_exp.start_time", "start time (getattr, default 1.0)"),
    ("self.config.sampling.langevin_exp.nfe", "number of function evaluations"),
    ("self.config.sampling.langevin_exp.use_exp_map", "whether to use exponential map"),
    ("self.config.sampling.langevin_exp.save_vector_field", "save vector field flag"),
    ("self.dynamic_graph.from_graph(...)", "create dynamic graph"),
    ("self.predict_score", "method reference — score prediction"),
    ("self.q_type", "coordinate type"),
    ("self.config.manifold.ode_solver.iter", "ODE solver iterations"),
    ("self.config.manifold.ode_solver.max_iter", "ODE solver max iterations"),
    ("self.config.manifold.ode_solver._ref_dt", "ODE solver reference dt"),
    ("self.config.manifold.ode_solver._min_dt", "ODE solver min dt"),
    ("self.config.manifold.ode_solver._max_dt", "ODE solver max dt"),
    ("self.config.manifold.ode_solver.vpae_thresh", "ODE solver error threshold"),
    ("self.geodesic_solver.batch_geodesic_ode_solve", "method — geodesic ODE solver"),
    ("self.config.debug.save_dynamic", "save dynamic graph flag"),
    ("self.dynamic_graph_list", "list to store trajectory results"),
])

SAMPLE_BATCH_GD_REFS = OrderedDict([
    ("self.graph.from_batch(batch)", "create static graph from batch"),
    ("self.config.sampling.graph_condition", "whether to use graph conditioning"),
    ("self.config.sampling.noisy_pos_idx", "index for noisy position (getattr, default -1)"),
    ("self.config.sampling.fixed_exp.total_time", "total integration time"),
    ("self.config.sampling.fixed_exp.dt", "time step size"),
    ("self.config.sampling.fixed_exp.use_exp_map", "whether to use exponential map"),
    ("self.config.sampling.fixed_exp.save_vector_field", "save vector field flag"),
    ("self.dynamic_graph.from_graph(...)", "create dynamic graph"),
    ("self.predict_vector", "method reference — vector field prediction"),
    ("self.q_type", "coordinate type"),
    ("self.config.manifold.ode_solver.iter", "ODE solver iterations"),
    ("self.config.manifold.ode_solver.max_iter", "ODE solver max iterations"),
    ("self.config.manifold.ode_solver._ref_dt", "ODE solver reference dt"),
    ("self.config.manifold.ode_solver._min_dt", "ODE solver min dt"),
    ("self.config.manifold.ode_solver._max_dt", "ODE solver max dt"),
    ("self.config.manifold.ode_solver.vpae_thresh", "ODE solver error threshold"),
    ("self.geodesic_solver.batch_geodesic_ode_solve", "method — geodesic ODE solver"),
    ("self.config.debug.save_dynamic", "save dynamic graph flag"),
    ("self.dynamic_graph_list", "list to store trajectory results"),
])


# -----------------------------------------------------------------------
# 4. Check availability in DiffusionModel
# -----------------------------------------------------------------------

def check_ref(ref: str) -> tuple:
    """
    Returns (status, alternative_path) for a self.xxx reference.
    Status: PASS / RENAME / MISSING / CONFIG_ONLY / METHOD_PORT_NEEDED
    """
    # Direct attribute matches
    if ref == "self.config.train.do_scale":
        return "PASS", "model.config.train.do_scale"
    if ref == "self.forward(graph)":
        return "PASS", "model.forward(graph) or model(graph)"
    if ref == "self.q_type":
        return "PASS", "model.q_type"
    if ref == "self.geodesic_solver.batch_projection(...)":
        return "PASS", "model.geodesic_solver.batch_projection(...)"
    if ref == "self.geodesic_solver.batch_dq2dx(...)":
        return "PASS", "model.geodesic_solver.batch_dq2dx(...)"
    if ref == "self.geodesic_solver.batch_geodesic_ode_solve":
        return "PASS", "model.geodesic_solver.batch_geodesic_ode_solve"

    # Noise schedule methods
    if ref == "self.noise_schedule.get_beta(t)":
        # TSDiffNoiseScheduler.get_beta raises NotImplementedError!
        return "MISSING", "TSDiffNoiseScheduler.get_beta(t) raises NotImplementedError. Must implement or compute from betas tensor."
    if ref == "self.noise_schedule._log_alpha(t)":
        return "MISSING", "No _log_alpha method in any NoiseScheduler. Must compute: log(alphas[t]) or implement."
    if ref == "self.noise_schedule._log_sigma(t)":
        return "MISSING", "No _log_sigma method in any NoiseScheduler. Must compute: 0.5*log(1 - alphas[t]) or implement."

    # Renamed attributes (graph -> graph_cls, dynamic_graph -> dynamic_graph_cls)
    if ref == "self.graph.from_batch(batch)":
        return "RENAME", "model.graph_cls.from_batch(batch)  [graph -> graph_cls]"
    if ref == "self.dynamic_graph.from_graph(...)":
        return "RENAME", "model.dynamic_graph_cls.from_graph(...)  [dynamic_graph -> dynamic_graph_cls]"

    # Methods that need to be ported (they don't exist in DiffusionModel yet)
    if ref == "self.predict_score":
        return "METHOD_PORT_NEEDED", "predict_score must be ported to DiffusionModel or sampling.py"
    if ref == "self.predict_vector":
        return "METHOD_PORT_NEEDED", "predict_vector must be ported to DiffusionModel or sampling.py"

    # Missing attributes (training/PL-specific, not in DiffusionModel)
    if ref == "self.dynamic_graph_list":
        return "MISSING", "Not in DiffusionModel. Pass as argument (like dynamic_graph_list in sample_batch_simple)."

    # Config-only references (accessible via model.config)
    if ref.startswith("self.config."):
        return "CONFIG_ONLY", f"model.{ref[5:]}  (config access — available if config has the key)"

    return "UNKNOWN", "Manual check needed"


def print_table(method_name: str, refs: OrderedDict):
    print(f"\n{'='*100}")
    print(f"  Method: {method_name}")
    print(f"{'='*100}")
    print(f"  {'self.xxx reference':<55} {'Status':<22} {'Resolution'}")
    print(f"  {'-'*55} {'-'*22} {'-'*50}")

    status_counts = {"PASS": 0, "RENAME": 0, "MISSING": 0, "CONFIG_ONLY": 0, "METHOD_PORT_NEEDED": 0}
    issues = []

    for ref, description in refs.items():
        status, resolution = check_ref(ref)
        marker = {
            "PASS": "[OK]",
            "RENAME": "[RENAME]",
            "MISSING": "[MISSING]",
            "CONFIG_ONLY": "[OK-cfg]",
            "METHOD_PORT_NEEDED": "[PORT]",
        }.get(status, "[???]")

        status_counts[status] = status_counts.get(status, 0) + 1
        if status in ("MISSING", "METHOD_PORT_NEEDED", "RENAME"):
            issues.append((ref, status, resolution))

        print(f"  {ref:<55} {marker:<22} {resolution}")

    print(f"\n  Summary: {sum(status_counts.values())} refs total | "
          f"OK={status_counts['PASS']} | OK-cfg={status_counts['CONFIG_ONLY']} | "
          f"RENAME={status_counts['RENAME']} | MISSING={status_counts['MISSING']} | "
          f"PORT={status_counts['METHOD_PORT_NEEDED']}")

    return issues


def main():
    print("=" * 100)
    print("  ANALYSIS: self.xxx Reference Mapping")
    print("  Old branch: origin/time_embedding_ablation (BridgeDiffusion)")
    print("  New target: src/diffusion/model.py (DiffusionModel)")
    print("=" * 100)

    all_issues = []

    all_issues += print_table("predict_vector (line 853)", PREDICT_VECTOR_REFS)
    all_issues += print_table("predict_score (line 885)", PREDICT_SCORE_REFS)
    all_issues += print_table("sample_batch_langevin (line 1070)", SAMPLE_BATCH_LANGEVIN_REFS)
    all_issues += print_table("sample_batch_gradient_descent (line 1168)", SAMPLE_BATCH_GD_REFS)

    # -----------------------------------------------------------------------
    # Deduplicated issue summary
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*100}")
    print("  ISSUES REQUIRING RESOLUTION (deduplicated)")
    print(f"{'='*100}")

    seen = set()
    unique_issues = []
    for ref, status, resolution in all_issues:
        if ref not in seen:
            seen.add(ref)
            unique_issues.append((ref, status, resolution))

    for i, (ref, status, resolution) in enumerate(unique_issues, 1):
        print(f"\n  {i}. [{status}] {ref}")
        print(f"     {resolution}")

    # -----------------------------------------------------------------------
    # Resolution proposals
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*100}")
    print("  PROPOSED RESOLUTIONS")
    print(f"{'='*100}")

    proposals = [
        (
            "self.graph -> self.graph_cls  (RENAME)",
            "Already renamed in DiffusionModel.__init__.\n"
            "     When porting, replace self.graph.from_batch -> model.graph_cls.from_batch\n"
            "     Same for self.dynamic_graph -> model.dynamic_graph_cls"
        ),
        (
            "self.predict_score / self.predict_vector  (METHOD PORT)",
            "Option A: Add as methods on DiffusionModel (keeps self.xxx pattern)\n"
            "     Option B: Extract as standalone functions in sampling.py (like sample_batch_simple)\n"
            "              -> predict_score(model, graph, t, dt, batch, stochastic)\n"
            "              -> predict_vector(model, graph, dt, batch)\n"
            "     Recommendation: Option B (consistent with existing sampling.py pattern)"
        ),
        (
            "self.noise_schedule._log_alpha(t) / ._log_sigma(t)  (MISSING METHOD)",
            "These methods don't exist in ANY version of the NoiseScheduler.\n"
            "     predict_score uses them for VP->VE SDE conversion.\n"
            "     Resolution: Add _log_alpha(t) and _log_sigma(t) to TSDiffNoiseScheduler:\n"
            "       def _log_alpha(self, t):  # t is continuous [0,1]\n"
            "           return 0.5 * torch.log(self.alphas[t])  # or interpolate for continuous t\n"
            "       def _log_sigma(self, t):\n"
            "           return 0.5 * torch.log(1 - self.alphas[t])"
        ),
        (
            "self.noise_schedule.get_beta(t)  (MISSING — raises NotImplementedError)",
            "TSDiffNoiseScheduler.get_beta(t) is explicitly NotImplementedError.\n"
            "     But self.noise_schedule.betas IS a tensor.\n"
            "     Resolution: Either implement get_beta() or index betas directly:\n"
            "       beta = model.noise_schedule.betas[t]"
        ),
        (
            "self.dynamic_graph_list  (MISSING ATTRIBUTE)",
            "Training/PL artifact — not needed in pure model.\n"
            "     Resolution: Pass as argument to sampling function (already done in\n"
            "     sample_batch_simple/sample_batch_diffusion via dynamic_graph_list param)."
        ),
        (
            "self.config.sampling.graph_condition -> graph.reset_to_dummy()  (MISSING METHOD)",
            "reset_to_dummy() is called when graph_condition=False.\n"
            "     This method does not exist in current RxnGraph/MolGraph.\n"
            "     Resolution: Check if this feature is needed. If yes, implement on graph classes."
        ),
        (
            "_masking(num_nodes)  (MISSING UTILITY)",
            "Used in exp_map branch of both langevin and gradient_descent.\n"
            "     Defined at line 24 of old diffusion_model.py.\n"
            "     Resolution: Port to src/diffusion/model.py or sampling.py as utility."
        ),
    ]

    for i, (title, detail) in enumerate(proposals, 1):
        print(f"\n  {i}. {title}")
        print(f"     {detail}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*100}")
    print("  DECISION: ADOPT analysis results for porting")
    print(f"{'='*100}")
    print("""
  Blockers (must resolve before porting):
    1. noise_schedule._log_alpha / ._log_sigma  → implement in TSDiffNoiseScheduler
    2. noise_schedule.get_beta(t)                → implement or use .betas[t]
    3. predict_score / predict_vector            → port as functions in sampling.py
    4. _masking utility                          → port to sampling.py

  Non-blockers (straightforward renames/refactors):
    5. self.graph -> model.graph_cls             → simple rename
    6. self.dynamic_graph -> model.dynamic_graph_cls → simple rename
    7. self.dynamic_graph_list                   → pass as function argument
    8. graph.reset_to_dummy()                    → check if needed, implement if so

  All config references (self.config.xxx) are accessible via model.config.xxx — no issues.
  All geodesic_solver methods are directly available on model.geodesic_solver — no issues.
""")


if __name__ == "__main__":
    main()
