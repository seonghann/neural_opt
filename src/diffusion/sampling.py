"""
Standalone sampling functions extracted from BridgeDiffusion.

All functions take a DiffusionModel instance as the first argument
instead of being methods on self.
"""

import torch
from torch_scatter import scatter_mean
from tqdm.auto import tqdm

from src.diffusion.model import center_pos, clip_norm
from src.model.geometry import get_distance


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _masking(num_nodes):
    """Create a flat boolean mask for padded batch tensors."""
    max_n = num_nodes.max()
    batch_size = len(num_nodes)
    mask = torch.arange(max_n, device=num_nodes.device).unsqueeze(0).expand(batch_size, -1)
    mask = mask < num_nodes.unsqueeze(1)
    return mask.reshape(-1)


# ---------------------------------------------------------------------------
# CFM sampling (flow-matching based)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_batch_simple(
    model,
    batch,
    config,
    stochastic=False,
    clip=None,
    num_cycles=1,
    dynamic_graph_list=None,
):
    """
    Sample molecular structures with CFM score function.

    Args:
        model: DiffusionModel instance
        batch: PyG batch from dataloader
        config: full OmegaConf config
        stochastic: whether to add noise during sampling
        clip: gradient clipping limit (None = no clipping)
        num_cycles: number of sampling cycles
        dynamic_graph_list: optional list to append DynamicGraph objects for saving

    Returns:
        List of per-molecule Data objects with .traj attribute
    """
    graph = model.graph_cls.from_batch(batch)
    if not getattr(config.sampling, 'graph_condition', True):
        graph.reset_to_dummy()

    pos = batch.pos[:, -1, :]
    pos = center_pos(pos, batch.batch)
    pos_init = pos.clone()

    t = torch.ones(batch.num_graphs, device=pos.device)
    t -= config.sampling.time_margin  # (G, )

    dt_base = torch.ones_like(t) / config.sampling.sde_steps
    dynamic_graph = model.dynamic_graph_cls.from_graph(graph, pos, pos_init, t)
    dynamic_graph.pos_traj.append(pos_init.to("cpu"))  # add initial position

    ## Set score function
    if config.sampling.score_type == "cfm":
        score_function = model.predict_cfm
    else:
        raise NotImplementedError(f"Unsupported score_type: {config.sampling.score_type}")

    print("[sample_batch_simple] start sampling")
    for cycle in range(num_cycles):
        print(f"Debug: cycle: {cycle + 1}/{num_cycles}")
        # Reset time for each cycle
        t = torch.ones(batch.num_graphs, device=pos.device) - config.sampling.time_margin  # (G, )
        dynamic_graph.update_graph(pos, t=t, append=False)
        dt = dt_base.clone()

        # Sampling loop (1 -> 0)
        while (t > 1e-6).any():
            print(f"t={t[0]}", end=", ")
            dt = dt.clip(max=t)

            dx = score_function(
                dynamic_graph,
                dt,
                stochastic=stochastic,
                debug=False,
                batch=batch,
            )[0]

            # Apply gradient clipping if specified
            if clip is not None:
                dx = clip_norm(dx, limit=clip)

            pos -= dx
            t -= dt
            pos = center_pos(pos, batch.batch)

            if config.debug.save_dynamic and not config.debug.save_dynamic_final_only:
                dynamic_graph.update_graph(pos, t=t, score=dx)
            else:
                dynamic_graph.update_graph(pos, t=t, append=False)

    print("\n[sample_batch_simple] sampling finished")

    if config.debug.save_dynamic and config.debug.save_dynamic_final_only:
        # Add final position to the trajectory
        dynamic_graph.update_graph(pos, t=t, score=dx)

    ## Save trajectory
    traj = torch.stack(dynamic_graph.pos_traj).transpose(0, 1).flip(dims=(1,))  # (N, T, 3)
    samples = []
    for i in range(batch.num_graphs):
        d = batch[i]
        traj_i = traj[(batch.batch == i).to("cpu")]
        d.traj = traj_i
        samples.append(d)

    ## Save dynamic_graph object
    if config.debug.save_dynamic and dynamic_graph_list is not None:
        dynamic_graph_list.append(dynamic_graph)

    return samples


# ---------------------------------------------------------------------------
# Score functions (standalone, used by Langevin / GradDescent samplers)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_vector(model, graph, dt, batch, **kwargs):
    """
    Vector field prediction: v(x, theta) ≈ x* - x.
    Displacement: dx = v(x, theta) * dt

    Args:
        model: DiffusionModel instance
        graph: DynamicGraph with current state
        dt: (G,) time step per graph
        batch: PyG batch

    Returns:
        dx: (N, 3), dq: (E,)
    """
    v_x, v_q = model.forward(graph)[:2]
    node2graph = batch.batch
    edge_index = graph.full_edge(upper_triangle=True)[0]
    edge2graph = node2graph.index_select(0, edge_index[0])
    dx = dt.index_select(0, node2graph).unsqueeze(-1) * v_x
    dq = dt.index_select(0, edge2graph) * v_q
    return dx, dq


@torch.no_grad()
def predict_score(model, graph, t, dt, batch, stochastic=True, **kwargs):
    """
    Diffusion score prediction with VP->VE SDE conversion.

    For VP case: f = d/dt(log alpha), g^2 = beta
    VE conversion: sigma_VE = sqrt((1 - alpha^2) / alpha^2), g_VE = sqrt(beta) / alpha

    Args:
        model: DiffusionModel instance
        graph: DynamicGraph with current state
        t: (G,) current time per graph
        dt: (G,) time step per graph
        batch: PyG batch
        stochastic: whether to add Langevin noise

    Returns:
        dx: (N, 3), dq: (E,)
    """
    v_x, v_q = model.forward(graph)[:2]
    node2graph = batch.batch
    edge_index = graph.full_edge(upper_triangle=True)[0]
    edge2graph = node2graph.index_select(0, edge_index[0])

    log_alpha = model.noise_schedule._log_alpha(t)
    log_sigma = model.noise_schedule._log_sigma(t)
    beta = model.noise_schedule.get_beta(t)

    # Coefficient for score term: beta * exp(-2 * log_sigma) * dt
    coeff1 = beta * (-2 * log_sigma).exp() * dt
    coeff1_node = coeff1.index_select(0, node2graph).unsqueeze(-1)
    coeff1_edge = coeff1.index_select(0, edge2graph)

    # Coefficient for noise term: g_VE * sqrt(dt)
    g_VE = beta.sqrt() * (-log_alpha).exp()
    coeff2 = g_VE * dt.sqrt()
    coeff2_edge = coeff2.index_select(0, edge2graph)

    num_nodes = batch.batch.bincount()
    if stochastic:
        Z = torch.randn_like(v_q)
        dw = model.geodesic_solver.batch_projection(
            Z, graph.pos, graph.atom_type, edge_index,
            node2graph, num_nodes, q_type=model.q_type, proj_type='manifold',
        )
        dq = v_q * coeff1_edge + dw * coeff2_edge
    else:
        dq = v_q * coeff1_edge * 0.5

    dx = model.geodesic_solver.batch_dq2dx(
        dq, graph.pos, graph.atom_type, edge_index,
        node2graph, num_nodes, q_type=model.q_type,
    ).reshape(-1, 3)

    return dx, dq


# ---------------------------------------------------------------------------
# Langevin dynamics sampling (reverse-time SDE)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_batch_langevin(model, batch, config, stochastic=True, dynamic_graph_list=None):
    """
    Langevin dynamics sampling (reverse-time SDE).

    Starts from start_time, steps backward to 0 using predict_score.
    Requires continuous noise scheduler (e.g., SigmoidDiffusionScheduler).

    Args:
        model: DiffusionModel instance
        batch: PyG batch from dataloader
        config: full OmegaConf config
        stochastic: whether to add Langevin noise
        dynamic_graph_list: optional list to append DynamicGraph objects for saving

    Returns:
        List of per-molecule Data objects with .traj attribute
    """
    graph = model.graph_cls.from_batch(batch)
    if not getattr(config.sampling, 'graph_condition', True):
        graph.reset_to_dummy()

    noisy_pos_idx = getattr(config.sampling, 'noisy_pos_idx', -1)
    pos = batch.pos[:, noisy_pos_idx, :]
    pos = center_pos(pos, batch.batch)
    pos_init = pos.clone()

    langevin_cfg = config.sampling.langevin_exp
    start_time = getattr(langevin_cfg, 'start_time', 1.0)
    t = torch.ones(batch.num_graphs, device=pos.device) * start_time
    dt = torch.ones_like(t) * start_time / langevin_cfg.nfe

    dynamic_graph = model.dynamic_graph_cls.from_graph(graph, pos, pos_init, t)
    dynamic_graph.pos_traj.append(pos.to("cpu"))

    # Setup exponential map solver if configured
    use_exp_map = getattr(langevin_cfg, 'use_exp_map', False)
    if use_exp_map:
        ode_cfg = config.manifold.ode_solver
        solve_kwargs = dict(
            q_type=model.q_type,
            num_iter=ode_cfg.iter, max_iter=ode_cfg.max_iter,
            ref_dt=ode_cfg._ref_dt, min_dt=ode_cfg._min_dt, max_dt=ode_cfg._max_dt,
            err_thresh=ode_cfg.vpae_thresh, verbose=0, method="RK4",
        )
        full_edge = dynamic_graph.full_edge(upper_triangle=True)[0]
        atom_type = dynamic_graph.atom_type
        node2graph = batch.batch
        num_nodes = batch.batch.bincount()

    print("[sample_batch_langevin] start sampling")
    for step in tqdm(range(langevin_cfg.nfe), desc="Langevin"):
        if not (t > 0).any():
            break
        dt = dt.clip(max=t)

        dx, dq = predict_score(
            model, dynamic_graph, t, dt,
            batch=batch, stochastic=stochastic,
        )

        if use_exp_map:
            init, last, iter_count, index_tensor, stats = model.geodesic_solver.batch_geodesic_ode_solve(
                pos, dq, full_edge, atom_type, node2graph, num_nodes, **solve_kwargs
            )
            pos = last["x"]
            pad_mask = _masking(num_nodes)
            pos = pos.reshape(-1, 3)[pad_mask]
        else:
            pos += dx

        t -= dt
        pos = center_pos(pos, batch.batch)

        save_vf = getattr(langevin_cfg, 'save_vector_field', False)
        if config.debug.save_dynamic and not config.debug.save_dynamic_final_only:
            if save_vf:
                dynamic_graph.update_graph(pos, t=t, score=dx)
            else:
                dynamic_graph.update_graph(pos, t=t)
        else:
            dynamic_graph.update_graph(pos, t=t, append=False)

    print("[sample_batch_langevin] sampling finished")

    if config.debug.save_dynamic and config.debug.save_dynamic_final_only:
        dynamic_graph.update_graph(pos, t=t, score=dx)

    # Build output samples
    traj = torch.stack(dynamic_graph.pos_traj).transpose(0, 1).flip(dims=(1,))
    samples = []
    for i in range(batch.num_graphs):
        d = batch[i]
        d.traj = traj[(batch.batch == i).to("cpu")]
        samples.append(d)

    if config.debug.save_dynamic and dynamic_graph_list is not None:
        dynamic_graph_list.append(dynamic_graph)

    return samples


# ---------------------------------------------------------------------------
# Diffusion sampling (DDPM / Langevin Dynamics)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_batch_diffusion(
    model,
    batch,
    config,
    stochastic=True,
    step_lr=0.0000010,
    clip=1000,
    clip_pos=None,
    sampling_type="ld",
    debug=False,
    start_from_time=None,
    dynamic_graph_list=None,
):
    """
    Diffusion-based sampling (DDPM or Langevin dynamics).

    Args:
        model: DiffusionModel instance
        batch: PyG batch from dataloader
        config: full OmegaConf config
        stochastic: whether to add noise during sampling
        step_lr: step size for Langevin dynamics
        clip: gradient clipping limit
        clip_pos: position clipping limit (None = no clipping)
        sampling_type: "ddpm" or "ld"
        debug: enable debug printing
        start_from_time: override start time (default: num_timesteps)
        dynamic_graph_list: optional list to append DynamicGraph objects for saving

    Returns:
        List of per-molecule Data objects with .traj attribute
    """
    alphas = model.noise_schedule.alphas
    sigmas = (1.0 - alphas).sqrt() / alphas.sqrt()

    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)
        return a

    graph = model.graph_cls.from_batch(batch)
    if not getattr(config.sampling, 'graph_condition', True):
        graph.reset_to_dummy()
    pos_init = batch.pos[:, -1, :]
    pos = torch.randn_like(pos_init) * sigmas[-1]

    ##################################################
    TYPE = 1
    if TYPE == 1:
        pos = batch.pos[:, -1, :].clone()
        print(f"Debug: pos_init is set to pos_T")
    elif TYPE == 2:
        pos = batch.pos[:, 0, :].clone()
        pos += torch.randn_like(pos) * 1e-1
        print(f"Debug: pos_init is set to pos_0 + 1e-1 eps")
    elif TYPE == 3:
        pos = batch.pos[:, -1, :].clone()
        pos += torch.randn_like(pos) * 1e-1
        print(f"Debug: pos_init is set to pos_T + 1e-1 eps")
    elif TYPE == 4:
        pos = batch.pos[:, 0, :].clone()
        print(f"Debug: pos_init is set to pos_0")
    ##################################################

    num_timesteps = model.noise_schedule.num_diffusion_timesteps
    t = torch.full(
        size=(batch.num_graphs,),
        fill_value=num_timesteps,
        dtype=torch.long,
        device=pos.device,
    )
    dynamic_graph = model.dynamic_graph_cls.from_graph(graph, pos, pos_init, t)
    dynamic_graph.pos_traj.append(pos.to("cpu"))

    if start_from_time is None:
        start_from_time = num_timesteps
    else:
        assert start_from_time <= num_timesteps

    edge_index = dynamic_graph.full_edge(upper_triangle=True)[0]
    node2graph = batch.batch
    edge2graph = node2graph.index_select(0, edge_index[0])
    num_nodes = batch.batch.bincount()

    #########################################################
    if debug:
        pos_ref = batch.pos[:, 0, :]
        rmsd_to_ref = torch.sqrt(scatter_mean(torch.sum((pos - pos_ref)** 2, dim=-1), node2graph))
        dm_ref = get_distance(pos_ref, edge_index)
        dm_pos = get_distance(pos, edge_index)
        dmae_to_ref = scatter_mean(abs(dm_ref - dm_pos), edge2graph)
        print(f"Initial rmsd to ref=\n{rmsd_to_ref}", flush=True)
        print(f"Initial dmae to ref=\n{dmae_to_ref}", flush=True)
        print(f"Initial rmsd to ref (mean)= {rmsd_to_ref.mean()}", flush=True)
        print(f"Initial dmae to ref (mean)= {dmae_to_ref.mean()}", flush=True)
    #########################################################

    seq = range(0, num_timesteps)
    seq_next = [-1] + list(seq[:-1])

    for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc="sample"):
        t = torch.full(
            size=(batch.num_graphs,),
            fill_value=i,
            dtype=torch.long,
            device=pos.device,
        )
        if (t > start_from_time).any():
            continue
        node_eq = model.forward(dynamic_graph.to("cuda"))[0]

        loss_weight_type = config.train.loss_weight
        if loss_weight_type is not None:
            assert loss_weight_type == "diffusion"
            node_eq /= sigmas[i]

        eps_pos = clip_norm(node_eq, limit=clip)

        #################################################
        if debug:
            ## Calculate reference score
            pos_0 = batch.pos[:, 0, :]
            q_gt = model.geodesic_solver.compute_d_or_q(pos_0, dynamic_graph.atom_type, edge_index, q_type=model.q_type)
            q_t = model.geodesic_solver.compute_d_or_q(pos, dynamic_graph.atom_type, edge_index, q_type=model.q_type)
            d_target = (q_gt - q_t) / sigmas[i]
            pos_target = (pos_0 - pos) / sigmas[i]
            pos_target, d_target = model.transform(
                pos_target, d_target, pos, dynamic_graph.atom_type, edge_index, node2graph, num_nodes, model.q_type,
            )
            rmsd = torch.sqrt(scatter_mean(torch.sum((pos_target - node_eq) ** 2, dim=-1), node2graph))
            denom = torch.sqrt(scatter_mean(torch.sum(pos_target ** 2, dim=-1), node2graph))
            pred_size = torch.sqrt(scatter_mean(torch.sum(node_eq ** 2, dim=-1), node2graph))
            perr = rmsd / denom
            print(f"Debug: rmsd={rmsd.detach()}", flush=True)
            print(f"Debug: pred_size={pred_size.detach()}", flush=True)
            print(f"Debug: denom={denom.detach()}", flush=True)
            print(f"Debug: perr={perr.detach()}", flush=True)
        #################################################

        # Update
        if stochastic:
            noise = torch.randn_like(pos)
        else:
            noise = 0.

        if sampling_type == "ddpm":
            b = model.noise_schedule.betas.to(pos.device)
            t = t[0]
            next_t = (torch.ones(1) * j).to(pos.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            atm1 = at_next
            beta_t = 1 - at / atm1
            e = -eps_pos
            pos_C = at.sqrt() * pos
            pos0_from_e = (1.0 / at).sqrt() * pos_C - (
                1.0 / at - 1
            ).sqrt() * e
            mean_eps = (
                (atm1.sqrt() * beta_t) * pos0_from_e
                + ((1 - beta_t).sqrt() * (1 - atm1)) * pos_C
            ) / (1.0 - at)
            mean = mean_eps
            mask = 1 - (t == 0).float()
            logvar = beta_t.log()

            pos_next = (mean + mask * torch.exp(0.5 * logvar) * noise) / atm1.sqrt()

        elif sampling_type == "ld":
            step_size = step_lr * (sigmas[i] / 0.01) ** 2
            pos_next = (
                pos
                + step_size * eps_pos / sigmas[i]
                + noise * torch.sqrt(step_size * 2)
            )

        pos = pos_next

        #########################################################
        if debug:
            pos_ref = batch.pos[:, 0, :]
            rmsd_to_ref = torch.sqrt(scatter_mean(torch.sum((pos - pos_ref)** 2, dim=-1), node2graph))
            dm_ref = get_distance(pos_ref, edge_index)
            dm_pos = get_distance(pos, edge_index)
            dmae_to_ref = scatter_mean(abs(dm_ref - dm_pos), edge2graph)
            print(f"t={i}: rmsd to ref=\n{rmsd_to_ref}")
            print(f"t={i}: dmae to ref=\n{dmae_to_ref}")
            print(f"t={i}: rmsd to ref (mean)= {rmsd_to_ref.mean()}")
            print(f"t={i}: dmae to ref (mean)= {dmae_to_ref.mean()}")
        #########################################################

        if config.debug.save_dynamic and not config.debug.save_dynamic_final_only:
            dynamic_graph.update_graph(pos, t=t, score=node_eq)
        else:
            dynamic_graph.update_graph(pos, t=t, append=False)

        if torch.isnan(pos).any():
            print("NaN detected. Please restart.")
            raise FloatingPointError()
        pos = center_pos(pos, batch.batch)
        if clip_pos is not None:
            pos = torch.clamp(pos, min=-clip_pos, max=clip_pos)

    if config.debug.save_dynamic and config.debug.save_dynamic_final_only:
        # Add final position to the trajectory
        dynamic_graph.update_graph(pos, t=t, score=node_eq)

    ## Save trajectory
    traj = torch.stack(dynamic_graph.pos_traj).transpose(0, 1).flip(dims=(1,))  # (N, T, 3)
    samples = []
    for i in range(batch.num_graphs):
        d = batch[i]
        traj_i = traj[(batch.batch == i).to("cpu")]
        d.traj = traj_i
        samples.append(d)

    ## Save dynamic_graph object
    if config.debug.save_dynamic and dynamic_graph_list is not None:
        dynamic_graph_list.append(dynamic_graph)

    return samples


# ---------------------------------------------------------------------------
# Gradient descent / fixed-point sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_batch_gradient_descent(model, batch, config, dynamic_graph_list=None):
    """
    Fixed-point / gradient descent sampling.

    Steps forward from t=0 to T using predict_vector.

    Args:
        model: DiffusionModel instance
        batch: PyG batch from dataloader
        config: full OmegaConf config
        dynamic_graph_list: optional list to append DynamicGraph objects for saving

    Returns:
        List of per-molecule Data objects with .traj attribute
    """
    graph = model.graph_cls.from_batch(batch)
    if not getattr(config.sampling, 'graph_condition', True):
        graph.reset_to_dummy()

    noisy_pos_idx = getattr(config.sampling, 'noisy_pos_idx', -1)
    pos = batch.pos[:, noisy_pos_idx, :]
    pos = center_pos(pos, batch.batch)
    pos_init = pos.clone()

    fixed_cfg = config.sampling.fixed_exp
    t = torch.zeros(batch.num_graphs, device=pos.device)
    T = fixed_cfg.total_time
    dt = torch.ones_like(t) * fixed_cfg.dt

    dynamic_graph = model.dynamic_graph_cls.from_graph(graph, pos, pos_init, t)
    dynamic_graph.pos_traj.append(pos.to("cpu"))

    # Setup exponential map solver if configured
    use_exp_map = getattr(fixed_cfg, 'use_exp_map', False)
    if use_exp_map:
        ode_cfg = config.manifold.ode_solver
        solve_kwargs = dict(
            q_type=model.q_type,
            num_iter=ode_cfg.iter, max_iter=ode_cfg.max_iter,
            ref_dt=ode_cfg._ref_dt, min_dt=ode_cfg._min_dt, max_dt=ode_cfg._max_dt,
            err_thresh=ode_cfg.vpae_thresh, verbose=0, method="RK4",
        )
        full_edge = dynamic_graph.full_edge(upper_triangle=True)[0]
        atom_type = dynamic_graph.atom_type
        node2graph = batch.batch
        num_nodes = batch.batch.bincount()

    num_steps = int(T / fixed_cfg.dt)
    print("[sample_batch_gradient_descent] start sampling")
    for step in tqdm(range(num_steps), desc="GradDescent"):
        if not (t < T).any():
            break
        dt = dt.clip(max=T - t)

        dx, dq = predict_vector(
            model, dynamic_graph, dt, batch=batch,
        )

        if use_exp_map:
            init, last, iter_count, index_tensor, stats = model.geodesic_solver.batch_geodesic_ode_solve(
                pos, dq, full_edge, atom_type, node2graph, num_nodes, **solve_kwargs
            )
            pos = last["x"]
            pad_mask = _masking(num_nodes)
            pos = pos.reshape(-1, 3)[pad_mask]
        else:
            pos += dx

        t += dt
        pos = center_pos(pos, batch.batch)

        save_vf = getattr(fixed_cfg, 'save_vector_field', False)
        if config.debug.save_dynamic and not config.debug.save_dynamic_final_only:
            if save_vf:
                dynamic_graph.update_graph(pos, t=t, score=dx)
            else:
                dynamic_graph.update_graph(pos, t=t)
        else:
            dynamic_graph.update_graph(pos, t=t, append=False)

    print("[sample_batch_gradient_descent] sampling finished")

    if config.debug.save_dynamic and config.debug.save_dynamic_final_only:
        dynamic_graph.update_graph(pos, t=t, score=dx)

    # Build output samples
    traj = torch.stack(dynamic_graph.pos_traj).transpose(0, 1).flip(dims=(1,))
    samples = []
    for i in range(batch.num_graphs):
        d = batch[i]
        d.traj = traj[(batch.batch == i).to("cpu")]
        samples.append(d)

    if config.debug.save_dynamic and dynamic_graph_list is not None:
        dynamic_graph_list.append(dynamic_graph)

    return samples
