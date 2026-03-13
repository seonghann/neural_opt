"""
DiffusionModel: Pure nn.Module extracted from BridgeDiffusion.

Contains core model logic (forward pass, noise sampling, score transformation)
without any PyTorch Lightning or training loop dependencies.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_sum

from src.manifold.graph import RxnGraph, DynamicRxnGraph, MolGraph, DynamicMolGraph
from src.manifold.solver import GeodesicSolver
from src.diffusion.noise_scheduler import load_noise_scheduler
from src.model.encoder import GeoDiffEncoder
from src.model.geometry import get_distance, eq_transform


# ---------------------------------------------------------------------------
# Utility functions (used by both model and sampling)
# ---------------------------------------------------------------------------

def center_pos(pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Center positions per graph in a batch."""
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center


def clip_norm(vec: torch.Tensor, limit: float, p: int = 2) -> torch.Tensor:
    """Clip vector norms to a maximum limit."""
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom


# ---------------------------------------------------------------------------
# DiffusionModel
# ---------------------------------------------------------------------------

class DiffusionModel(nn.Module):
    """
    Core diffusion model as a plain nn.Module.

    Encapsulates:
      - GeoDiffEncoder (self.NeuralNet)
      - GeodesicSolver
      - NoiseScheduler
      - Graph / DynamicGraph class dispatch
      - forward(), transform(), noise_sampling(), predict_cfm()
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.geodesic_solver = GeodesicSolver(config.manifold)

        if config.model.name == "geodiff":
            self.NeuralNet = GeoDiffEncoder(config.model)
        else:
            raise NotImplementedError(f"Model not supported: {config.model.name}")

        self.noise_schedule = load_noise_scheduler(config.diffusion)

        self.pred_type = config.model.pred_type
        self.q_type = config.manifold.ode_solver.q_type

        assert self.pred_type in ["edge", "node"], (
            f"pred_type should be 'edge' or 'node', not {self.pred_type}"
        )

        self.solver_threshold = config.manifold.ode_solver.vpae_thresh

        if config.dataset.type == "molecule":
            self.graph_cls = MolGraph
            self.dynamic_graph_cls = DynamicMolGraph
        elif config.dataset.type == "reaction":
            self.graph_cls = RxnGraph
            self.dynamic_graph_cls = DynamicRxnGraph
        else:
            raise ValueError(f"Dataset type not supported: {config.dataset.type}")

    # -----------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------

    def forward(self, graph):
        edge_index = graph.full_edge(upper_triangle=True)[0]

        node2graph = graph.batch
        edge2graph = node2graph.index_select(0, edge_index[0])
        num_nodes = node2graph.bincount()

        ## Prediction
        if self.pred_type == "node":
            pred_x = self.NeuralNet(graph).squeeze()
            pred_q = torch.zeros(*edge2graph.shape, dtype=pred_x.dtype, device=pred_x.device)
        elif self.pred_type == "edge":
            pred_q = self.NeuralNet(graph).squeeze()
            pred_x = torch.zeros((len(node2graph), 3), dtype=pred_q.dtype, device=pred_q.device)

            if self.q_type == "morse":
                dq_dd = self.geodesic_solver.dq_dd(
                    graph.pos,
                    graph.atom_type,
                    edge_index,
                    q_type=self.q_type,
                )
                pred_q = dq_dd * pred_q
        else:
            raise ValueError(f"pred_type should be 'edge' or 'node', not {self.pred_type}")

        if self.config.train.transform_at_forward:
            pred_x, pred_q = self.transform(
                pred_x,
                pred_q,
                graph.pos,
                graph.atom_type,
                edge_index,
                node2graph,
                num_nodes,
                self.q_type,
                rescale_dq=False,
            )
        return pred_x, pred_q, edge_index, node2graph, edge2graph

    # -----------------------------------------------------------------
    # Score transformation
    # -----------------------------------------------------------------

    def transform(
        self,
        score_x,
        score_q,
        pos,
        atom_type,
        edge_index,
        batch,
        num_nodes,
        q_type,
        rescale_dq=True,
    ):
        transform_type = self.config.train.transform

        assert transform_type in ["eq_transform", "projection_dx2dq", "projection_dq2dx", None]

        if transform_type == "eq_transform":
            assert self.q_type == "DM"
            assert self.pred_type == "edge" or not self.config.train.transform_at_forward

            edge_length = get_distance(pos, edge_index).unsqueeze(-1)
            score_q = score_q.unsqueeze(-1)
            score_x = eq_transform(score_q, pos, edge_index, edge_length)
            score_q = score_q.squeeze(-1)
        elif transform_type == "projection_dq2dx":
            assert self.pred_type == "edge" or not self.config.train.transform_at_forward

            if rescale_dq:
                edge2graph = batch.index_select(0, edge_index[0])
                norm_q = scatter_sum(score_q.square(), edge2graph).sqrt()

            score_q = self.geodesic_solver.batch_projection(
                score_q,
                pos,
                atom_type,
                edge_index,
                batch,
                num_nodes,
                q_type=q_type,
                proj_type="manifold",
            )
            if rescale_dq:
                norm_q_new = scatter_sum(score_q.square(), edge2graph).sqrt()
                # print(f"Debug: norm_q / norm_q_new (> 1.)={norm_q / norm_q_new}")
                rescale = (norm_q / norm_q_new).index_select(0, edge2graph)
                score_q *= rescale

            if self.config.train.lambda_x_train:
                score_x = self.geodesic_solver.batch_dq2dx(
                    score_q,
                    pos,
                    atom_type,
                    edge_index,
                    batch,
                    num_nodes,
                    q_type=q_type,
                ).reshape(-1, 3)
        elif transform_type == "projection_dx2dq":
            assert self.pred_type == "node"

            score_x = self.geodesic_solver.batch_projection(
                score_x,
                pos,
                atom_type,
                edge_index,
                batch,
                num_nodes,
                q_type=q_type,
                proj_type="euclidean",
            )
            score_q = self.geodesic_solver.batch_dx2dq(
                score_x,
                pos,
                atom_type,
                edge_index,
                batch,
                num_nodes,
                q_type=q_type,
            ).reshape(-1)
        elif transform_type is None:
            pass
        else:
            raise ValueError()
        return score_x, score_q

    # -----------------------------------------------------------------
    # Noise sampling (training targets)
    # -----------------------------------------------------------------

    def noise_sampling(self, data):
        """Make noised positions and target objectives."""
        if self.config.train.noise_type == "straight_to_x0":
            graph, pos, pos_init, tt, target_x, target_q = self.apply_straight_to_x0(
                data,
                do_scale=self.config.train.do_scale,
            )
        elif self.config.train.noise_type == "diffusion":
            graph, pos, pos_init, tt, target_x, target_q = self.apply_noise_diffusion(
                data,
                do_scale=self.config.train.do_scale,
            )
        elif self.config.train.noise_type == "diffusion_custom":
            graph, pos, pos_init, tt, target_x, target_q = self.apply_noise_diffusion_custom(data)
        else:
            raise NotImplementedError(f"Unsupported noise_type: {self.config.train.noise_type}")
        noisy_graph = self.dynamic_graph_cls.from_graph(graph, pos, pos_init, tt)
        return noisy_graph, target_x, target_q

    # -----------------------------------------------------------------
    # Loss weight
    # -----------------------------------------------------------------

    def get_loss_weight(self, time_step: torch.Tensor) -> torch.Tensor:
        """Weight of loss at each time step (lambda(t))."""
        loss_weight_type = self.config.train.loss_weight
        if loss_weight_type == "diffusion":
            a = self.noise_schedule.get_alpha(time_step, device=time_step.device)
            weight = a / (1.0 - a)
        elif loss_weight_type == "ddbm_h_transform":
            sigma2 = self.noise_schedule.get_sigma(time_step)
            weight = 1 / sigma2.square()
        elif loss_weight_type == "cfm":
            weight = 1 / time_step.square()
        elif loss_weight_type == "q_norm":
            raise NotImplementedError()
        else:
            weight = None
        return weight

    # -----------------------------------------------------------------
    # Noise application methods
    # -----------------------------------------------------------------

    def apply_noise_diffusion(self, data, do_scale=True):
        """Diffusion noise sampling (no bridge). Refer to GeoDiff, TSDiff."""
        assert self.noise_schedule.name == "TSDiffNoiseScheduler"

        graph = self.graph_cls.from_batch(data)
        edge_index = graph.full_edge(upper_triangle=True)[0]

        batch_size = len(data.ptr) - 1
        node2graph = graph.batch
        edge2graph = node2graph.index_select(0, edge_index[0])
        num_nodes = data.ptr[1:] - data.ptr[:-1]

        pos = data.pos[:, 0]
        pos = center_pos(pos, data.batch)
        device = pos.device

        t0 = self.config.diffusion.scheduler.t0
        t1 = self.config.diffusion.scheduler.t1
        time_step = torch.randint(t0, t1, size=(batch_size,), device=device)
        # print(f"Debug: time_step in [{min(time_step)}, {max(time_step)}]")
        time_step = time_step.sort()[0]
        a = self.noise_schedule.get_alpha(time_step, device=device)

        # Perturb pos
        a_pos = a.index_select(0, node2graph).unsqueeze(-1)
        pos_noise = torch.randn(size=pos.size(), device=device)
        pos_t = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        pos_t = center_pos(pos_t, data.batch)

        a_edge = a.index_select(0, edge2graph)

        pos_target = pos - pos_t
        q_gt = self.geodesic_solver.compute_d_or_q(pos, graph.atom_type, edge_index, q_type=self.q_type)
        q_t = self.geodesic_solver.compute_d_or_q(pos_t, graph.atom_type, edge_index, q_type=self.q_type)
        d_target = q_gt - q_t

        pos_target, d_target = self.transform(
            pos_target,
            d_target,
            pos_t,
            graph.atom_type,
            edge_index,
            node2graph,
            num_nodes,
            self.q_type,
        )

        if do_scale:
            assert self.q_type == "DM"
            pos_target *= a_pos.sqrt() / (1.0 - a_pos).sqrt()
            d_target *= a_edge.sqrt() / (1.0 - a_edge).sqrt()
        else:
            pass

        return graph, pos_t, pos_t, time_step, pos_target, d_target

    def apply_noise_diffusion_custom(self, data):
        """Using custom q_target, pos_target, and time_step from data."""
        assert self.noise_schedule.name == "TSDiffNoiseScheduler"

        graph = self.graph_cls.from_batch(data)
        edge_index = graph.full_edge(upper_triangle=True)[0]

        batch_size = len(data.ptr) - 1
        node2graph = graph.batch
        edge2graph = node2graph.index_select(0, edge_index[0])
        num_nodes = data.ptr[1:] - data.ptr[:-1]

        pos = data.pos[:, 0]
        pos = center_pos(pos, data.batch)
        device = pos.device

        time_step = data.time_step
        pos_t = data.pos[:, 1]

        q_target = data.q_target
        pos_target = data.pos[:, 2]
        return graph, pos_t, pos_t, time_step, pos_target, q_target

    def apply_straight_to_x0(self, data, do_scale=False):
        """No noised version. (Straight line approximation.)"""
        graph = self.graph_cls.from_batch(data)
        edge_index = graph.full_edge(upper_triangle=True)[0]

        node2graph = graph.batch
        edge2graph = node2graph.index_select(0, edge_index[0])
        num_nodes = data.ptr[1:] - data.ptr[:-1]

        device = data.pos.device
        pos_T = data.pos[:, -1]
        pos_0 = data.pos[:, 0]

        ## linear interpolated pos in euclidean
        batch_size = len(data.ptr) - 1
        time_step = torch.rand(size=(batch_size,), device=device)
        time_step = time_step.sort()[0]
        time_step_node = time_step.index_select(0, node2graph).unsqueeze(-1)

        mu_t = (1 - time_step_node) * pos_0 + time_step_node * pos_T

        pos_target = pos_0 - mu_t
        q_0 = self.geodesic_solver.compute_d_or_q(pos_0, graph.atom_type, edge_index, q_type=self.q_type)
        q_t = self.geodesic_solver.compute_d_or_q(mu_t, graph.atom_type, edge_index, q_type=self.q_type)
        q_target = q_0 - q_t

        pos_target, d_target = self.transform(
            pos_target,
            q_target,
            mu_t,
            graph.atom_type,
            edge_index,
            node2graph,
            num_nodes,
            self.q_type,
        )

        if do_scale:
            time_step_edge = time_step.index_select(0, edge2graph)
            pos_target /= time_step_node
            d_target /= time_step_edge
        else:
            pass

        return graph, mu_t, pos_T, time_step, pos_target, q_target

    # -----------------------------------------------------------------
    # CFM prediction (used during sampling)
    # -----------------------------------------------------------------

    @torch.no_grad()
    def predict_cfm(
        self,
        graph,
        dt,
        debug=False,
        batch=None,  # for debugging
        **kwargs,
    ):
        """
        CFM sampling: x_{t-dt} = x_t + (x_0 - x_t)_theta / t * dt
        """
        assert self.config.train.do_scale == False

        time_step = graph.t.unsqueeze(-1)
        dt = dt.unsqueeze(-1)
        score_x, score_q = self.forward(graph)[:2]
        score = score_x

        if debug:
            pos_0 = batch.pos[:, 0, :]
            pos_t = graph.pos
            edge_index = graph.full_edge(upper_triangle=True)[0]
            node2graph = batch.batch
            num_nodes = node2graph.bincount()
            pos_0 = center_pos(pos_0, batch.batch)
            pos_t = center_pos(pos_t, batch.batch)

            q_gt = self.geodesic_solver.compute_d_or_q(pos_0, graph.atom_type, edge_index, q_type=self.q_type)
            q_t = self.geodesic_solver.compute_d_or_q(pos_t, graph.atom_type, edge_index, q_type=self.q_type)
            score_ref, _ = self.transform((pos_0 - pos_t), (q_gt - q_t), pos_t, graph.atom_type, edge_index, node2graph, num_nodes, q_type=self.q_type, rescale_dq=False)

            square_err = (score - score_ref).square().sum(dim=-1)
            rmsd = scatter_mean(square_err, node2graph).sqrt()
            pred_norm = scatter_mean(score.square().sum(dim=-1), node2graph).sqrt()
            denom = scatter_mean(score_ref.square().sum(dim=-1), node2graph).sqrt()
            perr = rmsd / denom
            print(f"Debug: rmsd=\n{rmsd.detach()}")
            print(f"Debug: pred_norm=\n{pred_norm.detach()}")
            print(f"Debug: denom=\n{denom.detach()}")
            print(f"Debug: perr=\n{perr.detach()}")

        node2graph = batch.batch
        edge_index = graph.full_edge(upper_triangle=True)[0]
        edge2graph = node2graph.index_select(0, edge_index[0])

        score_x *= dt.index_select(0, node2graph) / time_step.index_select(0, node2graph)
        score_q *= (dt.index_select(0, edge2graph) / time_step.index_select(0, edge2graph)).squeeze(-1)

        return -score_x, -score_q
