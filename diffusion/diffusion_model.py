import time

import torch
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.data import Batch
import pytorch_lightning as pl

import wandb
from tqdm.auto import tqdm

from utils.wandb_utils import setup_wandb
from utils.rxn_graph import RxnGraph, DynamicRxnGraph, MolGraph, DynamicMolGraph
from utils.geodesic_solver import GeodesicSolver
from diffusion.noise_scheduler import load_noise_scheduler
from metrics.metrics import LossFunction, SamplingMetrics, TrainMetrics, ValidMetrics
from model import get_optimizer, get_scheduler
from model.geodiff_encoder import GeoDiffEncoder
from model.condensed_encoder import CondensedEncoderEpsNetwork
from model.equivariant_encoder import EquivariantEncoderEpsNetwork
from model_tsdiff.condensed_encoder import CondenseEncoderEpsNetwork as CondensedEncoderEpsNetwork2
from model_tsdiff.geometry import get_distance, eq_transform


def _masking(num_nodes):
    N = num_nodes.max()
    mask = torch.BoolTensor([True, False]).repeat(len(num_nodes)).to(num_nodes.device)
    num_repeats = torch.stack([num_nodes, N - num_nodes]).T.flatten()
    mask = mask.repeat_interleave(num_repeats)
    return mask


## TODO: Rename BridgeDiffusion class (e.g., UnifiedDiffusionModel, IntegratedDiffusionFramework)
class BridgeDiffusion(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.name is used for logging, add time to avoid name conflict
        # config.general.name  +  dd-mm-yy:hh-mm-ss
        self.name = config.general.name + time.strftime(":%d-%m-%y:%H-%M-%S")

        self.geodesic_solver = GeodesicSolver(config.manifold)

        if config.model.name == "equivariant":
            self.NeuralNet = EquivariantEncoderEpsNetwork(config.model, self.geodesic_solver)
        elif config.model.name == "condensed":
            self.NeuralNet = CondensedEncoderEpsNetwork(config.model, self.geodesic_solver)
        elif config.model.name == "condensed2":
            self.NeuralNet = CondensedEncoderEpsNetwork2(config.model)
        elif config.model.name == "geodiff":
            self.NeuralNet = GeoDiffEncoder(config.model)
        else:
            raise NotImplementedError

        self.noise_schedule = load_noise_scheduler(config.diffusion)
        self.dynamic_graph_list = []  # for storing results of all trajectories

        lambda_x_train = config.train.lambda_x_train
        lambda_q_train = config.train.lambda_q_train
        lambda_x_valid = config.train.lambda_x_valid
        lambda_q_valid = config.train.lambda_q_valid
        if lambda_x_train != lambda_x_valid:
            print(f"Warning: config.train.lambda_x_train != config.train.lambda_x_valid")
        if lambda_q_train != lambda_q_valid:
            print(f"Warning: config.train.lambda_q_train != config.train.lambda_q_valid")

        self.train_loss = LossFunction(lambda_x_train, lambda_q_train, "train")
        self.valid_loss = LossFunction(lambda_x_valid, lambda_q_valid, "valid")
        self.test_loss = LossFunction(lambda_x_valid, lambda_q_valid, "test")

        self.train_metrics = TrainMetrics(name='train')
        self.valid_metrics = ValidMetrics(self.geodesic_solver, 'valid', lambda_x_valid, lambda_q_valid)
        self.test_metrics = ValidMetrics(self.geodesic_solver, 'test', lambda_x_valid, lambda_q_valid)

        self.valid_sampling_metrics = SamplingMetrics(self.geodesic_solver, name='valid')
        self.test_sampling_metrics = SamplingMetrics(self.geodesic_solver, name='test')

        self.pred_type = self.config.model.pred_type
        self.q_type = self.config.manifold.ode_solver.q_type

        assert self.pred_type in ["edge", "node"], (
            f"pred_type should be 'edge' or 'node', not {self.pred_type}"
        )

        self.solver_threshold = config.manifold.ode_solver.vpae_thresh
        self.save_dir = config.debug.save_dir
        self.best_valid_loss = 1e9

        self.val_counter = 0
        self.test_counter = 0

        if self.config.dataset.type == "reaction":
            self.graph = RxnGraph
            self.dynamic_graph = DynamicRxnGraph
            assert self.config.model.name in ["equivariant", "condensed", "condensed2"]
        elif self.config.dataset.type == "molecule":
            self.graph = MolGraph
            self.dynamic_graph = DynamicMolGraph
            assert self.config.model.name in ["geodiff"]
        else:
            raise ValueError()
        return

    def transform_test(self, score_x, score_q, pos, atom_type, edge_index, batch, num_nodes, q_type):
        """Debugging. 각 transform의 값과 크기 비교."""

        score_x = score_x.clone()
        score_q = score_q.clone()

        node2graph = batch
        edge2graph = node2graph.index_select(0, edge_index[0])

        denom_x = scatter_mean(score_x.square().sum(dim=-1), node2graph).sqrt()
        denom_q = scatter_mean(score_q.square(), edge2graph).sqrt()

        assert self.pred_type == "edge"

        self.config.train.transform = "eq_transform"
        score_x1, score_q1 = self.transform(score_x, score_q, pos, atom_type, edge_index, node2graph, num_nodes, q_type)
        denom_x1 = scatter_mean(score_x1.square().sum(dim=-1), node2graph).sqrt()
        denom_q1 = scatter_mean(score_q1.square(), edge2graph).sqrt()

        self.config.train.transform = "projection_dq2dx"
        score_x2, score_q2 = self.transform(score_x, score_q, pos, atom_type, edge_index, node2graph, num_nodes, q_type)
        denom_x2 = scatter_mean(score_x2.square().sum(dim=-1), node2graph).sqrt()
        denom_q2 = scatter_mean(score_q2.square(), edge2graph).sqrt()

        print(f"Debug: denom_x=\n{denom_x}")
        print(f"Debug: denom_x1=\n{denom_x1}")
        print(f"Debug: denom_x2=\n{denom_x2}")
        print(f"Debug: denom_q=\n{denom_q}")
        print(f"Debug: denom_q1=\n{denom_q1}")
        print(f"Debug: denom_q2=\n{denom_q2}")
        exit("DEBUG: transform_test")
        return

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
        rescale_dq=True,  # rescale to have the same length with the input score_q
    ):
        transform_type = self.config.train.transform

        assert transform_type in ["eq_transform", "projection_dx2dq", "projection_dq2dx", None]

        if transform_type == "eq_transform":
            assert self.q_type == "DM"
            # assert self.pred_type == "edge"
            assert self.pred_type == "edge" or not self.config.train.transform_at_forward

            edge_length = get_distance(pos, edge_index).unsqueeze(-1)
            score_q = score_q.unsqueeze(-1)
            score_x = eq_transform(score_q, pos, edge_index, edge_length)
            score_q = score_q.squeeze(-1)
        elif transform_type == "projection_dq2dx":
            # assert self.pred_type == "edge"
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
                print(f"Debug: norm_q / norm_q_new (> 1.)={norm_q / norm_q_new}")
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

    def noise_sampling(self, data):
        ## Make noised positions and target objectives
        if self.config.train.noise_type == "manifold":
            graph, pos, pos_init, tt, target_x, target_q = self.apply_noise(data)
        elif self.config.train.noise_type == "euclidean":
            graph, pos, pos_init, tt, target_x, target_q = self.apply_noise_pos(data)
        elif self.config.train.noise_type == "straight_to_x0":
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
        elif self.config.train.noise_type == "diffusion_riemannian":
            graph, pos, pos_init, tt, target_x, target_q = self.apply_noise_diffusion_riemannian(data)
        elif self.config.train.noise_type == "diffusion_geodesic":
            graph, pos, pos_init, tt, target_x, target_q = self.apply_noise_diffusion_geodesic(data, use_traj=False)
            # graph, pos, pos_init, tt, target_x, target_q = self.apply_noise_diffusion_geodesic(data, use_traj=True)
        elif self.config.train.noise_type == "ddbm_h_transform":
            graph, pos, pos_init, tt, target_x, target_q = self.apply_noise_ddbm(
                data,
                objective="h_transform",
                do_scale=self.config.train.do_scale,
            )
        elif self.config.train.noise_type == "ddbm_score":
            graph, pos, pos_init, tt, target_x, target_q = self.apply_noise_ddbm(
                data,
                objective="score",
                do_scale=self.config.train.do_scale,
            )
        else:
            raise NotImplementedError
        noisy_graph = self.dynamic_graph.from_graph(graph, pos, pos_init, tt)
        print(f"Debug: tt={tt}")
        return noisy_graph, target_x, target_q

    def forward(self, graph):
        edge_index = graph.full_edge(upper_triangle=True)[0]

        node2graph = graph.batch
        edge2graph = node2graph.index_select(0, edge_index[0])
        num_nodes = node2graph.bincount()

        ## 2. Prediction
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
                # rescale_dq=True,
            )
        return pred_x, pred_q, edge_index, node2graph, edge2graph

    def get_loss_weight(
        self,
        time_step: torch.Tensor,
    ) -> torch.Tensor:
        """weight of loss at eact time step (\lambda(t))"""
        loss_weight_type = self.config.train.loss_weight
        if loss_weight_type == "diffusion":
            a = self.noise_schedule.get_alpha(time_step, device=time_step.device)
            weight = a / (1.0 - a)
        elif loss_weight_type == "ddbm_h_transform":
            sigma2 = self.noise_schedule.get_sigma(time_step)
            weight = 1 / sigma2.square()
        elif loss_weight_type == "cfm":
            weight = 1 / time_step.square()
        elif loss_weight_type == "q_norm":  # TODO: to do implementation
            raise NotImplementedError()
        else:
            weight = None
        return weight

    def apply_noise_diffusion(self, data, do_scale=True):
        """diffusion noise sampling (no bridge). refer to GeoDiff, TSDiff"""
        assert self.noise_schedule.name == "TSDiffNoiseScheduler"

        graph = self.graph.from_batch(data)
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
        print(f"Debug: time_step in [{min(time_step)}, {max(time_step)}]")
        time_step = time_step.sort()[0]
        a = self.noise_schedule.get_alpha(time_step, device=device)

        # Perterb pos
        a_pos = a.index_select(0, node2graph).unsqueeze(-1)
        pos_noise = torch.randn(size=pos.size(), device=device)
        pos_t = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        pos_t = center_pos(pos_t, data.batch)
        # pos_t = align_geom_batch(pos, pos_t, data.batch)

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
        """Using custom q_target, pos_target, and time_step from data"""
        assert self.noise_schedule.name == "TSDiffNoiseScheduler"

        graph = self.graph.from_batch(data)
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


    def apply_noise_diffusion_riemannian(self, data):
        """diffusion noise sampling (no bridge). refer to GeoDiff, TSDiff"""
        assert self.noise_schedule.name == "TSDiffNoiseScheduler"

        graph = self.graph.from_batch(data)
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
        print(f"Debug: time_step in [{min(time_step)}, {max(time_step)}]")
        time_step = time_step.sort()[0]
        a = self.noise_schedule.get_alpha(time_step, device=device)

        # a_pos = a.index_select(0, node2graph).unsqueeze(-1)
        a_edge = a.index_select(0, edge2graph)

        # Perterb pos
        q_noise = torch.randn(size=(edge_index.size(1),), device=device)
        q_noise *= (1.0 - a_edge).sqrt() / a_edge.sqrt()
        # pos_t, pos_target, q_target = self.ode_noise_sampling(
        pos_t, pos_target, q_target, graph, time_step = self.ode_noise_sampling(
            pos,
            q_noise,
            edge_index,
            graph.atom_type,
            node2graph,
            edge2graph,
            num_nodes,
            data,
            graph,
            time_step,
        )
        # pos_t = center_pos(pos_t, data.batch)

        return graph, pos_t, pos_t, time_step, pos_target, q_target

    def apply_xT_to_x0(self, data):
        graph = self.graph.from_batch(data)
        edge_index = graph.full_edge(upper_triangle=True)[0]

        node2graph = graph.batch
        edge2graph = node2graph.index_select(0, edge_index[0])
        num_nodes = data.ptr[1:] - data.ptr[:-1]

        pos = data.pos[:, 0]
        pos_T = data.pos[:, -1]
        device = pos.device

        q_0 = self.geodesic_solver.compute_d_or_q(pos, graph.atom_type, edge_index, q_type=self.q_type)
        q_T = self.geodesic_solver.compute_d_or_q(pos_T, graph.atom_type, edge_index, q_type=self.q_type)

        q_target = q_0 - q_T
        pos_target = pos - pos_T

        pos_target, q_target = self.transform(
            pos_target,
            q_target,
            pos_T,
            graph.atom_type,
            edge_index,
            node2graph,
            num_nodes,
            self.q_type,
        )
        t0 = self.config.diffusion.scheduler.t0
        t1 = self.config.diffusion.scheduler.t1
        batch_size = len(data.ptr) - 1
        time_step = torch.randint(t0, t1, size=(batch_size,), device=device)
        return graph, pos_T, pos_T, time_step, pos_target, q_target

    def apply_noise_diffusion_geodesic(self, data, use_traj=False):
        graph = self.graph.from_batch(data)
        edge_index = graph.full_edge(upper_triangle=True)[0]

        node2graph = graph.batch
        edge2graph = node2graph.index_select(0, edge_index[0])
        num_nodes = data.ptr[1:] - data.ptr[:-1]

        device = data.pos.device

        if use_traj:
            num_traj = data.geodesic_length.shape[-1]
            i_traj = torch.randint(1, num_traj, size=(1,)).item()
        else:
            i_traj = -1

        pos = data.pos[:, 0]
        pos_t = data.pos[:, i_traj]
        pos_tm1 = data.pos[:, i_traj - 1]

        ## calculate q_dot
        g_last_length = data.geodesic_length[:, i_traj]
        g_segment = data.geodesic_length[:, i_traj] - data.geodesic_length[:, i_traj - 1]
        ratio = (g_last_length / g_segment).unsqueeze(-1)

        # time_step = data.time_step

        q_t = self.geodesic_solver.compute_d_or_q(pos_t, graph.atom_type, edge_index, q_type=self.q_type)
        q_tm1 = self.geodesic_solver.compute_d_or_q(pos_tm1, graph.atom_type, edge_index, q_type=self.q_type)

        ratio_edge = ratio.index_select(0, edge2graph).squeeze(-1)
        ratio_node = ratio.index_select(0, node2graph)
        q_target = (q_tm1 - q_t) * ratio_edge
        pos_target = (pos_tm1 - pos_t) * ratio_node

        pos_target, q_target = self.transform(
            pos_target,
            q_target,
            pos_t,
            graph.atom_type,
            edge_index,
            node2graph,
            num_nodes,
            self.q_type,
        )
        return graph, pos_t, pos_t, data.time_step, pos_target, q_target

    def apply_straight_to_x0(self, data, do_scale=False):
        """No noised version. (straight line approximation.)"""
        graph = self.graph.from_batch(data)
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

    def apply_noise_ddbm(self, data, objective="h_transform", do_scale=False):
        assert self.config.diffusion.scheduler.name.lower() == "monomial"
        assert objective in ["h_transform", "score", None]

        graph = self.graph.from_batch(data)
        edge_index = graph.full_edge(upper_triangle=True)[0]

        node2graph = graph.batch
        edge2graph = node2graph.index_select(0, edge_index[0])
        num_nodes = data.ptr[1:] - data.ptr[:-1]

        device = data.pos.device
        pos_T = data.pos[:, -1]
        pos_0 = data.pos[:, 0]

        batch_size = len(data.ptr) - 1
        tt = torch.rand(size=(batch_size,), device=device)
        # margin = -0.3
        # print(f"Debug: time margin of {margin} is used.")
        # # tt = (1 + 2 * margin) * torch.rand(size=(batch_size,), device=device) - margin  # [-margin, 1 + margin]
        # tt = (1 + margin) * torch.rand(size=(batch_size,), device=device) - margin  # [-margin, 1]
        tt_node = tt.index_select(0, node2graph).unsqueeze(-1)

        sigma_hat2 = self.noise_schedule.get_sigma_hat(tt)  # (G, )
        sigma_hat2_node = sigma_hat2.index_select(0, node2graph).unsqueeze(-1)  # (N, )
        sigma_hat2_edge = sigma_hat2.index_select(0, edge2graph)

        sigma2 = self.noise_schedule.get_sigma(tt)
        sigma2_node = sigma2.index_select(0, node2graph).unsqueeze(-1)
        sigma2_edge = sigma2.index_select(0, edge2graph)
        SNR_ratio = self.noise_schedule.get_SNR(tt)
        SNR_ratio_node = SNR_ratio.index_select(0, node2graph).unsqueeze(-1)
        mu_t = SNR_ratio_node * pos_T + (1 - SNR_ratio_node) * pos_0

        eps_node = torch.randn_like(mu_t)
        noise = eps_node * sigma_hat2_node.sqrt()  # dq, (E, )
        pos_t = mu_t + noise
        pos_t = mu_t; print(f"Debug: pos_t is set to mu_t")

        pos_gt = pos_0 if objective == "h_transform" else mu_t

        pos_target = pos_gt - pos_t
        q_gt = self.geodesic_solver.compute_d_or_q(pos_gt, graph.atom_type, edge_index, q_type=self.q_type)
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
            if objective == "h_transform":
                pos_target /= sigma2_node
                d_target /= sigma2_edge
            elif objective == "score":
                pos_target /= sigma_hat2_node
                d_target /= sigma_hat2_edge
        else:
            pass

        assert self.config.train.lambda_q_train == 0
        assert self.config.train.lambda_q_valid == 0
        return graph, pos_t, pos_T, tt, pos_target, d_target

    def apply_noise_pos(self, data):
        graph = self.graph.from_batch(data)
        edge_index = graph.full_edge(upper_triangle=True)[0]

        node2graph = graph.batch
        edge2graph = node2graph.index_select(0, edge_index[0])
        num_nodes = data.ptr[1:] - data.ptr[:-1]

        # sampling time step
        t_index, tt, SNR_ratio = self.noise_level_sampling(data)  # (G, ), (G, ), (G, )

        t_index_node = t_index.index_select(0, node2graph)  # (N, )
        mean = data.pos[(torch.arange(len(t_index_node)), t_index_node)]  # (N, 3)
        pos_init = data.pos[:, -1]

        sigma_hat2 = self.noise_schedule.get_sigma_hat(tt)  # (G, )
        sigma_hat2_node = sigma_hat2.index_select(0, node2graph)  # (N, )

        eps_node = torch.randn_like(mean)
        noise = eps_node * sigma_hat2_node.sqrt().unsqueeze(-1)  # dq, (E, )

        pos_noise = mean + noise
        target_x = -eps_node
        target_q = self.geodesic_solver.batch_dx2dq(target_x, pos_noise, graph.atom_type, edge_index, node2graph, num_nodes, q_type=self.q_type).reshape(-1)

        return graph, pos_noise, pos_init, tt, target_x, target_q

    def ode_noise_sampling(
        self,
        pos,
        q_noise,
        edge_index,
        atom_type,
        node2graph,
        edge2graph,
        num_nodes,
        data,
        graph,
        time_step,
    ):
        """
        Performs ODE sampling.

        Parameters:
        -----------
        pos : torch.Tensor
            The initial positions of nodes in the graph.
        q_noise : torch.Tensor
            The noise tensor that will be added to q(pos).
        edge_index : torch.LongTensor
            The edge indices for the graph. This tensor typically has shape [2, num_edges], where each column represents
            a connection between two nodes.
        atom_type : torch.LongTensor
            A tensor representing the types of atoms or nodes in the graph. shape=[N,]
        node2graph : torch.LongTensor
            A tensor that maps each node to its corresponding graph index. shape=[N,]
        edge2graph : torch.LongTensor
            A tensor that maps each edge to its corresponding graph index. shape=[E,]
        num_nodes : torch.LongTensor
            The number of nodes in the batch of graphs. shape=[B,]
        data : Data object
            The input data containing the graph(s) information. This typically includes the node features, edge features, etc.
        graph : Graph object
            The graph structure containing all relevant information like adjacency lists, node attributes, etc.
        time_step : torch.LongTensor
            The time steps of noise scheduling. shape=[B,]
        """
        init, last, iter, index_tensor, stats = self.geodesic_solver.batch_geodesic_ode_solve(
            pos,
            q_noise,
            edge_index,
            atom_type,
            node2graph,
            num_nodes,
            q_type=self.q_type,
            num_iter=self.config.manifold.ode_solver.iter,
            max_iter=self.config.manifold.ode_solver.max_iter,
            ref_dt=self.config.manifold.ode_solver.ref_dt,
            min_dt=self.config.manifold.ode_solver.min_dt,
            max_dt=self.config.manifold.ode_solver.max_dt,
            err_thresh=self.config.manifold.ode_solver.vpae_thresh,
            verbose=0,
            method="Heun",
            pos_adjust_scaler=self.config.manifold.ode_solver.pos_adjust_scaler,
            pos_adjust_thresh=self.config.manifold.ode_solver.pos_adjust_thresh,
        )

        batch_pos_noise = last["x"]  # (B, n, 3)
        batch_x_dot = last["x_dot"]  # (B, n, 3)
        unbatch_node_mask = _masking(num_nodes)
        pos_noise = batch_pos_noise.reshape(-1, 3)[unbatch_node_mask]  # (N, 3)
        x_dot = batch_x_dot.reshape(-1, 3)[unbatch_node_mask]  # (N, 3)

        batch_q_dot = last["q_dot"]  # (B, e)
        batch_q_init = init["q"]  # (B, e)
        e = batch_q_dot.size(1)
        unbatch_edge_index = index_tensor[1] + index_tensor[0] * e
        q_dot = batch_q_dot.reshape(-1)[unbatch_edge_index]  # (E, )
        q_init = batch_q_init.reshape(-1)[unbatch_edge_index]  # (E, )

        # Check stability, percent error > threshold, then re-solve
        retry_index = stats["ban_index"].sort().values
        if len(retry_index) > 0:
            node_select = torch.isin(node2graph, retry_index)
            edge_select = torch.isin(edge2graph, retry_index)
            # _batch = torch.arange(len(retry_index), device=mean.device).repeat_interleave(num_nodes[retry_index])
            _batch = torch.arange(len(retry_index), device=pos.device).repeat_interleave(num_nodes[retry_index])
            _num_nodes = num_nodes[retry_index]
            _num_edges = _num_nodes * (_num_nodes - 1) // 2
            _ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=_num_nodes.device), _num_nodes.cumsum(0)])
            _full_edge = index_tensor[2:][:, edge_select] + _ptr[:-1].repeat_interleave(_num_edges)

            self.print(f"[Resolve] geodesic solver failed at {len(retry_index)}/{len(data)}, Retry...")
            _init, _last, _iter, _index_tensor, _stats = self.geodesic_solver.batch_geodesic_ode_solve(
                pos[node_select],
                q_noise[edge_select],
                _full_edge,
                atom_type[node_select],
                _batch,
                _num_nodes,
                q_type=self.q_type,
                num_iter=self.config.manifold.ode_solver.iter,
                max_iter=self.config.manifold.ode_solver.max_iter,
                ref_dt=self.config.manifold.ode_solver._ref_dt,
                min_dt=self.config.manifold.ode_solver._min_dt,
                max_dt=self.config.manifold.ode_solver._max_dt,
                err_thresh=self.config.manifold.ode_solver.vpae_thresh,
                verbose=0,
                method="RK4",
                pos_adjust_scaler=self.config.manifold.ode_solver.pos_adjust_scaler,
                pos_adjust_thresh=self.config.manifold.ode_solver.pos_adjust_thresh,
            )

            _batch_pos_noise = _last["x"]  # (b, n', 3)
            _batch_x_dot = _last["x_dot"]  # (b, n', 3)
            _unbatch_node_mask = _masking(_num_nodes)
            _pos_noise = _batch_pos_noise.reshape(-1, 3)[_unbatch_node_mask]  # (N', 3)
            _x_dot = _batch_x_dot.reshape(-1, 3)[_unbatch_node_mask]  # (N', 3)

            _batch_q_dot = _last["q_dot"]  # (b, e')
            _e = _batch_q_dot.size(1)
            _unbatch_edge_index = _index_tensor[1] + _index_tensor[0] * _e
            _q_dot = _batch_q_dot.reshape(-1)[_unbatch_edge_index]  # (E', )

            pos_noise[node_select] = _pos_noise
            x_dot[node_select] = _x_dot
            q_dot[edge_select] = _q_dot

            ban_index = _stats["ban_index"].sort().values
            ban_index = retry_index[ban_index]
        else:
            ban_index = torch.LongTensor([])

        # coeff = torch.ones_like(tt)
        # coeff_node = coeff.index_select(0, node2graph)  # (N, )
        # coeff_edge = coeff.index_select(0, edge2graph)  # (E, )
        # target_x = - x_dot * coeff_node.unsqueeze(-1)
        # target_q = - q_dot * coeff_edge
        target_x = - x_dot
        target_q = - q_dot

        if len(ban_index) > 0:
            ban_index = ban_index.to(torch.long)
            # rxn_idx = [data.rxn_idx[i] for i in ban_index]
            idx = [data.idx[i] for i in ban_index]
            # self.print(f"\n[Warning] geodesic solver failed at {len(ban_index)}/{len(data)}"
            #             f"\n\tidx: {idx}\n\ttime index: {t_index[ban_index].tolist()}")
            #             # f"\n\trxn_idx: {rxn_idx}\n\ttime index: {t_index[ban_index].tolist()}")
            self.print(f"\n[Warning] geodesic solver failed at {len(ban_index)}/{len(data)}\n\tidx: {idx}")
            ban_node_mask = torch.isin(node2graph, ban_index)
            ban_edge_mask = torch.isin(edge2graph, ban_index)
            # ban_batch_mask = torch.isin(torch.arange(len(data), device=mean.device), ban_index)
            ban_batch_mask = torch.isin(torch.arange(len(data), device=pos.device), ban_index)

            data = Batch.from_data_list(data[~ban_batch_mask])
            graph = self.graph.from_batch(data)

            pos_noise = pos_noise[~ban_node_mask]
            # x_dot = x_dot[~ban_node_mask]
            # pos_init = pos_init[~ban_node_mask]
            # tt = tt[~ban_batch_mask]
            time_step = time_step[~ban_batch_mask]
            target_x = target_x[~ban_node_mask]
            target_q = target_q[~ban_edge_mask]

        pos_noise = center_pos(pos_noise, data.batch)
        # return pos_noise, target_x, target_q
        return pos_noise, target_x, target_q, graph, time_step

    def configure_optimizers(self):
        ## Set optimizer and scheduler
        self.optim = get_optimizer(self.config.train.optimizer, self)
        print(f"Debug: self.optim={self.optim}")
        self.optim_scheduler = get_scheduler(self.config.train.scheduler, self.optim)
        return [self.optim]

    def on_train_epoch_start(self) -> None:
        self.print("Start Training Epoch ...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        if self.train_metrics is not None:
            self.train_metrics.reset()

    def training_step(self, data, i):
        graph, target_x, target_q = self.noise_sampling(data)
        pred_x, pred_q, edge_index, node2graph, edge2graph = self.forward(graph)

        if wandb.run:
            wandb.log({"train/lr": self.optim.param_groups[0]['lr']})

        loss = self.train_loss(
            pred_x=pred_x,
            pred_q=pred_q,
            true_x=target_x,
            true_q=target_q,
            merge_edge=edge2graph,
            merge_node=node2graph,
            weight=self.get_loss_weight(graph.t),
        )
        self.train_metrics(
            pred_x,
            pred_q,
            target_x,
            target_q,
            edge2graph,
            node2graph,
            log=True,
        )
        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        #############################################
        to_log = self.train_loss.log_epoch_metrics()
        print(f"Debug: to_log={to_log}")
        loss = list(to_log.values())[0]
        self.optim_scheduler.step(loss)
        print(f"Debug: loss={loss}")
        #############################################

        msg = f"Epoch {self.current_epoch} "
        for k, v in to_log.items():
            msg += f"\n\t{k}: {v: 0.6f}"
        self.print(msg + f"\n -- {time.time() - self.start_epoch_time:0.1f}s")
        # print(torch.cuda.memory_summary())

    def on_validation_epoch_start(self) -> None:
        self.print("Starting validation ...")
        self.valid_loss.reset()
        self.valid_metrics.reset()
        self.valid_sampling_metrics.reset()

    def validation_step(self, data, i):
        graph, target_x, target_q = self.noise_sampling(data)
        pred_x, pred_q, edge_index, node2graph, edge2graph = self.forward(graph)

        loss = self.valid_loss(
            pred_x=pred_x,
            pred_q=pred_q,
            true_x=target_x,
            true_q=target_q,
            merge_edge=edge2graph,
            merge_node=node2graph,
            weight=self.get_loss_weight(graph.t),
        )
        self.valid_metrics(
            pred_x,
            pred_q,
            target_x,
            target_q,
            edge2graph,
            node2graph,
            edge_index=edge_index,
            pos=graph.pos,
            log=True,
        )
        return {"loss": loss}

    def on_validation_epoch_end(self) -> None:
        to_log = self.valid_metrics.log_epoch_metrics()
        # print(to_log)
        loss = to_log["valid_epoch/loss"]
        # perr = to_log["valid_epoch/perr"]
        rmsd_perr = to_log["valid_epoch/pred_target_rmsd_perr"]
        norm_perr = to_log["valid_epoch/pred_target_norm_perr"]

        if loss < self.best_valid_loss:
            self.best_valid_loss = loss

        self.log("valid/loss", loss, sync_dist=True)
        # self.log("valid/perr", perr, sync_dist=True)
        self.log("valid/rmsd_perr", rmsd_perr, sync_dist=True)
        self.log("valid/norm_perr", norm_perr, sync_dist=True)
        self.val_counter += 1
        self.optim_scheduler.step(loss)
        msg = f"Epoch {self.current_epoch} "
        for k, v in to_log.items():
            msg += f"\n\t{k}: {v: 0.6f}"
        self.print(msg)
        if (self.val_counter) % self.config.train.sample_every_n_valid == 0:
            stochastic = self.config.sampling.stochastic
            samples = []

            start = time.time()
            for i, batch in enumerate(self.trainer.datamodule.val_dataloader()):
                if i % self.config.train.sample_every_n_batch == 0:
                    batch = batch.to(self.device)
                    if self.config.sampling.score_type == "diffusion":
                        batch_out = self.sample_batch_diffusion(
                            batch,
                            stochastic=stochastic,
                            start_from_time=self.config.sampling.start_from_time,
                        )
                    else:
                        batch_out = self.sample_batch_simple(
                            batch,
                            stochastic=stochastic,
                            num_cycles=self.config.sampling.num_cycles,
                        )
                        # batch_out = self.sample_batch(batch, stochastic=stochastic)
                    samples.extend(batch_out)
            self.print(f"Done. Sampling took {time.time() - start:0.1f}s")

            self.valid_sampling_metrics(
                samples, self.name,
                self.current_epoch,
                valid_counter=-1,
                test=True,
                local_rank=self.local_rank
            )
        print("Validation epoch end ends...")

    def on_test_epoch_start(self) -> None:
        self.print("Starting test ...")
        self.test_loss.reset()
        self.test_metrics.reset()
        self.test_sampling_metrics.reset()
        if self.local_rank == 0:
            setup_wandb(self.config)

    def test_step(self, data, i):
        graph, target_x, target_q = self.noise_sampling(data)
        pred_x, pred_q, edge_index, node2graph, edge2graph = self.forward(graph)

        loss = self.test_loss(
            pred_x=pred_x,
            pred_q=pred_q,
            true_x=target_x,
            true_q=target_q,
            merge_edge=edge2graph,
            merge_node=node2graph,
            weight=self.get_loss_weight(graph.t),
        )
        self.test_metrics(
            pred_x,
            pred_q,
            target_x,
            target_q,
            edge2graph,
            node2graph,
            edge_index=edge_index,
            pos=graph.pos,
            log=True,
        )
        return {'loss': loss}

    def test_step(self, data, i):
        print("Passing test_step")
        pass

    def on_test_epoch_end(self) -> None:
        to_log = self.test_metrics.log_epoch_metrics()
        loss = to_log["test_epoch/loss"]
        print(torch.cuda.memory_summary())
        msg = f"Epoch {self.current_epoch} "
        for k, v in to_log.items():
            msg += f"\n\t{k}: {v: 0.6f}"
        self.print(msg)
        self.test_counter += 1
        if self.test_counter % self.config.train.sample_every_n_valid == 0:
            stochastic = self.config.sampling.stochastic
            samples = []

            start = time.time()
            for i, batch in enumerate(tqdm(self.trainer.datamodule.test_dataloader(), total=len(self.trainer.datamodule.test_dataloader()))):
                # if i < self.config.sampling.batch_idx_start:
                #     continue
                # if i > self.config.sampling.batch_idx_end:
                #     break

                if i % self.config.train.sample_every_n_batch == 0:
                    batch = batch.to(self.device)
                    if self.config.sampling.score_type == "diffusion":
                        batch_out = self.sample_batch_diffusion(
                            batch,
                            stochastic=stochastic,
                            start_from_time=self.config.sampling.start_from_time,
                        )
                    else:
                        batch_out = self.sample_batch_simple(
                            batch,
                            stochastic=stochastic,
                            num_cycles=self.config.sampling.num_cycles,
                        )
                        # batch_out = self.sample_batch(batch, stochastic=stochastic)
                        # print(f"Debug: Run sample_batch with ODE!!!!!!!!!!!!!!!!!!!!!!!!")
                    samples.extend(batch_out)
            self.print(f"Done. Sampling took {time.time() - start:0.1f}s")
            self.test_sampling_metrics(samples, self.name, self.current_epoch, valid_counter=-1, test=True, local_rank=self.local_rank)
        print("Test epoch end ends...")

        ## Save dynamic_graph object
        if self.config.debug.save_dynamic:
            torch.save(self.dynamic_graph_list, self.config.debug.save_dynamic)
            print(f"Saved dynamic graph to {self.config.debug.save_dynamic}")
        return

    @torch.no_grad()
    def predict_ddbm_score(
        self,
        graph,
        dt,
        stochastic=False,
        h_transform=True,
        **kwargs,
    ):
        time_step = graph.t

        ## Predict score
        # s(x_t) = beta(t) * (mu_t - x_t) / sigma_hat(t)^2
        score = self.forward(graph)[1]

        coeff = torch.exp(
            torch.log(self.noise_schedule.get_beta(time_step)) -
            torch.log(self.noise_schedule.get_sigma_hat(time_step))
        )
        score *= coeff

        ## Predict h-transform
        if h_transform:
            h = self.get_h_transform(
                graph.pos,
                graph.pos_init,
                graph.t,
                edge_index,
                graph.atom_type,
            )
        retval = score - h

        raise NotImplementedError()
        return retval

    @torch.no_grad()
    def predict_ddbm_h_transform(
        self,
        graph,
        dt,
        stochastic=False,
        debug=False,
        batch=None,  # for debugging
        **kwargs,
    ):
        assert self.config.train.do_scale == False

        time_step = graph.t.unsqueeze(-1)
        dt = dt.unsqueeze(-1)
        beta_t = self.noise_schedule.get_beta(time_step)
        sigma2_t = self.noise_schedule.get_sigma(time_step)
        sigma2_0 = self.noise_schedule.get_sigma(torch.zeros_like(time_step))

        # h(x_t) = beta(t) * (x_0 - x_t) / (sigma(0)^2 - sigma(t)^2)
        h = self.forward(graph)[0]

        if debug:
            pos_0 = batch.pos[:, 0, :]
            pos_t = graph.pos
            edge_index = graph.full_edge(upper_triangle=True)[0]
            node2graph = batch.batch
            num_nodes = node2graph.bincount()

            # h_ref = pos_0 - graph.pos

            q_gt = self.geodesic_solver.compute_d_or_q(pos_0, graph.atom_type, edge_index, q_type=self.q_type)
            q_t = self.geodesic_solver.compute_d_or_q(pos_t, graph.atom_type, edge_index, q_type=self.q_type)
            h_ref, _ = self.transform(None, (q_gt - q_t), pos_t, graph.atom_type, edge_index, node2graph, num_nodes, q_type=self.q_type)

            square_err = (h - h_ref).square().sum(dim=-1)
            rmsd = scatter_mean(square_err, node2graph).sqrt()
            pred_norm = scatter_mean(h.square().sum(dim=-1), node2graph).sqrt()
            denom = scatter_mean(h_ref.square().sum(dim=-1), node2graph).sqrt()
            perr = rmsd / denom
            print(f"Debug: rmsd=\n{rmsd.detach()}")
            print(f"Debug: pred_norm=\n{pred_norm.detach()}")
            print(f"Debug: denom=\n{denom.detach()}")
            print(f"Debug: perr=\n{perr.detach()}")

            # h = h_ref; print(f"Debug: h_ref is set to h")

        h /= (sigma2_0 - sigma2_t)

        if stochastic:
            dw = (beta_t * dt).sqrt() * torch.randn_like(h)
        else:
            dw = 0.
        return beta_t * h * dt + dw

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
        CFM sampling: x_{t-dt} = x_t + (x_0 - x_t)_\theta / t * dt
        """
        assert self.config.train.do_scale == False

        time_step = graph.t.unsqueeze(-1)
        dt = dt.unsqueeze(-1)
        # score = self.forward(graph)[0]
        score_x, score_q = self.forward(graph)[:2]#[0]
        score = score_x

        if debug:
            pos_0 = batch.pos[:, 0, :]
            pos_t = graph.pos
            # edge_index, _, _ = graph.full_edge(upper_triangle=True)
            edge_index = graph.full_edge(upper_triangle=True)[0]
            node2graph = batch.batch
            num_nodes = node2graph.bincount()
            pos_0 = center_pos(pos_0, batch.batch)
            pos_t = center_pos(pos_t, batch.batch)

            q_gt = self.geodesic_solver.compute_d_or_q(pos_0, graph.atom_type, edge_index, q_type=self.q_type)
            q_t = self.geodesic_solver.compute_d_or_q(pos_t, graph.atom_type, edge_index, q_type=self.q_type)
            # score_ref, _ = self.transform(None, (q_gt - q_t), pos_t, graph.atom_type, edge_index, node2graph, num_nodes, q_type=self.q_type)
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

        # score /= time_step
        score_x *= dt.index_select(0, node2graph) / time_step.index_select(0, node2graph)
        score_q *= (dt.index_select(0, edge2graph) / time_step.index_select(0, edge2graph)).squeeze(-1)

        # return -score * dt
        # return -score * dt, -score_q * dt
        return -score_x, -score_q

    @torch.no_grad()
    def sample_batch_diffusion(
        self,
        batch,
        stochastic=True,
        step_lr=0.0000010,
        clip=1000,
        clip_pos=None,
        sampling_type="ld",
        debug=False,
        start_from_time=None,
    ):
        alphas = self.noise_schedule.alphas
        sigmas = (1.0 - alphas).sqrt() / alphas.sqrt()

        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)  # .view(-1, 1, 1, 1)
            return a

        graph = self.graph.from_batch(batch)
        pos_init = batch.pos[:, -1, :]
        pos = torch.randn_like(pos_init) * sigmas[-1]

        ##################################################
        TYPE = 1
        # TYPE = 0  # from random noise
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

        # t = torch.ones(batch.num_graphs, device=pos.device)
        num_timesteps = self.noise_schedule.num_diffusion_timesteps
        t = torch.full(
            size=(batch.num_graphs,),
            fill_value=num_timesteps,
            dtype=torch.long,
            device=pos.device,
        )
        dynamic_graph = self.dynamic_graph.from_graph(graph, pos, pos_init, t)
        dynamic_graph.pos_traj.append(pos.to("cpu"))

        if start_from_time is None:
            start_from_time = num_timesteps
        else:
            assert start_from_time <= num_timesteps

        # edge_index, _, _ = dynamic_graph.full_edge(upper_triangle=True)
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
        # pos = pos_init * sigmas[-1]

        for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc="sample"):
            t = torch.full(
                size=(batch.num_graphs,),
                fill_value=i,
                dtype=torch.long,
                device=pos.device,
            )
            # if (t > 1000).any():
            # if (t > 2000).any():
            #     continue
            if (t > start_from_time).any():
                continue
            # print(f"Debug: t={t[0]}")
            node_eq = self.forward(dynamic_graph.to("cuda"))[0]

            loss_weight_type = self.config.train.loss_weight
            if loss_weight_type is not None:
                # assert loss_weight_type == "tsdiff"
                assert loss_weight_type == "diffusion"
                node_eq /= sigmas[i]

            eps_pos = clip_norm(node_eq, limit=clip)

            #################################################
            if debug:
                ## Calculate reference score

                # d_gt = get_distance(batch.pos[:, 0, :], edge_index).unsqueeze(-1)
                # edge_length = get_distance(pos, edge_index).unsqueeze(-1)
                # d_perturbed = edge_length
                # d_target = (d_gt - d_perturbed) / sigmas[i]

                # sigmas_edge = sigmas[i].index_select(0, edge2graph).unsqueeze(-1)
                # d_target = (d_gt - d_perturbed) / sigmas_edge
                # pos_target = eq_transform(d_target, pos, edge_index, edge_length)

                pos_0 = batch.pos[:, 0, :]
                q_gt = self.geodesic_solver.compute_d_or_q(pos_0, dynamic_graph.atom_type, edge_index, q_type=self.q_type)
                q_t = self.geodesic_solver.compute_d_or_q(pos, dynamic_graph.atom_type, edge_index, q_type=self.q_type)
                d_target = (q_gt - q_t) / sigmas[i]
                pos_target = (pos_0 - pos) / sigmas[i]
                pos_target, d_target = self.transform(
                    pos_target, d_target, pos, dynamic_graph.atom_type, edge_index, node2graph, num_nodes, self.q_type,
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
            # sampling_type = kwargs.get("sampling_type", "ddpm")
            if stochastic:
                noise = torch.randn_like(pos)
            else:
                noise = 0.

            if sampling_type == "ddpm":
                # b = self.betas
                b = self.noise_schedule.betas.to(pos.device)
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

            if self.config.debug.save_dynamic and not self.config.debug.save_dynamic_final_only:
                dynamic_graph.update_graph(pos, score=node_eq, t=t)
            else:
                dynamic_graph.update_graph(pos, append=False)

            if torch.isnan(pos).any():
                print("NaN detected. Please restart.")
                raise FloatingPointError()
            pos = center_pos(pos, batch.batch)
            if clip_pos is not None:
                pos = torch.clamp(pos, min=-clip_pos, max=clip_pos)
            # pos_traj.append(pos.clone().cpu())

        if self.config.debug.save_dynamic and self.config.debug.save_dynamic_final_only:
            # Add final position to the trajectory
            dynamic_graph.update_graph(pos, score=node_eq, t=t)

        ## Save trajectory
        traj = torch.stack(dynamic_graph.pos_traj).transpose(0, 1).flip(dims=(1,))  # (N, T, 3)
        samples = []
        for i in range(batch.num_graphs):
            d = batch[i]
            traj_i = traj[(batch.batch == i).to("cpu")]
            d.traj = traj_i
            samples.append(d)

        ## Save dynamic_graph object
        if self.config.debug.save_dynamic:
            # self.dynamic_graph.append(dynamic_graph)
            self.dynamic_graph_list.append(dynamic_graph)
        return samples

    @torch.no_grad()
    def sample_batch_simple(self, batch, stochastic=False, clip=None, num_cycles=1):
        """
        Sample molecular structures with a specified score function.
        """
        # Make DynamicRxnGraph
        graph = self.graph.from_batch(batch)
        # node2graph = batch.batch

        pos = batch.pos[:, -1, :]
        pos = center_pos(pos, batch.batch)
        pos_init = pos.clone()

        t = torch.ones(batch.num_graphs, device=pos.device)
        t -= self.config.sampling.time_margin # (G, )

        dt_base = torch.ones_like(t) / self.config.sampling.sde_steps
        dynamic_graph = self.dynamic_graph.from_graph(graph, pos, pos_init, t)
        dynamic_graph.pos_traj.append(pos.to("cpu"))  # add initial position

        ## Set score function
        if self.config.sampling.score_type == "ddbm_score":
            score_function = self.predict_ddbm_score
        elif self.config.sampling.score_type == "ddbm_h_transform":
            score_function = self.predict_ddbm_h_transform
        elif self.config.sampling.score_type == "cfm":
            score_function = self.predict_cfm
        else:
            raise NotImplementedError()

        print("[sample_batch_simple] start sampling")
        for cycle in range(num_cycles):
            print(f"Debug: cycle: {cycle + 1}/{num_cycles}")
            # Reset time for each cycle
            t = torch.ones(batch.num_graphs, device=pos.device) - self.config.sampling.time_margin  # (G, )
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

                if self.config.debug.save_dynamic and not self.config.debug.save_dynamic_final_only:
                    dynamic_graph.update_graph(pos, score=dx, t=t)
                else:
                    dynamic_graph.update_graph(pos, append=False)

        print("\n[sample_batch_simple] sampling finished")

        if self.config.debug.save_dynamic and self.config.debug.save_dynamic_final_only:
            # Add final position to the trajectory
            dynamic_graph.update_graph(pos, score=dx, t=t)

        ## Save trajectory
        traj = torch.stack(dynamic_graph.pos_traj).transpose(0, 1).flip(dims=(1,))  # (N, T, 3)
        samples = []
        for i in range(batch.num_graphs):
            d = batch[i]
            traj_i = traj[(batch.batch == i).to("cpu")]
            d.traj = traj_i
            samples.append(d)

        ## Save dynamic_graph object
        if self.config.debug.save_dynamic:
            self.dynamic_graph_list.append(dynamic_graph)

        return samples

    @torch.no_grad()
    def sample_batch(self, batch, stochastic=True):
        # Geodesic ODE solve
        graph = self.graph.from_batch(batch)
        node2graph = batch.batch

        pos = batch.pos[:, -1, :]
        pos_init = batch.pos[:, -1, :]

        t = torch.ones(batch.num_graphs, device=pos.device) - self.config.sampling.time_margin # (G, )
        dynamic_graph = self.dynamic_graph.from_graph(graph, pos, pos_init, t)
        dynamic_graph.pos_traj.append(pos.to("cpu"))

        # full_edge, _, _ = dynamic_graph.full_edge(upper_triangle=True)
        full_edge = dynamic_graph.full_edge(upper_triangle=True)[0]
        edge2graph = node2graph.index_select(0, full_edge[0])
        num_nodes = batch.batch.bincount()
        t = torch.ones_like(full_edge[0]) - self.config.sampling.time_margin
        # dt = t * self.config.sampling.sde_dt
        dt = torch.ones_like(t) / self.config.sampling.sde_steps

        while (t > 1e-6).any():
            print(f"t={t[0]}")
            dt = dt.clip(max=t)

            # dq = self.predict_dq(
            #     batch,
            #     dynamic_graph,
            #     pos,
            #     pos_init,
            #     graph,
            #     full_edge,
            #     node2graph,
            #     edge2graph,
            #     num_nodes,
            #     t,
            #     dt,
            #     stochastic,
            #     pred_type=self.config.sampling.score_type,
            #     perr=0.0,
            # )
            dq = self.predict_cfm(
                dynamic_graph,
                dt,
                # debug=True,
                debug=False,
                batch=batch,
            )[1]

            # pos = dynamic_graph.pos

            init, last, iter, index_tensor, stats = self.geodesic_solver.batch_geodesic_ode_solve(
                pos,
                -dq,
                full_edge,
                dynamic_graph.atom_type,
                node2graph,
                num_nodes,
                q_type=self.q_type,
                num_iter=self.config.manifold.ode_solver.iter,
                max_iter=self.config.manifold.ode_solver.max_iter,
                ref_dt=self.config.manifold.ode_solver.ref_dt,
                min_dt=self.config.manifold.ode_solver.min_dt,
                max_dt=self.config.manifold.ode_solver.max_dt,
                err_thresh=self.config.manifold.ode_solver.vpae_thresh,
                verbose=0,
                method="Heun",
                pos_adjust_scaler=self.config.manifold.ode_solver.pos_adjust_scaler,
                pos_adjust_thresh=self.config.manifold.ode_solver.pos_adjust_thresh,
            )

            batch_pos_tm1 = last["x"]  # (B, n, 3)
            unbatch_node_mask = _masking(num_nodes)
            pos_tm1 = batch_pos_tm1.reshape(-1, 3)[unbatch_node_mask]  # (N, 3)

            retry_index = stats["ban_index"].sort().values
            if len(retry_index) > 0:
                node_select = torch.isin(node2graph, retry_index)
                edge_select = torch.isin(edge2graph, retry_index)
                _batch = torch.arange(len(retry_index), device=pos.device).repeat_interleave(num_nodes[retry_index])
                _num_nodes = num_nodes[retry_index]
                _num_edges = _num_nodes * (_num_nodes - 1) // 2
                _ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=_num_nodes.device), _num_nodes.cumsum(0)])
                _full_edge = index_tensor[2:][:, edge_select] + _ptr[:-1].repeat_interleave(_num_edges)

                self.print(f"[Resolve] geodesic solver failed at {len(retry_index)}/{len(batch)}, Retry...")
                _init, _last, _iter, _index_tensor, _stats = self.geodesic_solver.batch_geodesic_ode_solve(
                    pos[node_select],
                    dq[edge_select],
                    _full_edge,
                    dynamic_graph.atom_type[node_select],
                    _batch,
                    _num_nodes,
                    q_type=self.q_type,
                    num_iter=500,
                    max_iter=2000,
                    ref_dt=self.config.manifold.ode_solver._ref_dt,
                    min_dt=self.config.manifold.ode_solver._min_dt,
                    max_dt=self.config.manifold.ode_solver._max_dt,
                    err_thresh=self.config.manifold.ode_solver.vpae_thresh,
                    verbose=0,
                    method="RK4",
                    pos_adjust_scaler=self.config.manifold.ode_solver.pos_adjust_scaler,
                    pos_adjust_thresh=self.config.manifold.ode_solver.pos_adjust_thresh,
                )

                _batch_pos_tm1 = _last["x"]  # (b, n', 3)
                _unbatch_node_mask = _masking(_num_nodes)
                _pos_tm1 = _batch_pos_tm1.reshape(-1, 3)[_unbatch_node_mask]  # (N', 3)

                pos_tm1[node_select] = _pos_tm1

                ban_index = _stats["ban_index"].sort().values
                ban_index = retry_index[ban_index]
            else:
                ban_index = torch.LongTensor([])

            if len(ban_index) > 0:
                print(f"Debug: ban_index={ban_index}")
                # print(f"Debug: batch.rxn_idx={batch.rxn_idx}")
                print(f"Debug: batch.idx={batch.idx}")
                idx = batch.idx[ban_index]
                self.print(f"[Warning] geodesic solver failed solving reaction {idx}, at time {t[ban_index]}\n")
                ban_node_mask = torch.isin(node2graph, ban_index)
                pos_tm1[ban_node_mask] = pos[ban_node_mask] + torch.randn_like(pos[ban_node_mask]) * 1e-3

            t = t - dt
            pos = pos_tm1

            if self.config.debug.save_dynamic and not self.config.debug.save_dynamic_final_only:
                dynamic_graph.update_graph(pos, score=dq, t=t)
            else:
                dynamic_graph.update_graph(pos, append=False)

        if self.config.debug.save_dynamic and self.config.debug.save_dynamic_final_only:
            # Add final position to the trajectory
            dynamic_graph.update_graph(pos, score=dq, t=t)

        traj = torch.stack(dynamic_graph.pos_traj).transpose(0, 1).flip(dims=(1,))  # (N, T, 3)
        samples = []
        for i in range(batch.num_graphs):
            d = batch[i]
            traj_i = traj[(batch.batch == i).to("cpu")]
            d.traj = traj_i
            samples.append(d)

        ## Save dynamic_graph object
        if self.config.debug.save_dynamic:
            self.dynamic_graph_list.append(dynamic_graph)

        return samples

    def get_h_transform(self, pos, pos_init, t, edge_index, atom_type):
        if self.q_type == "DM":
            diff = self.geodesic_solver.compute_d(edge_index, pos_init) - self.geodesic_solver.compute_d(edge_index, pos)
        elif self.q_type == "morse":
            diff = self.geodesic_solver.compute_q(edge_index, atom_type, pos_init) - self.geodesic_solver.compute_q(edge_index, atom_type, pos)
        else:
            raise NotImplementedError

        # t = 1-t
        coeff = self.noise_schedule.get_sigma(torch.ones_like(t)) - self.noise_schedule.get_sigma(t)
        beta = self.noise_schedule.get_beta(t)
        coeff = torch.exp(torch.log(beta) - torch.log(coeff))
        h_transform = diff * coeff
        # return h_transform
        return -h_transform  # TODO: 부호 문제 있을 수 있음.

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Training Start")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()
        if self.local_rank == 0:
            setup_wandb(self.config)

    def noise_level_sampling(self, data):
        g_length = data.geodesic_length[:, 1: -1]  # (G, T-1)
        g_last_length = data.geodesic_length[:, -1:]
        SNR_ratio = g_length / g_last_length  # (G, T-1) SNR_ratio = SNR(1)/SNR(t)
        t = self.noise_schedule.get_time_from_SNRratio(SNR_ratio)  # (G, T-1)
        random_t = torch.rand(size=(t.size(0), 1), device=SNR_ratio.device)
        diff = abs(t - random_t)
        index = torch.argmin(diff, dim=1)

        t = t[(torch.arange(t.size(0)).to(index), index)]
        print(f"Debug: t={t}")
        sampled_SNR_ratio = SNR_ratio[(torch.arange(SNR_ratio.size(0)).to(index), index)]
        return index, t, sampled_SNR_ratio

    def apply_noise(self, data):
        graph = self.graph.from_batch(data)
        edge_index = graph.full_edge(upper_triangle=True)[0]

        node2graph = graph.batch
        edge2graph = node2graph.index_select(0, edge_index[0])
        num_nodes = data.ptr[1:] - data.ptr[:-1]

        # sampling time step
        t_index, tt, SNR_ratio = self.noise_level_sampling(data)  # (G, ), (G, ), (G, )

        t_index_node = t_index.index_select(0, node2graph)  # (N, )
        mean = data.pos[(torch.arange(len(t_index_node)), t_index_node)]  # (N, 3)
        pos_init = data.pos[:, -1]

        # sigma = SNR_ratio * self.noise_schedule.get_sigma(torch.ones_like(SNR_ratio))
        # sigma_hat = sigma * (1 - SNR_ratio)  # (G, )
        # NOTE : The above two is equivalent to
        sigma_hat = self.noise_schedule.get_sigma_hat(tt)  # (G, )
        sigma_hat_edge = sigma_hat.index_select(0, edge2graph)  # (E, )
        noise = torch.randn(size=(edge_index.size(1),), device=edge_index.device) * sigma_hat_edge.sqrt()  # dq, (E, )

        # apply noise
        init, last, iter, index_tensor, stats = self.geodesic_solver.batch_geodesic_ode_solve(
            mean,
            noise,
            edge_index,
            graph.atom_type,
            node2graph,
            num_nodes,
            q_type=self.q_type,
            num_iter=self.config.manifold.ode_solver.iter,
            max_iter=self.config.manifold.ode_solver.max_iter,
            ref_dt=self.config.manifold.ode_solver.ref_dt,
            min_dt=self.config.manifold.ode_solver.min_dt,
            max_dt=self.config.manifold.ode_solver.max_dt,
            err_thresh=self.config.manifold.ode_solver.vpae_thresh,
            verbose=0,
            method="Heun",
            pos_adjust_scaler=self.config.manifold.ode_solver.pos_adjust_scaler,
            pos_adjust_thresh=self.config.manifold.ode_solver.pos_adjust_thresh,
        )

        batch_pos_noise = last["x"]  # (B, n, 3)
        batch_x_dot = last["x_dot"]  # (B, n, 3)
        unbatch_node_mask = _masking(num_nodes)
        pos_noise = batch_pos_noise.reshape(-1, 3)[unbatch_node_mask]  # (N, 3)
        x_dot = batch_x_dot.reshape(-1, 3)[unbatch_node_mask]  # (N, 3)

        batch_q_dot = last["q_dot"]  # (B, e)
        # batch_q = last["q"]  # (B, e) # NOTE : for debugging
        batch_q_init = init["q"]  # (B, e)
        e = batch_q_dot.size(1)
        unbatch_edge_index = index_tensor[1] + index_tensor[0] * e
        q_dot = batch_q_dot.reshape(-1)[unbatch_edge_index]  # (E, )
        # q = batch_q.reshape(-1)[unbatch_edge_index]  # (E, ) # NOTE : for debugging
        q_init = batch_q_init.reshape(-1)[unbatch_edge_index]  # (E, )

        # Check stability, percent error > threshold, then re-solve
        retry_index = stats["ban_index"].sort().values
        if len(retry_index) > 0:
            node_select = torch.isin(node2graph, retry_index)
            edge_select = torch.isin(edge2graph, retry_index)
            _batch = torch.arange(len(retry_index), device=mean.device).repeat_interleave(num_nodes[retry_index])
            _num_nodes = num_nodes[retry_index]
            _num_edges = _num_nodes * (_num_nodes - 1) // 2
            _ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=_num_nodes.device), _num_nodes.cumsum(0)])
            _full_edge = index_tensor[2:][:, edge_select] + _ptr[:-1].repeat_interleave(_num_edges)

            self.print(f"[Resolve] geodesic solver failed at {len(retry_index)}/{len(data)}, Retry...")
            _init, _last, _iter, _index_tensor, _stats = self.geodesic_solver.batch_geodesic_ode_solve(
                mean[node_select],
                noise[edge_select],
                _full_edge,
                graph.atom_type[node_select],
                _batch,
                _num_nodes,
                q_type=self.q_type,
                num_iter=self.config.manifold.ode_solver.iter,
                max_iter=self.config.manifold.ode_solver.max_iter,
                ref_dt=self.config.manifold.ode_solver._ref_dt,
                min_dt=self.config.manifold.ode_solver._min_dt,
                max_dt=self.config.manifold.ode_solver._max_dt,
                err_thresh=self.config.manifold.ode_solver.vpae_thresh,
                verbose=0,
                method="RK4",
                pos_adjust_scaler=self.config.manifold.ode_solver.pos_adjust_scaler,
                pos_adjust_thresh=self.config.manifold.ode_solver.pos_adjust_thresh,
            )

            _batch_pos_noise = _last["x"]  # (b, n', 3)
            _batch_x_dot = _last["x_dot"]  # (b, n', 3)
            _unbatch_node_mask = _masking(_num_nodes)
            _pos_noise = _batch_pos_noise.reshape(-1, 3)[_unbatch_node_mask]  # (N', 3)
            _x_dot = _batch_x_dot.reshape(-1, 3)[_unbatch_node_mask]  # (N', 3)

            _batch_q_dot = _last["q_dot"]  # (b, e')
            # _batch_q = _last["q"]  # (b, e') # NOTE : for debugging
            _e = _batch_q_dot.size(1)
            _unbatch_edge_index = _index_tensor[1] + _index_tensor[0] * _e
            _q_dot = _batch_q_dot.reshape(-1)[_unbatch_edge_index]  # (E', )
            # _q = _batch_q.reshape(-1)[_unbatch_edge_index]  # (E', ) # NOTE : for debugging

            pos_noise[node_select] = _pos_noise
            x_dot[node_select] = _x_dot
            q_dot[edge_select] = _q_dot
            # q[edge_select] = _q  # NOTE : for debugging

            ban_index = _stats["ban_index"].sort().values
            ban_index = retry_index[ban_index]

        else:
            ban_index = torch.LongTensor([])

        # beta = self.noise_schedule.get_beta(tt)
        # coeff = beta / sigma_hat
        # coeff = 1 / sigma_hat
        coeff = torch.ones_like(tt)
        coeff_node = coeff.index_select(0, node2graph)  # (N, )
        coeff_edge = coeff.index_select(0, edge2graph)  # (E, )
        # target is not exactly the score function.
        # target = beta * score
        target_x = - x_dot * coeff_node.unsqueeze(-1)
        target_q = - q_dot * coeff_edge

        # target_q = (q_init - q) * coeff_edge  # NOTE : for debugging

        ##############################################################
        print(f"Debug: Target is set to straight vector from x_t to x_0 in apply_noise()")
        x_0 = data.pos[:, 0]
        x_dot = pos_noise - x_0
        q_dot = self.geodesic_solver.batch_dx2dq(
            x_dot,
            mean,
            graph.atom_type,
            edge_index,
            node2graph, num_nodes,
            q_type=self.q_type
        )

        target_x = - x_dot * coeff_node.unsqueeze(-1)
        target_q = - q_dot * coeff_edge
        ##############################################################


        if len(ban_index) > 0:
            ban_index = ban_index.to(torch.long)
            # rxn_idx = [data.rxn_idx[i] for i in ban_index]
            idx = [data.idx[i] for i in ban_index]
            self.print(f"\n[Warning] geodesic solver failed at {len(ban_index)}/{len(data)}"
                        f"\n\tidx: {idx}\n\ttime index: {t_index[ban_index].tolist()}")
                        # f"\n\trxn_idx: {rxn_idx}\n\ttime index: {t_index[ban_index].tolist()}")
            ban_node_mask = torch.isin(node2graph, ban_index)
            ban_edge_mask = torch.isin(edge2graph, ban_index)
            ban_batch_mask = torch.isin(torch.arange(len(data), device=mean.device), ban_index)

            data = Batch.from_data_list(data[~ban_batch_mask])
            graph = self.graph.from_batch(data)

            pos_noise = pos_noise[~ban_node_mask]
            x_dot = x_dot[~ban_node_mask]
            pos_init = pos_init[~ban_node_mask]
            tt = tt[~ban_batch_mask]
            target_x = target_x[~ban_node_mask]
            target_q = target_q[~ban_edge_mask]

        return graph, pos_noise, pos_init, tt, target_x, target_q

    @torch.no_grad()
    def predict_dq(
        self,
        batch,
        dynamic_graph,
        pos,
        pos_init,
        graph,
        full_edge,
        node2graph,
        edge2graph,
        num_nodes,
        t,
        dt,
        stochastic=True,
        pred_type="ddbm",
        perr=0.0,
    ):
        # assert pred_type in ["ddbm", "h_transform", "cfm"]
        beta = None

        ## Predict score term
        if pred_type == "ddbm":
            _, score, _, _, _= self.forward(dynamic_graph)

            sigma_hat = self.noise_schedule.get_sigma_hat(t)
            beta = self.noise_schedule.get_beta(t)
            coeff = torch.exp(torch.log(beta) - torch.log(sigma_hat))
            score *= coeff
        else:
            score = 0; print(f"Debug: score is set to 0")

        # ## Predict score term
        # if pred_type == "ddbm":
        #     score = self.forward(dynamic_graph.to("cuda"))
        #     if self.pred_type == "node":
        #         score = self.geodesic_solver.batch_dx2dq(score, pos, graph.atom_type, full_edge, node2graph, num_nodes, q_type=self.q_type).reshape(-1)
        #     elif self.pred_type == "edge":
        #         dq_dd = self.geodesic_solver.dq_dd(pos, dynamic_graph.atom_type, full_edge, q_type=self.q_type)
        #         if self.q_type == "morse":
        #             score = dq_dd * score
        #         if self.projection:
        #             score = self.geodesic_solver.batch_projection(score, pos, dynamic_graph.atom_type, full_edge, dynamic_graph.batch, num_nodes, q_type=self.q_type, proj_type="manifold")

        #     sigma_hat = self.noise_schedule.get_sigma_hat(t)
        #     beta = self.noise_schedule.get_beta(t)
        #     coeff = torch.exp(torch.log(beta) - torch.log(sigma_hat))
        #     score *= coeff
        # else:
        #     score = 0; print(f"Debug: score is set to 0")

        ## Predict h term
        if pred_type == "ddbm":
            h = self.get_h_transform(pos, pos_init, t, full_edge, dynamic_graph.atom_type)
            # h = 0.0; print(f"Debug: h_transform set to zero.")
        elif pred_type in ["h_transform", "cfm"]:
            h_pred= -self.forward(dynamic_graph.to("cuda")); print(f"Debug: h is calculated using NeuralNet")
            if self.pred_type == "node":
                if self.projection:
                    h_pred = self.geodesic_solver.batch_projection(h_pred, pos, dynamic_graph.atom_type, full_edge, dynamic_graph.batch, num_nodes, q_type=self.q_type, proj_type="euclidean")
                h_pred = self.geodesic_solver.batch_dx2dq(h_pred, pos, graph.atom_type, full_edge, node2graph, num_nodes, q_type=self.q_type).reshape(-1)
            elif self.pred_type == "edge":
                if self.q_type == "morse":
                    dq_dd = self.geodesic_solver.dq_dd(pos, dynamic_graph.atom_type, full_edge, q_type=self.q_type)
                    h_pred = dq_dd * h_pred
                if self.projection:
                    h_pred = self.geodesic_solver.batch_projection(h_pred, pos, dynamic_graph.atom_type, full_edge, dynamic_graph.batch, num_nodes, q_type=self.q_type, proj_type="manifold")
            else:
                raise NotImplementedError

            # Reference h term (for debugging)
            if pred_type == "h_transform":
                # Note: h_coeff = beta_t / (sigma_square_T - sigma_square_t)
                # Note: forward sampling에 대해서, h_transform을 학습시켰기 때문에 t <- 1 - t로 처리함.
                # Note: beta(t) sampling이 t=0.5에 대해서 대칭적이지 않으면 문제있을 수 있음.
                h_coeff = self.noise_schedule.get_sigma(torch.ones_like(1-t)) - self.noise_schedule.get_sigma(1-t)
                beta = self.noise_schedule.get_beta(1-t)
                h_coeff = torch.exp(torch.log(beta) - torch.log(h_coeff))

                # h_ref = self.get_h_transform(pos, batch.pos[:, 0, :], t, full_edge, dynamic_graph.atom_type); print(f"Debug: h is set to h_transform to x0")
                h_ref = self.get_h_transform(pos, batch.pos[:, 0, :], 1-t, full_edge, dynamic_graph.atom_type); print(f"Debug: h is set to h_transform to x0")
                if self.projection:
                    h_ref = self.geodesic_solver.batch_projection(h_ref, pos, dynamic_graph.atom_type, full_edge, dynamic_graph.batch, num_nodes, q_type=self.q_type, proj_type="manifold")
                # Note: h_ref = (xt - x0) * h_coeff

                # h_pred *= h_coeff
                h_pred *= self.noise_schedule.get_beta(1-t) / (self.noise_schedule.get_sigma(1-t) - self.noise_schedule.get_sigma(torch.zeros_like(t)))  # TEST: TSDiff sampling
            elif pred_type == "cfm":
                h_coeff = 1.0
                h_ref = self.geodesic_solver.batch_dx2dq(
                    batch.pos[:, -1, :] - batch.pos[:, 0, :],  # flow matching
                    pos,
                    dynamic_graph.atom_type,
                    full_edge,
                    node2graph, num_nodes,
                    q_type=self.q_type
                )
                if self.projection:
                    h_ref = self.geodesic_solver.batch_projection(h_ref, pos, dynamic_graph.atom_type, full_edge, dynamic_graph.batch, num_nodes, q_type=self.q_type, proj_type="manifold")
            else:
                raise NotImplementedError


            ## Print h-term error
            h_diff = (h_pred - h_ref) / h_coeff
            h_diff_norm = scatter_sum(h_diff.square(), edge2graph).sqrt()
            print(f"Debug: h_diff_norm={h_diff_norm}")
            # print(f"Debug: h_ref_norm(scaled)={scatter_sum((h_ref/h_coeff).square(), edge2graph).sqrt()}")
            h_ref_norm = scatter_sum((h_ref/h_coeff).square(), edge2graph).sqrt()
            print(f"Debug: h_ref_norm(scaled)={h_ref_norm}")
            perr_norm = h_diff_norm / h_ref_norm
            print(f"Debug: perr_norm={perr_norm}")
            print(f"Debug: perr_norm.mean()={perr_norm.mean()}")

            ## Add noise to h_ref
            ## NeuralNet이 가지는 perr을 가장하여 만듦.
            if perr:
                # perr = 1.0; print(f"Debug: perr={perr}")
                # perr = 0.0; print(f"Debug: perr={perr}")
                print(f"Debug: perr={perr}")
                h_ref_norm = scatter_sum(h_ref.square(), edge2graph).sqrt()
                noise_to_ref = torch.randn_like(h_ref)
                noise_to_ref_norm = scatter_sum(noise_to_ref.square(), edge2graph).sqrt()
                noise_to_ref *= h_ref_norm.mean() / noise_to_ref_norm.mean()
                noise_to_ref *= perr
                # noise_to_ref_norm = scatter_sum(noise_to_ref.square(), edge2graph).sqrt()
                # print(f"Debug: noise_to_ref_norm={noise_to_ref_norm}")
                noise_to_ref_norm = scatter_sum((noise_to_ref/h_coeff).square(), edge2graph).sqrt()
                print(f"Debug: noise_to_ref_norm={noise_to_ref_norm}")
                # print(f"Debug: h_ref=\n{h_ref / h_coeff}")
                h_ref += noise_to_ref

            # h = h_ref; print(f"Debug: h is set to h_ref")
            h = h_pred; print(f"Debug: h is set to h_pred")
            # if (t > 0.9).any():
            # # if (t > 0.8).any():
            #     print(f"Debug: h_ref is set to h")
            #     h = h_ref
            # else:
            #     print(f"Debug: h_pred is set to h")
            #     h = h_pred
            # print(f"Debug: h_ref=\n{h_ref / h_coeff}")
            # print(f"Debug: h_pred=\n{h_pred / h_coeff}")

        if stochastic:
            # dw = torch.sqrt(beta * dt) * torch.randn_like(score)
            dw = torch.sqrt(beta * dt) * torch.randn_like(h)
            # dt = dt * 100/70; print(f"Debug: dt * 100/70")
            # dw = 0; print(f"Debug: dw is set to 0")
            dq = - (score - h) * dt + dw
        else:
            dq = - (0.5 * score - h) * dt
        return dq


def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center


def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom


def align_geom_batch(refgeom, geom, batch):
    refgeom_list = refgeom.split(list(batch.bincount()))
    geom_list = geom.split(list(batch.bincount()))

    aligned_geom_list = []
    aligned_rmsd = []
    for g1, g2 in zip(refgeom_list, geom_list):
        rmsd, new_g2 = align_geom(g1, g2)
        aligned_geom_list.append(new_g2)
        aligned_rmsd.append(rmsd)
    return torch.cat(aligned_geom_list)


def align_geom(refgeom, geom):
    center = refgeom.mean(dim=0)
    ref2 = refgeom - center
    geom2 = geom - geom.mean(dim=0)
    cov = geom2.T @ ref2
    v, sv, w = torch.linalg.svd(cov)
    if torch.linalg.det(v) * torch.linalg.det(w) < 0:
        sv[-1] = -sv[-1]
        v[:, -1] = -v[:, -1]
    u = v @ w
    new_geom = geom2 @ u + center
    rmsd = (new_geom - refgeom).square().mean().sqrt()
    return rmsd, new_geom
