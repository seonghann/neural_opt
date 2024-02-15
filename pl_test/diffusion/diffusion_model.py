import wandb
import torch
import pytorch_lightning as pl
import time

import os.path as osp

from model.condensed_encoder import CondenseEncoderEpsNetwork
from utils.wandb_utils import setup_wandb
from utils.rxn_graph import RxnGraph, DynamicRxnGraph
from utils.geodesic_solver import GeodesicSolver
from diffusion.noise_scheduler import load_noise_scheduler
from metrics.metrics import LossFunction, SamplingMetrics, TrainMetrics, ValidMetrics


class BridgeDiffusion(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.name is used for logging, add time to avoid name conflict
        # config.general.name  +  dd-mm-yy:hh-mm-ss
        self.name = config.general.name + time.strftime(":%d-%m-%y:%H-%M-%S")
        self.NeuralNet = CondenseEncoderEpsNetwork(config.model)
        self.noise_schedule = load_noise_scheduler(config.diffusion)
        self.rxn_graph = RxnGraph
        self.dynamic_rxn_graph = DynamicRxnGraph
        self.geodesic_solver = GeodesicSolver(config.manifold)

        self.train_loss = LossFunction(config.train.lambda_train)
        self.train_metrics = TrainMetrics(name='train')  # TODO: define metrics
        self.valid_loss = LossFunction(config.train.lambda_valid)
        self.valid_metrics = ValidMetrics(self.geodesic_solver, name='valid')  # TODO: define metrics
        self.test_loss = LossFunction(config.train.lambda_valid)
        self.test_metrics = ValidMetrics(self.geodesic_solver, name='test')  # TODO: define metrics
        self.valid_sampling_metrics = SamplingMetrics(self.geodesic_solver, name='valid')
        self.test_sampling_metrics = SamplingMetrics(self.geodesic_solver, name='test')

        self.solver_threshold = config.manifold.ode_solver.accuracy_threshold
        self.save_dir = config.debug.save_dir
        self.best_valid_loss = 1e9

        self.val_counter = 0
        self.test_counter = 0

    def forward(self, noisy_rxn_graph):
        return self.NeuralNet(noisy_rxn_graph).squeeze()

    def training_step(self, data, i):
        rxn_graph, pos, pos_init, tt, target_x, target_q = self.apply_noise(data)
        noisy_rxn_graph = self.dynamic_rxn_graph.from_graph(rxn_graph, pos, pos_init, tt)

        full_edge, _, _ = noisy_rxn_graph.full_edge(upper_triangle=True)
        node2graph = noisy_rxn_graph.batch
        edge2graph = node2graph.index_select(0, full_edge[0])

        pred_q = self.forward(noisy_rxn_graph)
        pred_x = self.geodesic_solver.dq2dx(pred_q, pos, full_edge, rxn_graph.atom_type).reshape(-1, 3)

        loss = self.train_loss(
            pred_x=pred_x,
            pred_q=pred_q,
            true_x=target_x,
            true_q=target_q,
            merge_edge=edge2graph,
            merge_node=node2graph,
        )

        self.train_metrics(
            pred_x,
            pred_q,
            target_x,
            target_q,
            edge2graph,
            node2graph,
            rxn_graph.atom_type,
            rxn_graph.edge_feat_r,
            rxn_graph.edge_feat_p,
            log=True
        )
        return {"loss": loss}

    def noise_level_sampling(self, data):
        g_length = data.geodesic_length[:, 1:]  # (G, T-1)
        g_last_length = data.geodesic_length[:, -1:]
        SNR_ratio = g_length / g_last_length  # (G, T-1) SNR_ratio = SNR(1)/SNR(t)
        t = self.noise_schedule.get_time_from_SNRratio(SNR_ratio)  # (G, T-1)

        random_t = torch.rand(size=(t.size(0), 1), device=SNR_ratio.device)
        diff = abs(t - random_t)
        index = torch.argmin(diff, dim=1)

        t = t[(torch.arange(t.size(0)).to(index), index)]
        sampled_SNR_ratio = SNR_ratio[(torch.arange(SNR_ratio.size(0)).to(index), index)]
        return index, t, sampled_SNR_ratio

    def apply_noise(self, data):
        graph = self.rxn_graph.from_batch(data)
        full_edge, _, _ = graph.full_edge(upper_triangle=True)

        node2graph = graph.batch
        edge2graph = node2graph.index_select(0, full_edge[0])

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
        noise = torch.randn(size=(full_edge.size(1),), device=full_edge.device) * sigma_hat_edge  # dq, (E, )

        # apply noise
        pos_noise, x_dot, iter, total_dq, expected_dq = self.geodesic_solver.geodesic_ode_solve(
            mean,
            noise,
            full_edge,
            graph.atom_type,
        )

        # Check stability, percent error > threshold, then re-solve
        p_err = (abs(total_dq - expected_dq) / expected_dq) * 100
        if p_err > self.solver_threshold:
            self.print(f"[Warnning] geodesic solver percent error {p_err:0.2f}% > tol ({self.solver_threshold}%), Retry...")
            pos_noise, x_dot, iter, total_dq, expected_dq = self.geodesic_solver.geodesic_ode_solve(
                mean,
                noise,
                full_edge,
                graph.atom_type,
                num_iter=1000,
                ref_dt=5e-3,
                max_dt=5e-2
            )
            p_err = abs(total_dq - expected_dq) / expected_dq
            if p_err > self.solver_threshold:
                self.print(f"[Warnning] geodesic solver percent error {p_err:0.2f}% > tol ({self.solver_threshold}%), Terminate...")
                save_ = {
                    "mu": mean,
                    "noise": noise,
                    "edge": full_edge,
                    "atom_type": graph.atom_type
                }
                f = osp.join(self.save_dir, "error_batch.pt")
                torch.save(save_, f)
                # f = osp.join(self.save_dir, "abnormal_termination.pt")
                # self.save_model(f)
                raise ValueError("Unexpectedly inaccurate geodesic path.")

        beta = self.noise_schedule.get_beta(tt)
        coeff = beta / sigma_hat
        coeff_node = coeff.index_select(0, node2graph)  # (N, )
        # target is not exactly the score function.
        # target = beta * score
        target_x = - x_dot * coeff_node.unsqueeze(-1)
        target_q = self.geodesic_solver.dx2dq(target_x, pos_noise, full_edge, graph.atom_type)

        return graph, pos_noise, pos_init, tt, target_x, target_q

    def configure_optimizers(self):
        # set optimizer
        return torch.optim.AdamW(self.parameters(), lr=self.config.train.lr, amsgrad=True,
                                 weight_decay=self.config.train.weight_decay)

    def on_train_epoch_start(self) -> None:
        self.print("Start Training Epoch ...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        if self.train_metrics is not None:
            self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        metric_x, metric_q = self.train_loss.log_epoch_metrics()
        loss = metric_q + self.config.train.lambda_train * metric_x
        to_log = {"train/rmsd": metric_x, "train/norm": metric_q, "train/loss": loss}
        self.print(f"Epoch {self.current_epoch} Training Loss: {to_log['train/loss']: 0.3f}"
                   f" -- {time.time() - self.start_epoch_time:0.1f}s")
        print(torch.cuda.memory_summary())

    def on_validation_epoch_start(self) -> None:
        self.print("Starting validation ...")
        self.valid_loss.reset()
        self.valid_metrics.reset()
        self.valid_sampling_metrics.reset()

    def validation_step(self, data, i):
        rxn_graph, pos, pos_init, tt, target_x, target_q = self.apply_noise(data)
        noisy_rxn_graph = self.dynamic_rxn_graph.from_graph(rxn_graph, pos, pos_init, tt)

        full_edge, _, _ = noisy_rxn_graph.full_edge(upper_triangle=True)
        node2graph = noisy_rxn_graph.batch
        edge2graph = node2graph.index_select(0, full_edge[0])

        pred_q = self.forward(noisy_rxn_graph)
        pred_x = self.geodesic_solver.dq2dx(pred_q, pos, full_edge, rxn_graph.atom_type).reshape(-1, 3)

        loss = self.valid_loss(
            pred_x=pred_x,
            pred_q=pred_q,
            true_x=target_x,
            true_q=target_q,
            merge_edge=edge2graph,
            merge_node=node2graph,
        )
        self.valid_metrics(
            pred_x,
            pred_q,
            target_x,
            target_q,
            edge2graph,
            node2graph,
            rxn_graph.atom_type,
            rxn_graph.edge_feat_r,
            rxn_graph.edge_feat_p,
            edge_index=full_edge,
            pos=pos,
            log=True
        )
        return {"loss": loss}

    def on_validation_epoch_end(self) -> None:
        metric_x, metric_q, metric_proj = self.valid_metrics.log_epoch_metrics()
        loss = metric_q + self.config.train.lambda_valid * metric_x

        if loss < self.best_valid_loss:
            self.best_valid_loss = loss

        self.val_counter += 1
        self.print(f"Epoch {self.current_epoch} Validation Loss: {loss: 0.3f} Best loss: {self.best_valid_loss}")
        if (self.val_counter) % self.config.train.sample_every_n_valid == 0:
            start = time.time()
            samples = []

            for i, batch in enumerate(self.trainer.datamodule.val_dataloader()):
                if i % self.config.train.sample_every_n_batch == 0:
                    batch = batch.to(self.device)
                    batch_out = self.sample_batch(batch)
                    samples.extend(batch_out)

            self.valid_sampling_metrics(
                samples, self.name,
                self.current_epoch,
                valid_counter=-1,
                test=True,
                local_rank=self.local_rank
            )
            self.print(f"Done. Sampling took {time.time() - start:0.1f}s")
        print("Test epoch end ends...")

    def on_test_epoch_start(self) -> None:
        self.print("Starting test ...")
        self.test_loss.reset()
        self.test_metrics.reset()
        self.test_sampling_metrics.reset()
        if self.local_rank == 0:
            setup_wandb(self.config)

    def test_step(self, data, i):
        rxn_graph, pos, pos_init, tt, target_x, target_q = self.apply_noise(data)
        noisy_rxn_graph = self.dynamic_rxn_graph.from_graph(rxn_graph, pos, pos_init, tt)

        full_edge, _, _ = noisy_rxn_graph.full_edge(upper_triangle=True)
        node2graph = noisy_rxn_graph.batch
        edge2graph = node2graph.index_select(0, full_edge[0])

        pred_q = self.forward(noisy_rxn_graph)
        pred_x = self.geodesic_solver.dq2dx(pred_q, pos, full_edge, rxn_graph.atom_type).reshape(-1, 3)

        loss = self.test_loss(
            pred_x=pred_x,
            pred_q=pred_q,
            true_x=target_x,
            true_q=target_q,
            merge_edge=edge2graph,
            merge_node=node2graph,
        )
        self.test_metrics(
            pred_x,
            pred_q,
            target_x,
            target_q,
            edge2graph,
            node2graph,
            rxn_graph.atom_type,
            rxn_graph.edge_feat_r,
            rxn_graph.edge_feat_p,
            log=True
        )

        return {'loss': loss}

    def on_test_epoch_end(self) -> None:
        metric_x, metric_q, metric_proj = self.test_metrics.log_epoch_metrics()
        loss = metric_q + self.config.train.lambda_valid * metric_x
        print(torch.cuda.memory_summary())
        self.print(f"Epoch {self.current_epoch} Test Loss: {loss: 0.3f}")
        self.test_counter += 1
        if self.test_counter % self.config.train.sample_every_n_valid == 0:
            start = time.time()
            stochastic = self.config.general.sampling_stochastic == 'sde'
            samples = []
            for i, batch in enumerate(self.trainer.datamodule.test_dataloader()):
                if i % self.config.train.sample_every_n_batch == 0:
                    batch = batch.to(self.device)
                    batch_out = self.sample_batch(batch, stochastic=stochastic)
                    samples.extend(batch_out)

            self.test_sampling_metrics(samples, self.name, self.current_epoch, valid_counter=-1, test=True, local_rank=self.local_rank)
            self.print(f"Done. Sampling took {time.time() - start:0.1f}s")
        print("Test epoch end ends...")

    @torch.no_grad()
    def sample_batch(self, batch, stochastic=True):
        rxn_graph = self.rxn_graph.from_batch(batch)
        pos = batch.pos[:, -1, :]
        pos_init = batch.pos[:, -1, :]
        t = torch.ones(batch.num_graphs, device=pos.device)
        dynamic_rxn_graph = self.dynamic_rxn_graph.from_graph(rxn_graph, pos, pos_init, t)
        # dynamic_rxn_graph = dynamic_rxn_graph.to(pos.device)

        full_edge, _, _ = dynamic_rxn_graph.full_edge(upper_triangle=True)
        t = torch.ones_like(full_edge[0])
        dt = self.config.sampling.sde_dt

        while (t > 0).any():
            score = self.forward(dynamic_rxn_graph)
            # score already contains beta
            beta = self.noise_schedule.get_beta(t)
            if stochastic:
                # debug, check all variables' shape
                dw = torch.sqrt(beta * dt) * torch.randn_like(score)
                dq = score * dt + dw
            else:
                dq = 0.5 * score * dt

            pos = dynamic_rxn_graph.pos
            pos_tm1, pos_dot, iter, total_dq, q_dotnorm = self.geodesic_solver.geodesic_ode_solve(
                pos,
                dq,
                full_edge,
                dynamic_rxn_graph.atom_type,
                num_iter=self.config.sampling.gode_iter,
                ref_dt=self.config.sampling.gode_ref_dt,
                max_dt=self.config.sampling.gode_max_dt
            )

            p_err = abs(total_dq - q_dotnorm) / q_dotnorm
            if p_err > self.solver_threshold:
                pos_tm1, pos_dot, iter, total_dq, q_dotnorm = self.geodesic_solver.geodesic_ode_solve(
                    pos,
                    dq,
                    full_edge,
                    dynamic_rxn_graph.atom_type,
                    num_iter=1000,
                    ref_dt=5e-3,
                    max_dt=5e-2
                )
                p_err = abs(total_dq - q_dotnorm) / q_dotnorm
                if p_err > self.solver_threshold:
                    raise ValueError("Unexpectedly inaccurate geodesic path.")

            if (t < dt).any():
                dt = t

            t = t - dt
            pos = pos_tm1
            dynamic_rxn_graph.update_graph(pos, batch.batch, score=score, t=t)

        traj = torch.stack(dynamic_rxn_graph.pos_traj).transpose(0, 1).flip(dims=(1,))  # (N, T, 3)
        samples = []
        for i in range(batch.num_graphs):
            d = batch[i]
            traj_i = traj[(batch.batch == i).to("cpu")]
            d.traj = traj_i
            samples.append(d)

        return samples

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Training Start")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()
        setup_wandb(self.config)
