import torch
import pytorch_lightning as pl
import time

import os.path as osp

from model.condensed_encoder import CondensedEncoderEpsNetwork
from model.equivariant_encoder import EquivariantEncoderEpsNetwork
from utils.wandb_utils import setup_wandb
from utils.rxn_graph import RxnGraph, DynamicRxnGraph
from utils.geodesic_solver import GeodesicSolver
from diffusion.noise_scheduler import load_noise_scheduler
from metrics.metrics import LossFunction, SamplingMetrics, TrainMetrics, ValidMetrics

from torch_geometric.data import Batch

torch.set_num_threads(8)


def _masking(num_nodes):
    N = num_nodes.max()
    mask = torch.BoolTensor([True, False]).repeat(len(num_nodes)).to(num_nodes.device)
    num_repeats = torch.stack([num_nodes, N - num_nodes]).T.flatten()
    mask = mask.repeat_interleave(num_repeats)
    return mask


class BridgeDiffusion(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.name is used for logging, add time to avoid name conflict
        # config.general.name  +  dd-mm-yy:hh-mm-ss
        self.name = config.general.name + time.strftime(":%d-%m-%y:%H-%M-%S")

        if config.model.name == "equivariant":
            self.NeuralNet = EquivariantEncoderEpsNetwork(config.model)
        elif config.model.name == "condensed":
            self.NeuralNet = CondensedEncoderEpsNetwork(config.model)

        self.noise_schedule = load_noise_scheduler(config.diffusion)
        self.rxn_graph = RxnGraph
        self.dynamic_rxn_graph = DynamicRxnGraph
        self.geodesic_solver = GeodesicSolver(config.manifold)

        self.train_loss = LossFunction(lambda_x=config.train.lambda_x_train, lambda_q=config.train.lambda_q_train, name='train')
        self.train_metrics = TrainMetrics(name='train')  # TODO: define metrics
        self.valid_loss = LossFunction(lambda_x=config.train.lambda_x_valid, lambda_q=config.train.lambda_q_valid, name='valid')
        self.valid_metrics = ValidMetrics(self.geodesic_solver, name='valid', lambda_x=config.train.lambda_x_valid, lambda_q=config.train.lambda_q_valid)  # TODO: define metrics
        self.test_loss = LossFunction(lambda_x=config.train.lambda_x_valid, lambda_q=config.train.lambda_q_valid, name='test')
        self.test_metrics = ValidMetrics(self.geodesic_solver, name='test', lambda_x=config.train.lambda_x_valid, lambda_q=config.train.lambda_q_valid)  # TODO: define metrics
        self.valid_sampling_metrics = SamplingMetrics(self.geodesic_solver, name='valid')
        self.test_sampling_metrics = SamplingMetrics(self.geodesic_solver, name='test')

        self.projection = self.config.train.projection
        self.pred_type = self.config.model.pred_type
        self.q_type = self.config.manifold.ode_solver.q_type

        assert self.pred_type in ["edge", "node"], f"pred_type should be 'edge' or 'node', not {self.pred_type}"

        self.solver_threshold = config.manifold.ode_solver.vpae_thresh
        self.save_dir = config.debug.save_dir
        self.best_valid_loss = 1e9

        self.val_counter = 0
        self.test_counter = 0

    def forward(self, noisy_rxn_graph):
        print(f"Debug: self.optim.lr = {self.optim.param_groups[0]['lr']}")
        return self.NeuralNet(noisy_rxn_graph).squeeze()

    def training_step(self, data, i):
        print("===================================================")
        print("===================================================")
        print("===================================================")

        if self.config.train.noise_type == "manifold":
            rxn_graph, pos, pos_init, tt, target_x, target_q = self.apply_noise(data)
        elif self.config.train.noise_type == "euclidean":
            rxn_graph, pos, pos_init, tt, target_x, target_q = self.apply_noise_pos(data)

        noisy_rxn_graph = self.dynamic_rxn_graph.from_graph(rxn_graph, pos, pos_init, tt)

        full_edge, _, _ = noisy_rxn_graph.full_edge(upper_triangle=True)
        node2graph = noisy_rxn_graph.batch
        edge2graph = node2graph.index_select(0, full_edge[0])
        num_nodes = node2graph.bincount()

        if self.pred_type == "node":
            pred_x = self.forward(noisy_rxn_graph)
            if self.projection:
                pred_x = self.geodesic_solver.batch_projection(pred_x, pos, noisy_rxn_graph.atom_type, full_edge, node2graph, num_nodes, q_type=self.q_type, proj_type="euclidean")
            pred_q = self.geodesic_solver.batch_dx2dq(pred_x, pos, noisy_rxn_graph.atom_type, full_edge, node2graph, num_nodes, q_type=self.q_type).reshape(-1)

        elif self.pred_type == "edge":
            pred_q = self.forward(noisy_rxn_graph)
            dq_dd = self.geodesic_solver.dq_dd(pos, noisy_rxn_graph.atom_type, full_edge, q_type=self.q_type)
            if self.q_type == "morse":
                pred_q = dq_dd * pred_q
            if self.projection:
                pred_q = self.geodesic_solver.batch_projection(pred_q, pos, noisy_rxn_graph.atom_type, full_edge, node2graph, num_nodes, q_type=self.q_type, proj_type="manifold")
            pred_x = self.geodesic_solver.batch_dq2dx(pred_q, pos, noisy_rxn_graph.atom_type, full_edge, node2graph, num_nodes, q_type=self.q_type).reshape(-1, 3)
        else:
            raise ValueError(f"pred_type should be 'edge' or 'node', not {self.pred_type}")

        if pred_q.shape != target_q.shape:
            print(f"pred_q.shape : {pred_q.shape}")
            print(f"target_q.shape : {target_q.shape}")
            print(f"noisy_rxn_graph : {noisy_rxn_graph}")

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
        # NOTE: for debugging, fix time to near 0.5
        # random_t = torch.ones(size=(t.size(0), 1), device=SNR_ratio.device) * 0.5; print(f"Debug: Training debugging: Training is performed with only t=0.5")
        random_t = torch.rand(size=(t.size(0), 1), device=SNR_ratio.device)
        # random_t = 0.7 * random_t; print(f"Debug: t:[0, 0.7] 만 sampling!!!")
        # random_t = (0.3 + random_t) / 1.5; print(f"Debug: t:[0.2, 0.8] 만 sampling!!!")
        diff = abs(t - random_t)
        index = torch.argmin(diff, dim=1)

        t = t[(torch.arange(t.size(0)).to(index), index)]
        print(f"Debug: t={t}")
        sampled_SNR_ratio = SNR_ratio[(torch.arange(SNR_ratio.size(0)).to(index), index)]
        return index, t, sampled_SNR_ratio

    def apply_noise(self, data):
        graph = self.rxn_graph.from_batch(data)
        full_edge, _, _ = graph.full_edge(upper_triangle=True)

        node2graph = graph.batch
        edge2graph = node2graph.index_select(0, full_edge[0])
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
        noise = torch.randn(size=(full_edge.size(1),), device=full_edge.device) * sigma_hat_edge.sqrt()  # dq, (E, )

        # apply noise
        init, last, iter, index_tensor, stats = self.geodesic_solver.batch_geodesic_ode_solve(
            mean,
            noise,
            full_edge,
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

        if len(ban_index) > 0:
            ban_index = ban_index.to(torch.long)
            rxn_idx = [data.rxn_idx[i] for i in ban_index]
            self.print(f"\n[Warning] geodesic solver failed at {len(ban_index)}/{len(data)}"
                        f"\n\trxn_idx: {rxn_idx}\n\ttime index: {t_index[ban_index].tolist()}")
            ban_node_mask = torch.isin(node2graph, ban_index)
            ban_edge_mask = torch.isin(edge2graph, ban_index)
            ban_batch_mask = torch.isin(torch.arange(len(data), device=mean.device), ban_index)

            data = Batch.from_data_list(data[~ban_batch_mask])
            graph = self.rxn_graph.from_batch(data)

            pos_noise = pos_noise[~ban_node_mask]
            x_dot = x_dot[~ban_node_mask]
            pos_init = pos_init[~ban_node_mask]
            tt = tt[~ban_batch_mask]
            target_x = target_x[~ban_node_mask]
            target_q = target_q[~ban_edge_mask]

        return graph, pos_noise, pos_init, tt, target_x, target_q

    def apply_noise_pos(self, data):
        graph = self.rxn_graph.from_batch(data)
        full_edge, _, _ = graph.full_edge(upper_triangle=True)

        node2graph = graph.batch
        edge2graph = node2graph.index_select(0, full_edge[0])
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

        sigma_hat_node = sigma_hat.index_select(0, node2graph)  # (N, )
        # noise = torch.randn(size=(mean.size(),), device=full_edge.device) * sigma_hat_node.sqrt().unsqueeze(-1)  # dq, (E, )
        noise = torch.randn(size=mean.size(), device=full_edge.device) * sigma_hat_node.sqrt().unsqueeze(-1)  # dq, (E, )
        # print(f"Debug: mean={mean}")
        # print(f"Debug: noise={noise}")
        # print(f"Debug: sigma_hat.sqrt()={sigma_hat.sqrt()}")
        # print(f"Debug: tt={tt}")
        pos_noise = mean + noise

        # beta = self.noise_schedule.get_beta(tt)
        # coeff = beta / sigma_hat
        # coeff = 1 / sigma_hat
        coeff = torch.ones_like(tt)
        coeff_node = coeff.index_select(0, node2graph)  # (N, )
        # target is not exactly the score function.
        # target = beta * score
        target_x = - noise * coeff_node.unsqueeze(-1)
        target_q = self.geodesic_solver.batch_dx2dq(target_x, mean, graph.atom_type, full_edge, node2graph, num_nodes, q_type=self.q_type).reshape(-1)

        return graph, pos_noise, pos_init, tt, target_x, target_q

    def configure_optimizers(self):
        # set optimizer
        self.optim = torch.optim.AdamW(
            self.parameters(), lr=self.config.train.lr,
            amsgrad=True, weight_decay=self.config.train.weight_decay
        )
        self.optim_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim,
            mode='min',
            factor=0.7,
            patience=30,
            # threshold=0.01,
            min_lr=1e-5,
            # min_lr=1e-6,
            # min_lr=1e-3,
            # min_lr=1e-4,
        )
        return [self.optim]

    def on_train_epoch_start(self) -> None:
        self.print("Start Training Epoch ...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        if self.train_metrics is not None:
            self.train_metrics.reset()

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

        if self.config.train.noise_type == "manifold":
            rxn_graph, pos, pos_init, tt, target_x, target_q = self.apply_noise(data)
        elif self.config.train.noise_type == "euclidean":
            rxn_graph, pos, pos_init, tt, target_x, target_q = self.apply_noise_pos(data)

        noisy_rxn_graph = self.dynamic_rxn_graph.from_graph(rxn_graph, pos, pos_init, tt)

        full_edge, _, _ = noisy_rxn_graph.full_edge(upper_triangle=True)
        node2graph = noisy_rxn_graph.batch
        edge2graph = node2graph.index_select(0, full_edge[0])
        num_nodes = node2graph.bincount()

        if self.pred_type == "node":
            pred_x = self.forward(noisy_rxn_graph)
            if self.projection:
                pred_x = self.geodesic_solver.batch_projection(pred_x, pos, noisy_rxn_graph.atom_type, full_edge, node2graph, num_nodes, q_type=self.q_type, proj_type="euclidean")
            pred_q = self.geodesic_solver.batch_dx2dq(pred_x, pos, noisy_rxn_graph.atom_type, full_edge, node2graph, num_nodes, q_type=self.q_type).reshape(-1)

        elif self.pred_type == "edge":
            pred_q = self.forward(noisy_rxn_graph)
            dq_dd = self.geodesic_solver.dq_dd(pos, noisy_rxn_graph.atom_type, full_edge, q_type=self.q_type)
            if self.q_type == "morse":
                pred_q = dq_dd * pred_q
            if self.projection:
                pred_q = self.geodesic_solver.batch_projection(pred_q, pos, noisy_rxn_graph.atom_type, full_edge, node2graph, num_nodes, q_type=self.q_type, proj_type="manifold")
            pred_x = self.geodesic_solver.batch_dq2dx(pred_q, pos, noisy_rxn_graph.atom_type, full_edge, node2graph, num_nodes, q_type=self.q_type).reshape(-1, 3)
        else:
            raise ValueError(f"pred_type should be 'edge' or 'node', not {self.pred_type}")

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
        to_log = self.valid_metrics.log_epoch_metrics()
        # print(to_log)
        loss = to_log["valid_epoch/loss"]

        if loss < self.best_valid_loss:
            self.best_valid_loss = loss

        self.log("valid/loss", loss, sync_dist=True)
        self.val_counter += 1
        self.optim_scheduler.step(loss)
        msg = f"Epoch {self.current_epoch} "
        for k, v in to_log.items():
            msg += f"\n\t{k}: {v: 0.6f}"
        self.print(msg)
        if (self.val_counter) % self.config.train.sample_every_n_valid == 0:
            stochastic = self.config.sampling.stochastic
            start = time.time()
            samples = []

            for i, batch in enumerate(self.trainer.datamodule.val_dataloader()):
                if i % self.config.train.sample_every_n_batch == 0:
                    batch = batch.to(self.device)
                    batch_out = self.sample_batch(batch, stochastic=stochastic)
                    samples.extend(batch_out)

            self.valid_sampling_metrics(
                samples, self.name,
                self.current_epoch,
                valid_counter=-1,
                test=True,
                local_rank=self.local_rank
            )
            self.print(f"Done. Sampling took {time.time() - start:0.1f}s")
        print("Validation epoch end ends...")

    def on_test_epoch_start(self) -> None:
        self.print("Starting test ...")
        self.test_loss.reset()
        self.test_metrics.reset()
        self.test_sampling_metrics.reset()
        if self.local_rank == 0:
            setup_wandb(self.config)

    def test_step(self, data, i):

        if self.config.train.noise_type == "manifold":
            rxn_graph, pos, pos_init, tt, target_x, target_q = self.apply_noise(data)
        elif self.config.train.noise_type == "euclidean":
            rxn_graph, pos, pos_init, tt, target_x, target_q = self.apply_noise_pos(data)

        noisy_rxn_graph = self.dynamic_rxn_graph.from_graph(rxn_graph, pos, pos_init, tt)

        full_edge, _, _ = noisy_rxn_graph.full_edge(upper_triangle=True)
        node2graph = noisy_rxn_graph.batch
        edge2graph = node2graph.index_select(0, full_edge[0])
        num_nodes = node2graph.bincount()

        if self.pred_type == "node":
            pred_x = self.forward(noisy_rxn_graph)
            if self.projection:
                pred_x = self.geodesic_solver.batch_projection(pred_x, pos, noisy_rxn_graph.atom_type, full_edge, node2graph, num_nodes, q_type=self.q_type, proj_type="euclidean")
            pred_q = self.geodesic_solver.batch_dx2dq(pred_x, pos, noisy_rxn_graph.atom_type, full_edge, node2graph, num_nodes, q_type=self.q_type).reshape(-1)

        elif self.pred_type == "edge":
            pred_q = self.forward(noisy_rxn_graph)
            dq_dd = self.geodesic_solver.dq_dd(pos, noisy_rxn_graph.atom_type, full_edge, q_type=self.q_type)
            if self.q_type == "morse":
                pred_q = dq_dd * pred_q
            if self.projection:
                pred_q = self.geodesic_solver.batch_projection(pred_q, pos, noisy_rxn_graph.atom_type, full_edge, node2graph, num_nodes, q_type=self.q_type, proj_type="manifold")
            pred_x = self.geodesic_solver.batch_dq2dx(pred_q, pos, noisy_rxn_graph.atom_type, full_edge, node2graph, num_nodes, q_type=self.q_type).reshape(-1, 3)
        else:
            raise ValueError(f"pred_type should be 'edge' or 'node', not {self.pred_type}")

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
            edge_index=full_edge,
            pos=pos,
            log=True
        )

        return {'loss': loss}

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
            start = time.time()
            stochastic = self.config.sampling.stochastic
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
        t = torch.ones(batch.num_graphs, device=pos.device) - self.config.sampling.time_margin # (G, )
        dynamic_rxn_graph = self.dynamic_rxn_graph.from_graph(rxn_graph, pos, pos_init, t)
        # dynamic_rxn_graph = dynamic_rxn_graph.to(pos.device)

        full_edge, _, _ = dynamic_rxn_graph.full_edge(upper_triangle=True)
        node2graph = batch.batch
        edge2graph = node2graph.index_select(0, full_edge[0])
        num_nodes = batch.ptr[1:] - batch.ptr[:-1]
        t = torch.ones_like(full_edge[0])
        dt = t * self.config.sampling.sde_dt

        while (t > 1e-6).any():
            dt = dt.clip(max=t)
            score = self.forward(dynamic_rxn_graph.to("cuda"))
            if self.pred_type == "node":
                score = self.geodesic_solver.batch_dx2dq(score, pos, rxn_graph.atom_type, full_edge, node2graph, num_nodes, q_type=self.q_type).reshape(-1)
            elif self.pred_type == "edge":
                dq_dd = self.geodesic_solver.dq_dd(pos, dynamic_rxn_graph.atom_type, full_edge, q_type=self.q_type)
                if self.q_type == "morse":
                    score = dq_dd * score
                if self.projection:
                    score = self.geodesic_solver.batch_projection(score, pos, dynamic_rxn_graph.atom_type, full_edge, dynamic_rxn_graph.batch, dynamic_rxn_graph.num_nodes, q_type=self.q_type, proj_type="manifold")

            # TEST
            sigma_hat = self.noise_schedule.get_sigma_hat(t)
            beta = self.noise_schedule.get_beta(t)
            coeff = torch.exp(torch.log(beta) - torch.log(sigma_hat))
            score *= coeff
            h = self.get_h_transform(pos, pos_init, t, full_edge, dynamic_rxn_graph.atom_type)
            # score and h arleady contains beta
            if stochastic:
                # debug, check all variables' shape
                dw = torch.sqrt(beta * dt) * torch.randn_like(score)
                dq = - (score - h) * dt + dw
            else:
                dq = - (0.5 * score - h) * dt

            pos = dynamic_rxn_graph.pos

            init, last, iter, index_tensor, stats = self.geodesic_solver.batch_geodesic_ode_solve(
                pos,
                -dq,
                full_edge,
                dynamic_rxn_graph.atom_type,
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
                _ptr = torch.cat([torch.zeros(1, dtype=torch.long), _num_nodes.cumsum(0)])
                _full_edge = index_tensor[2:][:, edge_select] + _ptr[:-1].repeat_interleave(_num_edges)

                self.print(f"[Resolve] geodesic solver failed at {len(retry_index)}/{len(batch)}, Retry...")
                _init, _last, _iter, _index_tensor, _stats = self.geodesic_solver.batch_geodesic_ode_solve(
                    pos[node_select],
                    dq[edge_select],
                    _full_edge,
                    dynamic_rxn_graph.atom_type[node_select],
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

            if len(ban_index) > 0:
                rxn_idx = batch.rxn_idx[ban_index]
                self.print(f"[Warning] geodesic solver failed solving reaction {rxn_idx}, at time {t[ban_index]}\n")
                ban_node_mask = torch.isin(node2graph, ban_index)
                pos_tm1[ban_node_mask] = pos[ban_node_mask] + torch.randn_like(pos[ban_node_mask]) * 1e-3

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

    def get_h_transform(self, pos, pos_init, t, edge_index, atom_type):
        diff = self.geodesic_solver.compute_q(edge_index, atom_type, pos_init, q_type=self.q_type) - self.geodesic_solver.compute_q(edge_index, atom_type, pos, q_type=self.q_type)
        coeff = self.noise_schedule.get_sigma(torch.ones_like(t)) - self.noise_schedule.get_sigma(t)
        beta = self.noise_schedule.get_beta(t)
        coeff = torch.exp(torch.log(beta) - torch.log(coeff))
        h_transform = diff * coeff
        return h_transform

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Training Start")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()
        if self.local_rank == 0:
            setup_wandb(self.config)
