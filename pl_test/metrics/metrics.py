from torchmetrics import Metric, MetricCollection
from torch_scatter import scatter_mean, scatter_sum
import torch
import torch.nn as nn
import wandb


class LossFunction(nn.Module):
    def __init__(self, _lambda, name='train'):
        super().__init__()
        self.name = name
        self.metric_x = MetricRMSD()
        self.metric_q = MetricNorm()
        self._lambda = _lambda

    def reset(self,):
        for metric in [self.metric_x, self.metric_q]:
            metric.reset()

    def forward(self, pred_x, pred_q, true_x, true_q, merge_edge, merge_node, log=False):
        loss_x = self.metric_x(pred_x, true_x, merge_node)
        loss_q = self.metric_q(pred_q, true_q, merge_edge)
        if log:
            to_log = {}
            to_log[f"{self.name}/loss_x"] = loss_x.item()
            to_log[f"{self.name}/loss_q"] = loss_q.item()
            if wandb.run:
                wandb.log(to_log)
        return loss_q + self._lambda * loss_x

    def log_epoch_metrics(self,):
        epoch_metric_x = self.metric_x.compute()
        epoch_metric_q = self.metric_q.compute()

        to_log = {}
        to_log[f"{self.name}_epoch/loss_x"] = epoch_metric_x.item()
        to_log[f"{self.name}_epoch/loss_q"] = epoch_metric_q.item()
        if wandb.run:
            wandb.log(to_log)

        return epoch_metric_x, epoch_metric_q


class TrainMetrics(nn.Module):
    def __init__(self, name='train'):
        super().__init__()
        self.name = name
        self.rmsd_metrics = MetricRMSD()
        self.norm_metrics = MetricNorm()

    def forward(
        self,
        pred_x,
        pred_q,
        target_x,
        target_q,
        edge2graph,
        node2graph,
        atom_type,
        edge_r,
        edge_p,
        log=False
    ):
        self.rmsd_metrics(pred_x, target_x, node2graph)
        self.norm_metrics(pred_q, target_q, edge2graph)
        if log:
            to_log = {}
            to_log[f'{self.name}/rmsd'] = self.rmsd_metrics.compute()
            to_log[f'{self.name}/norm'] = self.norm_metrics.compute()
            if wandb.run:
                wandb.log(to_log)

    def reset(self):
        for metric in [self.rmsd_metrics, self.norm_metrics]:
            metric.reset()

    def log_epoch_metrics(self,):
        epoch_metric_x = self.rmsd_metrics.compute()
        epoch_metric_q = self.norm_metrics.compute()
        to_log = {}
        to_log["train_epoch/rmsd"] = epoch_metric_x
        to_log["train_epoch/norm"] = epoch_metric_q
        if wandb.run:
            wandb.log(to_log)
        return epoch_metric_x, epoch_metric_q


class ValidMetrics(nn.Module):
    def __init__(self, manifold, name='valid' ):
        super().__init__()
        self.name = name
        self.rmsd_metrics = MetricRMSD()
        self.norm_metrics = MetricNorm()
        self.proj_metrics = MetricProj()
        self.manifold = manifold

    def forward(
        self,
        pred_x,
        pred_q,
        target_x,
        target_q,
        edge2graph,
        node2graph,
        atom_type,
        edge_r,
        edge_p,
        edge_index=None,
        pos=None,
        log=False
    ):
        self.rmsd_metrics(pred_x, target_x, node2graph)
        self.norm_metrics(pred_q, target_q, edge2graph)
        J = self.manifold.jacobian_q(edge_index, atom_type, pos)
        J_inv = torch.linalg.pinv(J, rtol=1e-4, atol=self.manifold.svd_tol)
        proj_q = J @ J_inv @ pred_q
        self.proj_metrics(proj_q, pred_q, target_q, edge2graph)
        if log:
            to_log = {}
            to_log[f'{self.name}/rmsd'] = self.rmsd_metrics.compute()
            to_log[f'{self.name}/norm'] = self.norm_metrics.compute()
            to_log[f'{self.name}/proj'] = self.proj_metrics.compute()
            if wandb.run:
                wandb.log(to_log)

    def reset(self):
        for metric in [self.rmsd_metrics, self.norm_metrics, self.proj_metrics]:
            metric.reset()

    def log_epoch_metrics(self,):
        epoch_metric_x = self.rmsd_metrics.compute()
        epoch_metric_q = self.norm_metrics.compute()
        epoch_metric_proj = self.proj_metrics.compute()
        to_log = {}
        to_log[f"{self.name}_epoch/rmsd"] = epoch_metric_x
        to_log[f"{self.name}_epoch/norm"] = epoch_metric_q
        to_log[f"{self.name}_epoch/proj"] = epoch_metric_proj
        if wandb.run:
            wandb.log(to_log)
        return epoch_metric_x, epoch_metric_q, epoch_metric_proj


class SamplingMetrics(nn.Module):
    def __init__(self, manifold, name='test'):
        super().__init__()
        self.name = name
        self.rmsd_metrics = MetricRMSD()
        self.norm_metrics = MetricNorm()
        self.manifold = manifold

    def forward(self, samples, name, current_epoch, valid_counter=-1, test=True, local_rank=0):
        for sample in samples:
            x_gen = sample.traj[:, 0].to(sample.pos.device)
            x_true = sample.pos[:, 0]
            atom_type = sample.x
            edge_index = sample.edge_index
            q_gen = self.manifold.compute_q(edge_index, atom_type, x_gen)
            q_true = self.manifold.compute_q(edge_index, atom_type, x_true)

            nodes = torch.zeros_like(atom_type)
            edges = torch.zeros_like(edge_index[0])
            self.rmsd_metrics(x_gen, x_true, nodes)
            self.norm_metrics(q_gen, q_true, edges)
        rmsd = self.rmsd_metrics.compute()
        norm = self.norm_metrics.compute()

        if wandb.run:
            to_log = {}
            to_log[f"{self.name}_sampling/rmsd"] = rmsd
            to_log[f"{self.name}_sampling/norm"] = norm
            wandb.log(to_log)

    def reset(self):
        for metric in [self.rmsd_metrics, self.norm_metrics]:
            metric.reset()


class SquareErrorPerBond(Metric):
    # TODO :
    full_state_update = False
    def __init__(self, class_id):
        """
        class_id is set as bond type.
        (0, 1) means nonbond to single bond
        (1, 2) means single bond to double bond
        (4, 1) means aromatic bond to single bond
        """
        super().__init__()
        self.class_id = class_id
        self.add_state("total_square_error", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, pred, target, bond_type):
        pass


class MetricProj(Metric):
    def __init__(self, ):
        super().__init__()
        self.add_state("total_norm", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, proj, pred, target, merge):
        """
        Update state with predictions and targets.
        Args:
            proj: Projected predictions from model (E,)
            pred: Predictions from model (E,)
            target: Ground truth labels (E,)
            merge: Batch index for each edge (E,)
        """
        squre_err = (pred - proj) ** 2
        norm = torch.sqrt(scatter_sum(squre_err, merge))
        self.total_norm += norm.sum()
        self.total_samples += norm.numel()

    def compute(self):
        """
        Computes the metric based on the state collected.
        """
        return self.total_norm / self.total_samples


class MetricRMSD(Metric):
    def __init__(self, ):
        super().__init__()
        self.add_state("total_rmsd", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, pred, target, merge):
        """
        Update state with predictions and targets.
        Args:
            pred: Predictions from model (n, 3)
            target: Ground truth labels (n, 3)
            merge: Batch index for each atom (n,)
        """
        square_err = torch.sum((pred - target) ** 2, dim=-1)
        rmsd = torch.sqrt(scatter_mean(square_err, merge))
        self.total_rmsd += rmsd.sum()
        self.total_samples += rmsd.numel()

    def compute(self):
        """
        Computes the metric based on the state collected.
        """
        return self.total_rmsd / self.total_samples


class MetricNorm(Metric):
    def __init__(self, ):
        super().__init__()
        self.add_state("total_norm", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, pred, target, merge):
        """
        Update state with predictions and targets.
        Args:
            pred: Predictions from model (E,)
            target: Ground truth labels (E,)
            merge: Batch index for each edge (E,)
        """
        squre_err = (pred - target) ** 2
        norm = torch.sqrt(scatter_sum(squre_err, merge))
        self.total_norm += norm.sum()
        self.total_samples += norm.numel()

    def compute(self):
        """
        Computes the metric based on the state collected.
        """
        return self.total_norm / self.total_samples


if __name__ == "__main__":
    loss_fn = LossFunction(1)
    x = torch.randn(10, 3)
    x_target = torch.randn(10, 3)
    q = torch.randn(10)
    q_target = torch.randn(10)
    merge_x = torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    merge_q = torch.LongTensor([0, 0, 0, 0, 0, 1, 2, 2, 2, 2])
    for i in range(10):
        loss = loss_fn(x, q, x_target, q_target, merge_x, merge_q, log=True)
    a, b = loss_fn.log_epoch_metrics()
