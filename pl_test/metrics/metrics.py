from torchmetrics import Metric, MetricCollection
from torch_scatter import scatter_mean, scatter_sum
import torch
import torch.nn as nn
import numpy as np
import wandb

torch.set_printoptions(linewidth=200, edgeitems=1e10, precision=4, sci_mode=False)


class LossFunction(nn.Module):
    def __init__(self, lambda_x, lambda_q, name='train'):
        super().__init__()
        self.name = name
        self.metric_x = SquareLoss("euclidean")
        self.metric_q = SquareLoss("riemannian")
        self.lambda_x = lambda_x
        self.lambda_q = lambda_q

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
        # return loss_q + self._lambda * loss_x
        # return loss_x + self._lambda * loss_q
        return loss_x * self.lambda_x + loss_q * self.lambda_q

    def log_epoch_metrics(self,):
        to_log = {}
        loss_x = self.metric_x.compute().item()
        loss_q = self.metric_q.compute().item()
        # loss = loss_q + self._lambda * loss_x
        # loss = loss_x + self._lambda * loss_q
        loss = loss_x * self.lambda_x + loss_q * self.lambda_q
        to_log[f"{self.name}_epoch/loss_x"] = loss_x
        to_log[f"{self.name}_epoch/loss_q"] = loss_q
        to_log[f"{self.name}_epoch/loss"] = loss
        if wandb.run:
            wandb.log(to_log)
        return to_log


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
            for metric in [self.rmsd_metrics, self.norm_metrics]:
                for k, v in metric.compute().items():
                    to_log[f'{self.name}/{k}'] = v
            if wandb.run:
                wandb.log(to_log)

    def reset(self):
        for metric in [self.rmsd_metrics, self.norm_metrics]:
            metric.reset()

    def log_epoch_metrics(self,):
        to_log = {}
        for metric in [self.rmsd_metrics, self.norm_metrics]:
            for k, v in metric.compute().items():
                to_log[f'{self.name}/{k}'] = v
        if wandb.run:
            wandb.log(to_log)

        return to_log


class ValidMetrics(nn.Module):
    def __init__(self, manifold, name='valid', lambda_x=0.0, lambda_q=1.0):
        super().__init__()
        self.name = name
        self.rmsd_metrics = MetricRMSD()
        self.norm_metrics = MetricNorm()
        # self.proj_metrics = MetricProj()
        self.loss_x = SquareLoss("euclidean")
        self.loss_q = SquareLoss("riemannian")
        self.lambda_x = lambda_x
        self.lambda_q = lambda_q
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
        # TODO : Jacobian product should be done with bmm style
        # print(f"Warning: J is calculated with jacobian_q !")
        # J = self.manifold.jacobian_q(edge_index, atom_type, pos)
        # if self.q_type == "DM":
        #     J = self.manifold.jacobian_d(edge_index, atom_type, pos)
        # J_inv = torch.linalg.pinv(J, rtol=1e-4, atol=self.manifold.svd_tol)
        # proj_q = J @ J_inv @ pred_q
        # self.proj_metrics(proj_q, pred_q, target_q, edge2graph)
        loss_x = self.loss_x(pred_x, target_x, node2graph)
        loss_q = self.loss_q(pred_q, target_q, edge2graph)
        # loss = loss_q + self._lambda * loss_x
        # loss = loss_x + self._lambda * loss_q
        loss = loss_x * self.lambda_x + loss_q * self.lambda_q
        if log:
            to_log = {f"{self.name}/loss": loss.item()}
            # for metric in [self.rmsd_metrics, self.norm_metrics, self.proj_metrics]:
            for metric in [self.rmsd_metrics, self.norm_metrics]:
                for k, v in metric.compute().items():
                    to_log[f'{self.name}/{k}'] = v
            if wandb.run:
                wandb.log(to_log)

    def reset(self):
        # for metric in [self.rmsd_metrics, self.norm_metrics, self.proj_metrics]:
        for metric in [self.rmsd_metrics, self.norm_metrics]:
            metric.reset()

    def log_epoch_metrics(self,):
        loss_x = self.loss_x.compute()
        loss_q = self.loss_q.compute()
        # loss = loss_q + self._lambda * loss_x
        # loss = loss_x + self._lambda * loss_q
        loss = loss_x * self.lambda_x + loss_q * self.lambda_q
        to_log = {f"{self.name}_epoch/loss": loss}
        # for metric in [self.rmsd_metrics, self.norm_metrics, self.proj_metrics]:
        for metric in [self.rmsd_metrics, self.norm_metrics]:
            for k, v in metric.compute().items():
                to_log[f'{self.name}_epoch/{k}'] = v
        if wandb.run:
            wandb.log(to_log)

        return to_log


class SamplingMetrics(nn.Module):
    def __init__(self, manifold, name='test'):
        super().__init__()
        self.name = name
        self.dmae_metrics = MetricDMAE()
        self.rmsd_metrics = MetricRMSD()
        self.norm_metrics = MetricNorm()
        self.manifold = manifold

    def forward(self, samples, name, current_epoch, valid_counter=-1, test=True, local_rank=0):
        for sample in samples:
            x_gen = sample.traj[:, 0].to(sample.pos.device)
            x_true = sample.pos[:, 0]
            num_atoms = x_gen.size(0)
            atom_type = sample.x
            # edge_index = torch.LongTensor(np.triu_indices(num_atoms, k=1)) # full_edges
            edge_index = torch.LongTensor(np.triu_indices(num_atoms, k=1)).to(sample.pos.device) # full_edges
            q_gen = self.manifold.compute_q(edge_index, atom_type, x_gen)
            q_true = self.manifold.compute_q(edge_index, atom_type, x_true)
            d_gen = self.manifold.compute_d(edge_index, x_gen)
            d_true = self.manifold.compute_d(edge_index, x_true)

            nodes = torch.zeros_like(atom_type)
            edges = torch.zeros_like(edge_index[0])
            self.rmsd_metrics(x_gen, x_true, nodes)
            self.norm_metrics(q_gen, q_true, edges)
            self.dmae_metrics(d_gen, d_true, edges)

        if wandb.run:
            to_log = {}
            for metric in [self.rmsd_metrics, self.norm_metrics, self.dmae_metrics]:
                for k, v in metric.compute().items():
                    to_log[f"{self.name}_sampling/{k}"] = v
            wandb.log(to_log)

    def reset(self):
        for metric in [self.rmsd_metrics, self.norm_metrics, self.dmae_metrics]:
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


class SquareLoss(Metric):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.add_state(f"total_square_err_{name}", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, pred, target, merge):
        """
        Update state with predictions and targets.
        Args:
            pred: Predictions from model (n, 3) or (E,)
            target: Ground truth labels (n, 3) or (E,)
            merge: Batch index for each atom (n,) or edge (E,)
        """
        if self.name == "euclidean":
            square_err = torch.sum((pred - target) ** 2, dim=-1)
        elif self.name == "riemannian":
            square_err = (pred - target) ** 2
        square_err = scatter_sum(square_err, merge)
        # square_err = square_err.sqrt(); print(f"Debug: Using Norm error!!!! not norm square")

        state = self.__getstate__()
        state[f"total_square_err_{self.name}"] += square_err.sum()
        self.total_samples += square_err.numel()

    def compute(self):
        state = self.__getstate__()
        return state[f"total_square_err_{self.name}"] / self.total_samples


class MetricRMSD(Metric):
    def __init__(self, ):
        super().__init__()
        self.add_state("total_rmsd", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_perr_rmsd", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_pred_size_rmsd", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_target_size_rmsd", default=torch.tensor(0.), dist_reduce_fx="sum")
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

        denom = torch.sqrt(scatter_mean(torch.sum(target ** 2, dim=-1), merge))
        perr = rmsd / denom
        pred_size = torch.sqrt(scatter_mean(torch.sum(pred ** 2, dim=-1), merge))

        print(f"Debug: MetricRMSD.update ======================================")
        print(f"Debug: rmsd={rmsd.detach()}")
        print(f"Debug: denom={denom.detach()}")
        print(f"Debug: perr={perr.detach()}")
        print(f"Debug: perr.mean()={perr.mean()}")
        print(f"Debug: MetricRMSD.update ======================================")

        self.total_rmsd += rmsd.sum()
        self.total_perr_rmsd += perr.sum()
        self.total_pred_size_rmsd += pred_size.sum()
        self.total_target_size_rmsd += denom.sum()
        self.total_samples += rmsd.numel()

    def compute(self):
        """
        Computes the metric based on the state collected.
        """
        stats = {
            "pred_target_rmsd_err": self.total_rmsd / self.total_samples,
            "pred_target_rmsd_perr": self.total_perr_rmsd / self.total_samples,
            "pred_size_rmsd": self.total_pred_size_rmsd / self.total_samples,
            "target_size_rmsd": self.total_target_size_rmsd / self.total_samples,
        }
        return stats


class MetricDMAE(Metric):
    def __init__(self, ):
        super().__init__()
        self.add_state("total_dmae", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, pred, target, merge):
        """
        Update state with predictions and targets.
        Args:
            pred: Predictions from model (E,)
            target: Ground truth labels (E,)
            merge: Batch index for each edge (E,)
        """
        abs_err = (pred - target).abs()
        dmae = scatter_mean(abs_err, merge)

        self.total_dmae += dmae.sum()
        self.total_samples += dmae.numel()

    def compute(self):
        """
        Computes the metric based on the state collected.
        """
        stats = {
            "pred_target_dmae_err": self.total_dmae / self.total_samples,
        }
        return stats


class MetricNorm(Metric):
    def __init__(self, ):
        super().__init__()
        self.add_state("total_norm", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_perr_norm", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_pred_size_norm", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_target_size_norm", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, pred, target, merge):
        """
        Update state with predictions and targets.
        Args:
            pred: Predictions from model (E,)
            target: Ground truth labels (E,)
            merge: Batch index for each edge (E,)
        """
        square_err = (pred - target) ** 2
        norm_err = torch.sqrt(scatter_sum(square_err, merge))

        denom = torch.sqrt(scatter_sum(target ** 2, merge))
        perr = norm_err / denom
        pred_size = torch.sqrt(scatter_sum(pred ** 2, merge))
        print(f"Debug: MetricNorm.update ======================================")
        # print(f"Debug: pred=\n{pred.detach()}")
        # print(f"Debug: target=\n{target.detach()}")
        print(f"Debug: norm_err={norm_err.detach()}")
        print(f"Debug: denom={denom.detach()}")
        print(f"Debug: perr={norm_err / denom}")
        print(f"Debug: perr.mean()={perr.mean()}")
        print(f"Debug: MetricNorm.update ======================================")

        self.total_norm += norm_err.sum()
        self.total_perr_norm += perr.sum()
        self.total_pred_size_norm += pred_size.sum()
        self.total_target_size_norm += denom.sum()
        self.total_samples += norm_err.numel()

    def compute(self):
        """
        Computes the metric based on the state collected.
        """
        stats = {
            "pred_target_norm_err": self.total_norm / self.total_samples,
            "pred_target_norm_perr": self.total_perr_norm / self.total_samples,
            "pred_size_norm": self.total_pred_size_norm / self.total_samples,
            "target_size_norm": self.total_target_size_norm / self.total_samples
        }
        return stats


class MetricProj(Metric):
    def __init__(self, ):
        super().__init__()
        self.add_state("total_samples", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_proj_target_err", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_proj_target_perr", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_proj_pred_ratio", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_proj_pred_cos_angle", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_proj_target_cos_angle", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, proj, pred, target, merge):
        """
        Update state with predictions and targets.
        Args:
            proj: Projected predictions from model (E,)
            pred: Predictions from model (E,)
            target: Ground truth labels (E,)
            merge: Batch index for each edge (E,)
        """
        proj_target_err = (proj - target) ** 2
        proj_target_err = torch.sqrt(scatter_sum(proj_target_err, merge))

        denom = torch.sqrt(scatter_sum(target ** 2, merge))
        perr = proj_target_err / denom

        proj_size = torch.sqrt(scatter_sum(proj ** 2, merge))
        pred_size = torch.sqrt(scatter_sum(pred ** 2, merge))
        proj_ratio = proj_size / pred_size

        pred_proj_cos_angle = scatter_sum(proj * pred, merge) / (proj_size * pred_size)
        target_proj_cos_angle = scatter_sum(proj * target, merge) / (proj_size * denom)

        self.total_samples += proj_target_err.numel()
        self.total_proj_target_err += proj_target_err.sum()
        self.total_proj_target_perr += perr.sum()
        self.total_proj_pred_ratio += proj_ratio.sum()
        self.total_proj_pred_cos_angle += pred_proj_cos_angle.sum()
        self.total_proj_target_cos_angle += target_proj_cos_angle.sum()

    def compute(self):
        """
        Computes the metric based on the state collected.
        """
        stats = {
            "proj_target_err": self.total_proj_target_err / self.total_samples,
            "proj_target_perr": self.total_proj_target_perr / self.total_samples,
            "proj_pred_ratio": self.total_proj_pred_ratio / self.total_samples,
            "proj_pred_cos_angle": self.total_proj_pred_cos_angle / self.total_samples,
            "proj_target_cos_angle": self.total_proj_target_cos_angle / self.total_samples
        }
        return stats

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
