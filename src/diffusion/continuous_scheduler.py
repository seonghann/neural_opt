"""VP-type continuous-time diffusion noise scheduler with sigmoid beta schedule.

The beta(t) function follows a sigmoid curve between beta_start and beta_end,
providing smooth noise scheduling for variance-preserving (VP) diffusion processes.
All quantities (alpha, sigma, SNR, drift f, diffusion g) are computed analytically
from the continuous-time beta integral.
"""

import torch


class SigmoidDiffusionScheduler(torch.nn.Module):
    name = "SigmoidDiffusionScheduler"

    def __init__(self, c=12, beta_start=1.e-7, beta_end=10, alpha_1_thresh=1.0, EPS=1e-6):
        super().__init__()
        self.c = c
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.EPS = EPS
        half = torch.tensor([0.5])
        self.sigmoid_nhalf = self._sigmoid(-half)
        self.sigmoid_scaler = self._sigmoid(half) - self.sigmoid_nhalf
        self.integral_scalar = self._integrate_sigmoid(-half)
        one = torch.tensor([1.])
        alpha_1 = self.get_alpha(one)
        assert alpha_1 < alpha_1_thresh, f"alpha(1) = {alpha_1} > {alpha_1_thresh}"

    def _sigmoid(self, t):
        return 1 / (1 + torch.exp(- self.c * t))

    def get_beta(self, t):
        beta = (self._sigmoid(t - 0.5) - self.sigmoid_nhalf.to(t.device)) \
            * (self.beta_end - self.beta_start) \
            / self.sigmoid_scaler.to(t.device) + self.beta_start
        return beta

    def get_alpha(self, t):
        log_alpha = self._log_alpha(t)
        return torch.exp(log_alpha)

    def get_sigma(self, t):
        alpha_sq = torch.exp(self._log_alpha(t) * 2)
        sigma = torch.sqrt(1 - alpha_sq)
        return sigma

    def _integrate_sigmoid(self, t):
        return torch.log(1 + torch.exp(self.c * t)) / self.c

    def _integrate_beta(self, t):
        scaler_1 = (self.beta_end - self.beta_start) / self.sigmoid_scaler.to(t.device)
        sigmoid_term = scaler_1 * (self._integrate_sigmoid(t - 0.5) - self.integral_scalar.to(t.device))
        scaler_2 = self.beta_start - scaler_1 * (self.sigmoid_nhalf.to(t.device))
        linear_term = scaler_2 * t
        return sigmoid_term + linear_term + self.EPS

    def _log_alpha(self, t):
        return -0.5 * self._integrate_beta(t)

    def _log_sigma(self, t):
        log_alpha_sq = - self._integrate_beta(t)
        log_sigma_sq = torch.log(1 - torch.exp(log_alpha_sq))
        return 0.5 * log_sigma_sq

    def get_SNR(self, t):
        alpha = self.get_alpha(t)
        return alpha ** 2 / (1 - alpha ** 2)

    def get_f(self, t):
        return - 0.5 * self.get_beta(t)

    def get_g(self, t):
        g_sq = self.get_beta(t)
        return torch.sqrt(g_sq)
