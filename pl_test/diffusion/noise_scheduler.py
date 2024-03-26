from abc import *
import torch
import math


def load_noise_scheduler(config):
    """
    Args:
        config: diffusion config
    return:
        NoiseScheduler
    """
    name = config.scheduler.name.lower()
    if name == "bellcurve":
        scheduler = BellCurveNoiseScheduler(config.scheduler.sigma_min, config.scheduler.sigma_max, config.scheduler.beta_std)
    else:
        raise ValueError(f"Invalid scheduler name: {name}")

    return scheduler


def get_discrete_inverse(func, val, n=1000):
    """
    func is a monotonic increasing function in [0, 1]
    return the inverse of func of val
    Args:
        func (function): monotonic increasing function in [0, 1]
        val (torch.Tensor): value 1-d or 2-d (N, ) or (N, M)
    Returns:
        t (torch.Tensor): (N, ) or (N, M)
    """
    t = torch.linspace(0, 1, n, device=val.device)
    t = t.repeat(*val.shape, 1)
    print(t.shape)
    f_t = func(t)
    abs_diff = abs(f_t - val.unsqueeze(-1))
    idx = torch.argmin(abs_diff, dim=-1)
    print(idx.shape)
    t = t.index_select(-1, idx)
    print(t.shape)
    return t


class AbstractNoiseScheduler(metaclass=ABCMeta):
    def __init__(self, sigma_min=1e-7):
        self.sigma_min = sigma_min

    def get_beta(self, t):
        raise NotImplementedError

    def get_sigma(self, t):
        raise NotImplementedError

    def get_sigma_hat(self, t):
        raise NotImplementedError

    def get_SNR(self, t):
        raise NotImplementedError

    def get_time_from_SNRratio(self, SNR_ratio):
        raise NotImplementedError


class BellCurveNoiseScheduler(AbstractNoiseScheduler):
    def __init__(self, sigma_min=1e-7, sigma_max=1e-1, beta_std=0.125):
        self.beta_std = beta_std
        self.normalizer = 1 / (beta_std * math.sqrt(2 * math.pi))
        self.sigma_max = sigma_max
        super().__init__(sigma_min=sigma_min)

    def get_beta(self, t):
        beta = torch.exp(-((t - 0.5) / self.beta_std) ** 2 / 2)
        beta *= self.normalizer * self.sigma_max
        return beta

    def get_sigma(self, t):
        # sigma squre = \int_0^t beta(t) dt
        scaler = self.sigma_max / 2
        denominator = self.beta_std * math.sqrt(2)
        t0 = torch.zeros_like(t)
        integral_t = torch.special.erf((t - 0.5) / denominator)
        integral_0 = torch.special.erf((t0 - 0.5) / denominator)
        sigma = (integral_t - integral_0) * scaler
        sigma += self.sigma_min
        return sigma

    def get_sigma_hat(self, t):
        sigma_t = self.get_sigma(t)
        snr = self.get_SNR(t)
        sigma_hat = sigma_t * (1 - snr)
        return sigma_hat

    def get_SNR(self, t):
        """
        SNR(t) = 1 / sigma^2(t)
        return SNR(1)/SNR(t)
        """
        SNR = self.get_sigma(t) / self.get_sigma(torch.ones_like(t))
        return SNR

    def get_time_from_SNRratio(self, SNR_ratio, n=1000):
        """
        get t from SNR_ratio = (SNR(1)/SNR(t))
        Args:
            SNR_ratio (torch.Tensor): (N, )
        Returns:
            t (torch.Tensor): (N, )
        """
        denominator = self.beta_std * math.sqrt(2)
        t0 = torch.zeros_like(SNR_ratio)
        t1 = torch.ones_like(SNR_ratio)
        inv_inp = (SNR_ratio * self.get_sigma(t1) - self.sigma_min) * 2 / self.sigma_max
        inv_inp += torch.special.erf((t0 - 0.5) / denominator)
        t = denominator * torch.special.erfinv(inv_inp) + 0.5
        return t


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    scheduler = BellCurveNoiseScheduler()
    t = torch.linspace(0, 1, 1001)
    SNR = scheduler.get_SNR(t)
    t_inv = scheduler.get_time_from_SNRratio(SNR)
    diff = (t_inv - t)
    print(diff.max())
