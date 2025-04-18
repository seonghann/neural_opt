from typing import List
from abc import *

import math
import torch
import numpy as np


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
    elif name == "tsdiff":
        scheduler = TSDiffNoiseScheduler(config.scheduler.beta_start, config.scheduler.beta_end, config.scheduler.num_diffusion_timesteps)
    elif name == "dsm":
        scheduler = DSMNoiseScheduler(config.scheduler.sigma_start, config.scheduler.sigma_end)
    elif name == "monomial":
        scheduler = MonomialNoiseScheduler(config.scheduler.sigma_min, config.scheduler.sigma_max, order=config.scheduler.order)
    elif name == "none":
        scheduler = None
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
    def __init__(self, name):
        self.name = name
        return

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


class DSMNoiseScheduler(AbstractNoiseScheduler):
    def __init__(self, sigma_start=1e-4, sigma_end=1e-1, schedule_type="linear"):
        assert sigma_start < sigma_end

        super().__init__(name="DSMNoiseScheduler")
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.schedule_type = schedule_type
        return

    def get_beta(self, t):
        raise NotImplementedError

    def get_sigma(self, t):
        # sigma 
        if self.schedule_type == "linear":
            sigma = (self.sigma_end - self.sigma_start) * t + self.sigma_start
        else:
            raise NotImplementedError
        return sigma**2
        ## sigmas = np.exp(np.linspace(np.log(sigma_start, np.log(sigma_end)), num_noise_level))

    def get_sigma_hat(self, t):
        return self.get_sigma(t)

    def get_SNR(self, t):
        raise NotImplementedError

    def get_time_from_SNRratio(self, SNR_ratio):
        raise NotImplementedError


class TSDiffNoiseScheduler(AbstractNoiseScheduler):
    """Implement noise schedulers used in TSDiff"""
    # def __init__(self, beta_start=1e-7, beta_end=2e-3, schedule_type="sigmoid"):
    def __init__(
        self,
        beta_start=1e-7,
        beta_end=2e-3,
        num_diffusion_timesteps=5000,
        schedule_type="sigmoid",
        ):
        super().__init__(name="TSDiffNoiseScheduler")

        assert beta_start <= beta_end
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.schedule_type = schedule_type
        self.betas = get_beta_schedule(
            schedule_type,
            beta_start=beta_start,
            beta_end=beta_end,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
        self.betas = torch.tensor(self.betas, dtype=torch.float)
        self.alphas = (1 - self.betas).cumprod(dim=0)
        # self.beta_start = beta_start
        # self.beta_end = beta_end
        # self.schedule_type = schedule_type
        # assert schedule_type in ["sigmoid", "linear", "quad"]
        return

    def get_alpha(self, t: List[int], device="cpu"):
        return self.alphas.to(device).index_select(0, t)

    def get_beta(self, t):
        # if self.schedule_type == "sigmoid":
        #     # Transform ranges. t: [0, 1] -> _t: [-6, 6]
        #     t_start, t_end = -6, 6
        #     _t = (t_end - t_start) * (t - 0.5)
        #     betas = torch.sigmoid(_t) * (self.beta_end - self.beta_start) + self.beta_start
        #     return betas
        # else:
        #     raise NotImplementedError
        raise NotImplementedError

    # def get_sigma_sq(self, t):
    def get_sigma(self, t):
        # Return sigma square
        # if self.schedule_type == "sigmoid":
        #     # Transform ranges. t: [0, 1] -> _t: [-6, 6]
        #     t_start, t_end = -6, 6
        #     _t = (t_end - t_start) * (t - 0.5)

        #     retval = torch.log(torch.exp(_t) + 1) - torch.log(torch.exp(torch.full(_t.size(), t_start, device=t.device)) + 1)
        #     retval *= (self.beta_end - self.beta_start) / (t_end - t_start)
        #     retval += self.beta_start * t
        #     return retval
        # else:
        #     raise NotImplementedError
        raise NotImplementedError

    def get_sigma_hat(self, t):
        # return self.get_sigma(t)
        raise NotImplementedError

    def get_SNR(self, t):
        raise NotImplementedError

    def get_time_from_SNRratio(self, SNR_ratio):
        raise NotImplementedError


class MonomialNoiseScheduler(AbstractNoiseScheduler):
    def __init__(
        self,
        # sigma_min: float = 1e-6,  # min(start) of sigma square
        sigma_min: float = 1e-4,  # min(start) of sigma square
        sigma_max: float = 1e-2,  # max(end) of sigma square
        order: int = 1,
    ):
        super().__init__(name="MonomialNoiseScheduler")
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.order = order
        return

    def get_beta(self, t):
        beta = (self.sigma_max - self.sigma_min) * self.order * (t ** (self.order - 1))
        return beta

    def get_sigma(self, t):
        # Return sigma square
        s_sq = (self.sigma_max - self.sigma_min) * (t ** self.order) + self.sigma_min
        return s_sq

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
        raise NotImplementedError


class BellCurveNoiseScheduler(AbstractNoiseScheduler):
    def __init__(self, sigma_min=1e-7, sigma_max=1e-1, beta_std=0.125):
        super().__init__("BellCurveNoiseScheduler")
        self.beta_std = beta_std
        self.normalizer = 1 / (beta_std * math.sqrt(2 * math.pi))
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        return

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
        # NOTE: if SNR_ratio * self.get_sigma(t1)  < self.sigma_min, 'nan' appears.
        denominator = self.beta_std * math.sqrt(2)
        t0 = torch.zeros_like(SNR_ratio)
        t1 = torch.ones_like(SNR_ratio)
        inv_inp = (SNR_ratio * self.get_sigma(t1) - self.sigma_min) * 2 / self.sigma_max
        inv_inp += torch.special.erf((t0 - 0.5) / denominator)
        t = denominator * torch.special.erfinv(inv_inp) + 0.5
        return t


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


if __name__ == "__main__":
    sigma_min = 2e-06
    sigma_max = 0.001
    beta_std = 0.125
    time_margin = 0.0
    # dt = 0.05
    dt = 0.01

    scheduler = BellCurveNoiseScheduler(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        beta_std=beta_std,
    )
    # scheduler = MonomialNoiseScheduler(sigma_min=1e-4, sigma_max=1e-2, order=1)
    # scheduler = TSDiffNoiseScheduler(beta_start=1e-7, beta_end=2e-3, schedule_type="sigmoid")

    t = torch.arange(0, 1 - time_margin + 1e-10, dt)
    print(f"dt = {dt}")
    print(f"len(t) = {len(t)}")

    SNRTt = scheduler.get_SNR(t)
    betas = scheduler.get_beta(t)
    sigma_hat = scheduler.get_sigma_hat(t)
    sigma_square = scheduler.get_sigma(t)

    # t_inv = scheduler.get_time_from_SNRratio(SNR)
    # diff = (t_inv - t)
    # print(diff.max())

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 7, figsize=(20, 2))
    plt.subplots_adjust(hspace=0.5)
    plt.rcParams.update({'font.size': 10})  # Set the global font size to 12
    axs = axs.flatten()

    ax = axs[0];ax.plot(t, betas);ax.set_title(r"$\beta_t = g^2(t)$")
    ax = axs[1];ax.plot(t, sigma_square);ax.set_title("$\sigma_{t}^{2}$")
    ax = axs[2];ax.plot(t, SNRTt);ax.set_title(r"$\frac{SNR_{T}}{SNR_{t}}=\sigma_t^2/\sigma_T^2$")
    ax = axs[3];ax.set_title(r"$(\hat{\sigma}_t^2)^{-1}=\frac{1}{\sigma_t^2(1-\sigma_t^2/\sigma_T^2)}$");ax.plot(t, (1 / sigma_square / (1 - SNRTt)))
    ax = axs[4];ax.set_title(r"$\beta_t/\hat{\sigma}_{t}^2 dt=\frac{\beta_t}{\sigma_t^2(1-\sigma_t^2/\sigma_T^2)} dt$");ax.plot(t, (betas / sigma_square / (1 - SNRTt)) * 1 / len(betas))
    ax.axvline(0.9)
    ax = axs[5];ax.set_title(r"$\beta_t/\hat{\sigma}_{t}^2=\frac{\beta_t}{\sigma_t^2(1-\sigma_t^2/\sigma_T^2)}$");ax.plot(t, (betas / sigma_square / (1 - SNRTt)))
    ax = axs[6];ax.set_title(r"$(\hat{\sigma}_t^2)={\sigma_t^2(1-\sigma_t^2/\sigma_T^2)}$");ax.plot(t, (1 * sigma_square * (1 - SNRTt)))
    plt.tight_layout()
    plt.show()
