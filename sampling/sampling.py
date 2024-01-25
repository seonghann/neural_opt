import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy


class SamplingParams:
    def __init__(
        self,
        sampling_type,
        beta_std=0.125,
        sigma_max=1e-1,
        sigma_min=1e-7,
        order=None,
    ):
        assert sampling_type in ["bell-shaped", "monomial"]
        self.sampling_type = sampling_type
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.beta_std = beta_std
        self.order = order
        print(f"Debug: sampling_type = {sampling_type}")
        print("Debug: std_beta, sigma_max, sigma_min, order = ", beta_std, sigma_max, sigma_min, order)
        return

    def beta(self, t):
        if self.sampling_type == "monomial":
            b = (self.order + 1) * self.sigma_max * (t ** self.order)
        elif self.sampling_type == "bell-shaped":
            b = torch.exp(-((t - 0.5) / self.beta_std) ** 2 / 2)
            normalizer = 1 / (self.beta_std * np.sqrt(2 * np.pi))
            b *= normalizer
            b *= self.sigma_max
        else:
            raise NotImplementedError
        return b

    def sigma_square(self, t):
        if self.sampling_type == "monomial":
            s_sq = self.sigma_max * (t ** (self.order + 1)) + self.sigma_min
        elif self.sampling_type == "bell-shaped":
            erf_scaler = self.sigma_max / 2
            s_sq = erf_scaler * (torch.special.erf((t - 0.5) / (np.sqrt(2) * self.beta_std)) - scipy.special.erf((0 - 0.5) / (np.sqrt(2) * self.beta_std)))
            s_sq += self.sigma_min
        else:
            raise NotImplementedError
        return s_sq

    def SNR(self, t):
        """
        return SNR(T) / SNR(t)
        """
        return self.sigma_square(t) / self.sigma_square(torch.ones(1))


if __name__ == "__main__":
    ## Set sampling parameters
    sampling_type = ["bell-shaped", "monomial"][1]

    if sampling_type == "monomial":
        std_beta = None
        sigma_max = 0.01
        sigma_min = 5 * 1e-5
        sigma_min = 1e-6
        order = 2
    elif sampling_type == "bell-shaped":
        std_beta = 0.125
        sigma_max = 0.001
        sigma_min = 1e-6 * 2
        order = None
    else:
        raise NotImplementedError

    params = SamplingParams(
        sampling_type=sampling_type,
        beta_std=std_beta,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        order=order,
        )


    ## t samples
    num_time_steps = 20
    # num_time_steps = 100
    t = torch.linspace(0, 1, num_time_steps + 1)[:-1]
    # t = torch.linspace(1e-2, 1, num_time_steps + 1)[:-1]
    # t = torch.linspace(0, 0.9, num_time_steps + 1)[:-1]
    betas = params.beta(t)
    sigma_square = params.sigma_square(t)
    SNRTt = params.SNR(t)


    ## plot
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
    plt.show()

