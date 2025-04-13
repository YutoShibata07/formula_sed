from dataclasses import dataclass
import numpy as np

@dataclass
class GlobalParams:
    scale_exp_silent:float
    scale_exp_voiced:float
    scale_normal_silent:float
    scale_normal_voiced:float
    volume_variance: float
    volume_lengthscale: float
    volume_cov: np.ndarray
    harmonic_volume_initial_bias: float
    noise_volume_initial_bias: float
    f0_mel_variance: float
    f0_mel_lengthscale: float
    f0_mel_initial_bias: float
    harmonic_envelope_variance: float
    harmonic_envelope_lengthscale: float
    harmonic_envelope_initial_bias: float
    noise_distribution_freq_variance: float
    noise_distribution_freq_lengthscale: float
    noise_distribution_time_variance: float
    noise_distribution_time_lengthscale: float
    noise_distribution_initial_bias: float
    n_harmonics: int
    f0_quantize: bool
    ir_diminin: float