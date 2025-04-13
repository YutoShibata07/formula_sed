import numpy as np
from .global_params import GlobalParams
from .local_params import LocalParams


def sample_global_params(
    rng: np.random.Generator, small_variance: float
) -> GlobalParams:
    scale_exp_silent = 10 ** rng.uniform(-1, 0.5)  # 1e-1
    scale_exp_voiced = 10 ** rng.uniform(-1, 1)  # 1e1
    scale_normal_silent = small_variance
    scale_normal_voiced = small_variance
    volume_variance = small_variance
    volume_lengthscale = 1.0
    volume_cov = np.array([[1.0, 0.99], [0.99, 1.0]])
    harmonic_volume_initial_bias = rng.uniform(-4, 5)

    if harmonic_volume_initial_bias < -1:
        noise_volume_initial_bias = rng.uniform(5, 45)
    else:
        noise_volume_initial_bias = rng.uniform(-15, 45)

    f0_mel_variance = 30 ** rng.uniform(0, 2)
    f0_mel_lengthscale = 1e-2
    f0_mel_initial_bias = 50 ** rng.uniform(1, 2)
    harmonic_envelope_variance = 10 ** rng.uniform(-1, 1)
    harmonic_envelope_lengthscale = 10 ** rng.uniform(1, 3)
    harmonic_envelope_initial_bias = 10 ** rng.uniform(-1, 1)
    noise_distribution_freq_variance = 1.0
    noise_distribution_freq_lengthscale = 1e-1
    noise_distribution_time_variance = 1e2
    noise_distribution_time_lengthscale = 2.0
    noise_distribution_initial_bias = 10 ** rng.uniform(-1, 1)
    n_harmonics = rng.integers(5, 51)
    f0_quantize = rng.choice([False, True])
    ir_diminin = rng.uniform(-150.0, -5.0)

    return GlobalParams(
        scale_exp_silent=scale_exp_silent,
        scale_exp_voiced=scale_exp_voiced,
        scale_normal_silent=scale_normal_silent,
        scale_normal_voiced=scale_normal_voiced,
        volume_variance=volume_variance,
        volume_lengthscale=volume_lengthscale,
        volume_cov=volume_cov,
        harmonic_volume_initial_bias=harmonic_volume_initial_bias,
        noise_volume_initial_bias=noise_volume_initial_bias,
        f0_mel_variance=f0_mel_variance,
        f0_mel_lengthscale=f0_mel_lengthscale,
        f0_mel_initial_bias=f0_mel_initial_bias,
        harmonic_envelope_variance=harmonic_envelope_variance,
        harmonic_envelope_lengthscale=harmonic_envelope_lengthscale,
        harmonic_envelope_initial_bias=harmonic_envelope_initial_bias,
        noise_distribution_freq_variance=noise_distribution_freq_variance,
        noise_distribution_freq_lengthscale=noise_distribution_freq_lengthscale,
        noise_distribution_time_variance=noise_distribution_time_variance,
        noise_distribution_time_lengthscale=noise_distribution_time_lengthscale,
        noise_distribution_initial_bias=noise_distribution_initial_bias,
        n_harmonics=n_harmonics,
        f0_quantize=f0_quantize,
        ir_diminin=ir_diminin,
    )


def sample_local_params(rng: np.random.Generator, small_variance: float) -> LocalParams:
    hn_cor = rng.uniform(0.8, 1.0) * rng.choice([-1, 1])
    volume_hn_correlation = np.array([[1, hn_cor], [hn_cor, 1]])
    volume_hn_variance = 10 ** rng.uniform(0, 1)
    volume_hn_lengthscale = 1
    volume_section_correlation = np.array([[1.0, 0.99], [0.99, 1.0]])
    volume_section_variance = small_variance
    volume_section_lengthscale = 1
    f0_mel_variance = 10 ** rng.uniform(0, 4)
    f0_mel_lengthscale = 10 ** rng.uniform(-1, 1)
    section_f0_mel_variance = 10 ** rng.uniform(-1, 0)

    return LocalParams(
        hn_cor=hn_cor,
        volume_hn_correlation=volume_hn_correlation,
        volume_hn_variance=volume_hn_variance,
        volume_hn_lengthscale=volume_hn_lengthscale,
        volume_section_correlation=volume_section_correlation,
        volume_section_variance=volume_section_variance,
        volume_section_lengthscale=volume_section_lengthscale,
        f0_mel_variance=f0_mel_variance,
        f0_mel_lengthscale=f0_mel_lengthscale,
        section_f0_mel_variance=section_f0_mel_variance,
    )
