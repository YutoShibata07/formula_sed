import GPy
import numpy as np
from .kernels import kernels_list

SILENCE = -10 # mel-spec values when generated data is silent

# Boundaries to generate ground truth labels
boundaries_of_kernels = [kernel[0] for kernel in sorted(kernels_list)[1:]]
boundaries_scale_exp_voiced = [1, 5]
boundaries_global_harmonic_volume_initial_bias = [-1, 2]
boundaries_global_noise_volume_initial_bias = [5, 25]

boundaries_global_f0_mel_variance = [30**0.5, 30**1., 30**1.5]
boundaries_global_f0_mel_initial_bias = [50**1.25, 50**1.5, 50**1.75]
boundaries_global_harmonic_envelope_variance = [10**-0.5, 10**0., 10**0.5]

boundaries_global_harmonic_envelope_lengthscale = [10**1.5, 10**2., 10**2.5]
boundaries_global_harmonic_envelope_initial_bias = [10**-0.5, 10**0., 10**0.5]
boundaries_global_harmonic_envelope_center = "Not use"

boundaries_harmonic_envelope_kernel_name = boundaries_of_kernels
boundaries_global_noise_distribution_initial_bias = [10**-0.5, 10**0., 10**0.5]
boundaries_global_noise_distribution_mode = np.linspace(10, 90, 9)

boundaries_kernel_freq_name = boundaries_of_kernels
boundaries_hn_cor = [0.]
boundaries_local_volume_hn_variance = [10**0.25, 10**0.5, 10**0.75]

boundaries_local_volume_hn_K_name = boundaries_of_kernels
boundaries_local_f0_mel_variance = [10**1, 10**2, 10**3]
boundaries_local_f0_mel_lengthscale = [10**-0.5, 10**0., 10**0.5]

boundaries_local_f0_mel_kernel_name = boundaries_of_kernels
boundaries_n_harmonics = [10, 30]
boundaries_f0_quantize = [0.5]
boundaries_ir_diminin = [-100, -50]

