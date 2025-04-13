import numpy as np
from .synth_params.global_params import GlobalParams
from .synth_params.local_params import LocalParams
from .constants import boundaries_scale_exp_voiced, boundaries_global_harmonic_volume_initial_bias, boundaries_global_noise_volume_initial_bias, boundaries_global_f0_mel_variance, boundaries_global_f0_mel_initial_bias, boundaries_global_harmonic_envelope_variance, boundaries_global_harmonic_envelope_lengthscale, boundaries_global_harmonic_envelope_initial_bias, boundaries_harmonic_envelope_kernel_name, boundaries_global_noise_distribution_initial_bias, boundaries_global_noise_distribution_mode, boundaries_kernel_freq_name, boundaries_hn_cor, boundaries_local_volume_hn_variance, boundaries_local_volume_hn_K_name,boundaries_local_f0_mel_variance,boundaries_local_f0_mel_lengthscale,boundaries_local_f0_mel_kernel_name, boundaries_n_harmonics, boundaries_f0_quantize,boundaries_ir_diminin

def labelize(value, boundaries):
    """
        boundaries must be 1d list-like in an ascending order.
    """
    boundaries = sorted(boundaries)
    multi_label = np.zeros(len(boundaries) + 1)
    
    for enum, b in enumerate(boundaries):
        if value < b:
            multi_label[enum] = 1.
            return multi_label
    multi_label[-1] = 1.
    
    return multi_label

def labelize_kernel(k_name):
    kernel_dict = {"RBF":0, "Exponential":1, "Cosine":2, "Linear":3, "sde_Brownian":4, "Poly":5, "StdPeriodic":7, "PeriodicExponential":8}
    return kernel_dict[k_name]

def generate_continuous_targets(global_params: GlobalParams, local_params: LocalParams, harmonic_envelope_kernel_name: str, global_noise_distribution_mode:float, kernel_noise_freq_name:str, local_volume_hn_K_name:str, local_f0_mel_kernel_name:str)  -> np.ndarray:
    """
    Generate continuous targets based on the provided global and local parameters.

    Args:
        global_params: An instance of GlobalParams containing global parameters.
        local_params: An instance of LocalParams containing local parameters.

    Returns:
        A numpy array representing the generated continuous targets.
    """
    targets = np.array([global_params.scale_exp_voiced, global_params.harmonic_volume_initial_bias, global_params.noise_volume_initial_bias, global_params.f0_mel_variance, global_params.f0_mel_initial_bias, global_params.harmonic_envelope_variance, global_params.harmonic_envelope_lengthscale, global_params.harmonic_envelope_initial_bias, 
                        labelize_kernel(harmonic_envelope_kernel_name), global_params.noise_distribution_initial_bias, global_noise_distribution_mode, labelize_kernel(kernel_noise_freq_name), local_params.hn_cor, local_params.volume_hn_variance, labelize_kernel(local_volume_hn_K_name), 
                        local_params.f0_mel_variance, local_params.f0_mel_lengthscale, labelize_kernel(local_f0_mel_kernel_name), global_params.n_harmonics, global_params.f0_quantize, global_params.ir_diminin])

    return targets

def generate_discrete_targets(global_params: GlobalParams, local_params: LocalParams, harmonic_envelope_kernel_name: str, global_noise_distribution_mode:float, kernel_freq_name:str, local_volume_hn_K_name:str, local_f0_mel_kernel_name:str) -> dict:
    labels = {
            # Duration of voiced segment
            "label_scale_exp_voiced":
                labelize(global_params.scale_exp_voiced, boundaries_scale_exp_voiced),
            
            # Sharpness of global harmonic volume
            "label_global_harmonic_volume_initial_bias":
                labelize(global_params.harmonic_volume_initial_bias, boundaries_global_harmonic_volume_initial_bias), 
            
            # Sharpness of global noise volume
            "label_global_noise_volume_initial_bias":
                labelize(global_params.noise_volume_initial_bias, boundaries_global_noise_volume_initial_bias),
            
            # Global F0 variation range
            "label_global_f0_mel_variance":
                labelize(global_params.f0_mel_variance, boundaries_global_f0_mel_variance),
                    
            # # Harmonic pitch sharpness
            "label_global_f0_mel_initial_bias":
                labelize(global_params.f0_mel_initial_bias, boundaries_global_f0_mel_initial_bias),
            
            # Harmonic envelope variation range
            "label_global_harmonic_envelope_variance":
                labelize(global_params.harmonic_envelope_variance, boundaries_global_harmonic_envelope_variance),
            
            # Harmonic envelope lengthscale
            "label_global_harmonic_envelope_lengthscale":
                labelize(global_params.harmonic_envelope_lengthscale, boundaries_global_harmonic_envelope_lengthscale),
            
            # Harmonic envelope sharpness
            "label_global_harmonic_envelope_initial_bias":
                labelize(global_params.harmonic_envelope_initial_bias, boundaries_global_harmonic_envelope_initial_bias),

                    
            # Harmonic envelopne kernel
            "label_harmonic_envelope_kernel_name":
                labelize(harmonic_envelope_kernel_name, boundaries_harmonic_envelope_kernel_name),
                    
            # Sharpness of the noise distribution along the frequency axis
            "label_global_noise_distribution_initial_bias":
                labelize(global_params.noise_distribution_initial_bias, boundaries_global_noise_distribution_initial_bias),
                    
            # Spectral centroid of the noise distribution
            "label_global_noise_distribution_mode":
                labelize(global_noise_distribution_mode, boundaries_global_noise_distribution_mode),
                    
            # Kernel type of the inharmonic envelope
            "label_kernel_freq_name":
                labelize(kernel_freq_name, boundaries_kernel_freq_name),
                    
            # Local volume correlation between harmonic and inharmonic components
            "label_hn_cor":
                labelize(local_params.hn_cor, boundaries_hn_cor),
                    
            # Local volume variance
            "label_local_volume_hn_variance":
                labelize(local_params.volume_hn_variance, boundaries_local_volume_hn_variance),
                    
            # Kernel type of the local volume   
            "label_local_volume_hn_K_name":
                labelize(local_volume_hn_K_name, boundaries_local_volume_hn_K_name),
                    
            # Local F0 variance
            "label_local_f0_mel_variance":
                labelize(local_params.f0_mel_variance, boundaries_local_f0_mel_variance),
                    
            "label_local_f0_mel_lengthscale":
                labelize(local_params.f0_mel_lengthscale, boundaries_local_f0_mel_lengthscale),
                    
            "label_local_f0_mel_kernel_name":
                labelize(local_f0_mel_kernel_name, boundaries_local_f0_mel_kernel_name),
                            
            "label_n_harmonics":
                labelize(global_params.n_harmonics, boundaries_n_harmonics),
            "label_f0_quantize":
                labelize(global_params.f0_quantize, boundaries_f0_quantize),
            "label_ir_diminin":
                labelize(global_params.ir_diminin, boundaries_ir_diminin)
        }
    return labels
