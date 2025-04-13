import GPy
from constants import kernels_list
import numpy as np
import ddsp
from typing import Type, Tuple, List
from librosa import hz_to_note, note_to_hz, stft
from utils import mel_to_hz
from constants import SILENCE
from constants import boundaries_scale_exp_voiced, boundaries_global_harmonic_volume_initial_bias, boundaries_global_noise_volume_initial_bias, boundaries_global_f0_mel_variance, boundaries_global_f0_mel_initial_bias, boundaries_global_harmonic_envelope_variance, boundaries_global_harmonic_envelope_lengthscale, boundaries_global_harmonic_envelope_initial_bias, boundaries_harmonic_envelope_kernel_name, boundaries_global_noise_distribution_initial_bias, boundaries_global_noise_distribution_mode, boundaries_kernel_freq_name, boundaries_hn_cor, boundaries_local_volume_hn_variance, boundaries_local_volume_hn_K_name,boundaries_local_f0_mel_variance,boundaries_local_f0_mel_lengthscale,boundaries_local_f0_mel_kernel_name, boundaries_n_harmonics, boundaries_f0_quantize,boundaries_ir_diminin

n_kernel_kinds = len(kernels_list)

def kernel_sampler(rng=None, variance=None, lengthscale=None, period=None):
    kernels = [
        ["RBF", GPy.kern.RBF(1, variance, lengthscale)],  # variance, lengthscale
        ["Exponential", GPy.kern.Exponential(1, variance, lengthscale)],  # variance, lengthscale
        ["Cosine", GPy.kern.Cosine(1, variance, lengthscale)],  # variance, lengthscale
        ["Linear", GPy.kern.Linear(1, variance)],  # variances
        ["sde_Brownian", GPy.kern.sde_Brownian(1, variance)],  # variance
        ["Poly", GPy.kern.Poly(1, variance, lengthscale, bias=1., order=3.)],  # variance, scale, bias, order
        ["StdPeriodic", GPy.kern.StdPeriodic(1, variance, period, lengthscale)],  # variance, period, lengthscale
        ["PeriodicExponential", GPy.kern.PeriodicExponential(1, variance, lengthscale, period)],  # variance, lengthscale, period
    ]
    k = rng.integers(0, n_kernel_kinds)
    return kernels[k]

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

def sample_voiced_silence_durations(
    rng: np.random.Generator,
    repeat_times: int,
    scale_exp_silent: float,
    scale_exp_voiced: float,
    scale_normal_silent: float,
    scale_normal_voiced: float,
    duration: float,
    frames_per_sec: int,
    n_frames: int
) -> Tuple[List[int], List[int]]:
    """
    Sample the lengths of voiced and silence segments, measured in frames,
    and return both as a tuple.

    This function uses exponential distributions for the initial silent and
    voiced segments, and then uses normal distributions for subsequent segments
    while ensuring that durations remain positive. Finally, it appends an
    additional silent segment specified by `duration`.

    Args:
        rng: A random number generator (NumPy's >=1.17 recommended interface).
        repeat_times: The number of times to alternate between a silent
            segment and a voiced segment.
        scale_exp_silent: The scale parameter for the exponential distribution
            used to sample the first silent segment (in seconds).
        scale_exp_voiced: The scale parameter for the exponential distribution
            used to sample the first voiced segment (in seconds).
        scale_normal_silent: The standard deviation for the normal distribution
            used to sample subsequent silent segments (in seconds).
        scale_normal_voiced: The standard deviation for the normal distribution
            used to sample subsequent voiced segments (in seconds).
        duration: An additional duration in seconds to be appended as a final
            silent segment after all repeats.
        frames_per_sec: The number of frames per second (typically sample_rate // hop_size).
        n_frames: The maximum total number of frames allowed.

    Returns:
        A tuple (voiced_durations, silent_durations), where each is a list of
        segment lengths in frames. The final total of frames is capped at `n_frames`.
    """
    # verbose arrays
    _silent_durations = []
    _voiced_durations = []

    # set by sec
    for i in range(repeat_times):
        if i == 0:
            _silent_durations.append(rng.exponential(scale_exp_silent))
            _voiced_durations.append(rng.exponential(scale_exp_voiced))
            
        else:
            while True:
                candidate_silent_duration = rng.normal(_silent_durations[-1], scale_normal_silent)
                if candidate_silent_duration > 0:
                    break
            _silent_durations.append(candidate_silent_duration)
            
            while True:
                candidate_voiced_duration = rng.normal(_voiced_durations[-1], scale_normal_voiced)
                if candidate_voiced_duration > 0:
                    break
            _voiced_durations.append(candidate_voiced_duration)

    # padding
    _silent_durations.append(duration)

    # change to frames
    _silent_durations = [int(elm * frames_per_sec) for elm in _silent_durations]
    _voiced_durations = [int(elm * frames_per_sec) for elm in _voiced_durations]

    # arrays extracted from verbose arrays
    silent_durations = []
    voiced_durations = []
    total_frames = 0

    for i in range(repeat_times + 1):
        silent_durations.append(_silent_durations[i])
        total_frames += _silent_durations[i]
        if total_frames >= n_frames:
            silent_durations[-1] -= total_frames - n_frames
            break
            
        voiced_durations.append(_voiced_durations[i])
        total_frames += _voiced_durations[i]
        if total_frames >= n_frames:
            voiced_durations[-1] -= total_frames - n_frames
            break
    return voiced_durations, silent_durations

def generate_one_sample(rng: np.random.Generator, max_n_to_mix: int, n_samples:int, small_variance:float, duration:float, frames_per_sec:int, n_frames:int,  default_kernel: Type[GPy.kern.Kern], n_frequencies:int, noise_anchors:int, noise_vagueness:int, sample_rate:int, workid:int):
    n_to_mix = rng.integers(1, max_n_to_mix + 1) #the number of source audio to mix
    summed_wav = np.zeros(n_samples)
    i_mix = 0
    raw_target_list = []
    volume_list = []
    while i_mix < n_to_mix:
        try:
            """(1) Hyper-paramater settings
            """

            # for sections
            lam_poisson = rng.integers(1, 51) # 5
            # repeat_times = 50  
            while True:
                repeat_times = rng.poisson(lam_poisson) # 50
                if repeat_times > 0:
                    break
            scale_exp_silent = 10 ** rng.uniform(-1, 0.5) # 1e-1
            scale_exp_voiced = 10 ** rng.uniform(-1, 1) # 1e1
            scale_normal_silent = small_variance
            scale_normal_voiced = small_variance

            # for globals
            global_volume_variance = small_variance
            global_volume_lengthscale = 1  # 1 (fixed)
            global_volume_cov = np.array([[1, 0.99], [0.99, 1]])  # (fixed)
            global_harmonic_volume_initial_bias = rng.uniform(-4, 5) # -4~5 (boundary -1 / 2)
            if global_harmonic_volume_initial_bias < -1:
                global_noise_volume_initial_bias = rng.uniform(5, 45)  # -15~45 (boundary 5 / 25)
            else:
                global_noise_volume_initial_bias = rng.uniform(-15, 45)  # -15~45
            global_f0_mel_variance = 30 ** rng.uniform(0, 2)  # 1 ~ 900
            global_f0_mel_lengthscale = 1e-2  # (fixed)
            global_f0_mel_initial_bias = 50 ** rng.uniform(1, 2) # 430, 50~2500
            global_harmonic_envelope_variance = 10 ** rng.uniform(-1, 1)  # 0.1, to_consider_as_label
            global_harmonic_envelope_lengthscale = 10 ** rng.uniform(1, 3) # 5e2, to_consider_as_label
            global_harmonic_envelope_initial_bias = 10 ** rng.uniform(-1, 1)  # 0.2, to_consider_as_label
            global_noise_distribution_freq_variance = 1  # 1 (fixed)
            global_noise_distribution_freq_lengthscale = 1e-1  # 1e-1 (fixed)
            global_noise_distribution_time_variance = 1e2  # 1e2 (fixed)
            global_noise_distribution_time_lengthscale = 2  # 2 (fixed)
            global_noise_distribution_initial_bias = 10 ** rng.uniform(-1, 1)  # to_consider_as_label

            # for locals
            hn_cor = rng.uniform(0.8, 1.0) * rng.choice([-1, 1])
            local_volume_hn_correlation = np.array([[1, hn_cor], [hn_cor, 1]]) # np.array([[1, -0.99], [-0.99, 1]])
            local_volume_hn_variance = 10 ** rng.uniform(0, 1)
            local_volume_hn_lengthscale = 1  # to_consider_which_to_fix
            local_volume_section_correlation = np.array([[1, 0.99],[0.99, 1]])  # (fixed)
            local_volume_section_variance = small_variance
            local_volume_section_lengthscale = 1  # to_consider_which_to_fix
            local_f0_mel_variance = 10 ** rng.uniform(0, 4)  # 1e3
            local_f0_mel_lengthscale = 10 ** rng.uniform(-1, 1) # 1e-1
            local_section_f0_mel_variance = 10 ** rng.uniform(-1, 0) # 0.1

            # others
            n_harmonics = rng.integers(5, 51)  # 50, to_consider_as_label
            f0_quantize = rng.choice([False, True])
            ir_diminin = rng.uniform(-150.0, -5.0) 


            """(2) Sound generation
            """

            ### Sample voiced and unvoiced sections
            voiced_durations, silent_durations = sample_voiced_silence_durations(
                rng=rng,
                repeat_times=repeat_times,         
                scale_exp_silent=scale_exp_silent,   
                scale_exp_voiced=scale_exp_voiced,  
                scale_normal_silent=scale_normal_silent,
                scale_normal_voiced=scale_normal_voiced,
                duration=duration,           
                frames_per_sec=frames_per_sec,     
                n_frames=n_frames           
            )

            n_sections = len(voiced_durations)


            ### For voiced sections, set global characteristics

            # Global volume (harmonic & noise)
            global_volume_kernel = default_kernel(1, variance=global_volume_variance, lengthscale=global_volume_lengthscale)
            global_volume_W = np.linalg.cholesky(global_volume_cov)
            global_volume_icm = GPy.util.multioutput.ICM(
                input_dim=1, 
                num_outputs=2, 
                W_rank=2, 
                kernel=global_volume_kernel, 
                W=global_volume_W,
                kappa=1e-8*np.ones(2)
            )

            x = np.arange(repeat_times)
            x_ = np.stack((np.tile(x, 2), np.repeat(np.arange(2), repeat_times)), axis=-1)
            y_volume_means = np.random.multivariate_normal(np.zeros(repeat_times * 2), global_volume_icm.K(x_))
            harmonic_volume_means = y_volume_means[:repeat_times] + global_harmonic_volume_initial_bias
            noise_volume_means = y_volume_means[repeat_times:] + global_noise_volume_initial_bias

            # Global harmonic f0
            global_f0_mel_kernel = default_kernel(1, variance=global_f0_mel_variance, lengthscale=global_f0_mel_lengthscale)
            harmonic_f0_mel_means = np.random.multivariate_normal(np.zeros(repeat_times), global_f0_mel_kernel.K(x[:,None]) + 1e-8*np.identity(repeat_times))
            harmonic_f0_mel_means -=  harmonic_f0_mel_means.min()
            harmonic_f0_mel_means += global_f0_mel_initial_bias

            # Global envelope (harmonic <file-invariant>)
            harmonic_envelope_kernel_period = rng.uniform(0, 2*np.pi)
            harmonic_envelope_kernel_name, harmonic_envelope_kernel = kernel_sampler(
                rng=rng,
                variance=global_harmonic_envelope_variance, 
                lengthscale=global_harmonic_envelope_lengthscale, 
                period=harmonic_envelope_kernel_period)
            x = np.arange(n_frequencies)
            harmonic_envelope_mean = np.random.multivariate_normal(np.zeros(n_frequencies), harmonic_envelope_kernel.K(x[:,None]) + 1e-8*np.identity(n_frequencies))
            harmonic_envelope_mean -= harmonic_envelope_mean.min()
            harmonic_envelope_mean += global_harmonic_envelope_initial_bias
            harmonic_envelope_mean[0] = 0

            # Global envelope (noise <section-invariant, to interpolate>)
            # Correlations are introduced across multiple time points (noise_anchors) in the frequency-domain noise distribution using an Intrinsic Coregionalization Model (ICM).
            x = np.linspace(0, 1, n_frequencies // noise_vagueness)
            x_ = np.stack((np.tile(x, noise_anchors), np.repeat(np.arange(noise_anchors), n_frequencies // noise_vagueness)), axis=-1)
            kernel_freq_period = rng.uniform(0, 2*np.pi)
            kernel_freq_name, kernel_freq = kernel_sampler(
                rng=rng,
                variance=global_noise_distribution_freq_variance, 
                lengthscale=global_noise_distribution_freq_lengthscale,
                period=kernel_freq_period)
            kernel_time = default_kernel(1, variance=global_noise_distribution_time_variance, lengthscale=global_noise_distribution_time_lengthscale)
            W_noise_time = np.linalg.cholesky(kernel_time.K(np.linspace(0, 1, noise_anchors)[:,None]) + np.eye(noise_anchors) * 1e-8)
            # multi-output gaussian prosess
            icm_noise = GPy.util.multioutput.ICM(1, noise_anchors, kernel_freq, W_rank=noise_anchors, W=W_noise_time, kappa=1e-8*np.ones(noise_anchors))
            noise_cor = icm_noise.K(x_)
            # noise distribution (frequency * the number of noised section (noise_anchors))
            noise_distribution_to_interp = np.random.multivariate_normal(np.zeros(n_frequencies // noise_vagueness * noise_anchors), noise_cor)
            noise_distribution_to_interp = noise_distribution_to_interp.reshape(noise_anchors, -1).T
            noise_distribution_to_interp += -noise_distribution_to_interp.min(axis=0) + global_noise_distribution_initial_bias
            noise_distribution_to_interp = noise_distribution_to_interp / (noise_distribution_to_interp.sum(axis=0) + 1e-16)

            ### local harmonic f0
            local_f0_mels = []
            local_f0_hzs = [None for _ in range(n_sections)]
            x = np.linspace(-1, 1, 500)

            # generate initial sample (not resampled)
            local_f0_mel_kernel_period = rng.uniform(0, 2*np.pi)
            local_f0_mel_kernel_name, local_f0_mel_kernel = kernel_sampler(
                rng=rng,
                variance=local_f0_mel_variance, 
                lengthscale=local_f0_mel_lengthscale, 
                period=local_f0_mel_kernel_period)
            local_f0_mel = np.random.multivariate_normal(np.zeros(500), local_f0_mel_kernel.K(x[:,None]) + 1e-8*np.identity(500))
            local_f0_mels.append(local_f0_mel)

            # generate following samples (not resampled)
            local_section_f0_mel_kernel = default_kernel(
                1, 
                variance=local_section_f0_mel_variance, 
                lengthscale=local_section_f0_mel_variance
            )
            # Introduce frequency-wise correlations across local sections.
            for i in range(n_sections-1):
                diffs = np.random.multivariate_normal(np.zeros(500), local_section_f0_mel_kernel.K(x[:,None]) + 1e-8*np.identity(500))
                local_f0_mels.append(local_f0_mels[-1] + diffs)

            # resamplings and change to hz
            for e, _ in enumerate(voiced_durations):
                x = np.linspace(0, 1, voiced_durations[e])
                xp = np.linspace(0, 1, 500)
                local_f0_hzs[e] = mel_to_hz(np.interp(x, xp, local_f0_mels[e], left=0, right=0) + harmonic_f0_mel_means[e])
                local_f0_hzs[e] = np.where(local_f0_hzs[e] <= 0, 0, local_f0_hzs[e])
                if f0_quantize:
                    local_f0_hzs[e] = note_to_hz(hz_to_note(local_f0_hzs[e]))

            # final output
            harmonic_f0s = local_f0_hzs

                
            ### local harmonic and noise volume
            local_harmonic_volumes = []
            local_noise_volumes = []
            x = np.linspace(-1, 1, 500)
            x_ = np.stack((np.tile(x, 2), np.repeat(np.arange(2), 500)), axis=-1)


            local_volume_hn_correlation_W = np.linalg.cholesky(local_volume_hn_correlation)
            local_volume_hn_K_period = rng.uniform(0, 2*np.pi)
            local_volume_hn_K_name, local_volume_hn_K = kernel_sampler(
                rng=rng,
                variance=local_volume_hn_variance, 
                lengthscale=local_volume_hn_lengthscale, 
                period=local_volume_hn_K_period)
            local_volume_hn_icm = GPy.util.multioutput.ICM(
                input_dim=1, num_outputs=2, W_rank=2, kernel=local_volume_hn_K, W=local_volume_hn_correlation_W, kappa=1e-8*np.ones(2))

            # generate initial sample (not resampled)
            hn_volumes = np.random.multivariate_normal(np.zeros(1000), local_volume_hn_icm.K(x_))
            local_harmonic_volumes.append(hn_volumes[:500])
            local_noise_volumes.append(hn_volumes[500:])

            # generate following samples (not resampled)
            local_volume_section_correlation_W = np.linalg.cholesky(local_volume_section_correlation)
            local_volume_section_K = default_kernel(
                1, variance=local_volume_section_variance, lengthscale=local_volume_section_lengthscale)
            local_volume_section_icm = GPy.util.multioutput.ICM(
                input_dim=1, num_outputs=2, W_rank=2, kernel=local_volume_section_K, W=local_volume_section_correlation_W, kappa=1e-8*np.ones(2))
            for i in range(n_sections-1):
                hn_diffs = np.random.multivariate_normal(np.zeros(1000), local_volume_section_icm.K(x_))
                local_harmonic_volumes.append(local_harmonic_volumes[-1] + hn_diffs[:500])
                local_noise_volumes.append(local_noise_volumes[-1] + hn_diffs[500:])

            # resamplings and summation with global trend
            for e, _ in enumerate(voiced_durations):
                x = np.linspace(0, 1, voiced_durations[e])
                xp = np.linspace(0, 1, 500)
                local_harmonic_volumes[e] = np.interp(x, xp, local_harmonic_volumes[e]) + harmonic_volume_means[e]
                local_noise_volumes[e] = np.interp(x, xp, local_noise_volumes[e]) + noise_volume_means[e]

            # final output
            if len(local_harmonic_volumes) > 1:
                harmonic_volumes = np.array(local_harmonic_volumes, dtype=object)
            else:
                harmonic_volumes = np.array(local_harmonic_volumes)
            if len(local_noise_volumes) > 1:
                noise_volumes = np.array(local_noise_volumes, dtype=object)
            else:
                noise_volumes = np.array(local_noise_volumes)

            ### Local harmonic distributions
            local_harmonic_distributions = []
            for e, _ in enumerate(voiced_durations):
                harmonic_distribution = np.zeros([voiced_durations[e], n_harmonics])
                for h in range(n_harmonics):
                    harmonic_distribution[:,h] = np.interp(
                        harmonic_f0s[e] * (h+1), 
                        np.linspace(0, sample_rate / 2, n_frequencies),
                        harmonic_envelope_mean,
                        left = 0.,
                        right = 0.
                        )
                harmonic_distribution = harmonic_distribution / (harmonic_distribution.sum(axis=-1, keepdims=True) + 1e-16)
                local_harmonic_distributions.append(harmonic_distribution)

            ### Local noise distribution
            local_noise_distributions = []

            for e, _ in enumerate(voiced_durations):
                noise_distribution = np.zeros((n_frequencies // noise_vagueness, voiced_durations[e]))
                for h in range(n_frequencies // noise_vagueness):
                    x = np.linspace(0, 1, voiced_durations[e])
                    xp = np.linspace(0, 1, noise_anchors)
                    noise_distribution[h] = np.interp(x, xp, noise_distribution_to_interp[h])
                local_noise_distributions.append(noise_distribution)
                
            ## connected
            connected_harmonic_volumes = []
            connected_harmonic_f0s = []
            connected_noise_volumes = []
            for i in range(repeat_times + 1):
                if i < len(silent_durations):
                    connected_harmonic_volumes.extend([SILENCE for j in range(silent_durations[i])])
                    connected_harmonic_f0s.extend([SILENCE * 0 for j in range(silent_durations[i])])
                    connected_noise_volumes.extend([SILENCE for j in range(silent_durations[i])])
                else:
                    break
                    
                if i < n_sections:
                    connected_harmonic_volumes.extend(harmonic_volumes[i])
                    connected_harmonic_f0s.extend(harmonic_f0s[i])
                    connected_noise_volumes.extend(noise_volumes[i])
                else:
                    break
                
                
            ### generate connected harmonic distributions and noise distributions
            connected_harmonic_distributions = np.zeros((n_frames, n_harmonics))
            connected_noise_distributions = np.zeros((n_frequencies // noise_vagueness, n_frames))
            cur_frame = 0
            for i in range(repeat_times + 1):
                if i < len(silent_durations):
                    connected_harmonic_distributions[cur_frame : cur_frame + silent_durations[i], :] = 1 / n_harmonics
                    connected_noise_distributions[:, cur_frame : cur_frame + silent_durations[i]] = noise_vagueness / n_frequencies
                    cur_frame += silent_durations[i]
                else:
                    break
                    
                if i < n_sections:
                    connected_harmonic_distributions[cur_frame : cur_frame + voiced_durations[i], :] = local_harmonic_distributions[i]
                    connected_noise_distributions[:, cur_frame : cur_frame + voiced_durations[i]] = local_noise_distributions[i]
                    cur_frame += voiced_durations[i]
                else:
                    break

                    
            ### Reverb controls
            n_fade_in = 16 * 10
            ir_size = n_samples
            n_fade_out = ir_size - n_fade_in

            ir = 0.1 * np.random.randn(ir_size)
            ir[:n_fade_in] *= np.linspace(0.5, 1.0, n_fade_in)
            ir[n_fade_in:] *= np.exp(np.linspace(0.0, ir_diminin, n_fade_out))
            ir = ir[np.newaxis, :]


            ### Change to DDSP inputs
            amps = np.array(connected_harmonic_volumes)[np.newaxis, :, np.newaxis]  # Amplitude [batch, n_frames, 1]
            f0_hz = np.array(connected_harmonic_f0s)[np.newaxis, :, np.newaxis]  # Fundamental frequency in Hz [batch, n_frames, 1]
            harmonic_distribution = connected_harmonic_distributions[np.newaxis, :, :]  # Harmonic Distribution [batch, n_frames, n_harmonics]
            noise_amps = np.array(connected_noise_volumes)[np.newaxis, :, np.newaxis]  # Noise amplitude [batch, n_frames, 1]
            noise_distribution = connected_noise_distributions.T[np.newaxis, :, :]  # Noise distributions [batch, n_frames, freqs]
            ir = ir

            ### DDSP generation
            inputs = {
                'amps': amps, 
                'harmonic_distribution': harmonic_distribution,
                'f0_hz': f0_hz,
                'noise_amps': noise_amps,
                'noise_distribution': noise_distribution,
                'ir': ir,
            }
            inputs = {k: v.astype(np.float32) for k, v in inputs.items()}

            harmonic = ddsp.synths.Harmonic(n_samples=n_samples, use_angular_cumsum=True)
            noise = ddsp.synths.FilteredNoise(n_samples=n_samples, initial_bias=0)
            reverb = ddsp.effects.Reverb()

            # Python signal processor chain
            audio_harmonic = harmonic(inputs['amps'],
                                    inputs['harmonic_distribution'],
                                    inputs['f0_hz'])
                
            audio_noise = noise(inputs['noise_amps'],
                                inputs['noise_distribution'])
                
            audio_dry = audio_harmonic + audio_noise
                
            audio_wet = reverb(audio_dry, inputs['ir'])
            audio_wet /= np.abs(audio_wet).max()

            generated_audio = audio_wet[0]
       
            global_noise_distribution_mode = np.argmax(np.mean(noise_distribution_to_interp, axis=1)) # noise_distribution_to_interp: (freq', time=(noise_anchors))

            raw_targets = np.array([scale_exp_voiced, global_harmonic_volume_initial_bias, global_noise_volume_initial_bias, global_f0_mel_variance, global_f0_mel_initial_bias, global_harmonic_envelope_variance, global_harmonic_envelope_lengthscale, global_harmonic_envelope_initial_bias, 
                        labelize_kernel(harmonic_envelope_kernel_name), global_noise_distribution_initial_bias, global_noise_distribution_mode, labelize_kernel(kernel_freq_name), hn_cor, local_volume_hn_variance, labelize_kernel(local_volume_hn_K_name), 
                        local_f0_mel_variance, local_f0_mel_lengthscale, labelize_kernel(local_f0_mel_kernel_name), n_harmonics, f0_quantize, ir_diminin])
            # label information
            labels = {
                # Duration of voiced segment
                "label_scale_exp_voiced":
                    labelize(scale_exp_voiced, boundaries_scale_exp_voiced),
                
                # Sharpness of global harmonic volume
                "label_global_harmonic_volume_initial_bias":
                    labelize(global_harmonic_volume_initial_bias, boundaries_global_harmonic_volume_initial_bias), 
                
                # Sharpness of global noise volume
                "label_global_noise_volume_initial_bias":
                    labelize(global_noise_volume_initial_bias, boundaries_global_noise_volume_initial_bias),
                
                # Global F0 variation range
                "label_global_f0_mel_variance":
                    labelize(global_f0_mel_variance, boundaries_global_f0_mel_variance),
                        
                # # Harmonic pitch sharpness
                "label_global_f0_mel_initial_bias":
                    labelize(global_f0_mel_initial_bias, boundaries_global_f0_mel_initial_bias),
                
                # Harmonic envelope variation range
                "label_global_harmonic_envelope_variance":
                    labelize(global_harmonic_envelope_variance, boundaries_global_harmonic_envelope_variance),
                
                # Harmonic envelope lengthscale
                "label_global_harmonic_envelope_lengthscale":
                    labelize(global_harmonic_envelope_lengthscale, boundaries_global_harmonic_envelope_lengthscale),
                
                # Harmonic envelope sharpness
                "label_global_harmonic_envelope_initial_bias":
                    labelize(global_harmonic_envelope_initial_bias, boundaries_global_harmonic_envelope_initial_bias),

                        
                # Harmonic envelopne kernel
                "label_harmonic_envelope_kernel_name":
                    labelize(harmonic_envelope_kernel_name, boundaries_harmonic_envelope_kernel_name),
                        
                # Sharpness of the noise distribution along the frequency axis
                "label_global_noise_distribution_initial_bias":
                    labelize(global_noise_distribution_initial_bias, boundaries_global_noise_distribution_initial_bias),
                        
                # Spectral centroid of the noise distribution
                "label_global_noise_distribution_mode":
                    labelize(global_noise_distribution_mode, boundaries_global_noise_distribution_mode),
                        
                # Kernel type of the inharmonic envelope
                "label_kernel_freq_name":
                    labelize(kernel_freq_name, boundaries_kernel_freq_name),
                        
                # Local volume correlation between harmonic and inharmonic components
                "label_hn_cor":
                    labelize(hn_cor, boundaries_hn_cor),
                        
                # Local volume variance
                "label_local_volume_hn_variance":
                    labelize(local_volume_hn_variance, boundaries_local_volume_hn_variance),
                        
                # Kernel type of the local volume   
                "label_local_volume_hn_K_name":
                    labelize(local_volume_hn_K_name, boundaries_local_volume_hn_K_name),
                        
                # Local F0 variance
                "label_local_f0_mel_variance":
                    labelize(local_f0_mel_variance, boundaries_local_f0_mel_variance),
                        
                "label_local_f0_mel_lengthscale":
                    labelize(local_f0_mel_lengthscale, boundaries_local_f0_mel_lengthscale),
                        
                "label_local_f0_mel_kernel_name":
                    labelize(local_f0_mel_kernel_name, boundaries_local_f0_mel_kernel_name),
                                
                "label_n_harmonics":
                    labelize(n_harmonics, boundaries_n_harmonics),
                "label_f0_quantize":
                    labelize(f0_quantize, boundaries_f0_quantize),
                "label_ir_diminin":
                    labelize(ir_diminin, boundaries_ir_diminin)
            }

            # when noise is dominant, unlabel harmonics (and vice versa)
            if np.argmax(labels["label_global_harmonic_volume_initial_bias"]) == 0 and np.argmax(labels["label_global_noise_volume_initial_bias"]) == 2:
                labels["label_global_f0_mel_variance"] *= 0
                labels["label_global_f0_mel_initial_bias"] *= 0
                labels["label_global_harmonic_envelope_variance"] *= 0
                labels["label_global_harmonic_envelope_lengthscale"] *= 0
                labels["label_global_harmonic_envelope_initial_bias"] *= 0
                labels["label_harmonic_envelope_kernel_name"] *= 0
                labels["label_hn_cor"] *= 0
                labels["label_local_f0_mel_variance"] *= 0
                labels["label_local_f0_mel_lengthscale"] *= 0
                labels["label_local_f0_mel_kernel_name"] *= 0
                labels["label_n_harmonics"] *= 0
                labels["label_f0_quantize"] *= 0
            elif np.argmax(labels["label_global_harmonic_volume_initial_bias"]) == 2 and np.argmax(labels["label_global_noise_volume_initial_bias"]) == 0:
                labels["label_global_noise_distribution_initial_bias"] *= 0
                labels["label_global_noise_distribution_mode"] *= 0
                labels["label_kernel_freq_name"] *= 0
                labels["label_hn_cor"] *= 0


            # for 100ms label ([1025, 101] -> 100]), 10sec / 100 = 100ms.
            # powerspec = np.abs(stft(audio_wet[0], n_fft=2048, hop_length=1600)) ** 2
            # power = np.interp(np.linspace(0, 100, 100), np.linspace(0, 100, 101), np.sum(powerspec, axis=0))

            # for 16ms label ([1025, 626] -> 625]), 10sec / 625 = 16ms.
            powerspec = np.abs(stft(audio_wet[0], n_fft=2048, hop_length=256)) ** 2
            power = np.interp(np.linspace(0, 100, 625), np.linspace(0, 100, 626), np.sum(powerspec, axis=0))
            power_db = 10 * np.log10(power / np.mean(power))
            volume_list.append(power_db)
            volume_on_frames = np.where(power_db > -10, 1, 0)
            label_all = np.concatenate([*labels.values()])
            raw_targets = raw_targets[:, None] * volume_on_frames
            aligned_label = label_all[:,None] * volume_on_frames
            raw_target_list.append(raw_targets)
            sn_db_power = rng.uniform(-5, 5)
            sn_rate_power = 0.001 * (10 ** (sn_db_power / 10))
            summed_wav += generated_audio * np.sqrt(sn_rate_power)

            if i_mix == 0:
                summed_label = aligned_label[None, ...]  # 1 x labels x frames
            else:
                summed_label = np.concatenate([summed_label, aligned_label[None, ...]])
            
            i_mix += 1

        except Exception:
            print(f"Work id {workid}: Skipped (Some error happened).")
    
    summed_label = summed_label.sum(axis=0)
    summed_label = np.where(summed_label > 0, 1, 0).astype(np.int8)

    return summed_wav, summed_label, raw_target_list, volume_list