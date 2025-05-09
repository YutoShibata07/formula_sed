import GPy
import numpy as np
import ddsp
from typing import Type
from librosa import hz_to_note, note_to_hz, stft
from .utils import mel_to_hz
from .constants import SILENCE
from .synth_params import sample_global_params, sample_local_params
from .kernels import kernel_sampler
from .duration import sample_voiced_silence_durations
from .labeling import generate_continuous_targets, generate_discrete_targets


def generate_one_sample(
    rng: np.random.Generator,
    max_n_to_mix: int,
    n_samples: int,
    small_variance: float,
    duration: float,
    frames_per_sec: int,
    n_frames: int,
    default_kernel: Type[GPy.kern.Kern],
    n_frequencies: int,
    noise_anchors: int,
    noise_vagueness: int,
    sample_rate: int,
    workid: int,
):
    n_to_mix = rng.integers(1, max_n_to_mix + 1)  # the number of source audio to mix
    summed_wav = np.zeros(n_samples)
    i_mix = 0
    raw_target_list = []
    volume_list = []
    while i_mix < n_to_mix:
        try:
            """(1) Hyper-paramater settings
            """

            # for sections
            lam_poisson = rng.integers(1, 51)  # 5
            # repeat_times = 50
            while True:
                repeat_times = rng.poisson(lam_poisson)  # 50
                if repeat_times > 0:
                    break

            # for globals
            global_params = sample_global_params(rng, small_variance=small_variance)

            # for locals
            local_params = sample_local_params(rng=rng, small_variance=small_variance)

            """(2) Sound generation
            """

            ### Sample voiced and unvoiced sections
            voiced_durations, silent_durations = sample_voiced_silence_durations(
                rng=rng,
                repeat_times=repeat_times,
                scale_exp_silent=global_params.scale_exp_silent,
                scale_exp_voiced=global_params.scale_exp_voiced,
                scale_normal_silent=global_params.scale_normal_silent,
                scale_normal_voiced=global_params.scale_normal_voiced,
                duration=duration,
                frames_per_sec=frames_per_sec,
                n_frames=n_frames,
            )

            n_sections = len(voiced_durations)

            ### For voiced sections, set global characteristics

            # Global volume (harmonic & noise)
            global_volume_kernel = default_kernel(
                1,
                variance=global_params.volume_variance,
                lengthscale=global_params.volume_lengthscale,
            )
            global_volume_W = np.linalg.cholesky(global_params.volume_cov)
            global_volume_icm = GPy.util.multioutput.ICM(
                input_dim=1,
                num_outputs=2,
                W_rank=2,
                kernel=global_volume_kernel,
                W=global_volume_W,
                kappa=1e-8 * np.ones(2),
            )

            x = np.arange(repeat_times)
            x_ = np.stack(
                (np.tile(x, 2), np.repeat(np.arange(2), repeat_times)), axis=-1
            )
            y_volume_means = np.random.multivariate_normal(
                np.zeros(repeat_times * 2), global_volume_icm.K(x_)
            )
            harmonic_volume_means = (
                y_volume_means[:repeat_times]
                + global_params.harmonic_volume_initial_bias
            )
            noise_volume_means = (
                y_volume_means[repeat_times:] + global_params.noise_volume_initial_bias
            )

            # Global harmonic f0
            global_f0_mel_kernel = default_kernel(
                1,
                variance=global_params.f0_mel_variance,
                lengthscale=global_params.f0_mel_lengthscale,
            )
            harmonic_f0_mel_means = np.random.multivariate_normal(
                np.zeros(repeat_times),
                global_f0_mel_kernel.K(x[:, None]) + 1e-8 * np.identity(repeat_times),
            )
            harmonic_f0_mel_means -= harmonic_f0_mel_means.min()
            harmonic_f0_mel_means += global_params.f0_mel_initial_bias

            # Global envelope (harmonic <file-invariant>)
            harmonic_envelope_kernel_period = rng.uniform(0, 2 * np.pi)
            harmonic_envelope_kernel_name, harmonic_envelope_kernel = kernel_sampler(
                rng=rng,
                variance=global_params.harmonic_envelope_variance,
                lengthscale=global_params.harmonic_envelope_lengthscale,
                period=harmonic_envelope_kernel_period,
            )
            x = np.arange(n_frequencies)
            harmonic_envelope_mean = np.random.multivariate_normal(
                np.zeros(n_frequencies),
                harmonic_envelope_kernel.K(x[:, None])
                + 1e-8 * np.identity(n_frequencies),
            )
            harmonic_envelope_mean -= harmonic_envelope_mean.min()
            harmonic_envelope_mean += global_params.harmonic_envelope_initial_bias
            harmonic_envelope_mean[0] = 0

            # Global envelope (noise <section-invariant, to interpolate>)
            # Correlations are introduced across multiple time points (noise_anchors) in the frequency-domain noise distribution using an Intrinsic Coregionalization Model (ICM).
            x = np.linspace(0, 1, n_frequencies // noise_vagueness)
            x_ = np.stack(
                (
                    np.tile(x, noise_anchors),
                    np.repeat(
                        np.arange(noise_anchors), n_frequencies // noise_vagueness
                    ),
                ),
                axis=-1,
            )
            kernel_freq_period = rng.uniform(0, 2 * np.pi)
            kernel_freq_name, kernel_freq = kernel_sampler(
                rng=rng,
                variance=global_params.noise_distribution_freq_variance,
                lengthscale=global_params.noise_distribution_freq_lengthscale,
                period=kernel_freq_period,
            )
            kernel_time = default_kernel(
                1,
                variance=global_params.noise_distribution_time_variance,
                lengthscale=global_params.noise_distribution_time_lengthscale,
            )
            W_noise_time = np.linalg.cholesky(
                kernel_time.K(np.linspace(0, 1, noise_anchors)[:, None])
                + np.eye(noise_anchors) * 1e-8
            )
            # multi-output gaussian prosess
            icm_noise = GPy.util.multioutput.ICM(
                1,
                noise_anchors,
                kernel_freq,
                W_rank=noise_anchors,
                W=W_noise_time,
                kappa=1e-8 * np.ones(noise_anchors),
            )
            noise_cor = icm_noise.K(x_)
            # noise distribution (frequency * the number of noised section (noise_anchors))
            noise_distribution_to_interp = np.random.multivariate_normal(
                np.zeros(n_frequencies // noise_vagueness * noise_anchors), noise_cor
            )
            noise_distribution_to_interp = noise_distribution_to_interp.reshape(
                noise_anchors, -1
            ).T
            noise_distribution_to_interp += (
                -noise_distribution_to_interp.min(axis=0)
                + global_params.noise_distribution_initial_bias
            )
            noise_distribution_to_interp = noise_distribution_to_interp / (
                noise_distribution_to_interp.sum(axis=0) + 1e-16
            )

            ### local harmonic f0
            local_f0_mels = []
            local_f0_hzs = [None for _ in range(n_sections)]
            x = np.linspace(-1, 1, 500)

            # generate initial sample (not resampled)
            local_f0_mel_kernel_period = rng.uniform(0, 2 * np.pi)
            local_f0_mel_kernel_name, local_f0_mel_kernel = kernel_sampler(
                rng=rng,
                variance=local_params.f0_mel_variance,
                lengthscale=local_params.f0_mel_lengthscale,
                period=local_f0_mel_kernel_period,
            )
            local_f0_mel = np.random.multivariate_normal(
                np.zeros(500),
                local_f0_mel_kernel.K(x[:, None]) + 1e-8 * np.identity(500),
            )
            local_f0_mels.append(local_f0_mel)

            # generate following samples (not resampled)
            local_section_f0_mel_kernel = default_kernel(
                1,
                variance=local_params.f0_mel_variance,
                lengthscale=local_params.f0_mel_variance,
            )
            # Introduce frequency-wise correlations across local sections.
            for i in range(n_sections - 1):
                diffs = np.random.multivariate_normal(
                    np.zeros(500),
                    local_section_f0_mel_kernel.K(x[:, None]) + 1e-8 * np.identity(500),
                )
                local_f0_mels.append(local_f0_mels[-1] + diffs)

            # resamplings and change to hz
            for e, _ in enumerate(voiced_durations):
                x = np.linspace(0, 1, voiced_durations[e])
                xp = np.linspace(0, 1, 500)
                local_f0_hzs[e] = mel_to_hz(
                    np.interp(x, xp, local_f0_mels[e], left=0, right=0)
                    + harmonic_f0_mel_means[e]
                )
                local_f0_hzs[e] = np.where(local_f0_hzs[e] <= 0, 0, local_f0_hzs[e])
                if global_params.f0_quantize:
                    local_f0_hzs[e] = note_to_hz(hz_to_note(local_f0_hzs[e]))

            # final output
            harmonic_f0s = local_f0_hzs

            ### local harmonic and noise volume
            local_harmonic_volumes = []
            local_noise_volumes = []
            x = np.linspace(-1, 1, 500)
            x_ = np.stack((np.tile(x, 2), np.repeat(np.arange(2), 500)), axis=-1)

            local_volume_hn_correlation_W = np.linalg.cholesky(
                local_params.volume_hn_correlation
            )
            local_volume_hn_K_period = rng.uniform(0, 2 * np.pi)
            local_volume_hn_K_name, local_volume_hn_K = kernel_sampler(
                rng=rng,
                variance=local_params.volume_hn_variance,
                lengthscale=local_params.volume_hn_lengthscale,
                period=local_volume_hn_K_period,
            )
            local_volume_hn_icm = GPy.util.multioutput.ICM(
                input_dim=1,
                num_outputs=2,
                W_rank=2,
                kernel=local_volume_hn_K,
                W=local_volume_hn_correlation_W,
                kappa=1e-8 * np.ones(2),
            )

            # generate initial sample (not resampled)
            hn_volumes = np.random.multivariate_normal(
                np.zeros(1000), local_volume_hn_icm.K(x_)
            )
            local_harmonic_volumes.append(hn_volumes[:500])
            local_noise_volumes.append(hn_volumes[500:])

            # generate following samples (not resampled)
            local_volume_section_correlation_W = np.linalg.cholesky(
                local_params.volume_section_correlation
            )
            local_volume_section_K = default_kernel(
                1,
                variance=local_params.volume_section_variance,
                lengthscale=local_params.volume_section_lengthscale,
            )
            local_volume_section_icm = GPy.util.multioutput.ICM(
                input_dim=1,
                num_outputs=2,
                W_rank=2,
                kernel=local_volume_section_K,
                W=local_volume_section_correlation_W,
                kappa=1e-8 * np.ones(2),
            )
            for i in range(n_sections - 1):
                hn_diffs = np.random.multivariate_normal(
                    np.zeros(1000), local_volume_section_icm.K(x_)
                )
                local_harmonic_volumes.append(
                    local_harmonic_volumes[-1] + hn_diffs[:500]
                )
                local_noise_volumes.append(local_noise_volumes[-1] + hn_diffs[500:])

            # resamplings and summation with global trend
            for e, _ in enumerate(voiced_durations):
                x = np.linspace(0, 1, voiced_durations[e])
                xp = np.linspace(0, 1, 500)
                local_harmonic_volumes[e] = (
                    np.interp(x, xp, local_harmonic_volumes[e])
                    + harmonic_volume_means[e]
                )
                local_noise_volumes[e] = (
                    np.interp(x, xp, local_noise_volumes[e]) + noise_volume_means[e]
                )

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
                harmonic_distribution = np.zeros(
                    [voiced_durations[e], global_params.n_harmonics]
                )
                for h in range(global_params.n_harmonics):
                    harmonic_distribution[:, h] = np.interp(
                        harmonic_f0s[e] * (h + 1),
                        np.linspace(0, sample_rate / 2, n_frequencies),
                        harmonic_envelope_mean,
                        left=0.0,
                        right=0.0,
                    )
                harmonic_distribution = harmonic_distribution / (
                    harmonic_distribution.sum(axis=-1, keepdims=True) + 1e-16
                )
                local_harmonic_distributions.append(harmonic_distribution)

            ### Local noise distribution
            local_noise_distributions = []

            for e, _ in enumerate(voiced_durations):
                noise_distribution = np.zeros(
                    (n_frequencies // noise_vagueness, voiced_durations[e])
                )
                for h in range(n_frequencies // noise_vagueness):
                    x = np.linspace(0, 1, voiced_durations[e])
                    xp = np.linspace(0, 1, noise_anchors)
                    noise_distribution[h] = np.interp(
                        x, xp, noise_distribution_to_interp[h]
                    )
                local_noise_distributions.append(noise_distribution)

            ## connected
            connected_harmonic_volumes = []
            connected_harmonic_f0s = []
            connected_noise_volumes = []
            for i in range(repeat_times + 1):
                if i < len(silent_durations):
                    connected_harmonic_volumes.extend(
                        [SILENCE for j in range(silent_durations[i])]
                    )
                    connected_harmonic_f0s.extend(
                        [SILENCE * 0 for j in range(silent_durations[i])]
                    )
                    connected_noise_volumes.extend(
                        [SILENCE for j in range(silent_durations[i])]
                    )
                else:
                    break

                if i < n_sections:
                    connected_harmonic_volumes.extend(harmonic_volumes[i])
                    connected_harmonic_f0s.extend(harmonic_f0s[i])
                    connected_noise_volumes.extend(noise_volumes[i])
                else:
                    break

            ### generate connected harmonic distributions and noise distributions
            connected_harmonic_distributions = np.zeros(
                (n_frames, global_params.n_harmonics)
            )
            connected_noise_distributions = np.zeros(
                (n_frequencies // noise_vagueness, n_frames)
            )
            cur_frame = 0
            for i in range(repeat_times + 1):
                if i < len(silent_durations):
                    connected_harmonic_distributions[
                        cur_frame : cur_frame + silent_durations[i], :
                    ] = 1 / global_params.n_harmonics
                    connected_noise_distributions[
                        :, cur_frame : cur_frame + silent_durations[i]
                    ] = noise_vagueness / n_frequencies
                    cur_frame += silent_durations[i]
                else:
                    break

                if i < n_sections:
                    connected_harmonic_distributions[
                        cur_frame : cur_frame + voiced_durations[i], :
                    ] = local_harmonic_distributions[i]
                    connected_noise_distributions[
                        :, cur_frame : cur_frame + voiced_durations[i]
                    ] = local_noise_distributions[i]
                    cur_frame += voiced_durations[i]
                else:
                    break

            ### Reverb controls
            n_fade_in = 16 * 10
            ir_size = n_samples
            n_fade_out = ir_size - n_fade_in

            ir = 0.1 * np.random.randn(ir_size)
            ir[:n_fade_in] *= np.linspace(0.5, 1.0, n_fade_in)
            ir[n_fade_in:] *= np.exp(
                np.linspace(0.0, global_params.ir_diminin, n_fade_out)
            )
            ir = ir[np.newaxis, :]

            ### Change to DDSP inputs
            amps = np.array(connected_harmonic_volumes)[
                np.newaxis, :, np.newaxis
            ]  # Amplitude [batch, n_frames, 1]
            f0_hz = np.array(connected_harmonic_f0s)[
                np.newaxis, :, np.newaxis
            ]  # Fundamental frequency in Hz [batch, n_frames, 1]
            harmonic_distribution = connected_harmonic_distributions[
                np.newaxis, :, :
            ]  # Harmonic Distribution [batch, n_frames, n_harmonics]
            noise_amps = np.array(connected_noise_volumes)[
                np.newaxis, :, np.newaxis
            ]  # Noise amplitude [batch, n_frames, 1]
            noise_distribution = connected_noise_distributions.T[
                np.newaxis, :, :
            ]  # Noise distributions [batch, n_frames, freqs]
            ir = ir

            ### DDSP generation
            inputs = {
                "amps": amps,
                "harmonic_distribution": harmonic_distribution,
                "f0_hz": f0_hz,
                "noise_amps": noise_amps,
                "noise_distribution": noise_distribution,
                "ir": ir,
            }
            inputs = {k: v.astype(np.float32) for k, v in inputs.items()}

            harmonic = ddsp.synths.Harmonic(
                n_samples=n_samples, use_angular_cumsum=True
            )
            noise = ddsp.synths.FilteredNoise(n_samples=n_samples, initial_bias=0)
            reverb = ddsp.effects.Reverb()

            # Python signal processor chain
            audio_harmonic = harmonic(
                inputs["amps"], inputs["harmonic_distribution"], inputs["f0_hz"]
            )

            audio_noise = noise(inputs["noise_amps"], inputs["noise_distribution"])

            audio_dry = audio_harmonic + audio_noise

            audio_wet = reverb(audio_dry, inputs["ir"])
            audio_wet /= np.abs(audio_wet).max()

            generated_audio = audio_wet[0]

            global_noise_distribution_mode = np.argmax(
                np.mean(noise_distribution_to_interp, axis=1)
            )  # noise_distribution_to_interp: (freq', time=(noise_anchors))

            raw_targets = generate_continuous_targets(
                global_params=global_params,
                local_params=local_params,
                harmonic_envelope_kernel_name=harmonic_envelope_kernel_name,
                global_noise_distribution_mode=global_noise_distribution_mode,
                kernel_noise_freq_name=kernel_freq_name,
                local_volume_hn_K_name=local_volume_hn_K_name,
                local_f0_mel_kernel_name=local_f0_mel_kernel_name,
            )
            # label information
            labels = generate_discrete_targets(
                global_params=global_params,
                local_params=local_params,
                harmonic_envelope_kernel_name=harmonic_envelope_kernel_name,
                global_noise_distribution_mode=global_noise_distribution_mode,
                kernel_freq_name=kernel_freq_name,
                local_volume_hn_K_name=local_volume_hn_K_name,
                local_f0_mel_kernel_name=local_f0_mel_kernel_name,
            )

            # when noise is dominant, unlabel harmonics (and vice versa)
            if (
                np.argmax(labels["label_global_harmonic_volume_initial_bias"]) == 0
                and np.argmax(labels["label_global_noise_volume_initial_bias"]) == 2
            ):
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
            elif (
                np.argmax(labels["label_global_harmonic_volume_initial_bias"]) == 2
                and np.argmax(labels["label_global_noise_volume_initial_bias"]) == 0
            ):
                labels["label_global_noise_distribution_initial_bias"] *= 0
                labels["label_global_noise_distribution_mode"] *= 0
                labels["label_kernel_freq_name"] *= 0
                labels["label_hn_cor"] *= 0

            # for 100ms label ([1025, 101] -> 100]), 10sec / 100 = 100ms.
            # powerspec = np.abs(stft(audio_wet[0], n_fft=2048, hop_length=1600)) ** 2
            # power = np.interp(np.linspace(0, 100, 100), np.linspace(0, 100, 101), np.sum(powerspec, axis=0))

            # for 16ms label ([1025, 626] -> 625]), 10sec / 625 = 16ms.
            powerspec = np.abs(stft(audio_wet[0], n_fft=2048, hop_length=256)) ** 2
            power = np.interp(
                np.linspace(0, 100, 625),
                np.linspace(0, 100, 626),
                np.sum(powerspec, axis=0),
            )
            power_db = 10 * np.log10(power / np.mean(power))
            volume_list.append(power_db)
            volume_on_frames = np.where(power_db > -10, 1, 0)
            label_all = np.concatenate([*labels.values()])
            raw_targets = raw_targets[:, None] * volume_on_frames
            aligned_label = label_all[:, None] * volume_on_frames
            raw_target_list.append(raw_targets)
            sn_db_power = rng.uniform(-5, 5)
            sn_rate_power = 0.001 * (10 ** (sn_db_power / 10))
            summed_wav += generated_audio * np.sqrt(sn_rate_power)

            if i_mix == 0:
                summed_label = aligned_label[None, ...]  # 1 x labels x frames
            else:
                summed_label = np.concatenate([summed_label, aligned_label[None, ...]])

            i_mix += 1

        except Exception as e:
            print(f"Work id {workid}: {e}")
            print(f"Work id {workid}: Skipped (Some error happened).")

    summed_label = summed_label.sum(axis=0)
    summed_label = np.where(summed_label > 0, 1, 0).astype(np.int8)
    assert np.all(np.isin(summed_label, [0, 1])), (
        "summed_label contains values other than 0 and 1."
    )

    return summed_wav, summed_label, raw_target_list, volume_list
