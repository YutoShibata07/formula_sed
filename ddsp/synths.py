"""Reference: DDSP: Differentiable Digital Signal Processing
(https://github.com/magenta/ddsp)
"""

import numpy as np

from ddsp import core
from ddsp import processors


class Harmonic(processors.Processor):
    """Synthesize audio with a bank of harmonic sinusoidal oscillators."""

    def __init__(
        self,
        n_samples=64000,
        sample_rate=16000,
        scale_fn=core.exp_sigmoid,
        normalize_below_nyquist=True,
        amp_resample_method="window",
        use_angular_cumsum=False,
    ):
        """Constructor.

        Args:
            n_samples: Fixed length of output audio.
            sample_rate: Samples per a second.
            scale_fn: Scale function for amplitude and harmonic distribution inputs.
            normalize_below_nyquist: Remove harmonics above the nyquist frequency
                and normalize the remaining harmonic distribution to sum to 1.0.
            amp_resample_method: Mode with which to resample amplitude envelopes.
                Must be in ['nearest', 'linear', 'cubic', 'window']. 'window' uses
                overlapping windows (only for upsampling) which is smoother
                for amplitude envelopes with large frame sizes.
            use_angular_cumsum: Use angular cumulative sum on accumulating phase
                instead of tf.cumsum. If synthesized examples are longer than ~100k
                audio samples, consider use_angular_cumsum to avoid accumulating
                noticible phase errors due to the limited precision of tf.cumsum.
                However, using angular cumulative sum is slower on accelerators.
            name: Synth name.
        """
        super().__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.scale_fn = scale_fn
        self.normalize_below_nyquist = normalize_below_nyquist
        self.amp_resample_method = amp_resample_method
        self.use_angular_cumsum = use_angular_cumsum

    def get_controls(self, amplitudes, harmonic_distribution, f0_hz):
        """Convert network output tensors into a dictionary of synthesizer controls.

        Args:
            amplitudes: 3-D Tensor of synthesizer controls, of shape
                [batch, time, 1].
            harmonic_distribution: 3-D Tensor of synthesizer controls, of shape
                [batch, time, n_harmonics].
            f0_hz: Fundamental frequencies in hertz. Shape [batch, time, 1].

        Returns:
            controls: Dictionary of tensors of synthesizer controls.
        """
        # Scale the amplitudes.
        if self.scale_fn is not None:
            amplitudes = self.scale_fn(amplitudes)
            # harmonic_distribution = self.scale_fn(harmonic_distribution)

        # harmonic_distribution = core.normalize_harmonics(
        #     harmonic_distribution, f0_hz,
        #     self.sample_rate if self.normalize_below_nyquist else None)

        return {
            "amplitudes": amplitudes,
            "harmonic_distribution": harmonic_distribution,
            "f0_hz": f0_hz,
        }

    def get_signal(self, amplitudes, harmonic_distribution, f0_hz):
        """Synthesize audio with harmonic synthesizer from controls.

        Args:
            amplitudes: Amplitude tensor of shape [batch, n_frames, 1]. Expects
                float32 that is strictly positive.
            harmonic_distribution: Tensor of shape [batch, n_frames, n_harmonics].
                Expects float32 that is strictly positive and normalized in the last
                dimension.
            f0_hz: The fundamental frequency in Hertz. Tensor of shape [batch,
                n_frames, 1].

        Returns:
            signal: A tensor of harmonic waves of shape [batch, n_samples].
        """
        signal = core.harmonic_synthesis(
            frequencies=f0_hz,
            amplitudes=amplitudes,
            harmonic_distribution=harmonic_distribution,
            n_samples=self.n_samples,
            sample_rate=self.sample_rate,
            amp_resample_method=self.amp_resample_method,
            use_angular_cumsum=self.use_angular_cumsum,
        )
        return signal


class FilteredNoise(processors.Processor):
    """Synthesize audio by filtering white noise."""

    def __init__(
        self,
        n_samples=64000,
        window_size=257,
        scale_fn=core.exp_sigmoid,
        initial_bias=-5.0,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.window_size = window_size
        self.scale_fn = scale_fn
        self.initial_bias = initial_bias

    def get_controls(self, noise_amps, noise_distribution):
        """Convert network outputs into a dictionary of synthesizer controls.

        Args:
        noise_amps: 3-D Tensor of synthesizer controls, of shape
            [batch, time, 1].
        noise_distribution: 3-D Tensor of synthesizer controls, of shape
            [batch, time, n_filter_banks].

        Returns:
        controls: Dictionary of tensors of synthesizer controls.
        """
        # Scale the magnitudes.
        if self.scale_fn is not None:
            noise_amps = self.scale_fn(noise_amps + self.initial_bias)

        return {"noise_amps": noise_amps, "noise_distribution": noise_distribution}

    def get_signal(self, noise_amps, noise_distribution):
        """Synthesize audio with filtered white noise.

        Args:
        noise_amps: Amplitude tensor of shape [batch, n_frames, 1]. Expects
            float32 that is strictly positive.
        noise_distribution: Tensor of shape [batch, n_frames, n_filter_banks].
            Expects float32 that is strictly positive and normalized in the last
            dimension.

        Returns:
        signal: A tensor of noise waves of shape [batch, n_samples, 1].
        """
        batch_size = int(noise_amps.shape[0])
        rng = np.random.default_rng()
        signal = rng.uniform(low=-1.0, high=1.0, size=[batch_size, self.n_samples])
        magnitudes = noise_amps * noise_distribution
        magnitudes *= noise_distribution.shape[-1] / 10 * 2
        return core.frequency_filter(signal, magnitudes, window_size=self.window_size)
