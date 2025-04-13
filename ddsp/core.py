"""Reference: DDSP: Differentiable Digital Signal Processing
(https://github.com/magenta/ddsp)
"""

import numpy as np
from scipy import fftpack
from scipy.fft import irfft, rfft
from scipy.interpolate import interp1d
from scipy.signal.windows import hann


# new
def np_sigmoid(a):
    return 1 / (1 + np.exp(-a))


# line 31
def np_float32(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    else:
        return np.array(x, dtype=np.float32)


# line 153
def pad_axis(x, padding=(0, 0), axis=0, **pad_kwargs):
    """Pads only one axis of a tensor.

    Args:
        x: Input tensor.
        padding: Tuple of number of samples to pad (before, after).
        axis: Which axis to pad.
        **pad_kwargs: Other kwargs to pass to tf.pad.

    Returns:
        A tensor padded with padding along axis.
    """
    n_end_dims = len(x.shape) - axis - 1
    n_end_dims *= n_end_dims > 0
    paddings = [[0, 0]] * axis + [list(padding)] + [[0, 0]] * n_end_dims
    return np.pad(x, paddings, **pad_kwargs)


# line 207
def safe_divide(numerator, denominator, eps=1e-7):
    """Avoid dividing by zero by adding a small epsilon."""
    safe_denominator = np.where(denominator == 0.0, eps, denominator)
    return numerator / safe_denominator


# line 387
def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
    """Exponentiated Sigmoid pointwise nonlinearity.

    Bounds input to [threshold, max_value] with slope given by exponent.

    Args:
        x: Input tensor.
        exponent: In nonlinear regime (away from x=0), the output varies by this
            factor for every change of x by 1.0.
        max_value: Limiting value at x=inf.
        threshold: Limiting value at x=-inf. Stablizes training when outputs are
            pushed to 0.

    Returns:
        A tensor with pointwise nonlinearity applied.
    """
    x = np_float32(x)
    return max_value * np_sigmoid(x) ** np.log(exponent) + threshold


# line 573
def resample(inputs, n_timesteps, method="linear", add_endpoint=True):
    """Interpolates a tensor from n_frames to n_timesteps.

    Args:
        inputs: Framewise 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_frames],
            [batch_size, n_frames], [batch_size, n_frames, channels], or
            [batch_size, n_frames, n_freq, channels].
        n_timesteps: Time resolution of the output signal.
        method: Type of resampling, must be in ['nearest', 'linear', 'cubic',
            'window']. Linear and cubic ar typical bilinear, bicubic interpolation.
            'window' uses overlapping windows (only for upsampling) which is smoother
            for amplitude envelopes with large frame sizes.
        add_endpoint: Hold the last timestep for an additional step as the endpoint.
            Then, n_timesteps is divided evenly into n_frames segments. If false, use
            the last timestep as the endpoint, producing (n_frames - 1) segments with
            each having a length of n_timesteps / (n_frames - 1).
    Returns:
        Interpolated 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_timesteps],
            [batch_size, n_timesteps], [batch_size, n_timesteps, channels], or
            [batch_size, n_timesteps, n_freqs, channels].

    Raises:
        ValueError: If method is 'window' and input is 4-D.
        ValueError: If method is not one of 'nearest', 'linear', 'cubic', or
            'window'.
    """

    inputs = np_float32(inputs)
    is_1d = len(inputs.shape) == 1
    is_2d = len(inputs.shape) == 2
    is_4d = len(inputs.shape) == 4

    # Ensure inputs are at least 3d.
    if is_1d:
        inputs = inputs[np.newaxis, :, np.newaxis]
    elif is_2d:
        inputs = inputs[:, :, np.newaxis]

    def _resize_by_interp(method):
        """Closure around tf.image.resize."""
        # Image resize needs 4-D input. Add/remove extra axis if not 4-D.
        outputs = inputs[:, :, np.newaxis, :] if not is_4d else inputs

        # Modified part
        x = np.arange(outputs.shape[1])
        interp_func = interp1d(x, outputs, kind=method, axis=1)
        xnew = np.linspace(x[0], x[-1], n_timesteps)
        outputs = interp_func(xnew)
        return outputs[:, :, 0, :] if not is_4d else outputs

    # Perform resampling.
    if method in ("nearest", "linear", "cubic"):
        outputs = _resize_by_interp(method)
    elif method == "window":
        outputs = upsample_with_windows(inputs, n_timesteps, add_endpoint)
    else:
        raise ValueError(
            "Method ({}) is invalid. Must be one of {}.".format(
                method, "['nearest', 'linear', 'cubic', 'window']"
            )
        )

    # Return outputs to the same dimensionality of the inputs.
    if is_1d:
        outputs = outputs[0, :, 0]
    elif is_2d:
        outputs = outputs[:, :, 0]

    return outputs


# line 645
def upsample_with_windows(inputs, n_timesteps, add_endpoint=True):
    """Upsample a series of frames using using overlapping hann windows.

    Good for amplitude envelopes.
    Args:
        inputs: Framewise 3-D tensor. Shape [batch_size, n_frames, n_channels].
        n_timesteps: The time resolution of the output signal.
        add_endpoint: Hold the last timestep for an additional step as the endpoint.
            Then, n_timesteps is divided evenly into n_frames segments. If false, use
            the last timestep as the endpoint, producing (n_frames - 1) segments with
            each having a length of n_timesteps / (n_frames - 1).

    Returns:
        Upsampled 3-D tensor. Shape [batch_size, n_timesteps, n_channels].

    Raises:
        ValueError: If input does not have 3 dimensions.
        ValueError: If attempting to use function for downsampling.
        ValueError: If n_timesteps is not divisible by n_frames (if add_endpoint is
            true) or n_frames - 1 (if add_endpoint is false).
    """
    inputs = np_float32(inputs)

    if len(inputs.shape) != 3:
        raise ValueError(
            "Upsample_with_windows() only supports 3 dimensions, not {}.".format(
                inputs.shape
            )
        )

    # Mimic behavior of tf.image.resize.
    # For forward (not endpointed), hold value for last interval.
    if add_endpoint:
        inputs = np.concatenate([inputs, inputs[:, -1:, :]], axis=1)

    n_frames = int(inputs.shape[1])
    n_intervals = n_frames - 1

    if n_frames >= n_timesteps:
        raise ValueError(
            "Upsample with windows cannot be used for downsampling"
            "More input frames ({}) than output timesteps ({})".format(
                n_frames, n_timesteps
            )
        )

    if n_timesteps % n_intervals != 0.0:
        minus_one = "" if add_endpoint else " - 1"
        raise ValueError(
            "For upsampling, the target the number of timesteps must be divisible "
            "by the number of input frames{}. (timesteps:{}, frames:{}, "
            "add_endpoint={}).".format(minus_one, n_timesteps, n_frames, add_endpoint)
        )

    # Constant overlap-add, half overlapping windows.
    hop_size = n_timesteps // n_intervals
    window_length = 2 * hop_size
    window = hann(window_length, sym=False)  # [window]

    # # Transpose for overlap_and_add.
    x = np.transpose(inputs, axes=[0, 2, 1])  # [batch_size, n_channels, n_frames]

    # Broadcast multiply.
    # Add dimension for windows [batch_size, n_channels, n_frames, window].
    x = x[:, :, :, np.newaxis]
    window = window[np.newaxis, np.newaxis, np.newaxis, :]
    x_windowed = x * window

    # new closure
    def _overlap_and_add(hop_size):
        oamat = np.zeros(
            (
                x_windowed.shape[0],
                x_windowed.shape[1],
                n_intervals * hop_size + x_windowed.shape[3],
            )
        )  # [batch_size, n_channels, n_timesteps]
        for i in range(n_intervals):
            oamat[:, :, i * hop_size : i * hop_size + x_windowed.shape[3]] += (
                x_windowed[:, :, i, :]
            )
        return oamat

    x = _overlap_and_add(hop_size)

    # # Transpose back.
    x = np.transpose(x, axes=[0, 2, 1])  # [batch_size, n_timesteps, n_channels]

    # Trim the rise and fall of the first and last window.
    return x[:, hop_size:-hop_size, :]


# line 800
def angular_cumsum(angular_frequency, chunk_size=1000):
    """Get phase by cumulative sumation of angular frequency.

    Custom cumsum splits first axis into chunks to avoid accumulation error.
    Just taking tf.sin(tf.cumsum(angular_frequency)) leads to accumulation of
    phase errors that are audible for long segments or at high sample rates. Also,
    in reduced precision settings, cumsum can overflow the threshold.

    During generation, if syntheiszed examples are longer than ~100k samples,
    consider using angular_sum to avoid noticible phase errors. This version is
    currently activated by global gin injection. Set the gin parameter
    `oscillator_bank.use_angular_cumsum=True` to activate.

    Given that we are going to take the sin of the accumulated phase anyways, we
    don't care about the phase modulo 2 pi. This code chops the incoming frequency
    into chunks, applies cumsum to each chunk, takes mod 2pi, and then stitches
    them back together by adding the cumulative values of the final step of each
    chunk to the next chunk.

    Seems to be ~30% faster on CPU, but at least 40% slower on TPU.

    Args:
        angular_frequency: Radians per a sample. Shape [batch, time, ...].
            If there is no batch dimension, one will be temporarily added.
        chunk_size: Number of samples per a chunk. to avoid overflow at low
            precision [chunk_size <= (accumulation_threshold / pi)].

    Returns:
        The accumulated phase in range [0, 2*pi], shape [batch, time, ...].
    """
    # Get tensor shapes.
    n_batch = angular_frequency.shape[0]
    n_time = angular_frequency.shape[1]
    n_dims = len(angular_frequency.shape)
    n_ch_dims = n_dims - 2

    # Pad if needed.
    remainder = n_time % chunk_size
    if remainder:
        pad_amount = chunk_size - remainder
        angular_frequency = pad_axis(angular_frequency, [0, pad_amount], axis=1)

    # Split input into chunks.
    length = angular_frequency.shape[1]
    n_chunks = int(length / chunk_size)
    chunks = np.reshape(
        angular_frequency, [n_batch, n_chunks, chunk_size] + [-1] * n_ch_dims
    )
    phase = np.cumsum(chunks, axis=2)

    # Add offsets.
    # Offset of the next row is the last entry of the previous row.
    offsets = phase[:, :, -1:, ...] % (2.0 * np.pi)
    offsets = pad_axis(offsets, [1, 0], axis=1)
    offsets = offsets[:, :-1, ...]

    # Offset is cumulative among the rows.
    offsets = np.cumsum(offsets, axis=1) % (2.0 * np.pi)
    phase = phase + offsets

    # Put back in original shape.
    phase = phase % (2.0 * np.pi)
    phase = np.reshape(phase, [n_batch, length] + [-1] * n_ch_dims)

    # Remove padding if added it.
    if remainder:
        phase = phase[:, :n_time]
    return phase


# line 869
def remove_above_nyquist(frequency_envelopes, amplitude_envelopes, sample_rate=16000):
    """Set amplitudes for oscillators above nyquist to 0.

    Args:
        frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape
        [batch_size, n_samples, n_sinusoids].
        amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size,
        n_samples, n_sinusoids].
        sample_rate: Sample rate in samples per a second.

    Returns:
        amplitude_envelopes: Sample-wise filtered oscillator amplitude.
        Shape [batch_size, n_samples, n_sinusoids].
    """
    frequency_envelopes = np_float32(frequency_envelopes)
    amplitude_envelopes = np_float32(amplitude_envelopes)

    amplitude_envelopes = np.where(
        np.greater_equal(frequency_envelopes, sample_rate / 2.0),
        np.zeros_like(amplitude_envelopes),
        amplitude_envelopes,
    )
    return amplitude_envelopes


# line 894
def normalize_harmonics(harmonic_distribution, f0_hz=None, sample_rate=None):
    """Normalize the harmonic distribution, optionally removing above nyquist."""
    # Bandlimit the harmonic distribution.
    if sample_rate is not None and f0_hz is not None:
        n_harmonics = int(harmonic_distribution.shape[-1])
        harmonic_frequencies = get_harmonic_frequencies(f0_hz, n_harmonics)
        harmonic_distribution = remove_above_nyquist(
            harmonic_frequencies, harmonic_distribution, sample_rate
        )

    # Normalize
    harmonic_distribution = safe_divide(
        harmonic_distribution, np.sum(harmonic_distribution, axis=-1, keepdims=True)
    )
    return harmonic_distribution


# line 912
def oscillator_bank(
    frequency_envelopes,
    amplitude_envelopes,
    sample_rate=16000,
    sum_sinusoids=True,
    use_angular_cumsum=False,
):
    """Generates audio from sample-wise frequencies for a bank of oscillators.

    Args:
        frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape
            [batch_size, n_samples, n_sinusoids].
        amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size,
            n_samples, n_sinusoids].
        sample_rate: Sample rate in samples per a second.
        sum_sinusoids: Add up audio from all the sinusoids.
        use_angular_cumsum: If synthesized examples are longer than ~100k audio
            samples, consider use_angular_cumsum to avoid accumulating noticible phase
            errors due to the limited precision of tf.cumsum. Unlike the rest of the
            library, this property can be set with global dependency injection with
            gin. Set the gin parameter `oscillator_bank.use_angular_cumsum=True`
            to activate. Avoids accumulation of errors for generation, but don't use
            usually for training because it is slower on accelerators.

    Returns:
        wav: Sample-wise audio. Shape [batch_size, n_samples, n_sinusoids] if
            sum_sinusoids=False, else shape is [batch_size, n_samples].
    """
    frequency_envelopes = np_float32(frequency_envelopes)
    amplitude_envelopes = np_float32(amplitude_envelopes)

    # Don't exceed Nyquist.
    amplitude_envelopes = remove_above_nyquist(
        frequency_envelopes, amplitude_envelopes, sample_rate
    )

    # Angular frequency, Hz -> radians per sample.
    omegas = frequency_envelopes * (2.0 * np.pi)  # rad / sec
    omegas = omegas / float(sample_rate)  # rad / sample

    # Accumulate phase and synthesize.
    if use_angular_cumsum:
        # Avoids accumulation errors.
        phases = angular_cumsum(omegas)
    else:
        phases = np.cumsum(omegas, axis=1)

    # Convert to waveforms.
    wavs = np.sin(phases)
    audio = amplitude_envelopes * wavs  # [mb, n_samples, n_sinusoids]
    if sum_sinusoids:
        audio = np.sum(audio, axis=-1)  # [mb, n_samples]
    return audio


# line 1028
def get_harmonic_frequencies(frequencies, n_harmonics):
    """Create integer multiples of the fundamental frequency.

    Args:
        frequencies: Fundamental frequencies (Hz). Shape [batch_size, :, 1].
        n_harmonics: Number of harmonics.

    Returns:
        harmonic_frequencies: Oscillator frequencies (Hz).
            Shape [batch_size, :, n_harmonics].
    """
    frequencies = np_float32(frequencies)

    f_ratios = np.linspace(1.0, float(n_harmonics), int(n_harmonics), dtype=np.float32)
    f_ratios = f_ratios[np.newaxis, np.newaxis, :]
    harmonic_frequencies = frequencies * f_ratios
    return harmonic_frequencies


# line 1048
def harmonic_synthesis(
    frequencies,
    amplitudes,
    harmonic_shifts=None,
    harmonic_distribution=None,
    n_samples=64000,
    sample_rate=16000,
    amp_resample_method="window",
    use_angular_cumsum=False,
):
    """Generate audio from frame-wise monophonic harmonic oscillator bank.

    Args:
        frequencies: Frame-wise fundamental frequency in Hz. Shape [batch_size,
            n_frames, 1].
        amplitudes: Frame-wise oscillator peak amplitude. Shape [batch_size,
            n_frames, 1].
        harmonic_shifts: Harmonic frequency variations (Hz), zero-centered. Total
            frequency of a harmonic is equal to (frequencies * harmonic_number * (1 +
            harmonic_shifts)). Shape [batch_size, n_frames, n_harmonics].
        harmonic_distribution: Harmonic amplitude variations, ranged zero to one.
            Total amplitude of a harmonic is equal to (amplitudes *
            harmonic_distribution). Shape [batch_size, n_frames, n_harmonics].
        n_samples: Total length of output audio. Interpolates and crops to this.
        sample_rate: Sample rate.
        amp_resample_method: Mode with which to resample amplitude envelopes.
        use_angular_cumsum: Use angular cumulative sum on accumulating phase
            instead of tf.cumsum. More accurate for inference.

    Returns:
        audio: Output audio. Shape [batch_size, n_samples, 1]
    """
    frequencies = np_float32(frequencies)
    amplitudes = np_float32(amplitudes)

    if harmonic_distribution is not None:
        harmonic_distribution = np_float32(harmonic_distribution)
        n_harmonics = int(harmonic_distribution.shape[-1])
    elif harmonic_shifts is not None:
        harmonic_shifts = np_float32(harmonic_shifts)
        n_harmonics = int(harmonic_shifts.shape[-1])
    else:
        n_harmonics = 1

    # Create harmonic frequencies [batch_size, n_frames, n_harmonics].
    harmonic_frequencies = get_harmonic_frequencies(frequencies, n_harmonics)
    if harmonic_shifts is not None:
        harmonic_frequencies *= 1.0 + harmonic_shifts

    # Create harmonic amplitudes [batch_size, n_frames, n_harmonics].
    if harmonic_distribution is not None:
        harmonic_amplitudes = amplitudes * harmonic_distribution
    else:
        harmonic_amplitudes = amplitudes

    # Create sample-wise envelopes.
    frequency_envelopes = resample(harmonic_frequencies, n_samples)  # cycles/sec
    amplitude_envelopes = resample(
        harmonic_amplitudes, n_samples, method=amp_resample_method
    )

    # Synthesize from harmonics [batch_size, n_samples].
    audio = oscillator_bank(
        frequency_envelopes,
        amplitude_envelopes,
        sample_rate=sample_rate,
        use_angular_cumsum=use_angular_cumsum,
    )
    return audio


# line 1317
def get_fft_size(frame_size, ir_size, power_of_2=True):
    """Calculate final size for efficient FFT.

    Args:
        frame_size: Size of the audio frame.
        ir_size: Size of the convolving impulse response.
        power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
            numbers. TPU requires power of 2, while GPU is more flexible.

    Returns:
        fft_size: Size for efficient FFT.
    """
    convolved_frame_size = ir_size + frame_size - 1
    if power_of_2:
        # Next power of 2.
        fft_size = int(2 ** np.ceil(np.log2(convolved_frame_size)))
    else:
        fft_size = int(fftpack.helper.next_fast_len(convolved_frame_size))
    return fft_size


# line 1338
def crop_and_compensate_delay(audio, audio_size, ir_size, padding, delay_compensation):
    """Crop audio output from convolution to compensate for group delay.

    Args:
        audio: Audio after convolution. Tensor of shape [batch, time_steps].
        audio_size: Initial size of the audio before convolution.
        ir_size: Size of the convolving impulse response.
        padding: Either 'valid' or 'same'. For 'same' the final output to be the
            same size as the input audio (audio_timesteps). For 'valid' the audio is
            extended to include the tail of the impulse response (audio_timesteps +
            ir_timesteps - 1).
        delay_compensation: Samples to crop from start of output audio to compensate
            for group delay of the impulse response. If delay_compensation < 0 it
            defaults to automatically calculating a constant group delay of the
            windowed linear phase filter from frequency_impulse_response().

    Returns:
        Tensor of cropped and shifted audio.

    Raises:
        ValueError: If padding is not either 'valid' or 'same'.
    """
    # Crop the output.
    if padding == "valid":
        crop_size = ir_size + audio_size - 1
    elif padding == "same":
        crop_size = audio_size
    else:
        raise ValueError(
            "Padding must be 'valid' or 'same', instead of {}.".format(padding)
        )

    # Compensate for the group delay of the filter by trimming the front.
    # For an impulse response produced by frequency_impulse_response(),
    # the group delay is constant because the filter is linear phase.
    total_size = int(audio.shape[-1])
    crop = total_size - crop_size
    start = (ir_size - 1) // 2 - 1 if delay_compensation < 0 else delay_compensation
    end = crop - start
    return audio[:, start:-end]


# line 1382
def fft_convolve(audio, impulse_response, padding="same", delay_compensation=-1):
    """Filter audio with frames of time-varying impulse responses.

    Time-varying filter. Given audio [batch, n_samples], and a series of impulse
    responses [batch, n_frames, n_impulse_response], splits the audio into frames,
    applies filters, and then overlap-and-adds audio back together.
    Applies non-windowed non-overlapping STFT/ISTFT to efficiently compute
    convolution for large impulse response sizes.

    Args:
        audio: Input audio. Tensor of shape [batch, audio_timesteps].
        impulse_response: Finite impulse response to convolve. Can either be a 2-D
            Tensor of shape [batch, ir_size], or a 3-D Tensor of shape [batch,
            ir_frames, ir_size]. A 2-D tensor will apply a single linear
            time-invariant filter to the audio. A 3-D Tensor will apply a linear
            time-varying filter. Automatically chops the audio into equally shaped
            blocks to match ir_frames.
        padding: Either 'valid' or 'same'. For 'same' the final output to be the
            same size as the input audio (audio_timesteps). For 'valid' the audio is
            extended to include the tail of the impulse response (audio_timesteps +
            ir_timesteps - 1).
        delay_compensation: Samples to crop from start of output audio to compensate
            for group delay of the impulse response. If delay_compensation is less
            than 0 it defaults to automatically calculating a constant group delay of
            the windowed linear phase filter from frequency_impulse_response().

    Returns:
        audio_out: Convolved audio. Tensor of shape
            [batch, audio_timesteps + ir_timesteps - 1] ('valid' padding) or shape
            [batch, audio_timesteps] ('same' padding).

    Raises:
        ValueError: If audio and impulse response have different batch size.
        ValueError: If audio cannot be split into evenly spaced frames. (i.e. the
            number of impulse response frames is on the order of the audio size and
            not a multiple of the audio size.)
    """
    audio, impulse_response = np_float32(audio), np_float32(impulse_response)

    # Get shapes of audio.
    batch_size, audio_size = audio.shape

    # Add a frame dimension to impulse response if it doesn't have one.
    ir_shape = impulse_response.shape
    if len(ir_shape) == 2:
        impulse_response = impulse_response[:, np.newaxis, :]

    # Broadcast impulse response.
    if ir_shape[0] == 1 and batch_size > 1:
        impulse_response = np.tile(impulse_response, (batch_size, 1, 1))

    # Get shapes of impulse response.
    ir_shape = impulse_response.shape
    batch_size_ir, n_ir_frames, ir_size = ir_shape

    # Validate that batch sizes match.
    if batch_size != batch_size_ir:
        raise ValueError(
            "Batch size of audio ({}) and impulse response ({}) must "
            "be the same.".format(batch_size, batch_size_ir)
        )

    # Cut audio into frames.
    frame_size = int(np.ceil(audio_size / n_ir_frames))
    hop_size = frame_size

    # new closure
    def _frame(frame_size, hop_size):
        n_frames = -(-audio_size // hop_size)
        mat = np.zeros((audio.shape[0], n_frames, frame_size))
        for i in range(n_frames):
            mat[:, i, :] = audio[
                :, i * frame_size : (i + 1) * frame_size
            ]  # [batch_size, n_frames, frame_size]
        return mat

    audio_frames = _frame(frame_size, hop_size)

    # Check that number of frames match.
    n_audio_frames = int(audio_frames.shape[1])
    if n_audio_frames != n_ir_frames:
        raise ValueError(
            "Number of Audio frames ({}) and impulse response frames ({}) do not "
            "match. For small hop size = ceil(audio_size / n_ir_frames), "
            "number of impulse response frames must be a multiple of the audio "
            "size.".format(n_audio_frames, n_ir_frames)
        )

    # Pad and FFT the audio and impulse responses.
    fft_size = get_fft_size(frame_size, ir_size, power_of_2=True)
    audio_fft = rfft(audio_frames, fft_size)
    ir_fft = rfft(impulse_response, fft_size)

    # Multiply the FFTs (same as convolution in time).
    audio_ir_fft = np.multiply(audio_fft, ir_fft)

    # Take the IFFT to resynthesize audio.
    audio_frames_out = irfft(audio_ir_fft)

    # new closure
    def _overlap_and_add(hop_size):
        n_intervals = n_audio_frames - 1
        oamat = np.zeros(
            (
                audio_frames_out.shape[0],
                n_intervals * hop_size + audio_frames_out.shape[2],
            )
        )  # [batch_size, n_timesteps]
        for i in range(n_audio_frames):
            oamat[:, i * hop_size : i * hop_size + audio_frames_out.shape[2]] += (
                audio_frames_out[:, i, :]
            )
        return oamat

    audio_out = _overlap_and_add(hop_size)

    # Crop and shift the output audio.
    return crop_and_compensate_delay(
        audio_out, audio_size, ir_size, padding, delay_compensation
    )


# line 1477
def apply_window_to_impulse_response(impulse_response, window_size=0, causal=False):
    """Apply a window to an impulse response and put in causal form.

    Args:
        impulse_response: A series of impulse responses frames to window, of shape
            [batch, n_frames, ir_size].
        window_size: Size of the window to apply in the time domain. If window_size
            is less than 1, it defaults to the impulse_response size.
        causal: Impulse responnse input is in causal form (peak in the middle).

    Returns:
        impulse_response: Windowed impulse response in causal form, with last
            dimension cropped to window_size if window_size is greater than 0 and less
            than ir_size.
    """
    impulse_response = np_float32(impulse_response)

    # If IR is in causal form, put it in zero-phase form.
    if causal:
        impulse_response = np.fft.fftshift(impulse_response, axes=-1)

    # Get a window for better time/frequency resolution than rectangular.
    # Window defaults to IR size, cannot be bigger.
    ir_size = int(impulse_response.shape[-1])
    if (window_size <= 0) or (window_size > ir_size):
        window_size = ir_size
    window = hann(window_size, sym=False)

    # Zero pad the window and put in in zero-phase form.
    padding = ir_size - window_size
    if padding > 0:
        half_idx = (window_size + 1) // 2
        window = np.concatenate(
            [window[half_idx:], np.zeros([padding]), window[:half_idx]], axis=0
        )
    else:
        window = np.fft.fftshift(window, axes=-1)

    # Apply the window, to get new IR (both in zero-phase form).
    window = np.broadcast_to(window, impulse_response.shape)
    impulse_response = window * np.real(impulse_response)

    # Put IR in causal form and trim zero padding.
    if padding > 0:
        first_half_start = (ir_size - (half_idx - 1)) + 1
        second_half_end = half_idx + 1
        impulse_response = np.concatenate(
            [
                impulse_response[..., first_half_start:],
                impulse_response[..., :second_half_end],
            ],
            axis=-1,
        )
    else:
        impulse_response = np.fft.fftshift(impulse_response, axes=-1)

    return impulse_response


# line 1534
def frequency_impulse_response(magnitudes, window_size=0):
    """Get windowed impulse responses using the frequency sampling method.

    Follows the approach in:
    https://ccrma.stanford.edu/~jos/sasp/Windowing_Desired_Impulse_Response.html

    Args:
        magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
            n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
            last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
            f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
            audio into equally sized frames to match frames in magnitudes.
        window_size: Size of the window to apply in the time domain. If window_size
            is less than 1, it defaults to the impulse_response size.

    Returns:
        impulse_response: Time-domain FIR filter of shape
            [batch, frames, window_size] or [batch, window_size].

    Raises:
        ValueError: If window size is larger than fft size.
    """
    # Get the IR (zero-phase form).
    magnitudes = magnitudes + np.zeros_like(magnitudes) * 1.0j
    impulse_response = irfft(magnitudes)

    # Window and put in causal form.
    impulse_response = apply_window_to_impulse_response(impulse_response, window_size)

    return impulse_response


# line 1628
def frequency_filter(audio, magnitudes, window_size=0, padding="same"):
    """Filter audio with a finite impulse response filter.

    Args:
        audio: Input audio. Tensor of shape [batch, audio_timesteps].
        magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
            n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
            last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
            f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
            audio into equally sized frames to match frames in magnitudes.
        window_size: Size of the window to apply in the time domain. If window_size
            is less than 1, it is set as the default (n_frequencies).
        padding: Either 'valid' or 'same'. For 'same' the final output to be the
            same size as the input audio (audio_timesteps). For 'valid' the audio is
            extended to include the tail of the impulse response (audio_timesteps +
            window_size - 1).

    Returns:
        Filtered audio. Tensor of shape
            [batch, audio_timesteps + window_size - 1] ('valid' padding) or shape
            [batch, audio_timesteps] ('same' padding).
    """
    impulse_response = frequency_impulse_response(magnitudes, window_size=window_size)
    return fft_convolve(audio, impulse_response, padding=padding)
