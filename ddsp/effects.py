"""Reference: DDSP: Differentiable Digital Signal Processing
(https://github.com/magenta/ddsp)
"""

from ddsp import core
from ddsp import processors
from ddsp import synths
import numpy as np


np_float32 = core.np_float32


class Reverb(processors.Processor):
    """Convolutional (FIR) reverb."""

    def __init__(
        self, trainable=False, reverb_length=48000, add_dry=True, name="reverb"
    ):
        """Takes neural network outputs directly as the impulse response.

        Args:
            trainable: Learn the impulse_response as a single variable for the entire
                dataset.
            reverb_length: Length of the impulse response. Only used if
                trainable=True.
            add_dry: Add dry signal to reverberated signal on output.
            name: Name of processor module.
        """
        super().__init__()
        self._reverb_length = reverb_length
        self._add_dry = add_dry

    def _mask_dry_ir(self, ir):
        """Set first impulse response to zero to mask the dry signal."""
        # Make IR 2-D [batch, ir_size].
        if len(ir.shape) == 1:
            ir = ir[np.newaxis, :]  # Add a batch dimension
        if len(ir.shape) == 3:
            ir = ir[:, :, 0]  # Remove unnessary channel dimension.
        # Mask the dry signal.
        dry_mask = np.zeros([int(ir.shape[0]), 1], np.float32)
        return np.concatenate([dry_mask, ir[:, 1:]], axis=1)

    def get_controls(self, audio, ir=None):
        """Convert decoder outputs into ir response.

        Args:
            audio: Dry audio. 2-D Tensor of shape [batch, n_samples].
            ir: 3-D Tensor of shape [batch, ir_size, 1] or 2D Tensor of shape
                [batch, ir_size].

        Returns:
            controls: Dictionary of effect controls.

        Raises:
            ValueError: If trainable=False and ir is not provided.
        """
        if ir is None:
            raise ValueError('Must provide "ir" tensor if Reverb trainable=False.')

        return {"audio": audio, "ir": ir}

    def get_signal(self, audio, ir):
        """Apply impulse response.

        Args:
            audio: Dry audio, 2-D Tensor of shape [batch, n_samples].
            ir: 3-D Tensor of shape [batch, ir_size, 1] or 2D Tensor of shape
                [batch, ir_size].

        Returns:
            tensor of shape [batch, n_samples]
        """
        audio, ir = np_float32(audio), np_float32(ir)
        ir = self._mask_dry_ir(ir)
        wet = core.fft_convolve(audio, ir, padding="same", delay_compensation=0)
        return (wet + audio) if self._add_dry else wet
