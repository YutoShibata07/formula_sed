"""Reference: DDSP: Differentiable Digital Signal Processing
(https://github.com/magenta/ddsp)
"""


class Processor(object):
    """Abstract base class for signal processors.

    Since most effects / synths require specificly formatted control signals
    (such as amplitudes and frequenices), each processor implements a
    get_controls(inputs) method, where inputs are a variable number of tensor
    arguments that are typically neural network outputs. Check each child class
    for the class-specific arguments it expects. This gives a dictionary of
    controls that can then be passed to get_signal(controls). The
    get_outputs(inputs) method calls both in succession and returns a nested
    output dictionary with all controls and signals.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, *args, return_outputs_dict=False, **kwargs):
        """Convert input tensors arguments into a signal tensor."""
        # Don't use `training` or `mask` arguments from keras.Layer.
        for k in ["training", "mask"]:
            if k in kwargs:
                _ = kwargs.pop(k)

        controls = self.get_controls(*args, **kwargs)
        signal = self.get_signal(**controls)
        if return_outputs_dict:
            return dict(signal=signal, controls=controls)
        else:
            return signal

    def get_controls(self, *args, **kwargs):
        """Convert input tensor arguments into a dict of processor controls."""
        raise NotImplementedError

    def get_signal(self, *args, **kwargs):
        """Convert control tensors into a signal tensor."""
        raise NotImplementedError
