import numpy as np


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def hz_to_mel(hz):
    return 2595.0 * np.log10(1 + hz / 700.0)
