from dataclasses import dataclass
import numpy as np
@dataclass
class LocalParams:
    hn_cor: float
    volume_hn_correlation: np.ndarray
    volume_hn_variance: float
    volume_hn_lengthscale: float
    volume_section_correlation: np.ndarray
    volume_section_variance: float
    volume_section_lengthscale: float
    f0_mel_variance: float
    f0_mel_lengthscale: float
    section_f0_mel_variance: float