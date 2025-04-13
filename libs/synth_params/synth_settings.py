# params.py
from dataclasses import dataclass, field, asdict
from typing import Type, Optional, Dict, Any
import GPy


@dataclass
class SynthesisParams:
    sample_rate: int = 16000
    n_frames: int = 2500
    hop_size: int = 64
    n_frequencies: int = 1000
    noise_anchors: int = 10
    noise_vagueness: int = 10
    small_variance: float = 1e-2
    max_n_to_mix: int = 4
    default_kernel: Type[GPy.kern.Kern] = GPy.kern.RBF

    frames_per_sec: int = field(init=False)
    n_samples: int = field(init=False)
    duration: float = field(init=False)

    def __post_init__(self):
        self.frames_per_sec = self.sample_rate // self.hop_size
        self.n_samples = self.n_frames * self.hop_size
        self.duration = self.n_samples / self.sample_rate

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["default_kernel"] = self.default_kernel.__name__
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SynthesisParams":
        kernel_name = d.pop("default_kernel", "RBF")
        kernel_class = getattr(GPy.kern, kernel_name, GPy.kern.RBF)
        instance = cls(**d)
        instance.default_kernel = kernel_class
        return instance
