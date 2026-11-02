"""Small EEG preprocessing toolkit."""

from .csp import CSP
from .features import DEFAULT_BANDS, band_power
from .filters import bandpass, notch
from .synthetic import make_synthetic_mi
from .windows import sliding_windows

__all__ = [
    "CSP",
    "DEFAULT_BANDS",
    "band_power",
    "bandpass",
    "notch",
    "make_synthetic_mi",
    "sliding_windows",
]
__version__ = "0.1.0"
