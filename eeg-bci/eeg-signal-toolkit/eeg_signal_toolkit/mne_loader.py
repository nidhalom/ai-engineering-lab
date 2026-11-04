"""Optional loader for MNE's sample dataset (needs ``pip install mne``; downloads ~1.5 GB once)."""

from __future__ import annotations

from pathlib import Path


def load_mne_sample(tmin: float = 0.0, tmax: float = 0.6):
    """Left vs right auditory epochs from MNE's ``sample`` dataset, as (X, y, fs).

    This is auditory data, not motor imagery. It only shows how to feed real recordings into
    the toolkit.
    """
    try:
        import mne
    except ImportError as exc:
        raise ImportError("install the optional dependency: pip install mne") from exc

    root = Path(mne.datasets.sample.data_path())
    raw = mne.io.read_raw_fif(root / "MEG" / "sample" / "sample_audvis_raw.fif", preload=True, verbose=False)
    raw.pick("eeg")
    events = mne.find_events(raw, stim_channel="STI 014", verbose=False)
    epochs = mne.Epochs(raw, events, {"left": 1, "right": 2}, tmin, tmax, baseline=None, preload=True, verbose=False)
    y = (epochs.events[:, 2] == 2).astype(int)
    return epochs.get_data(), y, raw.info["sfreq"]
