####################################################################################################
#                                         seq_file.py                                                #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2026-09-17                                                                              #
#                                                                                                  #
# Purpose: Reads a Pulseq ".seq" file into the real gradient waveform and k-space trajectory it     #
#          commands, so the augmentations can be driven by an actual sequence instead of an         #
#          analytically-generated one. Built on "pypulseq"                                          #
#          (github.com/imr-framework/pypulseq, MIT, pip-installable), lazily imported so a          #
#          NumPy-only Augmentrum install never needs it. See                                        #
#          "augmentrum.sampling.kspace_sampling.SeqFile" for the "Trajectory" this feeds.            #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


__all__ = ['SeqFileData', 'load_seq_file', 'GAMMA_HZ_PER_T']

#: 1H gyromagnetic ratio / 2*pi, Hz/T - pypulseq's native gradient convention
#: is Hz/m, so this is what converts it to Augmentrum's T/m.
GAMMA_HZ_PER_T = 42.57747892e6


#**************************************************************************************************#
#                                       Class SeqFileData                                           #
#**************************************************************************************************#
#                                                                                                  #
# Everything read out of one ".seq" file - gradient waveform, k-space trajectory, and definitions.  #
#                                                                                                  #
#**************************************************************************************************#
@dataclass
class SeqFileData:
    """
    Gradient waveform, k-space trajectory and declared definitions read from
    one Pulseq ".seq" file.

    Two time axes coexist, on purpose:

    * "t_grid"/"gradients_t_per_m" - the real gradient waveform, on a
      **uniform** raster ("dt" apart) spanning the whole sequence.
    * "t_adc"/"k_traj_m" - the k-space trajectory at the actual ADC sample
      times, from pypulseq's own "Sequence.calculate_kspace()" (not
      re-derived by integrating the gradient waveform here) - this is where
      the sequence actually measures, and is what a reconstruction needs.

    Attributes:
        dt: Gradient raster spacing, seconds ("Sequence.grad_raster_time").
        t_grid: "(T,)" seconds, uniformly spaced, matching "gradients_t_per_m".
        gradients_t_per_m: "(3, T)" real gradient waveform "(Gx, Gy, Gz)",
            T/m, on "t_grid" - converted from pypulseq's native Hz/m.
        t_adc: "(L,)" seconds, the sequence's own ADC sample times.
        k_traj_m: "(3, L)" nominal (commanded) k-space trajectory at "t_adc",
            cycles/m - pypulseq's own convention, which matches the "cycles/m"
            convention "augmentrum.sampling.kspace_sampling.Trajectory"
            subclasses already return.
        definitions: The ".seq" file's "DEFINITIONS" section, verbatim
            (e.g. "B0", "FOV") - keys are whatever the file declares; nothing
            here is guessed for a key it omits.
    """

    dt: float
    t_grid: np.ndarray
    gradients_t_per_m: np.ndarray
    t_adc: np.ndarray
    k_traj_m: np.ndarray
    definitions: Dict[str, Any] = field(default_factory=dict)


#**************************************************************************************************#
#                                         load_seq_file                                             #
#**************************************************************************************************#
#                                                                                                  #
# Read a ".seq" file via pypulseq.                                                                  #
#                                                                                                  #
#**************************************************************************************************#
def load_seq_file(seq_file: str, gamma_hz_per_t: float = GAMMA_HZ_PER_T) -> SeqFileData:
    """
    Read a Pulseq ".seq" file into a :class:`SeqFileData`.

    Lazily imports "pypulseq" - installing it is only needed to actually call
    this function, never to import Augmentrum.

    Args:
        seq_file: Path to a ".seq" file.
        gamma_hz_per_t: Gyromagnetic ratio / 2*pi, Hz/T, used to convert
            pypulseq's native Hz/m gradient convention to Augmentrum's T/m.

    Returns:
        A populated :class:`SeqFileData`.

    Raises:
        ImportError: pypulseq is not installed.
    """
    try:
        import pypulseq as pp
    except ImportError as exc:
        raise ImportError(
            "Reading a .seq file needs the optional dependency pypulseq. "
            "Install it with `pip install pypulseq`."
        ) from exc

    seq = pp.Sequence()
    seq.read(seq_file)

    dt = float(seq.grad_raster_time)
    duration_s, _, _ = seq.duration()
    wave_data, _, _, t_adc, _ = seq.waveforms_and_times()
    k_traj_adc, _, _, _, t_adc_k = seq.calculate_kspace()

    n_pts = int(round(float(duration_s) / dt)) + 1
    t_grid = np.arange(n_pts, dtype=np.float64) * dt

    grads_hz_per_m = np.zeros((3, n_pts), dtype=np.float64)
    for axis, w in enumerate(wave_data):
        if w.size == 0:
            continue
        t_breakpoints, amplitude = w[0], w[1]
        grads_hz_per_m[axis] = np.interp(t_grid, t_breakpoints, amplitude, left=0.0, right=0.0)

    return SeqFileData(
        dt=dt,
        t_grid=t_grid,
        gradients_t_per_m=grads_hz_per_m / float(gamma_hz_per_t),
        t_adc=np.asarray(t_adc_k, dtype=np.float64),
        k_traj_m=np.asarray(k_traj_adc, dtype=np.float64),
        definitions=dict(seq.definitions),
    )
