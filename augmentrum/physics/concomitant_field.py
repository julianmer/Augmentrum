####################################################################################################
#                                     concomitant_field.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2026-09-05                                                                              #
#                                                                                                  #
# Purpose: The Maxwell/concomitant gradient field - the second-order field that div(B)=0/curl(B)=0 #
#          mandates must accompany any real linear gradient. GIRF-Sim's own                        #
#          "ConcomitantFieldPhase" is an interface placeholder ("NOT IMPLEMENTED... requires an     #
#          explicit model"); this fills it in with the standard symmetric-coil formula, verified    #
#          against primary literature rather than reconstructed from memory (see the class          #
#          docstring for the citations and the check performed).                                    #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from dataclasses import dataclass
from typing import Tuple

import numpy as np


__all__ = [
    'ConcomitantFieldPhase',
    'concomitant_field_coefficients',
    'concomitant_field_basis',
    'CONCOMITANT_BASIS_TERMS',
]

#: Order the four basis terms are returned/consumed in, everywhere in this module.
CONCOMITANT_BASIS_TERMS: Tuple[str, ...] = ('z2', 'r2_xy', 'xz', 'yz')

#: 1H gyromagnetic ratio / 2*pi, Hz/T - matches girf_mrsi_extensions.GAMMA_HZ_PER_T exactly, so a
#: concomitant term composed with GIRF-Sim's own phase terms shares one gamma convention.
GAMMA_HZ_PER_T = 42.57747892e6


#**************************************************************************************************#
#                                    concomitant_field_coefficients                                #
#**************************************************************************************************#
#                                                                                                  #
# The four scalar coefficients that turn a real gradient waveform into a concomitant phase.        #
#                                                                                                  #
#**************************************************************************************************#
def concomitant_field_coefficients(gradients_t_per_m: np.ndarray, dt: float, b0_tesla: float,
                                   gamma_hz_per_t: float = GAMMA_HZ_PER_T) -> np.ndarray:
    """
    Accumulated concomitant-field phase coefficients along a gradient waveform.

    The concomitant (Maxwell) field for a standard symmetric transverse
    gradient coil pair, z the main/longitudinal B0 axis, leading order in
    G/B0, is (Bernstein, Zhou, Polzin, King, Ganin, Pelc & Glover,
    "Concomitant gradient terms in phase contrast MR: analysis and
    correction," Magn Reson Med 1998;39(2):300-308; the symmetric-coil case
    is Eq. [2] of that paper with the coil offsets set to zero and the
    mixing parameter alpha=1/2 - independently re-derived here and
    cross-checked against two papers that cite and restate it, since the
    project this is written for explicitly forbids trusting a formula like
    this from memory):

        B_conc(x,y,z,t) = 1/(2*B0) * [ (Gx(t)*z - 0.5*Gz(t)*x)^2
                                        + (Gy(t)*z - 0.5*Gz(t)*y)^2 ]
                        = 1/(2*B0) * [ (Gx(t)^2 + Gy(t)^2)*z^2
                                       + 0.25*Gz(t)^2*(x^2+y^2)
                                       - Gx(t)*Gz(t)*x*z - Gy(t)*Gz(t)*y*z ]

    (Gx=Gy=0 reduces this to the well-known "Gz^2*(x^2+y^2)/(8*B0)" special
    case.) This is the general (Maxwell-pair, no vendor-specific asymmetric
    offsets) form - the only version defensible without vendor gradient-coil
    calibration data, matching the discipline GIRF-Sim itself applies to
    "GradientNonlinearityDisplacement".

    Converting to Hz ("f_conc = gamma_hz_per_t * B_conc") and accumulating
    phase along the waveform ("phi = 2*pi*cumsum(f_conc)*dt") separates into
    a spatial basis - "[z^2, x^2+y^2, x*z, y*z]" - times four scalar
    coefficients that depend only on time. This function returns those four
    coefficients, already integrated (one cumulative sum, not the
    instantaneous field), so a caller only ever needs a plain dot product
    with :func:`concomitant_field_basis` to get a phase in radians:

        phase(x,y,z,t) = coeffs(t) . concomitant_field_basis(x,y,z)

    Args:
        gradients_t_per_m: Gradient waveform "(Gx, Gy, Gz)", shape "(3, T)",
            in T/m, on a uniform raster.
        dt: Raster spacing, seconds.
        b0_tesla: Main field strength, Tesla (positive).
        gamma_hz_per_t: Gyromagnetic ratio / 2*pi, Hz/T. Defaults to the same
            constant "girf_mrsi_extensions.GAMMA_HZ_PER_T" uses, so a
            concomitant term composed alongside GIRF-Sim's own phase terms
            shares one convention.

    Returns:
        "(4, T)" accumulated coefficients, ordered as
        :data:`CONCOMITANT_BASIS_TERMS` ("z2", "r2_xy", "xz", "yz").
    """
    g = np.asarray(gradients_t_per_m, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] != 3:
        raise ValueError(f"gradients_t_per_m must be (3, T), got {g.shape}.")
    if b0_tesla <= 0:
        raise ValueError(f"b0_tesla must be positive, got {b0_tesla}.")

    gx, gy, gz = g[0], g[1], g[2]
    scale = 2.0 * np.pi * float(gamma_hz_per_t) / (2.0 * float(b0_tesla))

    inst_z2 = scale * (gx ** 2 + gy ** 2)
    inst_r2 = scale * 0.25 * gz ** 2
    inst_xz = -scale * gx * gz
    inst_yz = -scale * gy * gz

    inst = np.stack([inst_z2, inst_r2, inst_xz, inst_yz], axis=0)
    return np.cumsum(inst, axis=-1) * float(dt)


#**************************************************************************************************#
#                                     concomitant_field_basis                                       #
#**************************************************************************************************#
#                                                                                                  #
# The spatial half of the same separation: [z^2, x^2+y^2, xz, yz], in raw meters.                  #
#                                                                                                  #
#**************************************************************************************************#
def concomitant_field_basis(positions: np.ndarray) -> np.ndarray:
    """
    The four spatial basis functions the concomitant phase is linear in.

    Evaluated at raw physical positions in meters - unlike
    "girf_synthetic.real_solid_harmonics", there is no "radius"
    normalization here, because the formula's prefactors
    (:func:`concomitant_field_coefficients`) are already in SI units (T/m,
    T) and normalizing position would silently rescale them.

    Args:
        positions: "(N, 3)" positions in meters, "(x, y, z)" with z the
            main/longitudinal B0 axis.

    Returns:
        "(N, 4)", columns ordered as :data:`CONCOMITANT_BASIS_TERMS`.
    """
    p = np.asarray(positions, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got {p.shape}.")
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    return np.stack([z ** 2, x ** 2 + y ** 2, x * z, y * z], axis=-1)


#**************************************************************************************************#
#                                    Class ConcomitantFieldPhase                                    #
#**************************************************************************************************#
#                                                                                                  #
# The GIRF-Sim "PhaseTerm" this project fills in - see module docstring for the physics.           #
#                                                                                                  #
#**************************************************************************************************#
@dataclass
class ConcomitantFieldPhase:
    """
    "phi_Maxwell(r,t)": concomitant (Maxwell) gradient-field phase.

    Fills in GIRF-Sim's "girf_mrsi_extensions.ConcomitantFieldPhase", which
    is an interface placeholder that unconditionally raises
    "NotImplementedError". This class is a real "PhaseTerm" implementation -
    see :func:`concomitant_field_coefficients` for the formula and its
    citations - built from the actual gradient waveform (never GIRF-Sim's
    harmonic-coefficient representation, which is not a function of the real
    G(t) the concomitant field is algebraically determined by).

    Only implements the symmetric (Maxwell-pair) coil case - no vendor
    asymmetric-coil offsets, which would need calibration data this project
    does not fabricate (the same discipline applied to
    "GradientNonlinearityDisplacement").

    Duck-types "girf_mrsi_extensions.PhaseTerm" (a "name" attribute and a
    "phase(positions, t) -> [N,T]" method) rather than subclassing it, so
    this module imports and runs with plain NumPy - no GIRF-Sim or torch
    dependency - while still composing directly into GIRF-Sim's
    "ForwardOperator.phase_terms", which never checks "isinstance".

    Args:
        gradients_t_per_m: This shot's own gradient waveform "(Gx,Gy,Gz)",
            "(3, T)" T/m, on a uniform raster - e.g.
            "seq_girf.BatchedShotGradients.gradients_t_per_m[shot]".
        dt: Raster spacing, seconds.
        b0_tesla: Main field strength, Tesla.
        gamma_hz_per_t: Gyromagnetic ratio / 2*pi, Hz/T.

    Examples:
        >>> term = ConcomitantFieldPhase(gradients, dt=1e-5, b0_tesla=3.0)
        >>> term.phase(positions, t).shape
        (N, T)
    """

    gradients_t_per_m: np.ndarray
    dt: float
    b0_tesla: float
    gamma_hz_per_t: float = GAMMA_HZ_PER_T
    name: str = 'concomitant_maxwell'

    def phase(self, positions: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Concomitant phase at every position and time sample.

        Args:
            positions: "(N, 3)" meters.
            t: "(T,)" seconds, uniformly spaced (only "t[1]-t[0]" is used as
                "dt" for the internal accumulation, matching
                "gradients_t_per_m"'s own raster).

        Returns:
            "(N, T)" radians.
        """
        t = np.asarray(t, dtype=np.float64)
        n_t = t.shape[0]
        g = np.asarray(self.gradients_t_per_m, dtype=np.float64)
        if g.shape[-1] != n_t:
            raise ValueError(
                f"gradients_t_per_m has {g.shape[-1]} samples but t has {n_t}; "
                f"they must describe the same raster."
            )
        coeffs = concomitant_field_coefficients(g, self.dt, self.b0_tesla, self.gamma_hz_per_t)
        basis = concomitant_field_basis(positions)          # (N, 4)
        return basis @ coeffs                                # (N, T)
