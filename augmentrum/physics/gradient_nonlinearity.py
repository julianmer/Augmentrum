####################################################################################################
#                                  gradient_nonlinearity.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2026-09-06                                                                              #
#                                                                                                  #
# Purpose: Gradient-coil nonlinearity - the spatial displacement caused by a real gradient coil's  #
#          deviation from an ideal linear field, away from isocenter. Built on "gradunwarp"        #
#          (github.com/Washington-University/gradunwarp, MIT, pip-installable), which parses       #
#          vendor spherical-harmonic coefficient files and evaluates the resulting displacement    #
#          field - reused rather than reimplemented, since the exact vendor file format and the    #
#          coefficients themselves are not something to reconstruct from memory or a paper.         #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import os
import warnings
from typing import Optional

import numpy as np


__all__ = [
    'default_coeff_file',
    'load_coefficients',
    'perturb_coefficients',
    'gradient_nonlinearity_displacement_m',
    'GradientNonlinearityDisplacement',
]


#**************************************************************************************************#
#                                        default coefficients                                       #
#**************************************************************************************************#
def default_coeff_file() -> Optional[str]:
    """
    gradunwarp's own bundled test fixture - a SYNTHETIC placeholder coil,
    not any real scanner.

    No real vendor coefficient file is public: Siemens/GE/Philips gradient
    calibration files are proprietary, site-specific data obtained from the
    vendor under license, and neither GIRF-Sim, gradunwarp nor HCPpipelines
    bundles or references one (confirmed by inspecting all three). The one
    file gradunwarp ships is its own regression-test fixture
    ("gradunwarp/core/tests/data/gradunwarp_coeffs.grad"), whose own header
    reads "Name: dummy.grad ... Description: Defines Legendre coefficients
    for dummy Gradient Coil" - i.e. a fabricated toy coil, deliberately not
    presented as measured hardware.

    Using it as this module's default keeps "no real coefficients supplied"
    from being a hard failure (the earlier discipline for this term was to
    always raise instead - see git history / AUGMENTRUM_HANDOVER.md), while
    never pretending it is calibration data: :func:`load_coefficients` warns
    every time it is used, and callers who want a real scanner's distortion
    must supply "gradient_nonlinearity_coeffs_file" explicitly.
    """
    try:
        import gradunwarp
    except ImportError:
        return None
    path = os.path.join(os.path.dirname(gradunwarp.__file__),
                        'core', 'tests', 'data', 'gradunwarp_coeffs.grad')
    return path if os.path.exists(path) else None


#**************************************************************************************************#
#                                        loading / perturbing                                       #
#**************************************************************************************************#
def load_coefficients(coeffs_file: Optional[str] = None, vendor: str = 'siemens'):
    """
    Load a vendor gradient-coefficient file via gradunwarp.

    Args:
        coeffs_file: Path to a ".grad" (preferred) or ".coef" file. "None"
            uses gradunwarp's own bundled synthetic test fixture - see
            :func:`default_coeff_file` - and warns every time, since that
            file does not describe any real scanner.
        vendor: Only "'siemens'" is functional in gradunwarp today; its own
            GE path unconditionally raises (an unfinished stub upstream,
            not something to work around here).

    Returns:
        A "gradunwarp.core.coeffs.Coeffs" namedtuple
        ("alpha_x, alpha_y, alpha_z, beta_x, beta_y, beta_z, R0_m").
    """
    try:
        from gradunwarp.core import coeffs as gw_coeffs
    except ImportError as exc:
        raise ImportError(
            "Gradient-nonlinearity modeling needs the optional dependency "
            "gradunwarp. Install it with `pip install gradunwarp`."
        ) from exc

    if coeffs_file is None:
        coeffs_file = default_coeff_file()
        if coeffs_file is None:
            raise ImportError(
                "No gradient_nonlinearity_coeffs_file was given, and "
                "gradunwarp's own bundled test fixture could not be found "
                "(check the gradunwarp install)."
            )
        warnings.warn(
            "GradientNonlinearityDisplacement is using gradunwarp's own "
            "'gradunwarp_coeffs.grad' test fixture - a fabricated dummy "
            "coil (see its header: 'Description: Defines Legendre "
            "coefficients ... for dummy Gradient Coil'), NOT a real "
            "scanner's calibration. Pass gradient_nonlinearity_coeffs_file "
            "to use a real vendor coefficient file.",
            UserWarning, stacklevel=3,
        )

    return gw_coeffs.get_coefficients(vendor, coeffs_file)


def perturb_coefficients(coeffs, relative_std: float, rng: np.random.Generator):
    """
    A copy of *coeffs* with independent multiplicative noise on every
    already-nonzero term - one way to sample "a coil like this one, but not
    exactly this one" (manufacturing tolerance / a different unit of the
    same design), stochastically, the same way GIRF-Sim's own kernels are
    sampled by default rather than fixed.

    Only entries the base coefficient set already has nonzero are
    perturbed. Every "Coeffs" array is allocated at a fixed maximum order
    (see e.g. "beta_z"'s un-truncated 100x100 shape in gradunwarp's own
    parser) and is mostly structural zero padding, not real coil terms;
    perturbing those would fabricate high-order terms the coil design never
    had, rather than vary the terms it does.

    Args:
        coeffs: A "gradunwarp.core.coeffs.Coeffs" namedtuple.
        relative_std: Standard deviation of the multiplicative noise, as a
            fraction of each coefficient's own value (e.g. 0.05 = 5%). 0
            leaves every coefficient exactly as given.
        rng: NumPy generator.

    Returns:
        A new "Coeffs" with perturbed arrays; "R0_m" unchanged.
    """
    from gradunwarp.core.coeffs import Coeffs

    if relative_std < 0:
        raise ValueError(f"relative_std must be >= 0, got {relative_std}.")

    def _perturb(arr):
        arr = np.asarray(arr, dtype=np.float64).copy()
        if relative_std == 0:
            return arr
        mask = arr != 0
        noise = rng.standard_normal(int(mask.sum()))
        arr[mask] = arr[mask] * (1.0 + relative_std * noise)
        return arr

    return Coeffs(
        alpha_x=_perturb(coeffs.alpha_x), alpha_y=_perturb(coeffs.alpha_y),
        alpha_z=_perturb(coeffs.alpha_z), beta_x=_perturb(coeffs.beta_x),
        beta_y=_perturb(coeffs.beta_y), beta_z=_perturb(coeffs.beta_z),
        R0_m=coeffs.R0_m,
    )


#**************************************************************************************************#
#                                         displacement field                                        #
#**************************************************************************************************#
def gradient_nonlinearity_displacement_m(coeffs, positions_m: np.ndarray,
                                         vendor: str = 'siemens') -> np.ndarray:
    """
    The spatial displacement field a gradient coil's nonlinearity causes.

    Wraps "gradunwarp.core.unwarp_resample.eval_spherical_harmonics", which
    works in millimeters (confirmed from its source: "R0 = coeffs.R0_m *
    1000", and its only caller converts its own meter-based FOV to mm
    before use) - the conversion to/from Augmentrum's own meters convention
    happens entirely inside this function.

    Args:
        coeffs: A "gradunwarp.core.coeffs.Coeffs" namedtuple (see
            :func:`load_coefficients`/:func:`perturb_coefficients`).
        positions_m: "(N, 3)" positions in meters.
        vendor: Passed through to gradunwarp; only "'siemens'" is functional.

    Returns:
        "(N, 3)" displacement, in meters, evaluated at "positions_m".

    Note:
        gradunwarp evaluates spherical coordinates via
        "x += 0.0001 mm" as a singularity-avoidance step at r=0; combined
        with :func:`perturb_coefficients`, this means a perturbed
        coefficient set can show a small nonzero displacement exactly at
        isocenter even though the true physical distortion there is zero -
        a known quirk of the underlying library's r=0 handling, not
        something this wrapper corrects.
    """
    try:
        from gradunwarp.core.unwarp_resample import eval_spherical_harmonics
        from gradunwarp.core.utils import CoordsVector
    except ImportError as exc:
        raise ImportError(
            "Gradient-nonlinearity modeling needs the optional dependency "
            "gradunwarp. Install it with `pip install gradunwarp`."
        ) from exc

    p = np.asarray(positions_m, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"positions_m must be (N, 3), got {p.shape}.")

    p_mm = p * 1000.0
    vxyz = CoordsVector(x=p_mm[:, 0].copy(), y=p_mm[:, 1].copy(), z=p_mm[:, 2].copy())
    displacement_mm, _ = eval_spherical_harmonics(coeffs, vendor, vxyz)

    return np.stack([displacement_mm.x, displacement_mm.y, displacement_mm.z], axis=-1) / 1000.0


#**************************************************************************************************#
#                              Class GradientNonlinearityDisplacement                               #
#**************************************************************************************************#
#                                                                                                  #
# The GIRF-Sim "DisplacementTerm" this project fills in.                                           #
#                                                                                                  #
#**************************************************************************************************#
class GradientNonlinearityDisplacement:
    """
    "r_eff = r - Delta_r_GNL(r)": gradient-coil-nonlinearity displacement.

    Fills in GIRF-Sim's "girf_mrsi_extensions.GradientNonlinearityDisplacement",
    which is an interface placeholder that unconditionally raises
    "NotImplementedError" because no vendor coefficients were available in
    that project. Here, real coefficients come from
    :func:`load_coefficients` (a user-supplied vendor file, or - with an
    explicit warning - gradunwarp's own synthetic test coil).

    Sign convention: gradunwarp's own "Unwarper" class (its CLI's "-w/--warp"
    path) applies the forward (distorting) direction as
    "new_xyz = xyz + polarity * displacement" with "polarity = -1" - i.e.
    "r_distorted = r - Delta_r(r)" - the reverse of its default *correction*
    convention ("polarity = +1"), to first order. This class uses the same
    "minus" sign for the same reason: it introduces the distortion a real
    coil would cause, rather than removing one.

    Duck-types "girf_mrsi_extensions.DisplacementTerm" (a "name" attribute
    and a "displace(positions) -> positions" method) rather than
    subclassing it, so this module runs with plain NumPy + gradunwarp - no
    GIRF-Sim/torch dependency needed to use it standalone.

    Args:
        coeffs: A "gradunwarp.core.coeffs.Coeffs" namedtuple.
        vendor: Passed through to gradunwarp; only "'siemens'" is functional.

    Examples:
        >>> coeffs = load_coefficients("my_scanner.grad")
        >>> term = GradientNonlinearityDisplacement(coeffs)
        >>> term.displace(positions_m).shape
        (N, 3)
    """

    def __init__(self, coeffs, vendor: str = 'siemens'):
        self.coeffs = coeffs
        self.vendor = vendor
        self.name = 'gradient_nonlinearity'

    def displace(self, positions: np.ndarray) -> np.ndarray:
        """"(N, 3)" meters in, effective (displaced) "(N, 3)" meters out."""
        displacement = gradient_nonlinearity_displacement_m(self.coeffs, positions, self.vendor)
        return np.asarray(positions, dtype=np.float64) - displacement
