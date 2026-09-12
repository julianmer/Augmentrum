####################################################################################################
#                                        gnl_field.py                                                #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2026-09-12                                                                              #
#                                                                                                  #
# Purpose: A physically-direct gradient-coil-nonlinearity (GNL) model - the spatially-varying      #
#          phase GNL contributes to the MR encoding itself, never an image-space warp and never a  #
#          single globally-modified k-space trajectory (see augmentrum_gnl_virtual_scanner_        #
#          handover.md). Reuses gradunwarp's own per-harmonic-term math (verified directly against #
#          its source, "gradunwarp.core.unwarp_resample.siemens_B") rather than re-deriving it, and #
#          its "Coeffs" namedtuple as the one common representation every source mode produces.     #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


__all__ = [
    'GNL_MAX_SUPPORTED_ORDER',
    'term_index',
    'basis_matrix',
    'coefficient_matrix',
    'sample_coefficients',
    'perturb_coefficients_by_order',
    'GradientNonlinearityPhase',
]

#: Siemens gradient-coil calibration files rarely exceed order 9-10 in practice
#: (see the handover doc); this is a documentation/sanity bound, not a hard
#: mathematical one - "term_index"/"basis_matrix" work for any order.
GNL_MAX_SUPPORTED_ORDER = 9


#**************************************************************************************************#
#                                  the per-harmonic-term basis                                      #
#**************************************************************************************************#
def _cart2sph_mm(x_mm: np.ndarray, y_mm: np.ndarray, z_mm: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    "(r, cos_theta, phi)" from Cartesian mm coordinates.

    Deliberately byte-for-byte identical to "gradunwarp.core.unwarp_resample.
    cart2sph" (same "x += 0.0001" singularity-avoidance epsilon, same
    millimeter convention) so a coefficient set evaluated through this module
    reproduces gradunwarp's own aggregate "eval_spherical_harmonics" output
    exactly - see "tests/physics/test_gnl_field.py"'s parity test.
    """
    x = x_mm + 0.0001
    r = np.sqrt(x * x + y_mm * y_mm + z_mm * z_mm)
    cos_theta = z_mm / r
    phi = np.arctan2(y_mm / r, x / r)
    return r, cos_theta, phi


def _normfact(n: int, m: int) -> float:
    """Siemens' associated-Legendre normalization - the same conditional
    "gradunwarp.core.unwarp_resample.siemens_B" applies (no factor for m=0)."""
    if m == 0:
        return 1.0
    return ((-1.0) ** m) * math.sqrt(
        float((2 * n + 1) * math.factorial(n - m)) / float(2 * math.factorial(n + m)))


def term_index(max_order: int) -> List[Tuple[int, int, bool]]:
    """
    Every real spherical-harmonic term up to "max_order", as "(n, m, is_beta)".

    "is_beta=False" is the alpha/cos(m*phi) term (present for every m,
    including m=0 - "cos(0)=1"); "is_beta=True" is the beta/sin(m*phi) term
    (only for m>=1, since "sin(0*phi)=0" makes the m=0 beta term always
    vanish - it is never included, matching "siemens_B"'s own m-loop). Order
    "max_order" gives "(max_order+1)**2" terms total.

    This ordering is what "basis_matrix"/"coefficient_matrix" build their
    matching axes against - callers should treat it as an opaque, stable key
    rather than relying on its exact sequence.
    """
    if max_order < 0:
        raise ValueError(f"max_order must be >= 0, got {max_order}.")
    terms = []
    for n in range(max_order + 1):
        for m in range(n + 1):
            terms.append((n, m, False))
            if m > 0:
                terms.append((n, m, True))
    return terms


def basis_matrix(positions_m: np.ndarray, R0_m: float, max_order: int
                 ) -> Tuple[np.ndarray, List[Tuple[int, int, bool]]]:
    """
    The spatial half of the GNL decomposition: "Phi_term(r)", one column per
    harmonic term, in meters (pre-scaled by "R0_m" so a raw dimensionless
    alpha/beta coefficient directly gives a displacement contribution in
    meters - see :func:`coefficient_matrix` and the module docstring's
    derivation).

    "Phi_{(n,m,False)}(r) = R0 * (r/R0)^n * P_n^m(cos_theta) * normfact(n,m) * cos(m*phi)"
    "Phi_{(n,m,True)}(r)  = R0 * (r/R0)^n * P_n^m(cos_theta) * normfact(n,m) * sin(m*phi)"

    - the per-term pieces of "gradunwarp.core.unwarp_resample.siemens_B"'s own
    sum, evaluated once and cached by the caller (see
    :class:`GradientNonlinearityPhase`) rather than re-evaluated per time
    sample, which is the whole point of this decomposition.

    Args:
        positions_m: "(N, 3)" positions in meters.
        R0_m: Reference radius, meters (the coefficient set's own "R0_m").
        max_order: Highest harmonic degree "n" to include.

    Returns:
        "(Phi, terms)" - "Phi" is "(N, n_harm)" real, "terms" is
        :func:`term_index`'s own list (same order as "Phi"'s columns).
    """
    from scipy import special as scipy_special

    p = np.asarray(positions_m, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"positions_m must be (N, 3), got {p.shape}.")
    if R0_m <= 0:
        raise ValueError(f"R0_m must be positive, got {R0_m}.")

    x_mm, y_mm, z_mm = p[:, 0] * 1000.0, p[:, 1] * 1000.0, p[:, 2] * 1000.0
    r_mm, cos_theta, phi = _cart2sph_mm(x_mm, y_mm, z_mm)
    R0_mm = R0_m * 1000.0

    terms = term_index(max_order)
    Phi = np.empty((p.shape[0], len(terms)), dtype=np.float64)

    radial_legendre: Dict[Tuple[int, int], np.ndarray] = {}
    for i, (n, m, is_beta) in enumerate(terms):
        key = (n, m)
        common = radial_legendre.get(key)
        if common is None:
            radial = np.power(r_mm / R0_mm, n)
            legendre = scipy_special.lpmv(m, n, cos_theta) * _normfact(n, m)
            common = radial * legendre
            radial_legendre[key] = common
        trig = np.sin(m * phi) if is_beta else np.cos(m * phi)
        Phi[:, i] = R0_m * common * trig

    return Phi, terms


def coefficient_matrix(coeffs, terms: List[Tuple[int, int, bool]]) -> np.ndarray:
    """
    "(n_harm, 3)" raw alpha/beta coefficients, one row per :func:`term_index`
    entry, one column per physical gradient axis (x, y, z) - dimensionless,
    straight from a "gradunwarp.core.coeffs.Coeffs" namedtuple.

    A term beyond a given axis array's own allocated order (e.g. the
    un-truncated, mostly-zero "beta_z" gradunwarp itself sometimes returns -
    see "gradient_nonlinearity.py"'s own docstring note on this) reads as 0,
    not an index error - the axis simply has no coefficient at that order.
    """
    axis_alpha = (coeffs.alpha_x, coeffs.alpha_y, coeffs.alpha_z)
    axis_beta = (coeffs.beta_x, coeffs.beta_y, coeffs.beta_z)

    C = np.zeros((len(terms), 3), dtype=np.float64)
    for i, (n, m, is_beta) in enumerate(terms):
        arrays = axis_beta if is_beta else axis_alpha
        for j, arr in enumerate(arrays):
            if n < arr.shape[0] and m < arr.shape[1]:
                C[i, j] = arr[n, m]
    return C


#**************************************************************************************************#
#                                    stochastic coefficients                                        #
#**************************************************************************************************#
def _default_std(order: int, base_std: float, decay: float) -> float:
    return base_std * (decay ** order)


def sample_coefficients(max_order: int = 9,
                        std_by_order: Optional[Dict[int, float]] = None,
                        std_by_axis: Optional[Dict[str, float]] = None,
                        distribution: str = 'gaussian',
                        scale: float = 1.0,
                        base_std: float = 1e-3,
                        decay: float = 0.5,
                        include_linear_terms: bool = False,
                        R0_m: float = 0.25,
                        seed: Optional[int] = None):
    """
    A "Coeffs" namedtuple sampled from scratch - no calibration file at all.

    Every alpha/beta entry is drawn independently as
    "N(0, scale * std_by_axis.get(axis,1) * std_by_order.get(order, base_std*decay**order))".
    The default (unset "std_by_order") decays geometrically with harmonic
    order, per the handover's "sensible defaults with decreasing magnitude at
    increasing harmonic order" - these defaults are a reasonable *shape*, not
    a calibrated physical magnitude for any real coil; override
    "std_by_order"/"scale" for a specific target system.

    Args:
        max_order: Highest harmonic degree to populate (<= 9 is the
            documented, typically-sufficient range - see
            :data:`GNL_MAX_SUPPORTED_ORDER`; higher works but is untested
            against any real calibration file).
        std_by_order: "{order: std}" overrides of the default decay.
        std_by_axis: "{'x'|'y'|'z': multiplier}" per-axis scaling.
        distribution: Only "'gaussian'" is implemented.
        scale: Global multiplier on every std.
        base_std, decay: The default decay's own parameters, used for any
            order not present in "std_by_order".
        include_linear_terms: Whether order 0 (constant) and order 1
            (linear - gradient amplitude/rotation calibration, not really
            "nonlinearity") are populated at all. "False" (default, matching
            the handover's own suggested default) zeroes them.
        R0_m: Reference radius for the sampled coefficient set, meters.
        seed: "None" draws a fresh realization; a fixed seed reproduces it.

    Returns:
        A "gradunwarp.core.coeffs.Coeffs" namedtuple.
    """
    if distribution != 'gaussian':
        raise ValueError(f"distribution must be 'gaussian' (only one implemented), got {distribution!r}.")

    try:
        from gradunwarp.core.coeffs import Coeffs
    except ImportError as exc:
        raise ImportError(
            "GNL modeling needs the optional dependency gradunwarp. "
            "Install it with `pip install gradunwarp`."
        ) from exc

    rng = np.random.default_rng(seed)
    n = max_order + 1
    arrays: Dict[str, np.ndarray] = {}

    for axis in ('x', 'y', 'z'):
        axis_scale = float((std_by_axis or {}).get(axis, 1.0))
        alpha = np.zeros((n, n), dtype=np.float64)
        beta = np.zeros((n, n), dtype=np.float64)
        for order in range(n):
            if not include_linear_terms and order <= 1:
                continue
            std = (std_by_order or {}).get(order, _default_std(order, base_std, decay))
            std *= scale * axis_scale
            for m in range(order + 1):
                alpha[order, m] = rng.normal(0.0, std)
                if m > 0:
                    beta[order, m] = rng.normal(0.0, std)
        arrays[f'alpha_{axis}'], arrays[f'beta_{axis}'] = alpha, beta

    return Coeffs(
        alpha_x=arrays['alpha_x'], alpha_y=arrays['alpha_y'], alpha_z=arrays['alpha_z'],
        beta_x=arrays['beta_x'], beta_y=arrays['beta_y'], beta_z=arrays['beta_z'],
        R0_m=R0_m,
    )


def perturb_coefficients_by_order(coeffs, std_by_order: Dict[int, float],
                                  std_by_axis: Optional[Dict[str, float]] = None,
                                  seed: Optional[int] = None):
    """
    A copy of *coeffs* with additive Gaussian noise, magnitude controlled per
    harmonic order (and optionally per axis) - the "measured_stochastic"
    source: start from a real calibration, add *physically constrained*
    uncertainty rather than perturbing an arbitrary image deformation.

    Unlike :func:`augmentrum.physics.gradient_nonlinearity.perturb_coefficients`
    (a flat, order-independent multiplicative perturbation, still available
    and unchanged), this is additive and keyed by order - "std_by_order[n]"
    is an absolute standard deviation for every "alpha/beta" entry at degree
    "n", so different orders (which in a real coil can have very different
    calibration confidence) can be controlled independently.

    Args:
        coeffs: A "gradunwarp.core.coeffs.Coeffs" namedtuple.
        std_by_order: "{order: absolute_std}". Orders not present are left
            exactly as measured (zero added noise) - the handover's own
            "preserve measured coefficients exactly when stochastic mode is
            disabled" extended per-order rather than all-or-nothing.
        std_by_axis: "{'x'|'y'|'z': multiplier}" per-axis scaling of the
            order's own std.
        seed: "None" draws a fresh perturbation; a fixed seed reproduces it.

    Returns:
        A new "Coeffs"; "R0_m" unchanged.
    """
    from gradunwarp.core.coeffs import Coeffs

    rng = np.random.default_rng(seed)

    def _perturb(arr, axis):
        arr = np.asarray(arr, dtype=np.float64).copy()
        axis_scale = float((std_by_axis or {}).get(axis, 1.0))
        for order, std in std_by_order.items():
            if order >= arr.shape[0]:
                continue
            noise = rng.normal(0.0, std * axis_scale, size=arr.shape[1])
            arr[order, :] += noise
        return arr

    return Coeffs(
        alpha_x=_perturb(coeffs.alpha_x, 'x'), alpha_y=_perturb(coeffs.alpha_y, 'y'),
        alpha_z=_perturb(coeffs.alpha_z, 'z'), beta_x=_perturb(coeffs.beta_x, 'x'),
        beta_y=_perturb(coeffs.beta_y, 'y'), beta_z=_perturb(coeffs.beta_z, 'z'),
        R0_m=coeffs.R0_m,
    )


#**************************************************************************************************#
#                                Class GradientNonlinearityPhase                                    #
#**************************************************************************************************#
#                                                                                                  #
# The GNL contribution to the encoding phase itself - never an image warp, never a modified        #
# trajectory.                                                                                       #
#                                                                                                  #
#**************************************************************************************************#
class GradientNonlinearityPhase:
    """
    "phi_GNL(r,t) = -2*pi * k(t) . Delta_r(r)": gradient-nonlinearity phase,
    entered directly into the encoding rather than applied as a coordinate
    substitution.

    Derivation (see the module/plan docstring for the full algebra): the
    standard gradwarp displacement is
    "Delta_r_j(r) = sum_term c_j[term] * Phi_term(r)" per physical gradient
    axis "j" (:func:`basis_matrix`/:func:`coefficient_matrix`). Substituting
    the true position "r - Delta_r(r)" for the nominal one in the encoding
    exponential "exp(-i*2*pi*k(t).r)" (the same "warp" sign convention
    "GradientNonlinearityDisplacement" in "gradient_nonlinearity.py" uses,
    and for the same reason - see its docstring) gives an *additive* phase
    "-2*pi*k(t).Delta_r(r) = sum_term Phi_term(r) * q[term,t]", with
    "q[term,t] = -2*pi * sum_j k_j(t) * c_j[term]" - a "[n_harm,3] @ [3,T]"
    matmul against the trajectory already available from "GIRFModule", never
    a per-sample harmonic re-evaluation.

    Duck-types "girf_mrsi_extensions.PhaseTerm" (a "name" attribute and a
    "phase(positions, t) -> [N,T]" method, returned as a torch tensor since
    "ForwardOperator.total_phase" accumulates every term's output by "+="
    into a torch tensor) - composes directly into
    "ForwardOperator(phase_terms=[..., GradientNonlinearityPhase(...)])"
    alongside "GIRFGlobalPhase"/"GIRFNonlinearPhase"/"ConcomitantFieldPhase".

    Args:
        coeffs: A "gradunwarp.core.coeffs.Coeffs" namedtuple - the "truth" or
            "correction" GNL model, see "gnl_virtual_scanner.py".
        k_traj: "(3, T)" cycles/m - the trajectory this term's "q[term,t]" is
            built against (typically "k_actual", so GNL acts on the gradient
            system's real, GIRF-perturbed trajectory - construct a separate
            instance if a caller specifically wants it against "k_nominal").
        max_order: Highest harmonic degree to include - must not exceed what
            *coeffs*'s own arrays actually carry (higher orders silently
            read as zero coefficients via :func:`coefficient_matrix`, so
            this mainly controls basis-evaluation cost).
        name: Term name, for provenance/debugging.

    Examples:
        >>> term = GradientNonlinearityPhase(coeffs, k_actual, max_order=9)
        >>> op = ForwardOperator(positions, k_nominal, dk_girf, phase_terms=[term])
        >>> result = op.forward(phantom, dt)
    """

    def __init__(self, coeffs, k_traj: np.ndarray, max_order: int = 9,
                name: str = 'gradient_nonlinearity'):
        self.coeffs = coeffs
        self.k_traj = np.asarray(k_traj, dtype=np.float64)
        if self.k_traj.ndim != 2 or self.k_traj.shape[0] != 3:
            raise ValueError(f"k_traj must be (3, T), got {self.k_traj.shape}.")
        self.max_order = int(max_order)
        self.name = name

        self._phi_basis: Optional[np.ndarray] = None   # (N, n_harm), first-call cache
        self._q: Optional[np.ndarray] = None            # (n_harm, T), first-call cache
        self.last_terms_: Optional[List[Tuple[int, int, bool]]] = None

    def phase(self, positions, t):
        """
        Args:
            positions: "(N, 3)" meters - torch tensor (as "ForwardOperator"
                supplies) or a plain array.
            t: "(T,)" - only its length is checked against "k_traj"; the
                actual times are irrelevant here (GNL phase is driven by the
                trajectory "k_traj" this term was built with, not directly
                by "t" - see the class docstring's derivation).

        Returns:
            "(N, T)" radians, as a torch tensor.
        """
        import torch

        t_len = int(t.shape[0]) if hasattr(t, 'shape') else len(t)
        if self.k_traj.shape[1] != t_len:
            raise ValueError(
                f"k_traj has {self.k_traj.shape[1]} samples but t has {t_len}; "
                f"they must describe the same trajectory."
            )

        positions_np = (positions.detach().cpu().numpy() if hasattr(positions, 'detach')
                       else np.asarray(positions, dtype=np.float64))

        if self._phi_basis is None:
            self._phi_basis, self.last_terms_ = basis_matrix(
                positions_np, self.coeffs.R0_m, self.max_order)
        if self._q is None:
            C = coefficient_matrix(self.coeffs, self.last_terms_)          # (n_harm, 3)
            self._q = -2.0 * np.pi * (C @ self.k_traj)                     # (n_harm, T)

        phi = self._phi_basis @ self._q                                    # (N, T)
        return torch.from_numpy(phi.astype(np.float32))
