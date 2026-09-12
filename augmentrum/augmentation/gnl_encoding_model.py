####################################################################################################
#                                   gnl_encoding_model.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2026-09-12                                                                              #
#                                                                                                  #
# Purpose: A physically-direct gradient-nonlinearity (GNL) encoding model - see                    #
#          augmentrum_gnl_virtual_scanner_handover.md. Separate from and complementary to           #
#          GIRFArtifacts.include_gradient_nonlinearity (the fast, approximate image-space warp):    #
#          this one enters GNL directly into the encoding phase of an exact direct-NUDFT forward    #
#          model (GIRF-Sim's ForwardOperator), so it is slower but never shortcuts the physics       #
#          into an image warp or a single globally-modified trajectory. Composes with GIRF-Sim's    #
#          own eddy-current/concomitant-field phase terms, all independently ablatable.             #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from augmentrum.core.base_module import BaseModule
from augmentrum.processing.domain import Domain
from nifti_mrs_plus import Backend
from nifti_mrs_plus import ops


__all__ = ['GNLEncodingModel']

#: realism_level -> default max_harmonic_order, unless explicitly overridden.
_LEVEL_DEFAULT_ORDER: Dict[int, int] = {1: 3, 2: 5, 3: 7, 4: 9}
#: realism_level -> default source, unless explicitly overridden.
_LEVEL_DEFAULT_SOURCE: Dict[int, str] = {
    1: 'stochastic', 2: 'measured', 3: 'measured_stochastic', 4: 'measured_stochastic',
}


#**************************************************************************************************#
#                                    Class GNLEncodingModel                                         #
#**************************************************************************************************#
#                                                                                                  #
# Physically-direct gradient-nonlinearity forward model: GNL enters the encoding phase, never an   #
# image warp, never a single globally-modified trajectory.                                          #
#                                                                                                  #
#**************************************************************************************************#
class GNLEncodingModel(BaseModule):
    """
    Gradient-coil-nonlinearity (GNL), entered directly into the MR encoding
    phase - a physically-direct alternative to
    "GIRFArtifacts.include_gradient_nonlinearity".

    That module treats GNL as an image-space warp: fast, and exact for a
    single call, but a shortcut nonetheless (see its own class docstring for
    why an image warp - or a single modified k-space trajectory - cannot be
    an *exact* representation of a genuinely nonlinear spatial distortion).
    This module instead enters GNL as a spatially-varying **phase** directly
    into the encoding integral,

        s(t) = integral rho(r) C(r) exp(-i*phi_total(r,t)) dr,
        phi_total = phi_nominal + phi_B0 + phi_GNL + phi_GIRF,

    via GIRF-Sim's own composable "ForwardOperator" (exact direct NUDFT, not
    "GriddingNUFFT") - the same architecture already used for
    "augmentrum.physics.concomitant_field.ConcomitantFieldPhase". This is
    therefore the slower, exact-encoding tool, not a fast training-time
    augmentation; use "GIRFArtifacts.include_gradient_nonlinearity" when
    speed matters more than the physics being followed to the letter. It is
    a from-first-principles encoding model, not a validated scanner
    simulator - see the "Scope" note below.

    The physics: gradunwarp's own displacement field
    "Delta_r_j(r) = R0 sum_(n,m) [alpha_j(n,m)cos(m phi) + beta_j(n,m)sin(m phi)]
    (r/R0)^n P_n^m(cos theta)" (per gradient axis j - see
    "augmentrum.physics.gnl_field" for the verified-against-gradunwarp
    per-term decomposition) gives, substituted into the encoding exponential,
    an additive phase "phi_GNL(r,t) = -2*pi*k(t).Delta_r(r) = sum_n Phi_n(r)
    q[n,t]" - see "gnl_field.GradientNonlinearityPhase"'s own docstring for
    the full derivation. "Phi_n(r)" is cached once per position grid; "q[n,t]"
    is one small matmul against the trajectory, never a per-sample harmonic
    re-evaluation.

    Ablation
    --------
    Every term is independently toggleable, and none is implied by another:
    "include_gnl" (this module's own point - default on), "include_girf_phase"
    /"include_concomitant" (GIRF-Sim's eddy-current/Maxwell terms - default
    off; opt in to compose them), and "include_trajectory_error" (the
    GIRF-predicted "dk_girf" - default off). As with "GIRFArtifacts",
    "ForwardOperator.forward()" uses "dk_girf" **unconditionally** for the
    encoding trajectory once given, so "include_trajectory_error=False"
    genuinely passes an all-zero "dk_girf" - it is not merely "skip a phase
    term". A "GNL only" condition is the default; a "GIRF phase + GNL, no
    trajectory error" condition needs exactly the two flags set and nothing
    else assumed. Disabling "include_gnl" also fully relaxes every GNL-only
    constructor requirement (e.g. "calibration_path") - no term's
    configuration is ever required just because another term is enabled.

    GNL sources
    -----------
    "source='measured'": load "calibration_path" (a real Siemens ".grad"/
    ".coef" file) via "gnl_field"/gradunwarp, used exactly as calibrated.
    "'measured_stochastic'": start from that calibration, add
    "coefficient_std_by_order" Gaussian uncertainty
    ("gnl_field.perturb_coefficients_by_order"). "'stochastic'": no
    calibration file at all - sample a full order-0..9 coefficient set
    directly ("gnl_field.sample_coefficients"), decaying by default with
    harmonic order. "truth_model" (a "gradunwarp.core.coeffs.Coeffs")
    overrides all of the above directly, for a caller who already has one.

    Realism levels (independent of "source" - presets, not constraints;
    every field above stays individually overridable)
    -----------------------------------------------------------------------
    1: idealized, low order (default "max_harmonic_order=3"), unit-test-grade.
    2: scanner-specific/measured, "correction_model" defaults to the truth
       model itself - no deliberate mismatch.
    3: measured + calibration uncertainty (default "'measured_stochastic'"),
       higher order (default 7); "correction_model" still defaults to truth.
    4: "correction_model" defaults to an **auto-derived, deliberately
       reduced** copy of the truth model (lower order, uncertainty
       stripped) unless a caller supplies their own - guaranteeing an honest
       nonzero residual after correction rather than a coincidentally
       perfect one. See "last_truth_model_"/"last_correction_model_" and
       "residual_displacement_m()" for the diagnostic this is meant to feed.

    Scope
    -----
    This models the GNL encoding physics directly and lets a caller compare
    a "truth" against a "correction" coefficient set - it is not a validated,
    end-to-end scanner simulator (no vendor timing/RF/receive-chain
    modeling, no experimental validation against real acquisitions). Treat
    "realism_level" as a naming convenience for a coefficient-fidelity
    preset, not a claim of overall simulation fidelity.

    Examples:
        >>> # GNL only, level 2 (measured, no deliberate mismatch)
        >>> gnl = GNLEncodingModel(seq_file="...", source='measured',
        ...                       calibration_path="site_coeffs.grad", realism_level=2)
        >>> out, _ = gnl(volume_plus)

        >>> # level 4: truth is richer than the correction model gradunwarp would see
        >>> gnl = GNLEncodingModel(seq_file="...", source='measured_stochastic',
        ...                       calibration_path="site_coeffs.grad", realism_level=4)
        >>> gnl.last_truth_model_, gnl.last_correction_model_   # after a call
    """

    SUPPORTED_BACKENDS = tuple(b for b in Backend if b is not Backend.NIFTI_LIST)

    SOURCES = ('measured', 'measured_stochastic', 'stochastic')
    REALISM_LEVELS = (1, 2, 3, 4)

    def __init__(self,
                 seq_file: str,
                 source: Optional[str] = None,
                 realism_level: int = 2,
                 calibration_path: Optional[str] = None,
                 max_harmonic_order: Optional[int] = None,
                 include_linear_terms: bool = False,
                 coefficient_distribution: str = 'gaussian',
                 coefficient_scale: float = 1.0,
                 coefficient_std_by_order: Optional[Dict[int, float]] = None,
                 coefficient_std_by_axis: Optional[Dict[str, float]] = None,
                 stochastic_seed: Optional[int] = None,
                 truth_model: Optional[Any] = None,
                 correction_model: Optional[Any] = None,
                 include_gnl: bool = True,
                 include_girf_phase: bool = False,
                 include_concomitant: bool = False,
                 include_trajectory_error: bool = False,
                 girf_seed: Optional[int] = 0,
                 girf_kernel_params: Optional[Dict[str, Any]] = None,
                 girf_order: int = 3,
                 girf_radius: float = 0.12,
                 b0_tesla: Optional[float] = None,
                 pixdim: Optional[Tuple[float, ...]] = None,
                 debug_outputs: bool = False,
                 dtype: Optional[str] = None):
        """
        seq_file: Path to a Pulseq ".seq" file - the trajectory GNL's phase is
                built against (and, if enabled, GIRF/concomitant terms too).
        source: "'measured'", "'measured_stochastic'" or "'stochastic'".
                "None" (default) uses "realism_level"'s own preset. Ignored
                entirely if "include_gnl=False".
        realism_level: 1-4, see class docstring. Only supplies *defaults* for
                "source"/"max_harmonic_order"/"correction_model" - every one
                stays independently overridable.
        calibration_path: Real vendor ".grad"/".coef" file - required for
                "source" "'measured'"/"'measured_stochastic'", but only when
                "include_gnl=True" (see Ablation above).
        max_harmonic_order: Highest spherical-harmonic degree (<=9
                documented/typical - see "gnl_field.GNL_MAX_SUPPORTED_ORDER").
                "None" uses "realism_level"'s default.
        include_linear_terms: Whether order 0/1 (translation/linear gain -
                gradient calibration, not really "nonlinearity") are
                populated by "'stochastic'" sampling. Ignored for "'measured'"
                /"'measured_stochastic'" (a real file's own order-0/1 terms,
                if any, are used as calibrated).
        coefficient_distribution, coefficient_scale, coefficient_std_by_order,
                coefficient_std_by_axis: Forwarded to
                "gnl_field.sample_coefficients"("'stochastic'") or
                "gnl_field.perturb_coefficients_by_order"
                ("'measured_stochastic'" - only "..._by_order"/"..._by_axis"
                apply there, an additive per-order perturbation).
        stochastic_seed: Seeds coefficient sampling/perturbation. "None"
                draws fresh; a fixed seed reproduces it.
        truth_model: A "gradunwarp.core.coeffs.Coeffs" - overrides "source"
                entirely if given (the truth GNL model is exactly this).
        correction_model: A "Coeffs" used for "residual_displacement_m()"'s
                comparison. "None" uses "realism_level"'s own default (the
                truth model itself at levels 1-3; an auto-reduced copy at
                level 4).
        include_gnl: This module's own term. Default on; see Ablation above.
        include_girf_phase, include_concomitant, include_trajectory_error:
                Compose GIRF-Sim's own eddy-current/order>=2 phase, the
                Maxwell/concomitant phase, and the GIRF-predicted trajectory
                error respectively - each independently ablatable, none
                implied by the others or by "include_gnl". "include_concomitant"
                reuses "augmentrum.physics.concomitant_field.ConcomitantFieldPhase";
                needs "b0_tesla".
        girf_seed, girf_kernel_params, girf_order, girf_radius: Synthetic-tier
                GIRF kernel controls, same meaning as in "GIRFArtifacts".
        b0_tesla: Main field strength, Tesla, for "include_concomitant". "None"
                reads the ".seq" file's own "definitions['B0']"; raises if
                neither is available and concomitant phase is requested.
        pixdim: Voxel size in mm per spatial axis. "None" reads it from the
                data automatically.
        debug_outputs: Populate the "last_*_" diagnostic attributes (harmonic
                coefficients, "Phi_n(r)"/"q[n,t]", accumulated phase samples,
                local effective gradient, truth-vs-correction residual) after
                each call. Left off by default - the full-resolution arrays
                can be large.
        dtype: Unused placeholder for the handover's suggested config surface
                (this module always computes in float64/complex64 internally,
                matching the rest of Augmentrum); accepted so a config dict
                written against the handover's own suggested field list does
                not need to special-case this one.
        """
        super().__init__()

        if realism_level not in self.REALISM_LEVELS:
            raise ValueError(f"realism_level must be one of {self.REALISM_LEVELS}, got {realism_level!r}.")
        resolved_source = source if source is not None else _LEVEL_DEFAULT_SOURCE[realism_level]
        if resolved_source not in self.SOURCES:
            raise ValueError(f"source must be one of {self.SOURCES}, got {resolved_source!r}.")
        # Ablation: source/calibration only matter at all when include_gnl is
        # actually on - disabling this term must never require GNL-specific
        # configuration, the same way disabling a term in GIRFArtifacts never
        # requires that term's own inputs.
        if include_gnl and resolved_source in ('measured', 'measured_stochastic') \
                and calibration_path is None and truth_model is None:
            raise ValueError(
                f"source={resolved_source!r} needs calibration_path (a real vendor "
                f"'.grad'/'.coef' file) or an explicit truth_model."
            )

        self.seq_file = seq_file
        self.source = resolved_source
        self.realism_level = int(realism_level)
        self.calibration_path = calibration_path
        self.max_harmonic_order = (int(max_harmonic_order) if max_harmonic_order is not None
                                   else _LEVEL_DEFAULT_ORDER[self.realism_level])
        self.include_linear_terms = bool(include_linear_terms)
        self.coefficient_distribution = coefficient_distribution
        self.coefficient_scale = float(coefficient_scale)
        self.coefficient_std_by_order = dict(coefficient_std_by_order or {})
        self.coefficient_std_by_axis = dict(coefficient_std_by_axis or {})
        self.stochastic_seed = stochastic_seed
        self.truth_model = truth_model
        self.correction_model = correction_model

        self.include_gnl = bool(include_gnl)
        self.include_girf_phase = bool(include_girf_phase)
        self.include_concomitant = bool(include_concomitant)
        self.include_trajectory_error = bool(include_trajectory_error)
        self.girf_seed = girf_seed
        self.girf_kernel_params = dict(girf_kernel_params or {})
        self.girf_order = int(girf_order)
        self.girf_radius = float(girf_radius)
        self.b0_tesla = None if b0_tesla is None else float(b0_tesla)
        self.pixdim = tuple(pixdim) if pixdim is not None else None
        self.debug_outputs = bool(debug_outputs)
        self.dtype = dtype

        # Populated after every call that ran - provenance and (if
        # debug_outputs) the full diagnostic set from the class docstring.
        self.last_definitions_: Optional[Dict[str, Any]] = None
        self.last_truth_model_: Optional[Any] = None
        self.last_correction_model_: Optional[Any] = None
        self.last_terms_: Optional[List[Tuple[int, int, bool]]] = None
        self.last_phi_basis_: Optional[np.ndarray] = None
        self.last_q_: Optional[np.ndarray] = None
        self.last_phase_samples_: Optional[np.ndarray] = None

    #**********#
    #   DOMAIN #
    #**********#
    @property
    def DOMAIN(self):
        if (self.include_gnl or self.include_girf_phase
                or self.include_concomitant or self.include_trajectory_error):
            return Domain(spatial='image')
        return None

    #**************************#
    #   coefficient sourcing   #
    #**************************#
    def _resolve_models(self) -> Tuple[Any, Any]:
        """"(truth, correction)" Coeffs, per "source"/"realism_level" and any
        explicit overrides - see class docstring."""
        from augmentrum.physics.gnl_field import sample_coefficients, perturb_coefficients_by_order
        from augmentrum.physics.gradient_nonlinearity import load_coefficients

        if self.truth_model is not None:
            truth = self.truth_model
        elif self.source == 'stochastic':
            truth = sample_coefficients(
                max_order=self.max_harmonic_order,
                std_by_order=self.coefficient_std_by_order or None,
                std_by_axis=self.coefficient_std_by_axis or None,
                distribution=self.coefficient_distribution,
                scale=self.coefficient_scale,
                include_linear_terms=self.include_linear_terms,
                seed=self.stochastic_seed,
            )
        else:
            measured = load_coefficients(self.calibration_path)
            if self.source == 'measured_stochastic':
                std_by_order = self.coefficient_std_by_order or {
                    n: 1e-4 * self.coefficient_scale for n in range(2, self.max_harmonic_order + 1)
                }
                truth = perturb_coefficients_by_order(
                    measured, std_by_order=std_by_order,
                    std_by_axis=self.coefficient_std_by_axis or None,
                    seed=self.stochastic_seed,
                )
            else:
                truth = measured

        if self.correction_model is not None:
            correction = self.correction_model
        elif self.realism_level == 4:
            correction = self._reduced_copy(truth, max_order=max(1, self.max_harmonic_order - 4))
        else:
            correction = truth

        return truth, correction

    @staticmethod
    def _reduced_copy(coeffs, max_order: int):
        """A noise-free copy of *coeffs*, truncated to "max_order" - the
        default "realism_level=4" correction model: less complete than the
        truth it is meant to imperfectly correct, deliberately."""
        from gradunwarp.core.coeffs import Coeffs

        def _truncate(arr):
            arr = np.array(arr, dtype=np.float64, copy=True)
            n = max_order + 1
            out = np.zeros_like(arr)
            k = min(n, arr.shape[0])
            out[:k, :k] = arr[:k, :k]
            return out

        return Coeffs(
            alpha_x=_truncate(coeffs.alpha_x), alpha_y=_truncate(coeffs.alpha_y),
            alpha_z=_truncate(coeffs.alpha_z), beta_x=_truncate(coeffs.beta_x),
            beta_y=_truncate(coeffs.beta_y), beta_z=_truncate(coeffs.beta_z),
            R0_m=coeffs.R0_m,
        )

    def residual_displacement_m(self, positions_m: np.ndarray) -> np.ndarray:
        """
        "(N, 3)" truth-vs-correction displacement residual at *positions_m* -
        what a real GradUnwarp correction (built from "last_correction_model_")
        would fail to remove from data generated with
        "last_truth_model_". Zero everywhere at "realism_level" 1-3 by
        default (truth == correction unless overridden); nonzero at level 4
        - see the class docstring and requirement 8 in
        "tests/augmentation/test_gnl_encoding_model.py".

        Only meaningful after a call that populated "last_truth_model_"/
        "last_correction_model_" (i.e. with "include_gnl=True").
        """
        from augmentrum.physics.gnl_field import basis_matrix, coefficient_matrix

        if self.last_truth_model_ is None or self.last_correction_model_ is None:
            raise RuntimeError(
                "residual_displacement_m needs a prior call with include_gnl=True "
                "to populate last_truth_model_/last_correction_model_."
            )
        order = self.max_harmonic_order
        Phi_t, terms_t = basis_matrix(positions_m, self.last_truth_model_.R0_m, order)
        Phi_c, terms_c = basis_matrix(positions_m, self.last_correction_model_.R0_m, order)
        C_t = coefficient_matrix(self.last_truth_model_, terms_t)
        C_c = coefficient_matrix(self.last_correction_model_, terms_c)
        return (Phi_t @ C_t) - (Phi_c @ C_c)

    #*******************#
    #   the girf model  #
    #*******************#
    @staticmethod
    def _girf_module():
        try:
            import girf_module as gmod
        except ImportError as exc:
            raise ImportError(
                "GNLEncodingModel needs the optional dependency GIRF-Sim "
                "(https://github.com/JohnLaMaster/GIRF-Sim). Install it with "
                "`pip install -e /path/to/GIRF-Sim`."
            ) from exc
        return gmod

    def _load_girf_module(self):
        gmod = self._girf_module()
        kernel_params = None
        if self.girf_kernel_params:
            import girf_synthetic as gs
            kernel_params = gs.GIRFKernelParams(**self.girf_kernel_params)
        return gmod.GIRFModule.from_synthetic(
            self.seq_file, kernel_params=kernel_params, seed=self.girf_seed,
            radius=self.girf_radius, order=self.girf_order,
        )

    #****************************#
    #   forward operator build   #
    #****************************#
    def _build_forward_operator(self, mod, positions_np: np.ndarray,
                                k_nominal: np.ndarray, dk_girf: np.ndarray):
        """The composable "ForwardOperator" this class assembles - every
        term independently ablatable, per the class docstring."""
        import torch
        import girf_mrsi_extensions as gmx
        from augmentrum.physics.concomitant_field import ConcomitantFieldPhase
        from augmentrum.physics.gnl_field import GradientNonlinearityPhase

        positions_t = torch.tensor(positions_np, dtype=torch.float32)
        k_nominal_t = torch.tensor(k_nominal, dtype=torch.float32)
        dk_girf_t = (torch.tensor(dk_girf, dtype=torch.float32)[None]
                    if self.include_trajectory_error
                    else torch.zeros((1, k_nominal.shape[0], k_nominal.shape[1]), dtype=torch.float32))

        k_actual = k_nominal + (dk_girf if self.include_trajectory_error else 0.0)

        phase_terms = []
        if self.include_girf_phase and mod.tier == 'synthetic' and mod.coeffs is not None:
            # mod.coeffs is [n_shots, n_basis, L]; move n_basis to the front
            # before flattening (shot-major, sample-minor - the same order
            # k_nominal/dk_girf are flattened in above) so a plain reshape
            # doesn't interleave the shot and basis axes.
            coeffs_flat = mod.coeffs.permute(1, 0, 2).reshape(1, mod.coeffs.shape[1], -1)
            phase_terms.append(gmx.GIRFGlobalPhase(coeffs_flat, dt=mod.dt))
            if self.girf_order >= 2:
                phase_terms.append(gmx.GIRFNonlinearPhase(
                    coeffs_flat, radius=self.girf_radius, dt=mod.dt, order=self.girf_order))
        if self.include_concomitant:
            b0 = self._b0_tesla(mod)
            grad_flat = mod.batched.gradients_t_per_m.permute(1, 0, 2).reshape(3, -1).numpy()
            phase_terms.append(ConcomitantFieldPhase(grad_flat, dt=mod.dt, b0_tesla=b0))
        if self.include_gnl:
            truth, correction = self._resolve_models()
            self.last_truth_model_, self.last_correction_model_ = truth, correction
            gnl_term = GradientNonlinearityPhase(truth, k_actual, max_order=self.max_harmonic_order)
            phase_terms.append(gnl_term)
            if self.debug_outputs:
                Phi = gnl_term._phi_basis
                if Phi is None:
                    # force evaluation once so the diagnostics below have something to show
                    gnl_term.phase(positions_t, torch.arange(k_actual.shape[1], dtype=torch.float32))
                self.last_phi_basis_ = gnl_term._phi_basis
                self.last_q_ = gnl_term._q
                self.last_terms_ = gnl_term.last_terms_

        operator = gmx.ForwardOperator(positions_t, k_nominal_t, dk_girf_t, phase_terms=phase_terms)
        return operator

    def _b0_tesla(self, mod) -> float:
        if self.b0_tesla is not None:
            return self.b0_tesla
        b0 = mod.definitions.get('B0')
        if b0 is None:
            raise ValueError(
                "GNLEncodingModel.include_concomitant needs a main field strength: "
                "pass b0_tesla explicitly, or use a .seq file whose DEFINITIONS "
                "section declares 'B0' (Tesla)."
            )
        return float(np.asarray(b0).reshape(-1)[0])

    #***************************#
    #   direct simulate() API   #
    #***************************#
    def simulate(self, phantom, matrix: Tuple[int, int, int],
                geometry: Optional[dict] = None, noise_std: float = 0.0,
                chunk_size: int = 4096):
        """
        The direct, low-level API: forward-simulate raw k-t signal from an
        explicit "girf_mrsi_extensions.SpectralPhantom" - for validation
        workflows and the unit tests, not tied to any Augmentrum NIfTI-MRS
        volume.

        Args:
            phantom: A "girf_mrsi_extensions.SpectralPhantom".
            matrix, geometry: The object's own spatial grid, for building
                "positions" - same convention as "GIRFArtifacts"/
                "FieldInhomogeneity" (fov_mm/pixdim, never the ".seq" file's
                own declared geometry).

        Returns:
            "girf_mrsi_extensions.JointKTMRSIResult". Its own trajectory
            sample spacing is the ".seq" file's own raster ("GIRFModule.dt")
            - there is no separate "dt" to pass in.
        """
        mod = self._load_girf_module()
        self.last_definitions_ = dict(mod.definitions)

        ndim = int(mod.k_nominal.shape[1])
        positions_np = self._positions(matrix, geometry).numpy()
        k_nominal = mod.k_nominal.permute(0, 2, 1).reshape(-1, ndim).numpy().astype(np.float64).T
        dk_girf = mod.dk_error.permute(0, 2, 1).reshape(-1, ndim).numpy().astype(np.float64).T

        operator = self._build_forward_operator(mod, positions_np, k_nominal, dk_girf)
        return operator.forward(phantom, mod.dt, noise_std=noise_std, chunk_size=chunk_size)

    #**************************#
    #   basemodule interface   #
    #**************************#
    def process_tensor(self, data_array, water_array=None,
                       backend: Backend = Backend.PYTORCH, **kwargs):
        """
        Treats an existing "(batch, X, Y, Z, T)" volume's own per-voxel FID
        as the phantom directly (bypassing "SpectralPhantom"'s parametric
        rho/T2*/freq model), via the exact direct-NUDFT path above.

        Cost note: since every phase term here (GNL included) varies only
        with *trajectory* sample, not spectral sample - the same "shared
        trajectory across the spectral axis" convention
        "GIRFArtifacts"/"FieldInhomogeneity" use - this runs one
        forward-NUDFT-then-adjoint pair **per spectral sample**, looping
        "T" times rather than folding "T" into the NUFFT's channel axis the
        way the fast "GriddingNUFFT"-based modules do. Appropriate for the
        small/validation-scale volumes this model is meant for, not full
        training-batch throughput.
        """
        if not (self.include_gnl or self.include_girf_phase
                or self.include_concomitant or self.include_trajectory_error):
            return data_array, water_array

        if data_array.ndim not in (5, 6):
            raise ValueError(
                "GNLEncodingModel expects (batch, X, Y, Z, T) in the NIfTI "
                "layout, or (batch, X, Y, Z, T, C) with a receive array, got "
                f"shape {tuple(data_array.shape)}."
            )
        if data_array.ndim == 6:
            return self._apply_per_coil(data_array, **kwargs), water_array

        matrix = tuple(int(s) for s in data_array.shape[1:4])
        geometry = kwargs.get('geometry')
        vol = ops.cast(data_array, 'complex64')
        out = self._apply_gnl(vol, matrix, geometry)
        return out, water_array

    def _apply_per_coil(self, data_array, **kwargs):
        n_coils = int(ops.shape(data_array)[5])
        shape = tuple(int(n) for n in ops.shape(data_array))[:5]
        per_coil = []
        for c in range(n_coils):
            volume = ops.reshape(ops.take(data_array, np.array([c]), axis=5), shape)
            artifacted, _ = self.process_tensor(volume, **kwargs)
            per_coil.append(artifacted)
        return ops.stack(per_coil, axis=5)

    def _apply_gnl(self, vol, matrix: Tuple[int, int, int], geometry: Optional[dict]):
        import girf_mrsi_extensions as gmx
        from augmentrum.sampling.kspace_reconstructor import GriddingNUFFT

        mod = self._load_girf_module()
        self.last_definitions_ = dict(mod.definitions)
        self._validate_seq_geometry(mod.definitions, matrix)

        nx, ny, nz = matrix
        ndim = int(mod.k_nominal.shape[1])
        im_size = (nx, ny) if ndim == 2 else (nx, ny, nz)

        positions_t = self._positions(matrix, geometry)
        positions_np = positions_t.numpy()
        k_nominal = mod.k_nominal.permute(0, 2, 1).reshape(-1, ndim).numpy().astype(np.float64).T
        dk_girf = mod.dk_error.permute(0, 2, 1).reshape(-1, ndim).numpy().astype(np.float64).T
        operator = self._build_forward_operator(mod, positions_np, k_nominal, dk_girf)

        kmax = self._kmax_from_geometry(matrix, geometry)[:ndim]
        coords_nominal = (k_nominal.T / (2.0 * kmax[None, :])).astype(np.float32)

        nufft = GriddingNUFFT(im_size, 2.0)
        n_batch = int(ops.shape(vol)[0])
        n_t = int(ops.shape(vol)[4])

        recon = []
        for b in range(n_batch):
            per_t = []
            for spec in range(n_t):
                snapshot = ops.reshape(
                    ops.take(ops.take(vol, np.array([b]), axis=0), np.array([spec]), axis=4),
                    (-1,),
                )
                rho = ops.cast(snapshot, 'complex64')
                rho_t = self._to_torch_complex(rho)
                phantom = gmx.SpectralPhantom.single_component(rho_t, t2star_s=1e12, freq_hz=0.0)
                result = operator.forward(phantom, mod.dt)
                d_np = result.d.numpy().astype(np.complex64)
                per_t.append(nufft.adjoint(d_np[None, :], coords_nominal))   # (1, *im_size)
            recon.append(ops.stack(per_t, axis=-1))   # (1, *im_size, T) -> collapse channel below

        out = ops.stack([ops.reshape(r, (nx, ny, nz, n_t)) for r in recon], axis=0)
        return self._match_scale(out, vol)

    @staticmethod
    def _to_torch_complex(x):
        import torch
        return torch.as_tensor(np.asarray(x), dtype=torch.complex64)

    @staticmethod
    def _match_scale(out, vol):
        num = ops.sqrt(ops.sum(ops.abs(vol) ** 2))
        den = ops.sqrt(ops.sum(ops.abs(out) ** 2))
        scale = num / ops.where(den > 1e-12, den, den * 0 + 1e-12)
        return out * ops.cast_like(scale, out)

    #****************#
    #   geometry     #
    #****************#
    def _positions(self, matrix, geometry):
        import girf_synthetic as gs

        nx, ny, nz = matrix
        vx, vy, vz = self._voxel_size_m(geometry)
        fov_x, fov_y, fov_z = nx * vx, ny * vy, nz * vz
        if nz == 1:
            return gs.make_grid_2d(nx, ny, fov_x, fov_y, z=0.0)
        return gs.make_grid_3d(nx, ny, nz, fov_x, fov_y, fov_z)

    def _voxel_size_m(self, geometry: Optional[dict]) -> Tuple[float, float, float]:
        if self.pixdim is not None:
            vx, vy, vz = (list(self.pixdim) + [1.0, 1.0, 1.0])[:3]
        elif geometry is not None and 'voxel_mm' in geometry:
            vx, vy, vz = geometry['voxel_mm']
        else:
            vx = vy = vz = 1.0
        return float(vx) / 1000.0, float(vy) / 1000.0, float(vz) / 1000.0

    def _kmax_from_geometry(self, matrix, geometry: Optional[dict]) -> np.ndarray:
        nx, ny, nz = matrix
        vx, vy, vz = self._voxel_size_m(geometry)
        fov = np.array([nx * vx, ny * vy, nz * vz], dtype=np.float64)
        n = np.array([nx, ny, nz], dtype=np.float64)
        fov = np.where(fov > 0, fov, 1.0)
        return (n / 2.0) / fov

    def _validate_seq_geometry(self, definitions: Dict[str, Any], matrix: Tuple[int, int, int]):
        seq_matrix = definitions.get('Matrix')
        if seq_matrix is None:
            return
        seq_matrix = [int(round(float(v))) for v in np.asarray(seq_matrix).reshape(-1)]
        data_matrix = [int(v) for v in matrix[:len(seq_matrix)]]
        if seq_matrix != data_matrix:
            raise ValueError(
                f"seq_file declares Matrix={seq_matrix} but the data's own "
                f"spatial matrix is {tuple(matrix)}."
            )
