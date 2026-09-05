####################################################################################################
#                                     girf_artifacts.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2026-09-05                                                                              #
#                                                                                                  #
# Purpose: Scanner-hardware artifacts driven by a real Pulseq ".seq" file: GIRF-induced k-space     #
#          trajectory error and phase (stochastic/synthetic, or a real measured GIRF), and the      #
#          concomitant (Maxwell) gradient field. Built on GIRF-Sim (github.com/JohnLaMaster/        #
#          GIRF-Sim) as an installed dependency, lazily imported so a NumPy-only Augmentrum          #
#          install never needs torch/GIRF-Sim on its account. Gradient-coil nonlinearity is out     #
#          of scope without vendor calibration data - see GIRFArtifacts' docstring.                 #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from augmentrum.core.base_module import BaseModule
from augmentrum.processing.domain import Domain
from nifti_mrs_plus import Backend
from nifti_mrs_plus import ops


__all__ = ['GIRFArtifacts']


#**************************************************************************************************#
#                                       Class GIRFArtifacts                                        #
#**************************************************************************************************#
#                                                                                                  #
# GIRF-induced trajectory/phase error and concomitant-field phase, driven by a real .seq file.      #
#                                                                                                  #
#**************************************************************************************************#
class GIRFArtifacts(BaseModule):
    """
    GIRF (gradient impulse response function) and concomitant-field artifacts.

    Built on `GIRF-Sim <https://github.com/JohnLaMaster/GIRF-Sim>`_
    (`girf_module.GIRFModule`, lazily imported - installing it is only
    needed to actually run this module, never to import Augmentrum). A
    "*.seq" file supplies the real gradient waveforms and k-space
    trajectory; a GIRF - stochastically sampled by default, fixed by a
    seed, or a real measured scanner response - predicts how that
    trajectory is actually distorted by the gradient hardware.

    Acquisition model
    ------------------
    Every k-space "shot" GIRF-Sim's own loader
    ("seq_girf.load_all_shots_batched") reports for the ".seq" file is used
    at its own full fine-raster resolution - one Augmentrum k-space sample
    per fine-raster point of that shot's own commanded gradient waveform,
    not collapsed to a single point. This whole set of samples is what one
    shot contributes to the trajectory, and - matching every other NUFFT
    path in Augmentrum (see "KspaceUndersampling"/"FieldInhomogeneity") -
    that same set of samples is measured identically for every spectral
    (FID) point: the gradient waveform is replayed unchanged once per
    spectral sample, so its GIRF/concomitant response is unchanged too.
    Consequently every artifact modeled here varies along the *trajectory
    sample* axis (position along a shot's own waveform, i.e. real elapsed
    time within that one shot) - never along the spectral/FID axis, and
    never across repeats of the same shot.

    Terms
    -----
    "include_trajectory_error"
        Swaps the commanded trajectory for the GIRF-predicted one when
        measuring, and reconstructs assuming the commanded trajectory was
        used - the same "dual trajectory" idea "KspaceUndersampling"
        already uses for its own affine dual-trajectory mode, just with an
        arbitrary (not affine) per-sample displacement. Exact - no
        approximation.
    "include_girf_phase"
        Order-0 (spatially uniform - "girf_mrsi_extensions.GIRFGlobalPhase")
        and order>=2 (spatially nonlinear -
        "girf_mrsi_extensions.GIRFNonlinearPhase") GIRF phase. Synthetic
        tier only: the measured (Bacon) tier's released data is order<=1,
        already captured entirely in the trajectory error above, and
        reusing it here would double-count it (this matches GIRF-Sim's own
        "GIRFModule.phase_correction" docstring). Order-0 is spatially
        uniform, so it commutes with the Fourier transform and is applied
        exactly, as a cheap post-hoc phase per trajectory sample; order>=2
        is position-dependent and uses the same approximation as
        "include_concomitant" below.
    "include_concomitant"
        The Maxwell/concomitant gradient field -
        "augmentrum.physics.concomitant_field.ConcomitantFieldPhase", built
        from the real gradient waveform ("Gx(t),Gy(t),Gz(t)") and "b0_tesla"
        - see that module for the physics and its citations. Also
        position-dependent.
    "include_gradient_nonlinearity"
        Vendor gradient-coil spherical-harmonic coefficients would be
        needed to model this (deviation from a linear gradient field) and
        are not available here - set this to "True" only to get a clear
        "NotImplementedError" rather than silently doing nothing; leave it
        "False" (default) to omit the term entirely.

    Why position-dependent terms need an approximation
    -----------------------------------------------------
    Order-0 GIRF phase is spatially uniform, so one shared value per
    trajectory sample is exact. Order>=2 GIRF phase and the concomitant
    phase are not: they are position-dependent, so - unlike
    "FieldInhomogeneity"'s B0 segments, which can share one forward NUFFT
    across every trajectory sample because a B0 segment's mask does not
    depend on which sample is being measured - a per-sample spatial pattern
    would need one forward NUFFT per trajectory sample to apply exactly.
    Instead, trajectory samples are grouped into "n_severity_segments"
    clusters by how large their combined order>=2 coefficients are (an
    adaptive quantile binning, exactly analogous to
    "FieldInhomogeneity"'s B0 frequency segmentation, just keyed on a
    per-sample coefficient vector instead of a per-voxel frequency); each
    cluster's member samples share one representative (mean) coefficient
    vector, giving "n_severity_segments" forward NUFFTs total rather than
    one per sample. Increasing "n_severity_segments" converges toward the
    per-sample-exact result.

    Not duplicated here: static B0/off-resonance
    -----------------------------------------------
    "girf_mrsi_extensions.StaticOffResonancePhase" uses a phase convention
    tied to GIRF-Sim's own joint k-t time axis, which does not match
    Augmentrum's FID time axis (see this class's acquisition-model note
    above). Chain the existing "~augmentrum.augmentation.FieldInhomogeneity"
    (its "b0_map") in the pipeline instead, rather than modeling B0 twice
    with two different time conventions.

    Ablation warning
    -----------------
    "include_trajectory_error=False" genuinely zeroes the trajectory error
    (uses the commanded trajectory for both measurement and
    reconstruction) - it is not merely "don't add a phase term" the way it
    would be if the trajectory error were folded into "include_girf_phase".
    A "concomitant field only" ablation condition needs
    "include_trajectory_error=False, include_girf_phase=False,
    include_concomitant=True" - passing only "include_girf_phase=False"
    would still leave the GIRF-perturbed trajectory active.

    Examples:
        >>> girf = GIRFArtifacts(seq_file="press_mrsi2d_spiral.seq", b0_tesla=3.0)
        >>> out, _ = girf(volume_plus)

        >>> # a real measured GIRF for a specific scanner
        >>> girf = GIRFArtifacts(seq_file="...", girf_mode='measured',
        ...                      bacon_data_dir="/path/to/zenodo/GIRFs")
    """

    SUPPORTED_BACKENDS = tuple(b for b in Backend if b is not Backend.NIFTI_LIST)

    GIRF_MODES = ('synthetic', 'measured')

    def __init__(self,
                 seq_file: str,
                 girf_mode: str = 'synthetic',
                 girf_seed: Optional[int] = 0,
                 kernel_params: Optional[Dict[str, Any]] = None,
                 bacon_data_dir: Optional[str] = None,
                 measured_girf: Optional[Any] = None,
                 b0_tesla: Optional[float] = None,
                 radius: float = 0.12,
                 order: int = 3,
                 include_girf_phase: bool = True,
                 include_concomitant: bool = True,
                 include_trajectory_error: bool = True,
                 include_gradient_nonlinearity: bool = False,
                 n_severity_segments: int = 16,
                 nufft_osf: float = 2.0,
                 nufft_impl: str = 'gridding',
                 pixdim: Optional[Tuple[float, ...]] = None):
        """
        seq_file: Path to a Pulseq ".seq" file - defines the gradient
                waveforms and k-space trajectory this module measures along.
        girf_mode: "'synthetic'" (default) samples a kernel bank; "'measured'"
                attaches a real scanner GIRF (order<=1 only).
        girf_seed: Synthetic tier only. "None" draws a fresh stochastic GIRF
                realization every call, which is what an augmentation wants;
                a fixed seed (default 0) reproduces the same realization.
        kernel_params: Synthetic tier only, forwarded to
                "girf_synthetic.GIRFKernelParams(**kernel_params)". "None"
                uses "girf_mrsi_extensions.trajectory_kernel_params()".
        bacon_data_dir: Measured tier: Bacon et al.'s released file layout.
                Exactly one of "bacon_data_dir"/"measured_girf" is required
                for "girf_mode='measured'".
        measured_girf: Measured tier: a "bacon_girf.BaconGIRF" for your own
                system's measured GIRF (see "BaconGIRF.from_arrays").
        b0_tesla: Main field strength, Tesla, for the concomitant term.
                "None" reads "seq_file"'s own "definitions['B0']"; raises if
                neither is available. Ignored if "include_concomitant=False".
        radius: Shared solid-harmonic normalization radius (meters),
                matching whatever the GIRF kernel bank / measured GIRF was
                built with.
        order: Synthetic-tier spatial harmonic order (0-3) for
                "include_girf_phase"'s order>=2 term.
        include_girf_phase: Order-0 + order>=2 GIRF phase. Synthetic tier
                only - has no effect on the measured tier (see class
                docstring).
        include_concomitant: The Maxwell/concomitant gradient-field phase.
        include_trajectory_error: Measure on the GIRF-predicted trajectory,
                reconstruct on the commanded one. See the ablation warning
                in the class docstring before turning this off for a
                "concomitant only" condition.
        include_gradient_nonlinearity: If "True", raises
                "NotImplementedError" - no vendor gradient-coil calibration
                data is available to model this. Leave "False" to omit the
                term.
        n_severity_segments: Trajectory samples sharing similar order>=2
                GIRF/concomitant phase are grouped into this many clusters
                (see class docstring); increase to converge toward the
                per-sample-exact result.
        nufft_osf, nufft_impl: NUFFT oversampling and interpolator, as in
                "KspaceUndersampling"/"FieldInhomogeneity".
        pixdim: Voxel size in mm per spatial axis, only used to validate
                against "seq_file"'s own declared geometry. Normally left
                "None" - read off the data automatically.
        """
        super().__init__()

        if girf_mode not in self.GIRF_MODES:
            raise ValueError(f"girf_mode must be one of {self.GIRF_MODES}, got {girf_mode!r}.")
        if girf_mode == 'measured' and (bacon_data_dir is None) == (measured_girf is None):
            raise ValueError(
                "girf_mode='measured' needs exactly one of bacon_data_dir or "
                "measured_girf, not both and not neither."
            )
        if nufft_impl not in ('gridding', 'interp'):
            raise ValueError(f"nufft_impl must be 'gridding' or 'interp', got {nufft_impl!r}.")
        if int(n_severity_segments) < 1:
            raise ValueError(f"n_severity_segments must be >= 1, got {n_severity_segments}.")

        self.seq_file = seq_file
        self.girf_mode = girf_mode
        self.girf_seed = girf_seed
        self.kernel_params = dict(kernel_params or {})
        self.bacon_data_dir = bacon_data_dir
        self.measured_girf = measured_girf
        self.b0_tesla = None if b0_tesla is None else float(b0_tesla)
        self.radius = float(radius)
        self.order = int(order)
        self.include_girf_phase = bool(include_girf_phase)
        self.include_concomitant = bool(include_concomitant)
        self.include_trajectory_error = bool(include_trajectory_error)
        self.include_gradient_nonlinearity = bool(include_gradient_nonlinearity)
        self.n_severity_segments = int(n_severity_segments)
        self.nufft_osf = float(nufft_osf)
        self.nufft_impl = nufft_impl
        self.pixdim = tuple(pixdim) if pixdim is not None else None

        if self.include_gradient_nonlinearity:
            raise NotImplementedError(
                "GIRFArtifacts.include_gradient_nonlinearity: gradient-coil "
                "nonlinearity needs vendor spherical-harmonic gradient-coil "
                "coefficients (Siemens/GE/Philips coil models), which are "
                "not available here. GIRF-Sim's own "
                "'GradientNonlinearityDisplacement' is an unconditional "
                "placeholder for the same reason - this project does not "
                "fabricate calibration data. Leave this False to omit the "
                "term; supply real vendor coefficients and extend this "
                "class to make it available."
            )

        # Populated after every call that ran - provenance and diagnostics.
        self.last_definitions_: Optional[Dict[str, Any]] = None
        self.last_severity_segments_: Optional[List[Dict[str, Any]]] = None

    #**********#
    #   DOMAIN #
    #**********#
    @property
    def DOMAIN(self):
        """
        Everything this module does is a voxel-wise (or trajectory) operator
        that has to run in the image domain - unless every term is disabled,
        in which case it is the identity and no domain move is worth forcing.
        """
        if (self.include_trajectory_error or self.include_girf_phase
                or self.include_concomitant):
            return Domain(spatial='image')
        return None

    #**************************#
    #   basemodule interface   #
    #**************************#
    def process_tensor(self, data_array, water_array=None,
                       backend: Backend = Backend.PYTORCH, **kwargs):
        """
        Apply GIRF trajectory/phase and concomitant-field artifacts.

        Args:
            data_array: "(batch, X, Y, Z, T)" complex, or
                "(batch, X, Y, Z, T, C)" to run against every receive-coil
                element identically (these are hardware/object properties,
                not receive-coil properties).
            water_array: Passed through unchanged.
            backend: Backend enum (unused; kept for the BaseModule signature).
            **kwargs: Absorbs "geometry" injected by BaseModule.

        Returns:
            "(artifacted_data, water_unchanged)", same shape and dtype in.
        """
        if not (self.include_trajectory_error or self.include_girf_phase
                or self.include_concomitant):
            return data_array, water_array

        if data_array.ndim not in (5, 6):
            raise ValueError(
                "GIRFArtifacts expects (batch, X, Y, Z, T) in the NIfTI "
                "layout, or (batch, X, Y, Z, T, C) with a receive array, got "
                f"shape {tuple(data_array.shape)}."
            )
        if data_array.ndim == 6:
            return self._apply_per_coil(data_array, **kwargs), water_array

        matrix = tuple(int(s) for s in data_array.shape[1:4])
        geometry = kwargs.get('geometry')
        vol = ops.cast(data_array, 'complex64')
        out = self._apply_girf(vol, matrix, geometry)
        return out, water_array

    #***********#
    #   coils   #
    #***********#
    def _apply_per_coil(self, data_array, **kwargs):
        """Same rationale as FieldInhomogeneity/KspaceUndersampling: GIRF and
        concomitant fields are properties of the gradient hardware and the
        object, not of the receive coil, so every coil sees the same
        artifact."""
        n_coils = int(ops.shape(data_array)[5])
        shape = tuple(int(n) for n in ops.shape(data_array))[:5]

        per_coil = []
        for c in range(n_coils):
            volume = ops.reshape(ops.take(data_array, np.array([c]), axis=5), shape)
            artifacted, _ = self.process_tensor(volume, **kwargs)
            per_coil.append(artifacted)

        return ops.stack(per_coil, axis=5)

    #*******************#
    #   the girf model  #
    #*******************#
    @staticmethod
    def _girf_module():
        """The girf_module package, with an actionable error when it is
        absent - mirrors KspaceReconstructor._tkbn()'s pattern for its own
        optional dependency."""
        try:
            import girf_module as gmod
        except ImportError as exc:
            raise ImportError(
                "GIRFArtifacts needs the optional dependency GIRF-Sim "
                "(https://github.com/JohnLaMaster/GIRF-Sim). Install it with "
                "`pip install -e /path/to/GIRF-Sim`."
            ) from exc
        return gmod

    def _load_girf_module(self):
        """Build this call's GIRFModule (fresh stochastic draw unless
        girf_seed is fixed)."""
        gmod = self._girf_module()

        if self.girf_mode == 'synthetic':
            kernel_params = None
            if self.kernel_params:
                import girf_synthetic as gs
                kernel_params = gs.GIRFKernelParams(**self.kernel_params)
            return gmod.GIRFModule.from_synthetic(
                self.seq_file, kernel_params=kernel_params, seed=self.girf_seed,
                radius=self.radius, order=self.order,
            )
        return gmod.GIRFModule.from_measured(
            self.seq_file, bacon_data_dir=self.bacon_data_dir,
            bacon=self.measured_girf,
        )

    def _apply_girf(self, vol, matrix: Tuple[int, int, int], geometry: Optional[dict]):
        """Measure `vol` along the .seq file's trajectory, with GIRF/
        concomitant artifacts, and reconstruct assuming the commanded
        trajectory - see the class docstring for the acquisition model."""
        from augmentrum.processing.interpolating import LinearInterpolator
        from augmentrum.sampling.kspace_reconstructor import GriddingNUFFT

        mod = self._load_girf_module()
        self.last_definitions_ = dict(mod.definitions)

        nx, ny, nz = matrix
        self._validate_seq_geometry(mod.definitions, matrix)

        ndim = int(mod.k_nominal.shape[1])
        im_size = (nx, ny) if ndim == 2 else (nx, ny, nz)

        # Every shot's own fine-raster trajectory, concatenated verbatim -
        # the "one Augmentrum k-space sample per trajectory sample" model
        # from the class docstring. "[n_shots, D, L] -> [n_shots, L, D] ->
        # [n_shots*L, D]" flattens shot-major, sample-minor - the same order
        # every other per-K-sample array below (girf0 phase, severity
        # coefficients) is built in, so they all index the same K axis.
        pts_nominal = mod.k_nominal.permute(0, 2, 1).reshape(-1, ndim).numpy().astype(np.float64)
        pts_actual = mod.k_actual.permute(0, 2, 1).reshape(-1, ndim).numpy().astype(np.float64)

        kmax = self._kmax_from_geometry(matrix, geometry)[:ndim]
        coords_nominal = (pts_nominal / (2.0 * kmax[None, :])).astype(np.float32)
        coords_actual = (pts_actual / (2.0 * kmax[None, :])).astype(np.float32)
        coords_forward = coords_actual if self.include_trajectory_error else coords_nominal

        interpolator = LinearInterpolator() if self.nufft_impl == 'interp' else None
        nufft = GriddingNUFFT(im_size, self.nufft_osf, interpolator=interpolator)
        gridder = GriddingNUFFT(im_size, self.nufft_osf) if interpolator is not None else nufft

        positions = self._positions(matrix, geometry, ndim)   # [N_voxels, 3] meters, torch

        girf0_phase = None
        if self.include_girf_phase and mod.tier == 'synthetic':
            girf0_phase = self._girf0_phase(mod)                          # [K_total] numpy

        severity_coeffs, severity_basis = self._severity_coefficients(mod, positions)

        n_batch = int(ops.shape(vol)[0])
        n_t = int(ops.shape(vol)[4])

        recon = []
        segments_info = []
        for b in range(n_batch):
            vol_b = ops.reshape(ops.take(vol, np.array([b]), axis=0), (1, nx, ny, nz, n_t))

            if severity_coeffs is None:
                plane_b = self._to_channel_stack(vol_b, ndim, matrix, n_t)
                kdata_total = nufft.forward(plane_b, coords_forward)
            else:
                kdata_total, info = self._segmented_forward(
                    nufft, vol_b, coords_forward, positions,
                    severity_coeffs, severity_basis, matrix, ndim, n_t,
                )
                segments_info.append(info)

            if girf0_phase is not None:
                phase_factor = np.exp(1j * girf0_phase).astype(np.complex64)
                kdata_total = kdata_total * ops.cast_like(
                    ops.match_backend(phase_factor[None, :], kdata_total), kdata_total)

            recon.append(gridder.adjoint(kdata_total, coords_nominal))

        self.last_severity_segments_ = segments_info or None

        out = ops.stack(recon, axis=0)
        if ndim == 2:
            out = ops.transpose(ops.reshape(out, (n_batch, nz, n_t, nx, ny)), (0, 3, 4, 1, 2))
        else:
            out = ops.transpose(out, (0, 2, 3, 4, 1))

        return self._match_scale(out, vol)

    #****************************#
    #   order-0 (global) phase   #
    #****************************#
    def _girf0_phase(self, mod) -> np.ndarray:
        """Accumulated order-0 GIRF phase, one value per K-sample - spatially
        uniform, so exact (no clustering needed)."""
        import girf_mrsi_extensions as gmx

        c0 = gmx.split_harmonic_coefficients(mod.coeffs, order=0)['c0'][:, 0, :].real  # [n_shots, L]
        phase = 2.0 * math.pi * np.cumsum(c0.numpy(), axis=-1) * mod.dt
        return phase.reshape(-1).astype(np.float64)

    #*****************************************#
    #   order>=2 GIRF + concomitant (approx)  #
    #*****************************************#
    def _severity_coefficients(self, mod, positions):
        """
        Per-K-sample coefficients for every enabled position-dependent term,
        concatenated into one feature vector per sample, plus the matching
        basis-evaluation function. Returns "(None, None)" if no
        position-dependent term is enabled.
        """
        import girf_mrsi_extensions as gs
        from augmentrum.physics.concomitant_field import (
            concomitant_field_basis, concomitant_field_coefficients,
        )

        pieces_coeffs = []          # each [n_features_i, K_total]
        basis_fns = []              # each positions -> [N_voxels, n_features_i]

        if self.include_concomitant:
            b0 = self._b0_tesla(mod)
            conc = np.concatenate([
                concomitant_field_coefficients(
                    shot_grad.numpy(), mod.dt, b0)
                for shot_grad in mod.batched.gradients_t_per_m
            ], axis=-1)                                      # [4, K_total]
            pieces_coeffs.append(conc)
            basis_fns.append(concomitant_field_basis)

        if self.include_girf_phase and mod.tier == 'synthetic' and self.order >= 2:
            parts = gs.split_harmonic_coefficients(mod.coeffs, order=self.order)
            if 'c_ho' in parts:
                c_ho = parts['c_ho'].real                     # [n_shots, n_ho, L]
                phase_ho = 2.0 * math.pi * np.cumsum(c_ho.numpy(), axis=-1) * mod.dt
                n_ho = phase_ho.shape[1]
                pieces_coeffs.append(phase_ho.transpose(1, 0, 2).reshape(n_ho, -1))

                order, radius = self.order, self.radius

                def _ho_basis(pos, order=order, radius=radius):
                    import torch
                    import girf_synthetic as gsyn
                    pos_t = torch.from_numpy(np.ascontiguousarray(pos, dtype=np.float32))
                    return gsyn.real_solid_harmonics(pos_t, order=order, radius=radius)[:, 4:].numpy()

                basis_fns.append(_ho_basis)

        if not pieces_coeffs:
            return None, None

        coeffs = np.concatenate(pieces_coeffs, axis=0)   # [F_total, K_total]

        def basis(pos):
            return np.concatenate([fn(pos) for fn in basis_fns], axis=-1)  # [N_voxels, F_total]

        return coeffs, basis

    def _segmented_forward(self, nufft, vol_b, coords, positions,
                           coeffs: np.ndarray, basis_fn, matrix, ndim, n_t):
        """
        Group K-samples into severity clusters and accumulate their forward
        NUFFT contributions - the position-dependent-phase analogue of
        FieldInhomogeneity's B0 segmentation (see class docstring).

        The phase multiply happens on the ORIGINAL "(1, X, Y, Z, T)" volume,
        before it is folded into the NUFFT's channel-stack form - a 2-D
        trajectory folds Z into the channel axis (see "_to_channel_stack"),
        and the concomitant/GIRF-nonlinear phase genuinely varies with Z
        even though it is constant across T, so multiplying beforehand is
        what makes that vary-with-Z-constant-with-T pattern come out right
        automatically once the Z axis is folded away.
        """
        nx, ny, nz = matrix
        severity = np.sum(coeffs ** 2, axis=0)             # [K_total]
        bin_idx, edges = self._severity_bins(severity, self.n_severity_segments)

        positions_np = positions.numpy()
        basis = basis_fn(positions_np)                     # [N_voxels, F_total]

        # positions is a flattened (Y, X) [nz=1: girf_synthetic.make_grid_2d]
        # or (Z, Y, X) [make_grid_3d] grid - reshape to that order, transpose
        # to this module's own (X, Y, Z) axis order, and restore the
        # singleton Z axis for the nz=1 case so phi_q always broadcasts
        # against vol_b's (1, X, Y, Z, T).
        def _phi(flat):
            if nz == 1:
                return flat.reshape(ny, nx).T[:, :, None]           # (nx, ny, 1)
            return flat.reshape(nz, ny, nx).transpose(2, 1, 0)      # (nx, ny, nz)

        kdata_total = None
        info = {'n_bins': 0, 'bin_sizes': []}
        for q in range(len(edges) - 1):
            mask_q = (bin_idx == q)
            n_q = int(mask_q.sum())
            info['bin_sizes'].append(n_q)
            if n_q == 0:
                continue
            info['n_bins'] += 1

            rep_coeffs = coeffs[:, mask_q].mean(axis=1)     # [F_total]
            phi_q = _phi(basis @ rep_coeffs)                # (nx, ny, nz)
            phase_map = np.exp(1j * phi_q).astype(np.complex64)[None, :, :, :, None]

            vol_q = vol_b * ops.cast_like(ops.match_backend(phase_map, vol_b), vol_b)
            plane_q = self._to_channel_stack(vol_q, ndim, matrix, n_t)

            kdata_q = nufft.forward(plane_q, coords)
            mask_arr = np.asarray(mask_q, dtype=np.complex64)
            kdata_q = kdata_q * ops.cast_like(ops.match_backend(mask_arr[None, :], kdata_q), kdata_q)

            kdata_total = kdata_q if kdata_total is None else kdata_total + kdata_q

        if kdata_total is None:
            plane_b = self._to_channel_stack(vol_b, ndim, matrix, n_t)
            kdata_total = nufft.forward(plane_b, coords) * 0

        return kdata_total, info

    @staticmethod
    def _severity_bins(severity: np.ndarray, n_segments: int) -> Tuple[np.ndarray, np.ndarray]:
        """Adaptive (quantile) binning of a 1-D severity measure, mirroring
        FieldInhomogeneity._segment's strategy."""
        smin, smax = float(severity.min()), float(severity.max())
        if smax - smin < 1e-15:
            return np.zeros(severity.shape, dtype=np.int64), np.array([smin, smax + 1e-9])

        q = max(1, int(n_segments))
        edges = np.unique(np.quantile(severity, np.linspace(0.0, 1.0, q + 1)))
        if edges.size < 2:
            edges = np.array([smin, smax])
        edges = edges.copy()
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        bin_idx = np.clip(np.digitize(severity, edges[1:-1], right=False), 0, edges.size - 2)
        return bin_idx, edges

    #********************#
    #   array shaping    #
    #********************#
    @staticmethod
    def _to_channel_stack(x, ndim: int, matrix: Tuple[int, int, int], n_t: int):
        """"(1, X, Y, Z, T) -> (channels, *im_size)", matching
        FieldInhomogeneity/KspaceUndersampling's own NUFFT channel layout."""
        nx, ny, nz = matrix
        if ndim == 2:
            return ops.reshape(ops.transpose(x, (0, 3, 4, 1, 2)), (nz * n_t, nx, ny))
        return ops.reshape(ops.transpose(x, (0, 4, 1, 2, 3)), (n_t, nx, ny, nz))

    @staticmethod
    def _match_scale(out, vol):
        num = ops.sqrt(ops.sum(ops.abs(vol) ** 2))
        den = ops.sqrt(ops.sum(ops.abs(out) ** 2))
        scale = num / ops.where(den > 1e-12, den, den * 0 + 1e-12)
        return out * ops.cast_like(scale, out)

    #****************#
    #   geometry     #
    #****************#
    def _positions(self, matrix, geometry, ndim):
        """"[N_voxels, 3]" positions in meters, from the data's own geometry
        - never the .seq file's own declared FOV (see class docstring:
        the two are validated to agree, not silently reconciled)."""
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
        # Guard the singleton axis (nz=1, vz undefined/irrelevant): never
        # divide by a zero FOV even though that axis is never actually used.
        fov = np.where(fov > 0, fov, 1.0)
        return (n / 2.0) / fov

    def _validate_seq_geometry(self, definitions: Dict[str, Any], matrix: Tuple[int, int, int]):
        """The .seq file's own declared Matrix, if present, must agree with
        the data's own spatial matrix - simulating a real sequence against a
        differently-gridded phantom is physically meaningless, so this is
        flagged rather than silently reconciled."""
        seq_matrix = definitions.get('Matrix')
        if seq_matrix is None:
            return
        seq_matrix = [int(round(float(v))) for v in np.asarray(seq_matrix).reshape(-1)]
        data_matrix = [int(v) for v in matrix[:len(seq_matrix)]]
        if seq_matrix != data_matrix:
            raise ValueError(
                f"seq_file declares Matrix={seq_matrix} but the data's own "
                f"spatial matrix is {tuple(matrix)}. GIRFArtifacts measures "
                f"the data at the geometry the .seq file was designed for; "
                f"resample the data (or use a matching .seq file) rather "
                f"than silently reinterpreting the trajectory."
            )

    def _b0_tesla(self, mod) -> float:
        if self.b0_tesla is not None:
            return self.b0_tesla
        b0 = mod.definitions.get('B0')
        if b0 is None:
            raise ValueError(
                "GIRFArtifacts.include_concomitant needs a main field "
                "strength: pass b0_tesla explicitly, or use a .seq file "
                "whose DEFINITIONS section declares 'B0' (Tesla)."
            )
        return float(np.asarray(b0).reshape(-1)[0])
