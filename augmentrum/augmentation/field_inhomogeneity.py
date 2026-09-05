####################################################################################################
#                                   field_inhomogeneity.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2026-09-04                                                                              #
#                                                                                                  #
# Purpose: B1+ transmit-field and segmented B0 off-resonance forward modeling. B1+ is a static     #
#          complex spatial modulation, applied directly to the image. B0 is approximated by a      #
#          handful of frequency segments, each measured along the acquisition trajectory and       #
#          given its own phase evolution along the FID, which captures voxel-dependent dephasing   #
#          without the cost of a fresh NUFFT per spectral sample. Gradient-system effects (GIRF,    #
#          gradient delays, eddy-current trajectory errors) are out of scope here and belong with   #
#          "ShotPerturbations" / a future trajectory-error module instead.                          #
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


__all__ = ['FieldInhomogeneity']


#**************************************************************************************************#
#                                    Class FieldInhomogeneity                                      #
#**************************************************************************************************#
#                                                                                                  #
# Complex B1+ modulation and segmented B0 off-resonance, applied to the image before it is         #
# measured along a k-space trajectory.                                                             #
#                                                                                                  #
#**************************************************************************************************#
class FieldInhomogeneity(BaseModule):
    """
    Complex B1+ modulation and segmented B0 off-resonance.

    Both maps are expected to arrive from *outside* this module - typically a
    dataloader that pairs each subject's spectroscopic data with a B0 map (e.g.
    from a dual-echo GRE) and a B1+ map (e.g. from a Bloch-Siegert or AFI scan).
    Neither map is synthesized here; supply "None" for whichever is not
    available and this module is the identity on it.

    B1+ - a static spatial modulation
    ----------------------------------
    The transmit field is complex, "B1+(r) = |B1+(r)| exp(i phi_B1(r))". Its
    magnitude scales the local signal amplitude and its phase adds a spatial
    phase - both are voxel-wise multiplications in the *image* domain, applied
    once, never as a function of k-space sample or spectral time:

        I_B1(r) = I(r) * A_B1(r) * exp(i phi_B1(r))

    "b1_mode='complex_linear'" takes "A_B1(r) = |B1+(r)| / b1_reference"
    directly. "b1_mode='nonlinear_flip'" instead maps the same ratio through a
    nominal flip angle, "A_B1(r) = sin(alpha * b(r)) / sin(alpha)" with
    "b(r) = |B1+(r)| / b1_reference" - the small-flip-angle regime where signal
    is not linear in transmit amplitude. "b1_reference" defaults to 1.0, i.e.
    the supplied magnitude is already in the relative units the acquisition
    calibrated against.

    B0 - segmented off-resonance
    -----------------------------
    The physical effect of an off-resonance map "delta_f(r)" in Hz is
    "P_B0(r, t) = exp(-i 2*pi*delta_f(r)*t)" applied along the FID: it shifts
    each voxel's apparent frequency, so a spatially varying "delta_f" dephases
    the object over the acquisition and broadens or attenuates the spectrum.

    Applying that literally would need one full-volume NUFFT per spectral
    sample - hundreds to thousands of them. Instead "delta_f(r)" is quantized
    into "b0_n_segments" frequency bins ("b0_segment_strategy='uniform'" spaces
    bins evenly across the map's range; "'adaptive'" uses quantiles of the
    map's own values, which keeps bins from sitting empty when "delta_f" is
    concentrated in a narrow band). Each bin "q" contributes a spatial mask
    "M_q" and a representative frequency "f_q" (the mean "delta_f" inside it).
    The masked object "I_q(r) = I_B1(r) * M_q(r)" is measured with ONE forward
    NUFFT along the acquisition trajectory - the spectral axis rides in the
    NUFFT's channel slot exactly as "KspaceUndersampling" does it, so this costs
    one transform per segment, not one per spectral sample - and the result is
    given its segment's own phase evolution along the FID before every segment
    is summed:

        D(t) = sum_q  NUFFT_k(t){ I_q } * exp(-i 2*pi*f_q*t)

    "t" here is the spectral (FID) sample time "t_n = n * dwell_time_s", not a
    k-space readout time: in phase-encoded MRSI one shot acquires the whole FID
    at one k-space position (see "KspaceUndersampling"), so every phase-encode
    step shares the same "t" axis and only the per-segment frequency "f_q"
    varies spatially. Using anything else - in particular applying B0 as a
    single static multiplication of the finished k-space array - throws away
    exactly the effect this module exists to model.

    This is a measurement model, not an accelerator: the trajectory it uses is
    fully sampled by default ("acceleration_factor=1.0"). Undersampling, noise
    and Cartesian masking stay "KspaceUndersampling"'s job and belong later in
    the pipeline; this module only shapes what the object looks like *before*
    that acquisition model runs. Likewise it never perturbs the trajectory
    itself - gradient delays, eddy currents and GIRF-induced trajectory error
    are a separate concern for "ShotPerturbations" / a future GIRF module.

    What this does and does not capture
    ------------------------------------
    Captures: voxel-dependent frequency shifts, B0 phase accrual, spatially
    varying phase, intravoxel dephasing, spectral broadening/signal loss, and
    the resulting off-resonance spatial artifacts.

    Does not capture: gradient-system response, gradient delays, eddy-current
    trajectory errors, or GIRF-induced trajectory distortion.

    Choosing "b0_n_segments"
    -------------------------
    Start with 16-32 and validate against a finer reference (increasing
    "b0_n_segments" should monotonically reduce the error against a
    per-voxel-exact reference). For a bin of width "delta_f" and the latest
    spectral sample time "t_max", the worst-case phase error the quantization
    introduces is "2*pi*(delta_f/2)*t_max" - see :meth:`max_phase_error_rad`.

    Args:
        b1_map: Complex (or real, magnitude-only) transmit field, shape
            "(X, Y, Z)" matching the data's spatial matrix (Z=1 for 2-D data),
            or "(batch, X, Y, Z)" for one map per subject. "None" (default)
            disables B1+ modulation entirely.
        b0_map: Real off-resonance map in Hz, same shape convention as
            "b1_map". "None" (default) disables B0 modeling entirely, and no
            NUFFT is run.
        b1_mode: "'complex_linear'" or "'nonlinear_flip'" (see above).
        b0_mode: Only "'segmented'" is implemented.
        b0_n_segments: Number of frequency bins approximating "delta_f(r)".
        b0_segment_strategy: "'uniform'" or "'adaptive'" (see above).
        b1_reference: Reference "|B1+|" the magnitude ratio is taken against.
            "None" (default) uses 1.0, i.e. "b1_map"'s magnitude is already
            relative.
        b1_flip_angle_deg: Nominal flip angle "alpha" for
            "b1_mode='nonlinear_flip'".
        trajectory, undersampling, traj_params, us_params, acceleration_factor:
            Forwarded to trajectory generation exactly as in
            "KspaceUndersampling" - see its docstring. "acceleration_factor"
            defaults to 1.0 (fully sampled): this module measures the field
            effects, it does not accelerate.
        nufft_osf, nufft_impl: NUFFT oversampling and interpolator, as in
            "KspaceUndersampling" ("'gridding'" or "'interp'").
        pixdim: Voxel size in mm per spatial axis, for trajectory shaping.
            Normally left "None" - read off the data automatically.
        dwell_time_s: FID sample spacing. "None" (default) reads "sw_hz" (or
            "geometry['dwell_time']") injected by "BaseModule"; only needed
            explicitly outside a pipeline call.
        traj_seed: RNG seed for trajectory/undersampling randomness. "None"
            draws a fresh pattern every call.

    Examples:
        >>> b1 = np.ones((32, 32, 1), dtype=np.complex64)
        >>> b0 = np.zeros((32, 32, 1), dtype=np.float64)
        >>> field = FieldInhomogeneity(b1_map=b1, b0_map=b0, b0_n_segments=8)
        >>> out, water = field(volume_plus)          # module call plans the domain move

        >>> # B1+ only - no NUFFT is run at all
        >>> FieldInhomogeneity(b1_map=b1)
    """

    SUPPORTED_BACKENDS = tuple(b for b in Backend if b is not Backend.NIFTI_LIST)

    B1_MODES = ('complex_linear', 'nonlinear_flip')
    B0_MODES = ('segmented',)
    SEGMENT_STRATEGIES = ('uniform', 'adaptive')

    def __init__(self,
                 b1_map: Optional[np.ndarray] = None,
                 b0_map: Optional[np.ndarray] = None,
                 b1_mode: str = 'complex_linear',
                 b0_mode: str = 'segmented',
                 b0_n_segments: int = 24,
                 b0_segment_strategy: str = 'uniform',
                 b1_reference: Optional[float] = None,
                 b1_flip_angle_deg: float = 90.0,
                 trajectory: str = 'golden_radial_2d',
                 undersampling: str = 'prefix',
                 traj_params: Optional[Dict[str, Any]] = None,
                 us_params: Optional[Dict[str, Any]] = None,
                 acceleration_factor: float = 1.0,
                 nufft_osf: float = 2.0,
                 nufft_impl: str = 'gridding',
                 pixdim: Optional[Tuple[float, ...]] = None,
                 dwell_time_s: Optional[float] = None,
                 traj_seed: Optional[int] = None):
        super().__init__()

        if b1_mode not in self.B1_MODES:
            raise ValueError(f"b1_mode must be one of {self.B1_MODES}, got {b1_mode!r}.")
        if b0_mode not in self.B0_MODES:
            raise ValueError(f"b0_mode must be one of {self.B0_MODES}, got {b0_mode!r}.")
        if b0_segment_strategy not in self.SEGMENT_STRATEGIES:
            raise ValueError(
                f"b0_segment_strategy must be one of {self.SEGMENT_STRATEGIES}, "
                f"got {b0_segment_strategy!r}.")
        if int(b0_n_segments) < 1:
            raise ValueError(f"b0_n_segments must be >= 1, got {b0_n_segments}.")
        if nufft_impl not in ('gridding', 'interp'):
            raise ValueError(f"nufft_impl must be 'gridding' or 'interp', got {nufft_impl!r}.")

        # Kept exactly as given (dtype and all) - shape is only known once data
        # arrives, so validation happens in process_tensor via "_as_map".
        self.b1_map = None if b1_map is None else np.asarray(b1_map)
        self.b0_map = None if b0_map is None else np.asarray(b0_map, dtype=np.float64)

        self.b1_mode = b1_mode
        self.b0_mode = b0_mode
        self.b0_n_segments = int(b0_n_segments)
        self.b0_segment_strategy = b0_segment_strategy
        self.b1_reference = None if b1_reference is None else float(b1_reference)
        self.b1_flip_angle_deg = float(b1_flip_angle_deg)

        self.trajectory = trajectory
        self.undersampling = undersampling
        self.traj_params = dict(traj_params or {})
        self.us_params = dict(us_params or {})
        self.acceleration_factor = float(acceleration_factor)
        self.nufft_osf = float(nufft_osf)
        self.nufft_impl = nufft_impl
        self.pixdim = tuple(pixdim) if pixdim is not None else None
        self.dwell_time_s = None if dwell_time_s is None else float(dwell_time_s)
        self.traj_seed = traj_seed

        # Populated after every call that actually ran the B0 model - provenance,
        # and useful for the "does b0_n_segments help" check in the test suite.
        self.last_meta_: Optional[Dict[str, Any]] = None
        self.last_b0_segments_: Optional[List[Dict[str, Any]]] = None

    @property
    def DOMAIN(self):
        """
        Both effects are voxel-wise multiplications - B1+ directly, B0's
        segments through the trajectory they are measured on - so this needs
        the image, never k-space. With both maps "None" this module is the
        identity, and forcing a domain move for that would be a wasted
        round-trip, so no domain is asked for in that case.
        """
        if self.b1_map is not None or self.b0_map is not None:
            return Domain(spatial='image')
        return None

    #**************************#
    #   basemodule interface   #
    #**************************#
    def process_tensor(self, data_array, water_array=None,
                       backend: Backend = Backend.PYTORCH, **kwargs):
        """
        Apply B1+ modulation and segmented B0 off-resonance to a batch.

        Args:
            data_array: "(batch, X, Y, Z, T)" complex, or "(batch, X, Y, Z, T, C)"
                to run the same field maps against every receive-coil element -
                B1+ is a transmit field and B0 a property of the object, so
                neither varies with the receive coil.
            water_array: Passed through unchanged.
            backend: Backend enum (unused; kept for the BaseModule signature).
            **kwargs: Absorbs "geometry" / "sw_hz" injected by BaseModule.

        Returns:
            "(modulated_data, water_unchanged)", same shape and dtype in.
        """
        if self.b1_map is None and self.b0_map is None:
            return data_array, water_array

        if data_array.ndim not in (5, 6):
            raise ValueError(
                "FieldInhomogeneity expects (batch, X, Y, Z, T) in the NIfTI "
                "layout, or (batch, X, Y, Z, T, C) with a receive array, got "
                f"shape {tuple(data_array.shape)}."
            )
        if data_array.ndim == 6:
            return self._apply_per_coil(data_array, **kwargs), water_array

        matrix = tuple(int(s) for s in data_array.shape[1:4])
        n_batch = int(data_array.shape[0])
        geometry = kwargs.get('geometry')

        vol = ops.cast(data_array, 'complex64')

        if self.b1_map is not None:
            factor = self._b1_factor(matrix, n_batch)                # (1|B, X, Y, Z)
            factor = factor[:, :, :, :, None]                        # broadcast over T
            vol = vol * ops.cast_like(ops.match_backend(factor, vol), vol)

        if self.b0_map is None:
            return vol, water_array

        dwell = self._resolve_dwell_time(kwargs.get('sw_hz'), geometry)
        out = self._apply_segmented_b0(vol, matrix, geometry, dwell)
        return out, water_array

    def _resolve_dwell_time(self, sw_hz, geometry) -> float:
        """The FID sample spacing the B0 phase evolves along."""
        if self.dwell_time_s is not None:
            return self.dwell_time_s
        if sw_hz:
            return 1.0 / float(sw_hz)
        if geometry is not None and geometry.get('dwell_time'):
            return float(geometry['dwell_time'])
        raise ValueError(
            "FieldInhomogeneity needs a dwell time for the B0 phase evolution: "
            "pass dwell_time_s explicitly, or call this module through a "
            "pipeline / NIfTI_MRS_Plus so 'sw_hz' is supplied automatically."
        )

    #***********#
    #   coils   #
    #***********#
    def _apply_per_coil(self, data_array, **kwargs):
        """
        Apply the same field maps to a receive array, one element at a time.

        Mirrors "KspaceUndersampling._apply_per_coil": B1+ is transmit and B0 is
        a property of the object, so every coil sees the identical modulation.
        Looping keeps memory bounded the same way it does there - the NUFFT
        already carries the spectral axis in its channel slot.
        """
        n_coils = int(ops.shape(data_array)[5])
        shape = tuple(int(n) for n in ops.shape(data_array))[:5]

        per_coil = []
        for c in range(n_coils):
            volume = ops.reshape(ops.take(data_array, np.array([c]), axis=5), shape)
            modulated, _ = self.process_tensor(volume, **kwargs)
            per_coil.append(modulated)

        return ops.stack(per_coil, axis=5)

    #*******************#
    #   b1+ modeling    #
    #*******************#
    def _b1_factor(self, matrix: Tuple[int, int, int], n_batch: int) -> np.ndarray:
        """The complex "(1|B, X, Y, Z)" factor "A_B1(r) * exp(i phi_B1(r))"."""
        b1 = self._as_map(self.b1_map, matrix, n_batch, 'b1_map')
        mag = np.abs(b1)
        ref = self.b1_reference if self.b1_reference is not None else 1.0
        b = mag / ref

        if self.b1_mode == 'complex_linear':
            amp = b
        else:
            alpha = math.radians(self.b1_flip_angle_deg)
            amp = np.sin(alpha * b) / math.sin(alpha)

        phase = np.angle(b1) if np.iscomplexobj(self.b1_map) else np.zeros_like(mag)
        return (amp * np.exp(1j * phase)).astype(np.complex128)

    #***********************#
    #   b0 segmentation     #
    #***********************#
    @staticmethod
    def _segment(b0_map: np.ndarray, n_segments: int,
                strategy: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize "b0_map" (Hz) into at most "n_segments" bins.

        Returns:
            "bin_idx", shaped like "b0_map", each voxel's segment index; and
            "f_q", the representative frequency (mean "delta_f") of each
            segment actually populated.
        """
        flat = np.asarray(b0_map, dtype=np.float64).reshape(-1)
        finite = np.isfinite(flat)
        values = flat[finite] if finite.any() else flat
        fmin, fmax = float(values.min()), float(values.max())

        # A flat map - most commonly all-zero - has nothing to segment: one bin
        # covers everything, and the whole model reduces to a plain NUFFT with
        # a constant (possibly zero) frequency shift.
        if fmax - fmin < 1e-9:
            return np.zeros(b0_map.shape, dtype=np.int64), np.array([fmin], dtype=np.float64)

        q = max(1, int(n_segments))
        if strategy == 'uniform':
            edges = np.linspace(fmin, fmax, q + 1)
        else:  # 'adaptive' - validated at construction, nothing else reaches here
            edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, q + 1)))
            if edges.size < 2:
                edges = np.array([fmin, fmax])

        edges = edges.copy()
        edges[0] -= 1e-6
        edges[-1] += 1e-6
        n_bins = edges.size - 1

        bin_idx_flat = np.clip(np.digitize(flat, edges[1:-1], right=False), 0, n_bins - 1)
        f_q = np.empty(n_bins, dtype=np.float64)
        for i in range(n_bins):
            sel = bin_idx_flat == i
            f_q[i] = flat[sel].mean() if sel.any() else 0.5 * (edges[i] + edges[i + 1])

        return bin_idx_flat.reshape(b0_map.shape), f_q

    @staticmethod
    def max_phase_error_rad(delta_f_hz: float, t_max_s: float) -> float:
        """
        Worst-case phase error from quantizing a bin of width "delta_f_hz".

        "2*pi * (delta_f_hz / 2) * t_max_s" - see the class docstring's
        "Choosing b0_n_segments" section. Evaluate at the map's own bin width
        (range / b0_n_segments for 'uniform') and the acquisition's latest
        spectral sample time to judge whether "b0_n_segments" is sufficient.
        """
        return 2.0 * math.pi * (float(delta_f_hz) / 2.0) * float(t_max_s)

    #*******************************#
    #   nufft forward measurement   #
    #*******************************#
    def _apply_segmented_b0(self, vol, matrix, geometry, dwell: float):
        """
        Measure "vol" along a fully-sampled trajectory, per B0 segment.

        For each batch element and each populated frequency segment: mask the
        object to that segment, forward-NUFFT it (the spectral axis rides in
        the NUFFT's channel slot, so this is one transform, not one per
        spectral sample), multiply by that segment's FID phase evolution, and
        accumulate. Summing the segments and taking the density-compensated
        adjoint approximates the exact voxel-wise
        "exp(-i 2*pi*delta_f(r)*t)" evolution - see the class docstring.
        """
        from augmentrum.processing.interpolating import LinearInterpolator
        from augmentrum.sampling.kspace_reconstructor import GriddingNUFFT
        from augmentrum.sampling.kspace_sampling import KspaceSampler

        n_batch = int(ops.shape(vol)[0])
        n_t = int(ops.shape(vol)[4])
        nx, ny, nz = matrix

        header = self._header_for_trajectory(matrix, geometry, dwell)
        rng = np.random.default_rng(self.traj_seed)
        us_params = dict(self.us_params)
        us_params.setdefault('seed', int(rng.integers(0, 2 ** 31 - 1)))

        shots, shot_mask, meta = KspaceSampler.get_kspace_shots_and_mask(
            header, self.trajectory, self.undersampling,
            float(self.acceleration_factor),
            traj_params=dict(self.traj_params), us_params=us_params, like=None,
        )
        self.last_meta_ = meta

        keep = np.flatnonzero(np.asarray(ops.to_numpy(shot_mask), dtype=bool))
        pts = np.concatenate([np.atleast_2d(np.asarray(shots[int(i)])) for i in keep])
        ndim = pts.shape[1]
        kmax = np.asarray(meta['kmax'][:ndim], dtype=np.float64)
        coords = (pts / (2.0 * kmax[None, :])).astype(np.float32)

        im_size = (nx, ny) if ndim == 2 else (nx, ny, nz)
        interpolator = LinearInterpolator() if self.nufft_impl == 'interp' else None
        nufft = GriddingNUFFT(im_size, self.nufft_osf, interpolator=interpolator)
        gridder = GriddingNUFFT(im_size, self.nufft_osf) if interpolator is not None else nufft

        t_vec = self._channel_time_axis(ndim, nz, n_t, dwell)
        b0_maps = self._as_map(self.b0_map, matrix, n_batch, 'b0_map')

        segments_info = []
        recon = []
        for b in range(n_batch):
            b0 = b0_maps[b if b0_maps.shape[0] > 1 else 0]
            bin_idx, f_q = self._segment(b0, self.b0_n_segments, self.b0_segment_strategy)

            vol_b = ops.reshape(ops.take(vol, np.array([b]), axis=0), (1, nx, ny, nz, n_t))

            kdata_total = None
            counts = []
            for q, f_hz in enumerate(f_q):
                mask_q = (bin_idx == q)
                counts.append(int(mask_q.sum()))
                if not mask_q.any():
                    continue

                mask_arr = mask_q.astype(np.complex64)[None, :, :, :, None]
                vol_bq = vol_b * ops.cast_like(ops.match_backend(mask_arr, vol_b), vol_b)
                plane_bq = self._to_channel_stack(vol_bq, ndim, matrix, n_t)

                kdata_q = nufft.forward(plane_bq, coords)
                phase = np.exp(-1j * 2.0 * math.pi * float(f_hz) * t_vec).astype(np.complex128)
                kdata_q = kdata_q * ops.cast_like(
                    ops.match_backend(phase[:, None], kdata_q), kdata_q)

                kdata_total = kdata_q if kdata_total is None else kdata_total + kdata_q

            segments_info.append({'f_q_hz': f_q.tolist(), 'voxel_counts': counts})

            if kdata_total is None:
                # Every voxel the map covers is empty (fully masked object) -
                # measure the (all-zero) object once so shapes stay consistent.
                kdata_total = nufft.forward(self._to_channel_stack(vol_b * 0, ndim, matrix, n_t),
                                           coords)

            recon.append(gridder.adjoint(kdata_total, coords))

        self.last_b0_segments_ = segments_info

        out = ops.stack(recon, axis=0)
        if ndim == 2:
            out = ops.transpose(ops.reshape(out, (n_batch, nz, n_t, nx, ny)), (0, 3, 4, 1, 2))
        else:
            out = ops.transpose(out, (0, 2, 3, 4, 1))

        return self._match_scale(out, vol)

    def _header_for_trajectory(self, matrix: Tuple[int, ...],
                               geometry: Optional[dict], dwell: float) -> dict:
        """Assemble the geometry header trajectory generation needs."""
        nx, ny = int(matrix[0]), int(matrix[1])
        nz = int(matrix[2]) if len(matrix) > 2 else 1

        if self.pixdim is not None:
            vx, vy, vz = (list(self.pixdim) + [1.0, 1.0, 1.0])[:3]
        elif geometry is not None and 'voxel_mm' in geometry:
            vx, vy, vz = geometry['voxel_mm']
        else:
            vx = vy = vz = 1.0

        sf = (geometry or {}).get('spectrometer_frequency') or 127.732434e6
        return {
            "dim":    [4, nx, ny, nz, 1, 1, 1, 1],
            "pixdim": [1.0, float(vx), float(vy), float(vz), float(dwell), 1.0, 1.0, 1.0],
            "DwellTime": float(dwell),
            "SpectrometerFrequency": [float(sf)],
        }

    #********************#
    #   array shaping    #
    #********************#
    @staticmethod
    def _as_map(map_arr: np.ndarray, matrix: Tuple[int, int, int],
               n_batch: int, name: str) -> np.ndarray:
        """
        *map_arr* as "(1, X, Y, Z)" (shared) or "(B, X, Y, Z)" (one per sample).

        Requires an exact match to the data's spatial matrix - including the
        trailing singleton Z axis for 2-D data - the same convention every
        other spatial map in Augmentrum uses (e.g. "coil_sampling.Supplied").
        """
        arr = np.asarray(map_arr)
        if arr.shape == matrix:
            return arr[None]
        if arr.shape == (n_batch,) + matrix:
            return arr
        raise ValueError(
            f"{name} must have shape {matrix} (shared across the batch) or "
            f"{(n_batch,) + matrix} (one per sample), got {arr.shape}."
        )

    @staticmethod
    def _to_channel_stack(x, ndim: int, matrix: Tuple[int, int, int], n_t: int):
        """
        "(1, X, Y, Z, T)" -> "(channels, *im_size)", the shape "GriddingNUFFT"
        forward/adjoint expects.

        Matches "KspaceUndersampling._nufft_gridding" exactly: a 2-D trajectory
        describes one plane, so Z joins the spectral axis in the channel slot
        (channel "c = z*n_t + t"); a 3-D trajectory samples the volume
        directly, so the channel is the spectral index alone.
        """
        nx, ny, nz = matrix
        if ndim == 2:
            return ops.reshape(ops.transpose(x, (0, 3, 4, 1, 2)), (nz * n_t, nx, ny))
        return ops.reshape(ops.transpose(x, (0, 4, 1, 2, 3)), (n_t, nx, ny, nz))

    @staticmethod
    def _channel_time_axis(ndim: int, nz: int, n_t: int, dwell: float) -> np.ndarray:
        """
        FID sample time "t_n = n * dwell", one value per NUFFT channel.

        The channel axis carries the spectral index (see "_to_channel_stack"),
        so this is what "exp(-i 2*pi*f_q*t)" is evaluated against - never a
        k-space readout time, which "KspaceUndersampling" already establishes
        does not exist per-sample in phase-encoded MRSI.
        """
        t = np.arange(n_t, dtype=np.float64) * float(dwell)
        return np.tile(t, nz) if ndim == 2 else t

    @staticmethod
    def _match_scale(out, vol):
        """
        Restore the input's overall scale.

        Same rationale as "KspaceUndersampling._match_scale": a
        density-compensated adjoint recovers the image only up to a constant
        that depends on the trajectory, the DCF and the grid size.
        """
        num = ops.sqrt(ops.sum(ops.abs(vol) ** 2))
        den = ops.sqrt(ops.sum(ops.abs(out) ** 2))
        scale = num / ops.where(den > 1e-12, den, den * 0 + 1e-12)
        return out * ops.cast_like(scale, out)
