####################################################################################################
#                                     kspace_sampling.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-03-25                                                                              #
#                                                                                                  #
# Purpose: Generates k-space sampling trajectories and undersampling masks for MRI/MRS             #
#          simulation pipelines, backend-agnostically across NumPy and PyTorch. Produces           #
#          coordinates and boolean masks only; gridding and reconstruction live elsewhere.         #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from augmentrum.core.base_module import BaseModule
from nifti_mrs_plus import Backend
from nifti_mrs_plus import ops
from nifti_mrs_plus.ops import is_torch


#***************#
#   constants   #
#***************#
GOLDEN_ANGLE_RAD: float = math.pi * (3.0 - math.sqrt(5.0))   # ≈ 2.39996 rad
GOLDEN_RATIO: float = (1.0 + math.sqrt(5.0)) / 2.0           # ≈ 1.61803


#**************************************************************************************************#
#                                       Class KspaceGeometry                                       #
#**************************************************************************************************#
#                                                                                                  #
# Imaging geometry — matrix, voxel size, field of view and the k-space extent that follows from    #
# them.                                                                                            #
#                                                                                                  #
#**************************************************************************************************#
class KspaceGeometry:
    """
    Imaging geometry — matrix, voxel size, field of view and the k-space extent
    that follows from them.

    Geometry belongs to the data, so prefer reading it off a NIfTI-MRS object
    rather than hand-assembling a header dict; then it cannot drift from the
    array it describes.

    >>> geom = KspaceGeometry.read_header_geometry(nifti_mrs_plus)
    >>> geom["fov_mm"]
    (179.2, 224.0, 128.0)
    """

    #: Standard NIfTI-MRS header field names (for documentation & parsing)
    NIFTI_MRS_FIELDS = {
        "pixdim":               "voxel sizes (mm) stored at pixdim[1:4]",
        "dim":                  "matrix dimensions stored at dim[1:4]",
        "qform_code":           "qform code (non-zero -> use qform)",
        "sform_code":           "sform code (non-zero -> use sform)",
        "SpectrometerFrequency": "transmitter center frequency [Hz] list",
        "DwellTime":            "ADC dwell time [s] per spectral point",
        "ResonantNucleus":      "nucleus label e.g. '1H'",
    }

    @staticmethod
    def header_dict_from_nifti(obj) -> Dict[str, Any]:
        """
        Build the header dict "read_header_geometry" expects from a NIfTI-MRS object.

        Geometry belongs to the data, so it should be read off the object rather than
        hand-assembled by callers.  This accepts anything Augmentrum carries data in:

          * "NIfTI_MRS_Plus" — a batch shares one geometry, so the first subject is
            used (its ".shape" has a leading batch axis, which is why the underlying
            "nifti_list" entry is read instead).
          * a bare "NIFTI_MRS" (or any object exposing ".shape" and ".header").

        Note the unit change: NIfTI-MRS stores "SpectrometerFrequency" in **MHz**,
        while this module's header dicts use **Hz**, so it is scaled by 1e6 here.
        """
        nifti = obj
        nifti_list = getattr(obj, 'nifti_list', None)
        if nifti_list is not None:
            if len(nifti_list) == 0:
                raise ValueError("Cannot read geometry from an empty NIfTI_MRS_Plus.")
            nifti = nifti_list[0]

        if not hasattr(nifti, 'shape'):
            raise TypeError(
                f"Expected a NIfTI_MRS_Plus or NIFTI_MRS object, got {type(obj).__name__}."
            )

        shape = tuple(int(s) for s in nifti.shape[:3])
        shape = shape + (1,) * (3 - len(shape))

        hdr: Dict[str, Any] = {"dim": [4, shape[0], shape[1], shape[2], 1, 1, 1, 1]}

        nhdr = getattr(nifti, 'header', None)
        pixdim = None
        if nhdr is not None:
            try:
                pixdim = [float(v) for v in nhdr['pixdim']]
            except Exception:
                pixdim = None
        if pixdim is None:
            raise KeyError(
                f"{type(nifti).__name__} has no readable 'pixdim' in its NIfTI header, "
                "so voxel size is unknown. Pass an explicit header dict instead."
            )
        hdr["pixdim"] = pixdim

        dwell = getattr(nifti, 'dwelltime', None)
        if dwell is not None:
            hdr["DwellTime"] = float(dwell)

        sf = getattr(nifti, 'spectrometer_frequency', None)
        if sf is not None:
            sf0 = sf[0] if hasattr(sf, '__getitem__') else sf
            hdr["SpectrometerFrequency"] = [float(sf0) * 1e6]   # MHz -> Hz

        nuc = getattr(nifti, 'nucleus', None)
        if nuc is not None:
            hdr["ResonantNucleus"] = nuc[0] if hasattr(nuc, '__getitem__') else nuc

        return hdr

    @staticmethod
    def read_header_geometry(header: Union[Dict[str, Any], Any]) -> dict:
        """
        Return imaging geometry fields from a NIfTI-MRS header dict or data object.

        Parameters
        ----------
        header : dict or NIfTI_MRS_Plus or NIFTI_MRS
            Passing a **data object** is the preferred form — geometry is then read
            straight off the data via "header_dict_from_nifti" and cannot drift from
            it.  A plain dict is still accepted for synthetic geometries and tests.

            Recognized dict fields (see _NIFTI_MRS_FIELDS):
              - "dim"    : array-like, dim[1..3] → matrix [nx, ny, nz]
              - "pixdim" : array-like, pixdim[1..3] → voxel size [vx, vy, vz] in mm
              - "SpectrometerFrequency" : list[float] → center frequency in Hz
              - "DwellTime"             : float → ADC dwell time in seconds
              Fallback keys also accepted: "matrix", "voxel_size_mm",
              "fov_mm", "dwell_time".

        Returns
        -------
        dict with keys:
            fov_mm   : tuple (fx, fy, fz) — field of view in mm
            matrix   : tuple (nx, ny, nz) — matrix size
            voxel_mm : tuple (vx, vy, vz) — voxel size in mm
            dwell_time : float or None
            spectrometer_frequency : float or None
            ndim     : 2 or 3 (spatial)
            inferred : list[str] — list of fields that were inferred rather than explicit
        """
        if not isinstance(header, dict):
            header = KspaceGeometry.header_dict_from_nifti(header)

        inferred: List[str] = []

        # ---- Matrix ----
        if "dim" in header:
            dim = header["dim"]
            nx, ny, nz = int(dim[1]), int(dim[2]), int(dim[3])
        elif "matrix" in header:
            m = header["matrix"]
            nx, ny, nz = int(m[0]), int(m[1]), int(m[2]) if len(m) > 2 else 1
            inferred.append("matrix from 'matrix' fallback key")
        else:
            raise KeyError(
                "Header must contain 'dim' (NIfTI-MRS standard, dim[1:4] = [nx,ny,nz]) "
                "or fallback key 'matrix'. Neither was found."
            )

        # ---- Voxel size ----
        if "pixdim" in header:
            pd = header["pixdim"]
            vx, vy, vz = float(pd[1]), float(pd[2]), float(pd[3])
        elif "voxel_size_mm" in header:
            vs = header["voxel_size_mm"]
            vx, vy, vz = (float(vs[0]), float(vs[1]),
                          float(vs[2]) if len(vs) > 2 else float(vs[0]))
            inferred.append("voxel_mm from 'voxel_size_mm' fallback key")
        elif "fov_mm" in header:
            # derive voxel from FOV / matrix
            fov = header["fov_mm"]
            vx = float(fov[0]) / nx
            vy = float(fov[1]) / ny
            vz = float(fov[2]) / nz if len(fov) > 2 else 1.0
            inferred.append("voxel_mm derived from fov_mm / matrix")
        else:
            raise KeyError(
                "Header must contain 'pixdim' (NIfTI-MRS standard, pixdim[1:3] = voxel mm) "
                "or fallback keys 'voxel_size_mm' or 'fov_mm'. None found."
            )

        fov_mm = (nx * vx, ny * vy, nz * vz)

        # ---- Dwell time ----
        dwell_time: Optional[float] = None
        if "DwellTime" in header:
            dwell_time = float(header["DwellTime"])
        elif "dwell_time" in header:
            dwell_time = float(header["dwell_time"])
            inferred.append("dwell_time from 'dwell_time' fallback key")

        # ---- Spectrometer frequency ----
        spec_freq: Optional[float] = None
        if "SpectrometerFrequency" in header:
            sf = header["SpectrometerFrequency"]
            spec_freq = float(sf[0]) if hasattr(sf, "__len__") else float(sf)

        ndim = 2 if nz == 1 else 3

        return {
            "fov_mm":                  fov_mm,
            "matrix":                  (nx, ny, nz),
            "voxel_mm":                (vx, vy, vz),
            "dwell_time":              dwell_time,
            "spectrometer_frequency":  spec_freq,
            "ndim":                    ndim,
            "inferred":                inferred,
        }

    @staticmethod
    def _kmax_from_geometry(geom: dict) -> Tuple[float, float, float]:
        """
        Compute kmax per axis in cycles/m.

        Formula: kmax_axis = (N_axis / 2) / (FOV_axis_mm / 1000)
                 equivalently kmax = N / (2 * FOV_m)
        For Nyquist sampling Δk = 1 / FOV_m, so kmax = (N/2) * Δk.
        """
        nx, ny, nz = geom["matrix"]
        fx_m, fy_m, fz_m = (v / 1000.0 for v in geom["fov_mm"])
        kmax_x = (nx / 2.0) / fx_m
        kmax_y = (ny / 2.0) / fy_m
        kmax_z = (nz / 2.0) / fz_m if nz > 1 else 0.0
        return kmax_x, kmax_y, kmax_z


#**************************************************************************************************#
#                                         Class Trajectory                                         #
#**************************************************************************************************#
#                                                                                                  #
# Base class for k-space trajectories.                                                             #
#                                                                                                  #
#**************************************************************************************************#
class Trajectory(ABC):
    """
    Base class for k-space trajectories.

    A trajectory is a *parameterized object*: construct it with the parameters
    that shape it, then ask it for coordinates in a given geometry. This
    replaces the previous arrangement where every trajectory was a private
    static method reached by "getattr" through a name -> method-name table,
    which meant parameters had no home and no per-trajectory validation point.

    Subclasses must set "NAME" and "NDIM" and implement
    "generate". Shared machinery (density placeholder, undersampling
    warning) lives here.

    >>> traj = TrajectoryRegistry.create('radial_2d', n_shots=64)
    >>> traj.NAME, traj.NDIM
    ('radial_2d', 2)
    """

    #: Registry key. Set by every concrete subclass.
    NAME: str = ""
    #: Coordinate dimensionality, 2 or 3.
    NDIM: int = 2

    def __init__(self, **params):
        self.params = dict(params)

    def __repr__(self):
        return f"{type(self).__name__}(name={self.NAME!r}, params={self.params!r})"

    #****************#
    #   generation   #
    #****************#

    @abstractmethod
    def generate(self, geometry: dict, like=None) -> Tuple[List[Any], Dict[str, Any]]:
        """Return "(shots, meta)" for *geometry*, built on *like*'s backend."""


    @staticmethod
    def _density_estimate(trajectory_name: str, n_shots: int,
                                      samples_per_shot: int, geom: dict) -> dict:
        """
        Return a placeholder density estimate dictionary.

        A proper DCF would require Voronoi cells or iterative pipe methods.
        This placeholder records the parameters so downstream code can
        recompute or override it.

        Returns
        -------
        dict with keys:
            type : str — 'analytic_placeholder'
            description : str
            n_shots : int
            samples_per_shot : int
            kmax : tuple[float]
            note : str
        """
        kmax = KspaceGeometry._kmax_from_geometry(geom)
        return {
            "type":             "analytic_placeholder",
            "description":      (
                f"Density placeholder for '{trajectory_name}'. "
                "Compute DCF externally (e.g. iterative pipe, Voronoi, or analytic)."
            ),
            "trajectory":       trajectory_name,
            "n_shots":          n_shots,
            "samples_per_shot": samples_per_shot,
            "kmax":             kmax,
            "note":             (
                "For Cartesian: density = 1 everywhere by definition. "
                "For radial: density ∝ |k| (center-heavy). "
                "For spirals: use gradient-delay-corrected NUFFT pipe density."
            ),
        }


    @staticmethod
    def _warn_if_undersampled(shots, meta, geom, like, name):
        """
        Warn when consecutive samples along a shot are more than one k-space bin
        apart, which means the readout is aliased.

        A trajectory is only coordinates, so nothing can fail here — an aliased
        readout produces a perfectly valid array that simply does not describe
        the curve it claims to. The symptom downstream is a rasterised mask with
        moire structure, or apparent counter-rotating arms, and it is easy to
        mistake for an undersampling artefact rather than a trajectory bug.
        """
        if not shots:
            return
        try:
            first = np.asarray(ops.to_numpy(shots[0]), dtype=np.float64)
            if first.ndim != 2 or first.shape[0] < 2:
                return
            gap = float(np.median(np.linalg.norm(np.diff(first, axis=0), axis=1)))
            # One Cartesian bin along the first axis: dk = 1 / FOV.
            dk = 1.0 / (geom["fov_mm"][0] / 1000.0)
        except Exception:
            return

        if dk > 0 and gap > 2.0 * dk:
            warnings.warn(
                f"Trajectory '{name}': consecutive samples are ~{gap / dk:.0f} k-space "
                f"bins apart, so each shot is aliased along its own readout. Increase "
                f"samples_per_shot, or reduce the winding (e.g. 'spiral_turns'). "
                f"Coordinates are still returned, but any mask or PSF derived from "
                f"them will show aliasing rather than the intended trajectory.",
                RuntimeWarning,
                stacklevel=3,
            )


    @staticmethod
    def shots_from_continuous(
        coords: Any,
        segmentation: Dict[str, Any],
        like=None,
    ) -> List[Any]:
        """
        Split a continuous k-space readout array into shots.

        Parameters
        ----------
        coords : array, shape (N_total_samples, Ndims)
            Continuous coordinate array in cycles/m.
        segmentation : dict
            Must contain one of:
              'n_shots' : int  — equal-length split into n_shots chunks.
              'shot_lengths' : list[int] — variable lengths per shot.
              'shot_boundaries' : list[int] — start indices for each shot
                  (last boundary implicitly at N_total_samples).
        like : array, optional

        Returns
        -------
        list of arrays, each shape (Nsamples_i, Ndims)

        Raises
        ------
        ValueError
            If segmentation dict lacks required keys or lengths are inconsistent.
        """
        total = ops.shape(coords)[0]
        cpu_coords = ops.to_numpy(coords)

        if "n_shots" in segmentation:
            n = int(segmentation["n_shots"])
            if total % n != 0:
                raise ValueError(
                    f"'n_shots'={n} does not evenly divide total samples={total}. "
                    "Use 'shot_lengths' or 'shot_boundaries' for unequal shots."
                )
            sps = total // n
            shots = [ops.asarray_like(like, cpu_coords[i * sps: (i + 1) * sps]) for i in range(n)]

        elif "shot_lengths" in segmentation:
            lengths = list(segmentation["shot_lengths"])
            if sum(lengths) != total:
                raise ValueError(
                    f"Sum of shot_lengths ({sum(lengths)}) != total samples ({total})."
                )
            shots = []
            start = 0
            for length in lengths:
                shots.append(ops.asarray_like(like, cpu_coords[start: start + length]))
                start += length

        elif "shot_boundaries" in segmentation:
            bounds = list(segmentation["shot_boundaries"]) + [total]
            shots = []
            for s, e in zip(bounds[:-1], bounds[1:]):
                shots.append(ops.asarray_like(like, cpu_coords[s:e]))

        else:
            raise ValueError(
                "segmentation must contain 'n_shots', 'shot_lengths', or 'shot_boundaries'. "
                f"Got keys: {list(segmentation.keys())}."
            )
        return shots


#**************************************************************************************************#
#                                     Class TrajectoryRegistry                                     #
#**************************************************************************************************#
#                                                                                                  #
# Name -> "Trajectory" subclass lookup.                                                     #
#                                                                                                  #
#**************************************************************************************************#
class TrajectoryRegistry:
    """
    Name -> "Trajectory" subclass lookup.

    Classes register themselves with the "register" decorator, so adding a
    trajectory means writing one class — there is no separate table to keep in
    sync, which is how "stack_of_cones" once stayed registered as a duplicate
    of "stack_of_spirals" long after it stopped meaning anything different.

    >>> 'radial_2d' in TrajectoryRegistry.available()
    True
    """

    _REGISTRY: Dict[str, type] = {}

    #****************#
    #   population   #
    #****************#

    @classmethod
    def register(cls, trajectory_cls: type) -> type:
        """Class decorator recording *trajectory_cls* under its "NAME"."""
        name = getattr(trajectory_cls, "NAME", "")
        if not name:
            raise ValueError(f"{trajectory_cls.__name__} must define a non-empty NAME.")
        if name in cls._REGISTRY:
            raise ValueError(
                f"Trajectory name '{name}' already registered to "
                f"{cls._REGISTRY[name].__name__}."
            )
        cls._REGISTRY[name] = trajectory_cls
        return trajectory_cls

    #****************#
    #   lookup       #
    #****************#

    @classmethod
    def available(cls) -> List[str]:
        """Every registered trajectory name, sorted."""
        return sorted(cls._REGISTRY)

    @classmethod
    def get(cls, name: str) -> type:
        """The class registered under *name*."""
        if name not in cls._REGISTRY:
            raise ValueError(
                f"Unknown trajectory '{name}'. Valid trajectories: {cls.available()}."
            )
        return cls._REGISTRY[name]

    @classmethod
    def create(cls, name: str, **params) -> "Trajectory":
        """Construct the trajectory registered under *name* with *params*."""
        return cls.get(name)(**params)

    #****************#
    #   generation   #
    #****************#

    @classmethod
    def generate(cls, name: str, header: Dict[str, Any], params: Dict[str, Any],
                 like=None) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Build a trajectory by name and return shot-organized coordinates.

        Parameters
        ----------
        name : str
            Registered trajectory name; see "available".
        header : dict
            Parsed NIfTI-MRS header, passed to "read_header_geometry".
        params : dict
            Trajectory-specific parameters, forwarded to the class constructor.
        like : array, optional
            Reference tensor fixing the backend and device of the result.

        Returns
        -------
        coords_per_shot : list of arrays, each (Nsamples, Ndims)
            Ndims is 2 or 3. All values in cycles/m.
        meta : dict
            trajectory_type, kmax, fov_mm, matrix, n_shots, samples_per_shot,
            density_estimate, params_used.

        Raises
        ------
        ValueError
            Unknown trajectory name, or geometry incompatible with it.
        KeyError
            Required header fields missing.
        TypeError
        """
        trajectory = cls.create(name, **(params or {}))
        geom = KspaceGeometry.read_header_geometry(header)
        if geom["inferred"]:
            warnings.warn(
                f"Some geometry fields were inferred: {geom['inferred']}. "
                "Verify header completeness for accurate k-space scaling.",
                UserWarning,
                stacklevel=2,
            )
        shots, meta = trajectory.generate(geom, like)
        Trajectory._warn_if_undersampled(shots, meta, geom, like, name)
        return shots, meta


#**************************************************************************************************#
#                                        Class Cartesian2D                                         #
#**************************************************************************************************#
#                                                                                                  #
# See "_build" for the geometry and parameters.                                              #
#                                                                                                  #
#**************************************************************************************************#
@TrajectoryRegistry.register
class Cartesian2D(Trajectory):
    """See "_build" for the geometry and parameters."""

    NAME = "cartesian_2d"
    NDIM = 2

    def generate(self, geometry, like=None):
        return self._build(geometry, self.params, like)

    @staticmethod
    def _build(geom: dict, params: dict, like=None) -> Tuple[List[Any], Dict[str, Any]]:
        """
        2D Cartesian trajectory.

        Each shot = one phase-encode line (ky) with full kx readout.

        Parameters (params)
        -------------------
        ordering : str
            'linear' (default) — ky from -N/2 to N/2-1
            'centric' — ky ordered from center outwards
        n_shots : int, optional
            Number of phase encodes to generate (default: ny).

        Output shot count: n_shots (= ny unless overridden).
        Sample count per shot: nx.
        """
        nx, ny, _ = geom["matrix"]
        fov_x_m = geom["fov_mm"][0] / 1000.0
        fov_y_m = geom["fov_mm"][1] / 1000.0

        ordering = params.get("ordering", "linear")
        n_shots = int(params.get("n_shots", ny))

        kx_line = ops.linspace_like(like, -nx / 2, nx / 2 - 1, nx, True) / fov_x_m   # (nx,) in 1/m

        if ordering == "centric":
            half = ny // 2
            ky_idx = []
            for i in range(max(half, ny - half)):
                if i < half:
                    ky_idx.extend([half - 1 - i, half + i] if half + i < ny else [half - 1 - i])
                elif half + i < ny:
                    ky_idx.append(half + i)
            ky_indices = np.array(ky_idx[:n_shots], dtype=np.float32)
        else:
            ky_indices = np.arange(n_shots, dtype=np.float32)

        # Map indices [0..ny-1] → k-space positions
        # ky = (index - ny/2) / fov_y_m
        ky_positions = (ky_indices - ny / 2.0) / fov_y_m

        shots = []
        for ky_val in ky_positions:
            ky_line = ops.full_like_shape(like, (nx,), float(ky_val))
            shot = ops.stack([kx_line, ky_line], axis=1)   # (nx, 2)
            shots.append(shot)

        kmax_x, kmax_y, _ = KspaceGeometry._kmax_from_geometry(geom)
        meta = {
            "trajectory_type":    "cartesian_2d",
            "ordering":           ordering,
            "kmax":               (kmax_x, kmax_y),
            "fov_mm":             geom["fov_mm"][:2],
            "matrix":             (nx, ny),
            "n_shots":            n_shots,
            "samples_per_shot":   nx,
            "density_estimate":   Trajectory._density_estimate("cartesian_2d", n_shots, nx, geom),
            "params_used":        params,
        }
        return shots, meta


#**************************************************************************************************#
#                                        Class Cartesian3D                                         #
#**************************************************************************************************#
#                                                                                                  #
# See "_build" for the geometry and parameters.                                              #
#                                                                                                  #
#**************************************************************************************************#
@TrajectoryRegistry.register
class Cartesian3D(Trajectory):
    """See "_build" for the geometry and parameters."""

    NAME = "cartesian_3d"
    NDIM = 3

    def generate(self, geometry, like=None):
        return self._build(geometry, self.params, like)

    @staticmethod
    def _build(geom: dict, params: dict, like=None) -> Tuple[List[Any], Dict[str, Any]]:
        """
        3D Cartesian trajectory.

        Each shot = one (ky, kz) phase-encode pair with full kx readout.

        Parameters (params)
        -------------------
        ordering : str
            'linear' (default), 'centric_ky', 'centric_kz', 'centric_both'
        n_shots : int, optional
            Total phase encodes (default: ny*nz).

        Output shot count: ny * nz (or n_shots).
        Sample count per shot: nx.
        """
        if geom["ndim"] < 3:
            raise ValueError(
                "cartesian_3d requires a 3D geometry (nz > 1). "
                "Header has nz=1. Use cartesian_2d for 2D acquisitions."
            )
        nx, ny, nz = geom["matrix"]
        fov_x_m = geom["fov_mm"][0] / 1000.0
        fov_y_m = geom["fov_mm"][1] / 1000.0
        fov_z_m = geom["fov_mm"][2] / 1000.0

        ordering = params.get("ordering", "linear")
        kx_line = ops.linspace_like(like, -nx / 2, nx / 2 - 1, nx) / fov_x_m

        ky_arr = (np.arange(ny, dtype=np.float32) - ny / 2.0) / fov_y_m
        kz_arr = (np.arange(nz, dtype=np.float32) - nz / 2.0) / fov_z_m

        if "centric" in ordering:
            def _centric_order(n):
                half = n // 2
                idx = []
                for i in range(max(half, n - half)):
                    if i < half:
                        idx.extend([half - 1 - i, half + i] if half + i < n else [half - 1 - i])
                    elif half + i < n:
                        idx.append(half + i)
                return idx

            if ordering in ("centric_ky", "centric_both"):
                ky_arr = ky_arr[_centric_order(ny)]
            if ordering in ("centric_kz", "centric_both"):
                kz_arr = kz_arr[_centric_order(nz)]

        n_max = int(params.get("n_shots", ny * nz))
        shots = []
        count = 0
        for kz_val in kz_arr:
            if count >= n_max:
                break
            for ky_val in ky_arr:
                if count >= n_max:
                    break
                ky_line = ops.full_like_shape(like, (nx,), float(ky_val))
                kz_line = ops.full_like_shape(like, (nx,), float(kz_val))
                shot = ops.stack([kx_line, ky_line, kz_line], axis=1)  # (nx, 3)
                shots.append(shot)
                count += 1

        kmax_x, kmax_y, kmax_z = KspaceGeometry._kmax_from_geometry(geom)
        n_shots = len(shots)
        meta = {
            "trajectory_type":    "cartesian_3d",
            "ordering":           ordering,
            "kmax":               (kmax_x, kmax_y, kmax_z),
            "fov_mm":             geom["fov_mm"],
            "matrix":             (nx, ny, nz),
            "n_shots":            n_shots,
            "samples_per_shot":   nx,
            "density_estimate":   Trajectory._density_estimate("cartesian_3d", n_shots, nx, geom),
            "params_used":        params,
        }
        return shots, meta


#**************************************************************************************************#
#                                          Class Radial2D                                          #
#**************************************************************************************************#
#                                                                                                  #
# Uniform radial spokes.                                                                           #
#                                                                                                  #
#**************************************************************************************************#
@TrajectoryRegistry.register
class Radial2D(Trajectory):
    """Uniform radial spokes. Full diameters over [0, pi), covering the disc exactly once."""

    NAME = "radial_2d"
    NDIM = 2
    USE_GOLDEN = False

    def generate(self, geometry, like=None):
        return self._build(geometry, self.params, like, use_golden=self.USE_GOLDEN)

    @staticmethod
    def _build(geom: dict, params: dict, like=None,
                        use_golden: bool = False) -> Tuple[List[Any], Dict[str, Any]]:
        """
        2D radial trajectory (uniform or golden-angle angular increment).

        Center-out by default: samples go from k=0 outward to kmax.

        Parameters (params)
        -------------------
        n_shots : int
            Number of spokes (default: π/2 * max(nx,ny) ≈ Nyquist angular rate).
        samples_per_shot : int
            Samples per spoke (default: max(nx, ny)).
        center_out : bool
            If True (default), spoke goes from center outward.
            If False, spoke is symmetric (–kmax → +kmax, center at midpoint).
        golden_angle_rad : float
            Golden angle override (default: GOLDEN_ANGLE_RAD).

        Output shot count: n_shots.
        Sample count per shot: samples_per_shot (constant).
        """
        nx, ny, _ = geom["matrix"]
        fov_x_m = geom["fov_mm"][0] / 1000.0
        fov_y_m = geom["fov_mm"][1] / 1000.0
        kmax_x = (nx / 2.0) / fov_x_m
        kmax_y = (ny / 2.0) / fov_y_m

        n_shots = int(params.get("n_shots", max(1, int(math.pi / 2 * max(nx, ny)))))
        sps = int(params.get("samples_per_shot", max(nx, ny)))
        # Full diameters by default (-kmax -> +kmax through the center).
        #
        # This has to agree with the angular range, and for the uniform case the
        # range is [0, pi). A center-out ray only covers the half-plane its
        # angle points into, so center-out spokes over [0, pi) rasterise to HALF
        # a disc — the other half is never visited and the reconstruction is
        # missing every conjugate line. Full diameters over [0, pi) cover the
        # disc exactly once, which is what "radial" means in MRI and what a
        # stack-of-stars readout actually acquires.
        #
        # Pass center_out=True only together with an angular range spanning 2*pi
        # (the golden-angle case does, since i*GA is taken modulo 2*pi).
        center_out = bool(params.get("center_out", False))
        ga = float(params.get("golden_angle_rad", GOLDEN_ANGLE_RAD))

        name = "golden_radial_2d" if use_golden else "radial_2d"
        shots = []

        for i in range(n_shots):
            if use_golden:
                angle = i * ga
            else:
                angle = i * math.pi / n_shots

            if center_out:
                r = ops.linspace_like(like, 0.0, 1.0, sps)
            else:
                r = ops.linspace_like(like, -1.0, 1.0, sps)

            kx = r * ops.cos(ops.asarray_like(like, [angle]))[0] * kmax_x
            ky = r * ops.sin(ops.asarray_like(like, [angle]))[0] * kmax_y
            shot = ops.stack([kx, ky], axis=1)
            shots.append(shot)

        kmax = (kmax_x, kmax_y)
        meta = {
            "trajectory_type":    name,
            "golden_angle_rad":   ga if use_golden else None,
            "kmax":               kmax,
            "fov_mm":             geom["fov_mm"][:2],
            "matrix":             (nx, ny),
            "n_shots":            n_shots,
            "samples_per_shot":   sps,
            "center_out":         center_out,
            "density_estimate":   Trajectory._density_estimate(name, n_shots, sps, geom),
            "params_used":        params,
        }
        return shots, meta


#**************************************************************************************************#
#                                       Class GoldenRadial2D                                       #
#**************************************************************************************************#
#                                                                                                  #
# Radial spokes advancing by the golden angle, so any prefix stays near-uniform.                   #
#                                                                                                  #
#**************************************************************************************************#
@TrajectoryRegistry.register
class GoldenRadial2D(Radial2D):
    """Radial spokes advancing by the golden angle, so any prefix stays near-uniform."""

    NAME = "golden_radial_2d"
    USE_GOLDEN = True


#**************************************************************************************************#
#                                          Class Spiral2D                                          #
#**************************************************************************************************#
#                                                                                                  #
# See "_build" for the geometry and parameters.                                              #
#                                                                                                  #
#**************************************************************************************************#
@TrajectoryRegistry.register
class Spiral2D(Trajectory):
    """See "_build" for the geometry and parameters."""

    NAME = "spiral_2d"
    NDIM = 2

    def generate(self, geometry, like=None):
        return self._build(geometry, self.params, like)

    @staticmethod
    def _build(geom: dict, params: dict, like=None) -> Tuple[List[Any], Dict[str, Any]]:
        """
        2D multi-shot Archimedean or variable-density spiral.

        Each shot = one spiral arm (interleaved rotation by 2π/n_shots).
        Center-out (starts at k=0, ends at kmax).

        Parameters (params)
        -------------------
        n_shots : int
            Number of interleaved arms (default: 1).
        samples_per_shot : int
            Samples per arm (default: nx * ny // n_shots).
        spiral_turns : float
            Number of full turns per arm. Default max(nx, ny) / (2 * n_shots),
            the interleaved-spiral relation: all arms together sweep the N/2
            turns needed to reach kmax at Nyquist spacing, so each of n arms
            makes N/(2n).

            The previous default, n_shots * max(nx,ny) / 4, was larger by a
            factor of 2 * n_shots^2 — 8192x at 256 arms on a 128 matrix. Nothing
            errored, because a trajectory is just coordinates; the arms simply
            wound so fast that consecutive samples landed ~50 k-space bins apart
            and the rasterised mask showed angular aliasing (apparent
            counter-rotating arms, and only a couple of visible lobes) rather
            than the spiral itself.
        vd_alpha : float
            Variable-density exponent ≥ 1 (1 = Archimedean, >1 = denser center).
            r(t) ∝ t^(1/vd_alpha).
        ordering : str
            How successive arms are rotated.
            'golden' (default) — arm i is rotated by i * golden angle, so ANY
                prefix of arms stays near-uniform in angle. This is what makes
                the trajectory usable with prefix-style undersampling.
            'linear' — arm i is rotated by i * 2π / n_shots. Uniform only when
                every arm is kept: retaining the first N/AF arms then leaves a
                contiguous angular wedge and the rest of k-space is unsampled,
                which shows up as a one-sided reconstruction.
        golden_angle_rad : float
            Golden angle override (default: GOLDEN_ANGLE_RAD).
        rotate_first : bool
            If False, arms are not rotated at all (every arm identical).
        """
        nx, ny, _ = geom["matrix"]
        fov_x_m = geom["fov_mm"][0] / 1000.0
        fov_y_m = geom["fov_mm"][1] / 1000.0
        kmax_x = (nx / 2.0) / fov_x_m
        kmax_y = (ny / 2.0) / fov_y_m

        n_shots = int(params.get("n_shots", 1))
        turns = float(params.get("spiral_turns",
                                 max(nx, ny) / (2.0 * max(n_shots, 1))))
        # Along-arm Nyquist. The parametrisation is uniform in t while the
        # path speed grows as (N/2)*sqrt(1 + (2*pi*turns*t)^2) bins per unit t,
        # so the binding constraint is the PEAK speed at t=1, not the mean arc
        # length: with only mean-rate sampling the outer turns still take
        # >1-bin steps, the rasterised mask shows dashed arms there, and the
        # zero-filled recon grows coherent streaks from the periodic gaps.
        w = 2.0 * math.pi * turns
        peak_bins = 0.5 * max(nx, ny) * math.sqrt(1.0 + w * w)
        sps = int(params.get("samples_per_shot",
                             max(nx * ny // max(n_shots, 1), 64,
                                 int(math.ceil(peak_bins)))))
        vd_alpha = float(params.get("vd_alpha", 1.0))
        rotate_first = bool(params.get("rotate_first", True))
        ordering = str(params.get("ordering", "golden")).lower()
        ga = float(params.get("golden_angle_rad", GOLDEN_ANGLE_RAD))
        if ordering not in ("golden", "linear"):
            raise ValueError(
                f"spiral_2d ordering must be 'golden' or 'linear', got {ordering!r}"
            )

        t = ops.linspace_like(like, 0.0, 1.0, sps)           # normalized parameter
        # r(t) ∝ t^(1/vd_alpha), ensures r ∈ [0, 1]
        r = t ** (1.0 / vd_alpha)
        theta_base = r * (2.0 * math.pi * turns)

        shots = []
        for i in range(n_shots):
            if not rotate_first:
                rot = 0.0
            elif ordering == "golden":
                rot = (i * ga) % (2.0 * math.pi)
            else:
                rot = 2.0 * math.pi * i / n_shots
            theta = theta_base + rot
            kx = r * ops.cos(theta) * kmax_x
            ky = r * ops.sin(theta) * kmax_y
            shot = ops.stack([kx, ky], axis=1)
            shots.append(shot)

        meta = {
            "trajectory_type":    "spiral_2d",
            "spiral_turns":       turns,
            "vd_alpha":           vd_alpha,
            "ordering":           ordering,
            "kmax":               (kmax_x, kmax_y),
            "fov_mm":             geom["fov_mm"][:2],
            "matrix":             (nx, ny),
            "n_shots":            n_shots,
            "samples_per_shot":   sps,
            "density_estimate":   Trajectory._density_estimate("spiral_2d", n_shots, sps, geom),
            "params_used":        params,
        }
        return shots, meta


#**************************************************************************************************#
#                                         Class Rosette2D                                          #
#**************************************************************************************************#
#                                                                                                  #
# See "_build" for the geometry and parameters.                                              #
#                                                                                                  #
#**************************************************************************************************#
@TrajectoryRegistry.register
class Rosette2D(Trajectory):
    """See "_build" for the geometry and parameters."""

    NAME = "rosette_2d"
    NDIM = 2

    def generate(self, geometry, like=None):
        return self._build(geometry, self.params, like)

    @staticmethod
    def _build(geom: dict, params: dict, like=None) -> Tuple[List[Any], Dict[str, Any]]:
        """
        2D Rosette (looped petal) trajectory.

        Parametric: kx = kmax * sin(ω1*t) * cos(ω2*t)
                    ky = kmax * sin(ω1*t) * sin(ω2*t)

        Parameters (params)
        -------------------
        n_shots : int
            Number of shots (each is one full rosette pass, default: 1).
        samples_per_shot : int
            Samples per shot (default: 512).
        omega1, omega2 : float
            Petal frequencies (default: 5, 4) — ratio determines petal count.
        rotation_per_shot : float
            Angular increment (rad) between shots (default: golden angle).
        """
        nx, ny, _ = geom["matrix"]
        fov_x_m = geom["fov_mm"][0] / 1000.0
        fov_y_m = geom["fov_mm"][1] / 1000.0
        kmax_x = (nx / 2.0) / fov_x_m
        kmax_y = (ny / 2.0) / fov_y_m

        n_shots = int(params.get("n_shots", 1))
        omega1 = float(params.get("omega1", 5.0))
        omega2 = float(params.get("omega2", 4.0))
        # Nyquist along the readout. The fastest term oscillates at (w1 + w2)
        # over t in [0, 2*pi], so the sample spacing kmax * (w1+w2) * 2*pi/sps
        # must stay under one bin, kmax / (N/2). A fixed default cannot satisfy
        # that — at 512 the petals alias into a mask that looks like a different
        # trajectory entirely.
        sps_nyquist = int(math.ceil(math.pi * (omega1 + omega2) * max(nx, ny)))
        sps = int(params.get("samples_per_shot", max(512, sps_nyquist)))
        rot_step = float(params.get("rotation_per_shot", GOLDEN_ANGLE_RAD))

        t = ops.linspace_like(like, 0.0, 2.0 * math.pi, sps)
        shots = []
        for i in range(n_shots):
            rot = i * rot_step
            kx = ops.sin(t * omega1) * ops.cos(t * omega2 + rot) * kmax_x
            ky = ops.sin(t * omega1) * ops.sin(t * omega2 + rot) * kmax_y
            shot = ops.stack([kx, ky], axis=1)
            shots.append(shot)

        meta = {
            "trajectory_type":    "rosette_2d",
            "omega1":             omega1,
            "omega2":             omega2,
            "kmax":               (kmax_x, kmax_y),
            "fov_mm":             geom["fov_mm"][:2],
            "matrix":             (nx, ny),
            "n_shots":            n_shots,
            "samples_per_shot":   sps,
            "density_estimate":   Trajectory._density_estimate("rosette_2d", n_shots, sps, geom),
            "params_used":        params,
        }
        return shots, meta


#**************************************************************************************************#
#                                      Class RosettePetals2D                                       #
#**************************************************************************************************#
#                                                                                                  #
# See "_build" for the geometry and parameters.                                              #
#                                                                                                  #
#**************************************************************************************************#
@TrajectoryRegistry.register
class RosettePetals2D(Trajectory):
    """See "_build" for the geometry and parameters."""

    NAME = "rosette_2d_petals"
    NDIM = 2

    def generate(self, geometry, like=None):
        return self._build(geometry, self.params, like)

    @staticmethod
    def _build(geom: dict, params: dict, like=None) -> Tuple[List[Any], Dict[str, Any]]:
        """
        2D Rosette where each individual petal is returned as a separate shot.

        Each petal traces a closed lemniscate-like path from the k-space origin
        to the tip and back, giving a center-out-center readout per petal:

            kx(t) = kmax_x · sin(t) · cos(t + φₖ)
            ky(t) = kmax_y · sin(t) · sin(t + φₖ)
            t ∈ [0, π],   φₖ = -π/2 + 2π·k/n_petals + start_angle_rad

        At t=0 and t=π: sin(t)=0, so both endpoints are at the origin.
        The petal tip lies at t=π/2, pointing in direction φₖ + π/2.
        With the default start_angle = -π/2, petal 0 points along +kx.

        Parameters (params)
        -------------------
        n_petals : int
            Number of petals / shots (default: 7).
        samples_per_shot : int
            Samples per petal (default: 512). Includes origin at both ends.
        start_angle_rad : float
            Extra azimuthal offset applied to all petals (default: 0).
            The built-in -π/2 offset already aligns petal 0 with +kx.

        Output shot count: n_petals.
        Sample count per shot: samples_per_shot (constant).
        """
        nx, ny = geom["matrix"][0], geom["matrix"][1]
        fov_x_m = geom["fov_mm"][0] / 1000.0
        fov_y_m = geom["fov_mm"][1] / 1000.0
        kmax_x = (nx / 2.0) / fov_x_m
        kmax_y = (ny / 2.0) / fov_y_m

        # 'n_shots' is accepted as an alias. Every other generator takes its
        # count as n_shots, and the undersampling/coverage helpers speak that
        # vocabulary, so reading only n_petals here made the shot count silently
        # unchangeable from generic code — the trajectory always returned 7.
        n_petals = int(params.get("n_petals", params.get("n_shots", 7)))
        sps = int(params.get("samples_per_shot", 512))
        start_angle = float(params.get("start_angle_rad", 0.0))

        t = ops.linspace_like(like, 0.0, math.pi, sps)   # one petal: origin → tip → origin
        shots = []
        for k in range(n_petals):
            # -π/2 aligns petal 0 with +kx; 2π*k/n_petals rotates CCW
            phi = -math.pi / 2.0 + 2.0 * math.pi * k / n_petals + start_angle
            kx = ops.sin(t) * ops.cos(t + phi) * kmax_x
            ky = ops.sin(t) * ops.sin(t + phi) * kmax_y
            shot = ops.stack([kx, ky], axis=1)
            shots.append(shot)

        kmax_xv, kmax_yv, _ = KspaceGeometry._kmax_from_geometry(geom)
        meta = {
            "trajectory_type":    "rosette_2d_petals",
            "n_petals":           n_petals,
            "kmax":               (kmax_xv, kmax_yv),
            "fov_mm":             geom["fov_mm"][:2],
            "matrix":             (nx, ny),
            "n_shots":            n_petals,
            "samples_per_shot":   sps,
            "center_out_center":  True,
            "density_estimate":   Trajectory._density_estimate(
                                      "rosette_2d_petals", n_petals, sps, geom),
            "params_used":        params,
        }
        return shots, meta


#**************************************************************************************************#
#                                     Class ConcentricRings2D                                      #
#**************************************************************************************************#
#                                                                                                  #
# See "_build" for the geometry and parameters.                                              #
#                                                                                                  #
#**************************************************************************************************#
@TrajectoryRegistry.register
class ConcentricRings2D(Trajectory):
    """See "_build" for the geometry and parameters."""

    NAME = "concentric_rings_2d"
    NDIM = 2

    def generate(self, geometry, like=None):
        return self._build(geometry, self.params, like)

    @staticmethod
    def _build(geom: dict, params: dict, like=None) -> Tuple[List[Any], Dict[str, Any]]:
        """
        2D Concentric Rings trajectory.

        A set of concentric circles in the kx–ky plane. Each ring is one shot.
        Rings are uniformly spaced from a minimum radius to kmax. Alternating
        rings are traversed in opposite directions (CW / CCW) to mimic an
        echo-planar acquisition pattern.

        Equivalent to the ECCENTRIC / concentric-ring MRSI acquisition but
        exposed directly for 2D use without a kz stack.

        Parameters (params)
        -------------------
        n_rings : int
            Number of concentric rings (default: ny // 2).
        samples_per_ring : int
            Samples per ring (default: proportional to circumference).
            Overrides the auto-computed circumference-proportional value.
        min_ring_radius : float
            Minimum ring radius as fraction of kmax (default: 1 / n_rings).

        Output shot count: n_rings.
        Sample count per shot: variable unless samples_per_ring is set.
        """
        shots, meta_eccentric = _Eccentric2D._build(geom, params, like)
        # Re-label trajectory type for clarity
        meta_eccentric["trajectory_type"] = "concentric_rings_2d"
        meta_eccentric["n_shots"] = len(shots)
        meta_eccentric["density_estimate"]["trajectory"] = "concentric_rings_2d"
        return shots, meta_eccentric


#**************************************************************************************************#
#                                     Class ConcentricShells3D                                     #
#**************************************************************************************************#
#                                                                                                  #
# See "_build" for the geometry and parameters.                                              #
#                                                                                                  #
#**************************************************************************************************#
@TrajectoryRegistry.register
class ConcentricShells3D(Trajectory):
    """See "_build" for the geometry and parameters."""

    NAME = "concentric_shells_3d"
    NDIM = 3

    def generate(self, geometry, like=None):
        return self._build(geometry, self.params, like)

    @staticmethod
    def _build(geom: dict, params: dict, like=None) -> Tuple[List[Any], Dict[str, Any]]:
        """
        3D Concentric Shells trajectory — the volumetric analogue of
        concentric_rings_2d.

        Nested ellipsoidal shells from a minimum radius out to kmax. Each shell
        is covered by circles of constant latitude, and each circle is one
        shot, exactly as each ring is one shot in 2D. Latitude count per shell
        and samples per circle both follow the grid Nyquist rate (spacing of
        one bin along polar arc and circumference respectively), and adjacent
        circles alternate traversal direction like the 2D rings.

        Parameters (params)
        -------------------
        n_shells : int
            Number of shells (default: max(nx, ny, nz) // 2, one per bin of
            radius). 'n_shots' is accepted as an alias; the per-shell latitude
            count is derived from geometry, so the resulting shot count is
            larger than n_shells.
        min_shell_radius : float
            Minimum shell radius as a fraction of kmax (default: 1 / n_shells).
        samples_per_ring : int
            Override for samples per circle (default: its circumference in
            bins).

        Output shot count: sum over shells of ceil(pi * r_shell_bins).
        Sample count per shot: variable (proportional to circumference).
        """
        if geom["ndim"] < 3:
            raise ValueError(
                "concentric_shells_3d requires a 3D geometry (nz > 1). "
                "Header has nz=1. Use concentric_rings_2d for 2D."
            )
        nx, ny, nz = geom["matrix"]
        fov_x_m, fov_y_m, fov_z_m = (v / 1000.0 for v in geom["fov_mm"])
        kmax_x = (nx / 2.0) / fov_x_m
        kmax_y = (ny / 2.0) / fov_y_m
        kmax_z = (nz / 2.0) / fov_z_m
        n_max = max(nx, ny, nz)

        n_shells = int(params.get("n_shells", params.get("n_shots", n_max // 2)))
        n_shells = max(1, n_shells)
        min_frac = float(params.get("min_shell_radius", 1.0 / n_shells))
        sps_override = params.get("samples_per_ring", None)

        shots = []
        total_samples = 0
        shots_per_shell = []
        for si in range(n_shells):
            frac = (min_frac + (1.0 - min_frac) * si / (n_shells - 1)
                    if n_shells > 1 else 1.0)
            r_bins = frac * (n_max / 2.0)
            # Polar Nyquist: one circle per bin of arc along the meridian.
            n_lat = max(1, int(round(math.pi * r_bins)))
            shots_per_shell.append(n_lat)
            for li in range(n_lat):
                theta = math.pi * (li + 0.5) / n_lat
                circ_bins = 2.0 * math.pi * r_bins * math.sin(theta)
                sps = int(sps_override) if sps_override is not None else \
                    max(6, int(round(circ_bins)))
                # Alternate direction shell-locally, as the 2D rings do.
                sign = 1.0 if li % 2 == 0 else -1.0
                phi = ops.linspace_like(like, 0.0, sign * 2.0 * math.pi, sps)
                sin_t = math.sin(theta)
                kx = ops.cos(phi) * (frac * sin_t * kmax_x)
                ky = ops.sin(phi) * (frac * sin_t * kmax_y)
                kz = ops.full_like_shape(like, (sps,), frac * math.cos(theta) * kmax_z)
                shots.append(ops.stack([kx, ky, kz], axis=1))
                total_samples += sps

        n_shots_total = len(shots)
        mean_sps = int(total_samples / max(n_shots_total, 1))
        meta = {
            "trajectory_type":    "concentric_shells_3d",
            "n_shells":           n_shells,
            "shots_per_shell":    shots_per_shell,
            "kmax":               (kmax_x, kmax_y, kmax_z),
            "fov_mm":             geom["fov_mm"],
            "matrix":             geom["matrix"],
            "n_shots":            n_shots_total,
            "samples_per_shot":   mean_sps,
            "density_estimate":   Trajectory._density_estimate(
                                      "concentric_shells_3d", n_shots_total, mean_sps, geom),
            "params_used":        params,
        }
        return shots, meta


#**************************************************************************************************#
#                                       Class Phyllotaxis3D                                        #
#**************************************************************************************************#
#                                                                                                  #
# See "_build" for the geometry and parameters.                                              #
#                                                                                                  #
#**************************************************************************************************#
@TrajectoryRegistry.register
class Phyllotaxis3D(Trajectory):
    """See "_build" for the geometry and parameters."""

    NAME = "3d_phyllotaxis"
    NDIM = 3

    def generate(self, geometry, like=None):
        return self._build(geometry, self.params, like)

    @staticmethod
    def _build(geom: dict, params: dict, like=None) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Fermat phyllotaxis 3D radial trajectory.

        Angular positions on unit sphere use sunflower / Fibonacci lattice,
        so any prefix of N shots provides near-uniform angular coverage.
        Center-out readout: each spoke starts at k=0 and extends to kmax.

        Parameters (params)
        -------------------
        n_shots : int
            Number of spokes (default: π/4 * max(nx,ny,nz)^2).
        samples_per_shot : int
            Samples per spoke (default: max(nx,ny,nz)).
        golden_angle_2d : bool
            Use 2D golden angle for azimuth increment (default: True).

        Output shot count: n_shots.
        Sample count per shot: samples_per_shot (constant).
        """
        if geom["ndim"] < 3:
            raise ValueError(
                "3d_phyllotaxis requires a 3D geometry (nz > 1). "
                "Header has nz=1. Use radial_2d or golden_radial_2d for 2D."
            )
        nx, ny, nz = geom["matrix"]
        fov_x_m, fov_y_m, fov_z_m = (v / 1000.0 for v in geom["fov_mm"])
        kmax_x = (nx / 2.0) / fov_x_m
        kmax_y = (ny / 2.0) / fov_y_m
        kmax_z = (nz / 2.0) / fov_z_m

        max_n = max(nx, ny, nz)
        n_shots = int(params.get("n_shots", max(1, int(math.pi / 4 * max_n ** 2))))
        sps = int(params.get("samples_per_shot", max_n))

        # Fermat phyllotaxis: spoke i at
        #   cos(elevation) = 1 - (2*i+1)/n_shots
        #   azimuth = i * golden_angle (3D variant: 2.399963...)
        shots = []
        r = ops.linspace_like(like, 0.0, 1.0, sps)   # center-out
        for i in range(n_shots):
            # Low-discrepancy elevation (golden-fraction), NOT the monotonic
            # lattice 1 - (2i+1)/n. The monotonic form walks from the north
            # pole to the south in shot order, so "any prefix is near-uniform"
            # — the property this trajectory exists for, and the one prefix-
            # style undersampling relies on — was false: a prefix kept a polar
            # cap, and the decimated trajectory rendered as a dome missing its
            # entire southern half. The golden-fraction sequence fills the
            # sphere at every prefix length.
            #
            # The elevation fraction must NOT reuse the golden ratio: the
            # azimuth i * golden-angle is itself a golden fraction of the
            # circle, so a phi-based elevation is a deterministic function of
            # the azimuth and every spoke direction falls on ONE 1-D spiral
            # curve — grid coverage collapsed to 3% when tried. The silver
            # ratio (sqrt(2) - 1) is rationally independent of phi, giving a
            # genuine 2-D low-discrepancy sequence on the sphere.
            cos_el = 1.0 - 2.0 * (((i + 0.5) * (math.sqrt(2.0) - 1.0)) % 1.0)
            cos_el = float(np.clip(cos_el, -1.0, 1.0))
            sin_el = math.sqrt(max(0.0, 1.0 - cos_el ** 2))
            azimuth = i * GOLDEN_ANGLE_RAD    # 2D golden angle for azimuth

            dx = sin_el * math.cos(azimuth)
            dy = sin_el * math.sin(azimuth)
            dz = cos_el

            kx = r * dx * kmax_x
            ky = r * dy * kmax_y
            kz = r * dz * kmax_z
            shot = ops.stack([kx, ky, kz], axis=1)
            shots.append(shot)

        meta = {
            "trajectory_type":    "3d_phyllotaxis",
            "golden_angle_rad":   GOLDEN_ANGLE_RAD,
            "kmax":               (kmax_x, kmax_y, kmax_z),
            "fov_mm":             geom["fov_mm"],
            "matrix":             (nx, ny, nz),
            "n_shots":            n_shots,
            "samples_per_shot":   sps,
            "center_out":         True,
            "density_estimate":   Trajectory._density_estimate("3d_phyllotaxis", n_shots, sps, geom),
            "params_used":        params,
        }
        return shots, meta


#**************************************************************************************************#
#                                          Class Cones3D                                           #
#**************************************************************************************************#
#                                                                                                  #
# See "_build" for the geometry and parameters.                                              #
#                                                                                                  #
#**************************************************************************************************#
@TrajectoryRegistry.register
class Cones3D(Trajectory):
    """See "_build" for the geometry and parameters."""

    NAME = "cones_3d"
    NDIM = 3

    def generate(self, geometry, like=None):
        return self._build(geometry, self.params, like)

    @staticmethod
    def _build(geom: dict, params: dict, like=None) -> Tuple[List[Any], Dict[str, Any]]:
        """
        3D cones trajectory: spiral arms arranged on cones at discrete polar angles.

        Each shot is a spiral arm on a cone surface. Cones are indexed by their
        polar angle θ measured from the +kz axis.

        By default the polar range is [0, π], covering both the upper (+kz) and
        lower (−kz) hemispheres symmetrically.  Set "polar_max=π/2" to restrict
        to the upper hemisphere only (legacy behavior).

        Parameters (params)
        -------------------
        n_cones : int
            Number of cone polar angles (default: 7).
        arms_per_cone : int
            Azimuthal arms per cone (default: 8).
        samples_per_shot : int
            Samples per arm (default: max(nx, ny, nz)).
        spiral_turns_per_arm : float
            Spiral turns per arm (default: 1.0).
        polar_min : float
            Minimum polar angle in radians (default: 0 = +kz axis).
        polar_max : float
            Maximum polar angle in radians (default: π = −kz axis).
            Set to π/2 to generate upper-hemisphere only.

        Output shot count: n_cones * arms_per_cone.

        Notes
        -----
        Polar angle θ is the standard physics convention (angle from +kz):
            θ = 0   → spoke along +kz
            θ = π/2 → spoke in the kxy plane  (kz = 0)
            θ = π   → spoke along −kz
        kz = r · cos(θ) · kmax_z, so negative-kz cones arise naturally for θ > π/2.
        """
        if geom["ndim"] < 3:
            raise ValueError(
                "cones_3d requires a 3D geometry (nz > 1). Header has nz=1."
            )
        nx, ny, nz = geom["matrix"]
        fov_x_m, fov_y_m, fov_z_m = (v / 1000.0 for v in geom["fov_mm"])
        kmax_x = (nx / 2.0) / fov_x_m
        kmax_y = (ny / 2.0) / fov_y_m
        kmax_z = (nz / 2.0) / fov_z_m

        arms_per_cone = int(params.get("arms_per_cone", 8))
        # 'n_shots' alias: shot count is n_cones * arms_per_cone, so a generic
        # caller asking for n_shots gets ceil(n_shots / arms) cones.
        if "n_cones" not in params and "n_shots" in params:
            params = {**params, "n_cones": -(-int(params["n_shots"]) // arms_per_cone)}
        n_cones       = int(params.get("n_cones", 14))
        sps           = int(params.get("samples_per_shot", max(nx, ny, nz)))
        turns         = float(params.get("spiral_turns_per_arm", 1.0))
        polar_min     = float(params.get("polar_min", 0.15*math.pi))#0.0))
        polar_max     = float(params.get("polar_max", 0.85*math.pi))

        cone_angles = np.linspace(polar_min, polar_max, n_cones, True)
        t = ops.linspace_like(like, 0.0, 1.0, sps)

        shots = []
        for ci, cone_ang in enumerate(cone_angles):
            sin_ca = math.sin(float(cone_ang))
            cos_ca = math.cos(float(cone_ang))   # negative for θ > π/2 → −kz cones
            for ai in range(arms_per_cone):
                az0 = ai * 2.0 * math.pi / arms_per_cone + ci * GOLDEN_ANGLE_RAD
                theta_spiral = t * (2.0 * math.pi * turns) + az0
                r = t  # center-out

                kx = r * ops.cos(theta_spiral) * sin_ca * kmax_x
                ky = r * ops.sin(theta_spiral) * sin_ca * kmax_y
                kz = r * cos_ca * kmax_z          # naturally negative when θ > π/2
                shot = ops.stack([kx, ky, kz], axis=1)
                shots.append(shot)

        n_shots = len(shots)
        meta = {
            "trajectory_type":    "cones_3d",
            "n_cones":            n_cones,
            "arms_per_cone":      arms_per_cone,
            "spiral_turns":       turns,
            "polar_min_deg":      math.degrees(polar_min),
            "polar_max_deg":      math.degrees(polar_max),
            "kmax":               (kmax_x, kmax_y, kmax_z),
            "fov_mm":             geom["fov_mm"],
            "matrix":             (nx, ny, nz),
            "n_shots":            n_shots,
            "samples_per_shot":   sps,
            "density_estimate":   Trajectory._density_estimate("cones_3d", n_shots, sps, geom),
            "params_used":        params,
        }
        return shots, meta


#**************************************************************************************************#
#                                       Class ConesRosette3D                                       #
#**************************************************************************************************#
#                                                                                                  #
# See "_build" for the geometry and parameters.                                              #
#                                                                                                  #
#**************************************************************************************************#
@TrajectoryRegistry.register
class ConesRosette3D(Trajectory):
    """See "_build" for the geometry and parameters."""

    NAME = "cones_3d_rosette"
    NDIM = 3

    def generate(self, geometry, like=None):
        return self._build(geometry, self.params, like)

    @staticmethod
    def _build(geom: dict, params: dict, like=None) -> Tuple[List[Any], Dict[str, Any]]:
        """
        3D Cones Rosette trajectory.

        Replaces the spiral readout arms of "cones_3d" with rosette petals.
        Like a 2D petal, each 3D petal is a *closed* trajectory that starts and
        ends at the k-space origin (center-out-center), tracing a lemniscate-like
        loop on the surface of the cone:

            t ∈ [0, π]  (both endpoints at k = 0)
            φⱼ = j · 2π / petals_per_cone  +  cone_idx · golden_angle_rad

            kx(t) = sin(t) · sin(θ_cone) · cos(t + φⱼ) · kmax_x
            ky(t) = sin(t) · sin(θ_cone) · sin(t + φⱼ) · kmax_y
            kz(t) = sin(t) · cos(θ_cone) · kmax_z

        Because kx, ky, kz all share the sin(t) envelope, the trajectory lies
        on the cone surface for all t and returns to the origin at t=0 and t=π.
        The azimuthal sweep (t + φⱼ) creates the petal lobe in the transverse
        plane. The golden-angle inter-cone offset spreads petals uniformly in
        azimuth across cone levels.

        By default the polar range is [0, π], covering both the upper (+kz) and
        lower (−kz) hemispheres. cos(θ) is negative for θ > π/2, which makes kz
        naturally negative — no special mirroring step is required.

        Parameters (params)
        -------------------
        n_cones : int
            Number of cone polar angles, spaced across [polar_min, polar_max]
            (default: 7). Includes cones in both hemispheres when polar_max > π/2.
        petals_per_cone : int
            Number of rosette petals per cone (default: 6).
        samples_per_shot : int
            Samples per petal (default: 512). Includes origin at both ends.
        polar_min : float
            Minimum polar angle in radians (default: 0 = +kz axis).
        polar_max : float
            Maximum polar angle in radians (default: π = −kz axis).
            Set to π/2 to generate upper-hemisphere only.

        Output shot count: n_cones * petals_per_cone.
        Sample count per shot: samples_per_shot (constant).

        Notes
        -----
        Same polar-angle convention as cones_3d:
            θ = 0   → cone along +kz  (kz > 0)
            θ = π/2 → equatorial cone (kz = 0)
            θ = π   → cone along −kz  (kz < 0)
        """
        if geom["ndim"] < 3:
            raise ValueError(
                "cones_3d_rosette requires a 3D geometry (nz > 1). "
                "Header has nz=1."
            )
        nx, ny, nz = geom["matrix"]
        fov_x_m, fov_y_m, fov_z_m = (v / 1000.0 for v in geom["fov_mm"])
        kmax_x = (nx / 2.0) / fov_x_m
        kmax_y = (ny / 2.0) / fov_y_m
        kmax_z = (nz / 2.0) / fov_z_m

        ppc       = int(params.get("petals_per_cone", 9))
        # 'n_shots' alias — see _traj_cones_3d.
        if "n_cones" not in params and "n_shots" in params:
            params = {**params, "n_cones": -(-int(params["n_shots"]) // ppc)}
        n_cones   = int(params.get("n_cones", 14))
        sps       = int(params.get("samples_per_shot", 512))
        polar_min = float(params.get("polar_min", 0.15*math.pi))#0.0))
        polar_max = float(params.get("polar_max", 0.85*math.pi))

        cone_angles = np.linspace(polar_min, polar_max, n_cones, True)
        t     = ops.linspace_like(like, 0.0, math.pi, sps)   # petal: 0 → π
        sin_t = ops.sin(t)                         # envelope — 0 at both ends

        shots = []
        for ci, theta_cone in enumerate(cone_angles):
            sin_ca = math.sin(float(theta_cone))
            cos_ca = math.cos(float(theta_cone))   # negative for θ > π/2 → −kz petals
            for pi_ in range(ppc):
                phi = pi_ * 2.0 * math.pi / ppc + ci * GOLDEN_ANGLE_RAD
                kx = sin_t * sin_ca * ops.cos(t + phi) * kmax_x
                ky = sin_t * sin_ca * ops.sin(t + phi) * kmax_y
                kz = sin_t * cos_ca * kmax_z       # sign follows cos(θ) naturally
                shot = ops.stack([kx, ky, kz], axis=1)
                shots.append(shot)

        n_shots = len(shots)
        kmax_xv, kmax_yv, kmax_zv = KspaceGeometry._kmax_from_geometry(geom)
        meta = {
            "trajectory_type":    "cones_3d_rosette",
            "n_cones":            n_cones,
            "petals_per_cone":    ppc,
            "polar_min_deg":      math.degrees(polar_min),
            "polar_max_deg":      math.degrees(polar_max),
            "kmax":               (kmax_xv, kmax_yv, kmax_zv),
            "fov_mm":             geom["fov_mm"],
            "matrix":             (nx, ny, nz),
            "n_shots":            n_shots,
            "samples_per_shot":   sps,
            "center_out_center":  True,
            "density_estimate":   Trajectory._density_estimate(
                                      "cones_3d_rosette", n_shots, sps, geom),
            "params_used":        params,
        }
        return shots, meta


#**************************************************************************************************#
#                                          Class Floret3D                                          #
#**************************************************************************************************#
#                                                                                                  #
# See "_build" for the geometry and parameters.                                              #
#                                                                                                  #
#**************************************************************************************************#
@TrajectoryRegistry.register
class Floret3D(Trajectory):
    """See "_build" for the geometry and parameters."""

    NAME = "floret_3d"
    NDIM = 3

    def generate(self, geometry, like=None):
        return self._build(geometry, self.params, like)

    @staticmethod
    def _build(geom: dict, params: dict, like=None) -> Tuple[List[Any], Dict[str, Any]]:
        """
        FLORET (Fermat-Loop-Over-Radial Echo Trajectory) 3D trajectory.

        Distributed spiral shells (hubs) arranged uniformly on the sphere via
        phyllotaxis ordering. Each hub is a set of interleaved spiral arms.

        Parameters (params)
        -------------------
        n_hubs : int
            Number of shell hubs (default: 5).
        arms_per_hub : int
            Spiral arms per hub (default: 15).
        samples_per_shot : int
            Samples per arm (default: max(nx, ny, nz)).
        spiral_turns : float
            Turns per arm (default: arms_per_hub / 2).
        hub_cone_angle_deg : float
            Half-angle of each hub cone in degrees (default: 35).

        Output shot count: n_hubs * arms_per_hub.
        """
        if geom["ndim"] < 3:
            raise ValueError("floret_3d requires a 3D geometry (nz > 1).")
        nx, ny, nz = geom["matrix"]
        fov_x_m, fov_y_m, fov_z_m = (v / 1000.0 for v in geom["fov_mm"])
        kmax_x = (nx / 2.0) / fov_x_m
        kmax_y = (ny / 2.0) / fov_y_m
        kmax_z = (nz / 2.0) / fov_z_m

        n_hubs = int(params.get("n_hubs", 3))
        # 'n_shots' alias: total shots = n_hubs * arms_per_hub, so a generic
        # caller asking for n_shots gets ceil(n_shots / n_hubs) arms per hub.
        if "arms_per_hub" not in params and "n_shots" in params:
            params = {**params, "arms_per_hub": -(-int(params["n_shots"]) // n_hubs)}
        aph = int(params.get("arms_per_hub", 15))
        # One loop per arm — the "Fermat loop" of the FLORET name. The old
        # default tied turns to the arm count (aph/2), so asking for more arms
        # wound each one faster: at 800 arms every arm made 400 turns while
        # samples_per_shot stayed at max(matrix)=96, i.e. ~4 samples per turn.
        # The arms aliased into scattered points and grid coverage saturated
        # near 5% however many were added.
        turns = float(params.get("spiral_turns", 1.0))
        # Along-arm Nyquist at the peak path speed, as in _traj_spiral_2d.
        w_arm = 2.0 * math.pi * turns
        peak_bins = 0.5 * max(nx, ny, nz) * math.sqrt(1.0 + w_arm * w_arm)
        sps = int(params.get("samples_per_shot",
                             max(max(nx, ny, nz), int(math.ceil(peak_bins)))))
        # Magic angle acos(1/sqrt(3)). With three orthogonal hub axes this is
        # the exact covering radius: the largest component of any unit vector
        # is at least 1/sqrt(3), so every direction lies within this angle of
        # some axis and the three double-cones tile the sphere with minimal
        # overlap. The previous 35 deg left most of the sphere unvisited.
        hub_ang_deg = float(params.get("hub_cone_angle_deg",
                                       math.degrees(math.acos(1.0 / math.sqrt(3.0)))))
        hub_ang_rad = math.radians(hub_ang_deg)

        t = ops.linspace_like(like, 0.0, 1.0, sps)

        # Hub axes. Three hubs get the published FLORET arrangement — the
        # cardinal axes — because that is what tiles the sphere at the magic
        # angle; other counts fall back to phyllotaxis placement.
        orthogonal_hubs = bool(params.get("orthogonal_hubs", n_hubs == 3))
        shots = []
        for h in range(n_hubs):
            if orthogonal_hubs:
                hub_axis = np.zeros(3); hub_axis[h % 3] = 1.0
            else:
                cos_el_hub = float(np.clip(1.0 - (2 * h + 1.0) / n_hubs, -1.0, 1.0))
                sin_el_hub = math.sqrt(max(0.0, 1.0 - cos_el_hub ** 2))
                az_hub = h * GOLDEN_ANGLE_RAD
                hub_axis = np.array([sin_el_hub * math.cos(az_hub),
                                     sin_el_hub * math.sin(az_hub),
                                     cos_el_hub])
            # local x = cross(z_global, hub_axis) or fallback
            z_global = np.array([0.0, 0.0, 1.0])
            local_x = np.cross(z_global, hub_axis)
            norm = np.linalg.norm(local_x)
            if norm < 1e-8:
                local_x = np.array([1.0, 0.0, 0.0])
            else:
                local_x = local_x / norm
            local_y = np.cross(hub_axis, local_x)

            for a in range(aph):
                az0 = a * GOLDEN_ANGLE_RAD
                theta = t * (2.0 * math.pi * turns) + az0
                r = t
                # Each arm gets its OWN polar angle, equal-area distributed
                # within the hub cone, and arms alternate between the +axis and
                # -axis lobes so a hub is a double cone.
                #
                # Both properties were missing and each capped coverage on its
                # own. A single fixed polar angle for every arm pinned the whole
                # hub to one cone SURFACE, so adding arms only crowded that
                # surface; and a positive-only polar angle made each hub a
                # single cone, leaving three hubs covering just the directions
                # near +x/+y/+z. Together they held grid coverage at ~6%
                # however many arms were requested.
                n_half = max(1, aph // 2)
                u_arm = ((a // 2) + 0.5) / n_half
                sgn = 1.0 if (a % 2 == 0) else -1.0
                cos_pol_a = 1.0 - u_arm * (1.0 - math.cos(hub_ang_rad))
                sin_pol_a = math.sqrt(max(0.0, 1.0 - cos_pol_a * cos_pol_a))
                arm_vec_x = r * ops.cos(theta) * sin_pol_a
                arm_vec_y = r * ops.sin(theta) * sin_pol_a
                arm_vec_z = r * (sgn * cos_pol_a)

                # Rotate from hub local frame to global
                kx = (arm_vec_x * local_x[0] + arm_vec_y * local_y[0]
                      + arm_vec_z * hub_axis[0]) * kmax_x
                ky = (arm_vec_x * local_x[1] + arm_vec_y * local_y[1]
                      + arm_vec_z * hub_axis[1]) * kmax_y
                kz = (arm_vec_x * local_x[2] + arm_vec_y * local_y[2]
                      + arm_vec_z * hub_axis[2]) * kmax_z
                shot = ops.stack([kx, ky, kz], axis=1)
                shots.append(shot)

        n_shots = len(shots)
        meta = {
            "trajectory_type":    "floret_3d",
            "n_hubs":             n_hubs,
            "orthogonal_hubs":    orthogonal_hubs,
            "arms_per_hub":       aph,
            "hub_cone_angle_deg": hub_ang_deg,
            "spiral_turns":       turns,
            "kmax":               (kmax_x, kmax_y, kmax_z),
            "fov_mm":             geom["fov_mm"],
            "matrix":             (nx, ny, nz),
            "n_shots":            n_shots,
            "samples_per_shot":   sps,
            "density_estimate":   Trajectory._density_estimate("floret_3d", n_shots, sps, geom),
            "params_used":        params,
        }
        return shots, meta


#**************************************************************************************************#
#                                        Class EggRosette3D                                        #
#**************************************************************************************************#
#                                                                                                  #
# See "_build" for the geometry and parameters.                                              #
#                                                                                                  #
#**************************************************************************************************#
@TrajectoryRegistry.register
class EggRosette3D(Trajectory):
    """See "_build" for the geometry and parameters."""

    NAME = "3d_egg_rosette"
    NDIM = 3

    def generate(self, geometry, like=None):
        return self._build(geometry, self.params, like)

    @staticmethod
    def _build(geom: dict, params: dict, like=None) -> Tuple[List[Any], Dict[str, Any]]:
        """
        3D Egg-Shaped Rosette trajectory.

        The 3D analogue of rosette_2d_petals, and the geometry of the
        egg-shaped modified rosette / PETALUTE family: every shot is ONE
        petal — a circle of radius kmax/2 passing exactly through the k-space
        center — whose tip direction walks a Fibonacci lattice over the FULL
        sphere with golden-angle azimuth, so any prefix of shots covers the
        egg near-uniformly and the interior is sampled along every petal.

        Petal (t in [0, pi], tip direction d, in-plane perpendicular q):

            k(t) = kmax_vec * [ petal_width * sin(t)cos(t) * q + sin(t)^2 * d ]

        which is the rosette_2d_petals curve lifted into the plane (d, q) and
        narrowed across the tip axis: petal_width = 1 gives the circular petal
        of the 2D variant, petal_width < 1 an egg-shaped one (long axis kmax/2
        towards the tip, short axis petal_width * kmax/2 across it).
        Componentwise kmax_vec additionally turns the whole envelope into the
        egg.

        Two earlier versions of this generator were wrong in instructive ways:
        the original placed every sample ON the sphere surface (radius
        identically kmax — a hollow shell, no center crossings); the first fix
        used a sin(w1 t)[cos(w2 t)u + sin(w2 t)p] "figure-eight" whose w1 = w2
        default degenerates to a twice-traced circle tipped along p, so tip
        directions clustered in a band and left a void on the opposite side.

        Parameters (params)
        -------------------
        n_shots : int
            Number of petals (default: 64). Golden ordering: any prefix stays
            near-uniform over the sphere.
        samples_per_shot : int
            Samples per petal (default: along-petal Nyquist,
            max(512, ceil(pi * max(n)))).
        rotation_per_shot : float
            Azimuth increment of the tip lattice (default: golden angle).
        petal_width : float
            Across-tip petal width relative to a circular petal (default 0.6,
            an egg; 1.0 recovers circles).

        Output shot count: n_shots.
        Sample count per shot: samples_per_shot (constant).
        """
        if geom["ndim"] < 3:
            raise ValueError(
                "3d_egg_rosette requires a 3D geometry (nz > 1). "
                "Header has nz=1."
            )
        nx, ny, nz = geom["matrix"]
        fov_x_m, fov_y_m, fov_z_m = (v / 1000.0 for v in geom["fov_mm"])
        kmax_x = (nx / 2.0) / fov_x_m
        kmax_y = (ny / 2.0) / fov_y_m
        kmax_z = (nz / 2.0) / fov_z_m
        max_n = max(nx, ny, nz)

        n_shots = int(params.get("n_shots", 64))
        sps = int(params.get("samples_per_shot",
                             max(512, int(math.ceil(math.pi * max_n)))))
        rot_per_shot = float(params.get("rotation_per_shot", GOLDEN_ANGLE_RAD))
        petal_width = float(params.get("petal_width", 0.6))

        t = ops.linspace_like(like, 0.0, math.pi, sps)
        env_q = ops.sin(t) * ops.cos(t) * petal_width   # across tip
        env_d = ops.sin(t) * ops.sin(t)                 # towards tip

        shots = []
        for n in range(n_shots):
            # Tip direction: low-discrepancy (golden-fraction) polar spacing
            # over the FULL sphere — see _traj_3d_phyllotaxis for why the
            # monotonic lattice would break prefix undersampling, and for why
            # the fraction uses the silver ratio rather than phi (a phi-based
            # elevation is correlated with the golden-angle azimuth and
            # collapses the tips onto a 1-D curve).
            cos_pol = 1.0 - 2.0 * (((n + 0.5) * (math.sqrt(2.0) - 1.0)) % 1.0)
            sin_pol = math.sqrt(max(0.0, 1.0 - cos_pol * cos_pol))
            az = n * rot_per_shot
            d = (sin_pol * math.cos(az), sin_pol * math.sin(az), cos_pol)
            # In-plane perpendicular, spun by the golden angle per shot so
            # petal planes do not share a common axis.
            ref = (0.0, 0.0, 1.0) if abs(d[2]) < 0.9 else (1.0, 0.0, 0.0)
            v0 = (d[1] * ref[2] - d[2] * ref[1],
                  d[2] * ref[0] - d[0] * ref[2],
                  d[0] * ref[1] - d[1] * ref[0])
            norm = math.sqrt(sum(c * c for c in v0)) or 1.0
            v0 = tuple(c / norm for c in v0)
            w0 = (d[1] * v0[2] - d[2] * v0[1],
                  d[2] * v0[0] - d[0] * v0[2],
                  d[0] * v0[1] - d[1] * v0[0])
            spin = n * rot_per_shot
            q = tuple(math.cos(spin) * v0[i] + math.sin(spin) * w0[i]
                      for i in range(3))

            kx = (env_q * q[0] + env_d * d[0]) * kmax_x
            ky = (env_q * q[1] + env_d * d[1]) * kmax_y
            kz = (env_q * q[2] + env_d * d[2]) * kmax_z
            shots.append(ops.stack([kx, ky, kz], axis=1))

        meta = {
            "trajectory_type":    "3d_egg_rosette",
            "rotation_per_shot":  rot_per_shot,
            "petal_width":        petal_width,
            "kmax":               (kmax_x, kmax_y, kmax_z),
            "fov_mm":             geom["fov_mm"],
            "matrix":             (nx, ny, nz),
            "n_shots":            n_shots,
            "samples_per_shot":   sps,
            "reference":          "egg-shaped modified rosette / PETALUTE family",
            "density_estimate":   Trajectory._density_estimate(
                                      "3d_egg_rosette", n_shots, sps, geom),
            "params_used":        params,
        }
        return shots, meta


#**************************************************************************************************#
#                                        Class _Eccentric2D                                        #
#**************************************************************************************************#
#                                                                                                  #
# Concentric-ring implementation shared by ConcentricRings2D and StackOfRings.                     #
#                                                                                                  #
#**************************************************************************************************#
class _Eccentric2D:
    """Concentric-ring implementation shared by ConcentricRings2D and StackOfRings."""

    @staticmethod
    def _build(geom: dict, params: dict, like=None) -> Tuple[List[Any], Dict[str, Any]]:
        """
        ECCENTRIC MRSI trajectory (Echo-planar Concentric Rings Trajectory).

        Concentric rings in k-space sampled with alternating directions,
        commonly used in MRSI. Each shot is one ring.

        Parameters (params)
        -------------------
        n_rings : int
            Number of concentric rings (default: ny//2).
        samples_per_ring : int
            Samples per ring (default: 2*π*ring_radius → proportional to circumference).
        min_ring_radius : float
            Minimum ring radius in normalized units (default: 1e-3).
        """
        nx, ny, _ = geom["matrix"]
        fov_x_m = geom["fov_mm"][0] / 1000.0
        fov_y_m = geom["fov_mm"][1] / 1000.0
        kmax_x = (nx / 2.0) / fov_x_m
        kmax_y = (ny / 2.0) / fov_y_m

        # 'n_shots' accepted as an alias — see _traj_rosette_2d_petals. One
        # ring is one shot, so n_shots and n_rings are the same quantity.
        n_rings = int(params.get("n_rings", params.get("n_shots", ny // 2)))
        min_r = float(params.get("min_ring_radius", 1e-3))
        radii = np.linspace(min_r, 1.0, n_rings, True)

        shots = []
        for idx, r_norm in enumerate(radii):
            # Samples proportional to circumference
            n_samp = int(params.get("samples_per_ring",
                                    max(8, int(2 * math.pi * r_norm * max(nx, ny) / 2))))
            theta = np.linspace(0, 2 * math.pi, n_samp, endpoint=True)
            if idx % 2 == 1:        # alternating CW/CCW for echo-planar style
                theta = theta[::-1]
            theta_b = ops.asarray_like(like, theta)
            kx = ops.cos(theta_b) * r_norm * kmax_x
            ky = ops.sin(theta_b) * r_norm * kmax_y
            shot = ops.stack([kx, ky], axis=1)
            shots.append(shot)

        meta = {
            "trajectory_type":    "ECCENTRIC",
            "n_rings":            n_rings,
            "kmax":               (kmax_x, kmax_y),
            "fov_mm":             geom["fov_mm"][:2],
            "matrix":             (nx, ny),
            "n_shots":            n_rings,
            "samples_per_shot":   "variable (∝ ring circumference)",
            "density_estimate":   Trajectory._density_estimate("ECCENTRIC", n_rings,
                                                                 ny // 2, geom),
            "params_used":        params,
        }
        return shots, meta


#**************************************************************************************************#
#                                          Class StackOf                                           #
#**************************************************************************************************#
#                                                                                                  #
# A 2-D in-plane trajectory replicated along kz.                                                   #
#                                                                                                  #
#**************************************************************************************************#
class StackOf(Trajectory):
    """A 2-D in-plane trajectory replicated along kz. Subclasses pick the in-plane pattern."""

    NDIM = 3
    #: Key into the in-plane dispatch table inside "_build".
    INPLANE: str = ""

    def generate(self, geometry, like=None):
        return self._build(geometry, self.params, self.INPLANE, like=like)

    @staticmethod
    def _build(geom: dict, params: dict, inplane_name: str,
               like=None) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Generic stack-of-2D hybrid: generate in-plane trajectory per kz slice.

        In-plane options: 'radial', 'golden_radial', 'spiral', 'cones', 'ECCENTRIC'.
        Shots are grouped: [kz0_shot0, kz0_shot1, ..., kz1_shot0, ...].

        Parameters (params)
        -------------------
        kz_ordering : str
            'linear' or 'centric' (default: 'linear').
        kz_range : tuple (kz_min_idx, kz_max_idx) optional
            Slice index range to include (default: all nz slices).
        plus all params for the in-plane trajectory generator.

        Output shot count: n_kz_slices * n_inplane_shots.
        """
        if geom["ndim"] < 3:
            raise ValueError(
                f"stack_of_{inplane_name} requires a 3D geometry (nz > 1). "
                f"Header has nz=1."
            )
        _, _, nz = geom["matrix"]
        fov_z_m = geom["fov_mm"][2] / 1000.0
        kmax_z = (nz / 2.0) / fov_z_m

        kz_ordering = params.get("kz_ordering", "linear")
        kz_range = params.get("kz_range", (0, nz))
        kz_min, kz_max = int(kz_range[0]), int(kz_range[1])
        n_kz = kz_max - kz_min

        kz_indices = np.arange(kz_min, kz_max, dtype=np.float32)
        kz_positions = (kz_indices - nz / 2.0) / fov_z_m

        if kz_ordering == "centric":
            half = n_kz // 2
            order = []
            for i in range(max(half, n_kz - half)):
                if i < half:
                    order.extend([half - 1 - i, half + i] if half + i < n_kz else [half - 1 - i])
                elif half + i < n_kz:
                    order.append(half + i)
            kz_positions = kz_positions[order[:n_kz]]

        # Generate in-plane geometry (treat as 2D)
        geom2d = {**geom, "ndim": 2, "matrix": (geom["matrix"][0], geom["matrix"][1], 1),
                   "fov_mm": (geom["fov_mm"][0], geom["fov_mm"][1], 1.0)}

        ip_map = {
            "radial":        lambda: Radial2D._build(geom2d, params, like, use_golden=False),
            "golden_radial": lambda: Radial2D._build(geom2d, params, like, use_golden=True),
            "spiral":        lambda: Spiral2D._build(geom2d, params, like),
            "ECCENTRIC":     lambda: _Eccentric2D._build(geom2d, params, like),
            "rings":         lambda: ConcentricRings2D._build(geom2d, params, like),
            "rosette":       lambda: Rosette2D._build(geom2d, params, like),
            "rosette_petals": lambda: RosettePetals2D._build(geom2d, params, like),
        }
        if inplane_name not in ip_map:
            raise ValueError(f"Unknown in-plane trajectory '{inplane_name}'. "
                             f"Valid: {list(ip_map.keys())}")
        inplane_shots, inplane_meta = ip_map[inplane_name]()

        all_shots = []
        for kz_val in kz_positions:
            kz_col = float(kz_val)
            for shot2d in inplane_shots:
                nrows = ops.shape(shot2d)[0]
                kz_col_arr = ops.full_like_shape(like, (nrows,), kz_col)
                shot3d = ops.concatenate([shot2d, ops.stack([kz_col_arr], axis=1)], axis=1)
                all_shots.append(shot3d)

        n_shots_total = len(all_shots)
        sps = ops.shape(all_shots[0])[0] if all_shots else 0
        traj_name = f"stack_of_{inplane_name}"
        kmax_x, kmax_y, _ = KspaceGeometry._kmax_from_geometry(geom)
        meta = {
            "trajectory_type":    traj_name,
            "inplane_trajectory": inplane_name,
            "inplane_meta":       inplane_meta,
            "kz_ordering":        kz_ordering,
            "n_kz_slices":        n_kz,
            "n_inplane_shots":    len(inplane_shots),
            "kmax":               (kmax_x, kmax_y, kmax_z),
            "fov_mm":             geom["fov_mm"],
            "matrix":             geom["matrix"],
            "n_shots":            n_shots_total,
            "samples_per_shot":   sps,
            "density_estimate":   Trajectory._density_estimate(traj_name, n_shots_total, sps, geom),
            "params_used":        params,
        }
        return all_shots, meta


#**************************************************************************************************#
#                                        Class StackOfStars                                        #
#**************************************************************************************************#
#                                                                                                  #
# Stack of radial in-plane shots along kz.                                                         #
#                                                                                                  #
#**************************************************************************************************#
@TrajectoryRegistry.register
class StackOfStars(StackOf):
    """Stack of radial in-plane shots along kz."""

    NAME = "stack_of_stars"
    INPLANE = "radial"


#**************************************************************************************************#
#                                       Class StackOfSpirals                                       #
#**************************************************************************************************#
#                                                                                                  #
# Stack of spiral in-plane shots along kz.                                                         #
#                                                                                                  #
#**************************************************************************************************#
@TrajectoryRegistry.register
class StackOfSpirals(StackOf):
    """Stack of spiral in-plane shots along kz."""

    NAME = "stack_of_spirals"
    INPLANE = "spiral"


#**************************************************************************************************#
#                                      Class StackOfRosettes                                       #
#**************************************************************************************************#
#                                                                                                  #
# Stack of rosette_petals in-plane shots along kz.                                                 #
#                                                                                                  #
#**************************************************************************************************#
@TrajectoryRegistry.register
class StackOfRosettes(StackOf):
    """Stack of rosette_petals in-plane shots along kz."""

    NAME = "stack_of_rosettes"
    INPLANE = "rosette_petals"


#**************************************************************************************************#
#                                        Class StackOfRings                                        #
#**************************************************************************************************#
#                                                                                                  #
# Stack of rings in-plane shots along kz.                                                          #
#                                                                                                  #
#**************************************************************************************************#
@TrajectoryRegistry.register
class StackOfRings(StackOf):
    """Stack of rings in-plane shots along kz."""

    NAME = "stack_of_rings"
    INPLANE = "rings"


#**************************************************************************************************#
#                                      Class ShotUndersampler                                      #
#**************************************************************************************************#
#                                                                                                  #
# Decide which shots of a trajectory are actually acquired.                                        #
#                                                                                                  #
#**************************************************************************************************#
class ShotUndersampler:
    """
    Decide which shots of a trajectory are actually acquired.

    >>> mask, _ = ShotUndersampler.undersample_shots(shots, "random_vd", 4.0, {}, like)

    "COMPAT" records which undersampling methods make sense for which
    trajectory families; pass "trajectory_name=''" to bypass the check.
    """

    #: trajectory -> undersampling methods that make sense for it
    COMPAT: Dict[str, List[str]] = {
        "cartesian_2d":      ["keep_acs", "poisson_disc_cartesian", "random_vd",
                              "drop_every", "prefix", "combined"],
        "cartesian_3d":      ["keep_acs", "poisson_disc_cartesian", "random_vd",
                              "drop_every", "prefix", "combined"],
        "radial_2d":         ["prefix", "golden_prefix", "drop_every",
                              "random_vd", "keep_acs", "combined"],
        "golden_radial_2d":  ["prefix", "golden_prefix", "drop_every",
                              "random_vd", "keep_acs", "combined"],
        "spiral_2d":         ["prefix", "drop_every", "variable_density_spiral",
                              "combined"],
        "rosette_2d":           ["prefix", "drop_every", "combined"],
        "rosette_2d_petals":    ["prefix", "drop_every", "combined"],
        "concentric_rings_2d":  ["prefix", "drop_every", "combined"],
        "stack_of_stars":    ["prefix", "drop_every", "random_vd", "keep_acs",
                              "combined"],
        "stack_of_spirals":  ["prefix", "drop_every", "combined"],
        "stack_of_rosettes":  ["prefix", "drop_every", "combined"],
        "stack_of_rings":    ["prefix", "drop_every", "combined"],
        "concentric_shells_3d": ["prefix", "drop_every", "combined"],
        "3d_phyllotaxis":    ["prefix", "golden_prefix", "drop_every",
                              "random_vd", "shell_based", "combined"],
        "cones_3d":          ["prefix", "drop_every", "shell_based", "combined"],
        "cones_3d_rosette":  ["prefix", "drop_every", "shell_based", "combined"],
        "floret_3d":         ["prefix", "drop_every", "shell_based", "combined"],
        "3d_egg_rosette":    ["prefix", "golden_prefix", "drop_every",
                              "shell_based", "combined"],
    }

    @staticmethod
    def _us_prefix(n_shots: int, af: float, params: dict) -> np.ndarray:
        """Keep first ceil(n_shots / acceleration_factor) shots."""
        n_keep = math.ceil(n_shots / af)
        mask = np.zeros(n_shots, dtype=bool)
        mask[:n_keep] = True
        return mask

    @staticmethod
    def _us_golden_prefix(n_shots: int, af: float, params: dict) -> np.ndarray:
        """
        Equivalent to prefix for golden-angle sorted trajectories.
        Alias of prefix; golden-angle ordering guarantees spatial uniformity.
        """
        return ShotUndersampler._us_prefix(n_shots, af, params)

    @staticmethod
    def _us_stride(n_shots: int, af: float, params: dict) -> np.ndarray:
        """
        Keep every M-th shot, M = round(acceleration_factor).

        The uniform-decimation counterpart to 'drop_every', which removes one
        shot in M and therefore *keeps* (M-1)/M — nearly everything at low
        acceleration. This keeps 1/M, which is what an acceleration factor
        normally means, while spreading the retained shots evenly over the
        acquisition order. For a trajectory whose shots are ordered by angle,
        that preserves full angular coverage where 'prefix' would leave a wedge.
        """
        m = int(params.get("stride", max(1, round(af))))
        mask = np.zeros(n_shots, dtype=bool)
        mask[::m] = True
        return mask

    @staticmethod
    def _us_drop_every(n_shots: int, af: float, params: dict) -> np.ndarray:
        """Drop every M-th shot. M = round(acceleration_factor) by default or params['drop_every_m']."""
        m = int(params.get("drop_every_m", max(1, round(af))))
        mask = np.ones(n_shots, dtype=bool)
        mask[::m] = False
        return mask

    @staticmethod
    def _us_random_vd(n_shots: int, af: float, params: dict,
                      coords_per_shot: List[Any], like=None) -> np.ndarray:
        """
        Variable-density random undersampling.
        Center shots (lower norm) are kept with higher probability.

        params
        ------
        vd_beta : float
            Density decay exponent. 0 = uniform random, >0 = center-weighted (default 2).
        acs_fraction : float
            Fraction of shots nearest to k-center always kept (default 0.05).
        """
        n_keep = math.ceil(n_shots / af)
        vd_beta = float(params.get("vd_beta", 2.0))
        acs_frac = float(params.get("acs_fraction", 0.05))

        # Compute shot center norms (use numpy via to_cpu)
        norms = []
        for shot in coords_per_shot:
            cpu = ops.to_numpy(shot)
            norms.append(float(np.linalg.norm(cpu.mean(axis=0))))
        norms = np.array(norms, dtype=np.float32)

        # Normalize to [0,1]
        if norms.max() > 0:
            norms_n = norms / norms.max()
        else:
            norms_n = norms

        # Probability: center gets high prob
        prob = np.exp(-vd_beta * norms_n)
        prob /= prob.sum()

        acs_n = max(0, int(acs_frac * n_shots))
        sorted_idx = np.argsort(norms)
        acs_idx = sorted_idx[:acs_n]

        outer_idx = np.setdiff1d(np.arange(n_shots), acs_idx)
        n_outer_keep = max(0, n_keep - acs_n)
        outer_prob = prob[outer_idx]
        outer_prob /= outer_prob.sum()
        chosen = np.random.choice(outer_idx, size=min(n_outer_keep, len(outer_idx)),
                                  replace=False, p=outer_prob)

        mask = np.zeros(n_shots, dtype=bool)
        mask[acs_idx] = True
        mask[chosen] = True
        return mask

    @staticmethod
    def _us_keep_acs(n_shots: int, af: float, params: dict,
                     coords_per_shot: List[Any], traj_name: str,
                     like=None) -> np.ndarray:
        """
        Keep shots whose center lies within the ACS region (innermost fraction of k-space).

        params
        ------
        acs_radius_fraction : float
            Fraction of kmax within which shots are kept (default 0.25).
        acs_n_lines : int
            For Cartesian: number of central lines to keep (overrides acs_radius_fraction).
        """
        acs_frac = float(params.get("acs_radius_fraction", 0.25))
        mask = np.zeros(n_shots, dtype=bool)

        norms = []
        for shot in coords_per_shot:
            cpu = ops.to_numpy(shot)
            norms.append(float(np.linalg.norm(cpu.mean(axis=0))))
        norms = np.array(norms, dtype=np.float32)

        if norms.max() > 0:
            norms_n = norms / norms.max()
        else:
            norms_n = np.zeros(n_shots, dtype=np.float32)

        if "acs_n_lines" in params and "cartesian" in traj_name:
            # Center the selection
            acs_n = int(params["acs_n_lines"])
            center_idx = np.argsort(norms_n)[:acs_n]
            mask[center_idx] = True
        else:
            mask[norms_n <= acs_frac] = True

        if not mask.any():
            warnings.warn(
                "keep_acs found no shots inside ACS region. "
                f"Increase acs_radius_fraction (currently {acs_frac}) or acs_n_lines.",
                UserWarning, stacklevel=3,
            )
        return mask

    @staticmethod
    def _us_poisson_disc_cartesian(n_shots: int, af: float, params: dict,
                                   coords_per_shot: List[Any],
                                   like=None) -> np.ndarray:
        """
        Variable-density Poisson-disc undersampling for Cartesian trajectories.

        Full Poisson-disc is expensive; this uses a simplified VD-random approach
        with a center-dense probability map as a practical stand-in compatible
        with MRI pipelines.

        params
        ------
        acs_lines : int
            Number of central lines always included (default: 8).  Note this is a
            floor on the retained count, so it caps the achievable acceleration at
            n_shots / acs_lines — with the old default of 24 on a 64-line grid,
            nothing above 2.67 was reachable no matter what was requested.
        vd_beta : float
            Density exponent (default: 3.0).
        seed : int, optional
            RNG seed.  Omit for a fresh draw on every call, which is what you want
            when this is used as an augmentation; pass an int only to reproduce a
            specific mask.
        """
        acs_lines = int(params.get("acs_lines", 8))
        vd_beta = float(params.get("vd_beta", 3.0))
        rng = np.random.default_rng(params.get("seed", None))

        # Phase encode index from shot center
        norms = []
        for shot in coords_per_shot:
            cpu = ops.to_numpy(shot)
            norms.append(float(np.abs(cpu[:, 1]).mean()))   # ky-like axis
        norms = np.array(norms, dtype=np.float32)

        if norms.max() > 0:
            norms_n = norms / norms.max()
        else:
            norms_n = np.zeros(n_shots, dtype=np.float32)

        sorted_idx = np.argsort(norms_n)
        acs_idx = sorted_idx[:acs_lines]

        n_keep = max(0, math.ceil(n_shots / af) - acs_lines)
        remaining = sorted_idx[acs_lines:]
        prob = np.exp(-vd_beta * norms_n[remaining])
        if prob.sum() > 0:
            prob /= prob.sum()
        else:
            prob = np.ones(len(remaining), dtype=np.float32) / len(remaining)

        chosen = rng.choice(remaining, size=min(n_keep, len(remaining)),
                            replace=False, p=prob)

        mask = np.zeros(n_shots, dtype=bool)
        mask[acs_idx] = True
        mask[chosen] = True
        return mask

    @staticmethod
    def _us_combined(n_shots: int, af: float, params: dict, coords_per_shot: List[Any], traj_name: str, like=None) -> np.ndarray:
        """
        Combined strategy: keep ACS + prefix/random_vd on remaining shots.

        params
        ------
        acs_radius_fraction : float (default 0.15)
        outer_method : str  'prefix' | 'random_vd' | 'drop_every' (default 'prefix')
        outer_af : float  (default = af, applied to outer shots only)
        """
        acs_mask = ShotUndersampler._us_keep_acs(n_shots, af, params, coords_per_shot, like, traj_name)
        outer_idx = np.where(~acs_mask)[0]
        outer_method = str(params.get("outer_method", "prefix"))
        outer_af = float(params.get("outer_af", af))

        outer_coords = [coords_per_shot[i] for i in outer_idx]
        n_outer = len(outer_idx)

        if outer_method == "prefix":
            outer_mask = ShotUndersampler._us_prefix(n_outer, outer_af, params)
        elif outer_method == "random_vd":
            outer_mask = ShotUndersampler._us_random_vd(n_outer, outer_af, params, outer_coords, like)
        elif outer_method == "drop_every":
            outer_mask = ShotUndersampler._us_drop_every(n_outer, outer_af, params)
        else:
            raise ValueError(
                f"combined outer_method '{outer_method}' not recognized. "
                "Use 'prefix', 'random_vd', or 'drop_every'."
            )

        mask = acs_mask.copy()
        mask[outer_idx[outer_mask]] = True
        return mask

    @staticmethod
    def _us_variable_density_spiral(n_shots: int, af: float, params: dict) -> np.ndarray:
        """
        Spiral undersampling by removing outer arms.
        Arms are assumed ordered center-to-outer; keep first ceil(n/acceleration_factor).
        """
        return ShotUndersampler._us_prefix(n_shots, af, params)

    @staticmethod
    def _us_shell_based(n_shots: int, af: float, params: dict,
                        coords_per_shot: List[Any], like=None) -> np.ndarray:
        """
        Shell-based undersampling for 3D trajectories: keep full inner shells,
        undersample outer shells progressively.

        params
        ------
        n_shells : int
            Number of concentric shells to divide shots into (default: 4).
        shell_afs : list[float]
            Per-shell acceleration factors from inner to outer
            (default: geometrically increasing from 1 to af).
        """
        n_shells = int(params.get("n_shells", 4))
        shell_afs_param = params.get("shell_afs", None)

        norms = np.array([float(np.linalg.norm(ops.to_numpy(s).mean(axis=0)))
                          for s in coords_per_shot], dtype=np.float32)
        if norms.max() > 0:
            norms_n = norms / norms.max()
        else:
            norms_n = np.zeros(n_shots, dtype=np.float32)

        shell_edges = np.linspace(0.0, 1.0, n_shells + 1)
        if shell_afs_param is None:
            shell_afs = np.geomspace(1.0, af, n_shells)
        else:
            shell_afs = np.asarray(shell_afs_param, dtype=np.float64)
            if len(shell_afs) != n_shells:
                raise ValueError(
                    f"shell_afs length ({len(shell_afs)}) must equal n_shells ({n_shells})."
                )

        mask = np.zeros(n_shots, dtype=bool)
        for si in range(n_shells):
            idx = np.where((norms_n >= shell_edges[si]) &
                           (norms_n < shell_edges[si + 1]))[0]
            n_keep = math.ceil(len(idx) / float(shell_afs[si]))
            chosen = idx[:n_keep]
            mask[chosen] = True

        return mask

    @staticmethod
    def undersample_shots(
        coords_per_shot: List[Any],
        method: str,
        acceleration_factor: float,
        params: Dict[str, Any],
        like=None,
        trajectory_name: str = "",
    ) -> Tuple[Any, Optional[List[Any]]]:
        """
        Apply an undersampling strategy to a list of shots.

        Parameters
        ----------
        coords_per_shot : list of arrays, each shape (Nsamples, Ndims)
            Full trajectory shots.
        method : str
            Undersampling method. One of:
            'prefix', 'golden_prefix', 'drop_every', 'poisson_disc_cartesian',
            'random_vd', 'keep_acs', 'variable_density_spiral',
            'shell_based', 'combined'.
        acceleration_factor : float
            Acceleration factor (≥ 1). Number of retained shots ≈ N / acceleration_factor.
            Exact: N_keep = ceil(N_total / acceleration_factor).
        params : dict
            Method-specific parameters (see each _us_* helper for keys).
        like : array, optional
        trajectory_name : str, optional
            Name of the trajectory (used for compatibility checks).

        Returns
        -------
        shot_mask : boolean array, shape (N_shots,)
            True → shot is included.
        per_sample_masks : list of boolean arrays | None
            One boolean array of shape (Nsamples,) per *included* shot.
            Currently None (per-shot masks deferred to downstream partial-Fourier
            or variable-density readout implementations).

        Raises
        ------
        ValueError
            If method is not supported, acceleration_factor < 1, or trajectory/method incompatible.
        TypeError
        """
        
        if acceleration_factor < 1.0:
            raise ValueError(
                f"acceleration_factor must be ≥ 1.0 (got {acceleration_factor}). "
                "acceleration_factor < 1 would require more shots than fully sampled — not valid."
            )

        n_shots = len(coords_per_shot)
        if n_shots == 0:
            raise ValueError("coords_per_shot is empty — no shots to undersample.")

        # Compatibility check
        if trajectory_name:
            compat = ShotUndersampler.COMPAT.get(trajectory_name, None)
            if compat is not None and method not in compat:
                raise ValueError(
                    f"Undersampling method '{method}' is not compatible with "
                    f"trajectory '{trajectory_name}'. "
                    f"Supported methods for this trajectory: {compat}. "
                    "Pass trajectory_name='' to bypass compatibility checking."
                )

        valid_methods = [
            "prefix", "golden_prefix", "stride", "drop_every", "poisson_disc_cartesian",
            "random_vd", "keep_acs", "variable_density_spiral", "shell_based", "combined",
        ]
        if method not in valid_methods:
            raise ValueError(
                f"Unknown undersampling method '{method}'. Valid: {valid_methods}."
            )

        traj_name = trajectory_name or ""

        if method == "prefix":
            mask_np = ShotUndersampler._us_prefix(n_shots, acceleration_factor, params)
        elif method == "golden_prefix":
            mask_np = ShotUndersampler._us_golden_prefix(n_shots, acceleration_factor, params)
        elif method == "stride":
            mask_np = ShotUndersampler._us_stride(n_shots, acceleration_factor, params)
        elif method == "drop_every":
            mask_np = ShotUndersampler._us_drop_every(n_shots, acceleration_factor, params)
        elif method == "poisson_disc_cartesian":
            mask_np = ShotUndersampler._us_poisson_disc_cartesian(n_shots, acceleration_factor, params, coords_per_shot, like)
        elif method == "random_vd":
            mask_np = ShotUndersampler._us_random_vd(n_shots, acceleration_factor, params, coords_per_shot, like)
        elif method == "keep_acs":
            mask_np = ShotUndersampler._us_keep_acs(n_shots, acceleration_factor, params, coords_per_shot, like, traj_name)
        elif method == "variable_density_spiral":
            mask_np = ShotUndersampler._us_variable_density_spiral(n_shots, acceleration_factor, params)
        elif method == "shell_based":
            mask_np = ShotUndersampler._us_shell_based(n_shots, acceleration_factor, params, coords_per_shot, like)
        elif method == "combined":
            mask_np = ShotUndersampler._us_combined(n_shots, acceleration_factor, params, coords_per_shot, like, traj_name)

        achieved_af = n_shots / max(mask_np.sum(), 1)
        if abs(achieved_af - acceleration_factor) > 0.5 * acceleration_factor:
            warnings.warn(
                f"Achieved acceleration_factor={achieved_af:.2f} differs substantially from requested acceleration_factor={acceleration_factor:.2f}. "
                "Consider adjusting 'drop_every_m', 'acs_lines', or 'acs_radius_fraction'.",
                UserWarning,
                stacklevel=2,
            )

        shot_mask = ops.cast_bool(ops.asarray_like(like, mask_np.astype(np.uint8)))
        return shot_mask, None   # per_sample_masks deferred → None


#**************************************************************************************************#
#                                          Class GridMask                                          #
#**************************************************************************************************#
#                                                                                                  #
# Boolean sampling masks on a Cartesian grid.                                                      #
#                                                                                                  #
#**************************************************************************************************#
class GridMask:
    """
    Boolean sampling masks on a Cartesian grid.

    For data already reconstructed onto a grid, undersampling can be applied as
    FFT -> mask -> IFFT, with no gridding or NUFFT. Masks are returned in
    "numpy.fft.fftshift" order, so they multiply a shifted spectrum directly.
    """

    @staticmethod
    def rasterize_shots_to_grid(
        coords_per_shot: List[Any],
        shot_mask: Any,
        fov_m: Tuple[float, ...],
        matrix: Tuple[int, ...],
    ) -> np.ndarray:
        """
        Rasterise the retained samples of a trajectory onto a Cartesian grid mask.

        Maps each retained k-space sample to its nearest grid bin using

            i = round(k [cycles/m] * FOV_m) + N // 2

        which is exact for "cartesian_2d" / "cartesian_3d" (whose coordinates are
        generated on exactly this lattice) and nearest-neighbor for everything else.

        The mapping goes through the *coordinates*, never the shot index. Shot index
        is not a k-space index: with "ordering='centric'" the "cartesian_2d"
        generator emits shots in a permuted order, so "shot_mask[s]" says nothing
        about which "ky" line "s" refers to.

        Parameters
        ----------
        coords_per_shot : list of arrays
            One (n_samples, n_dims) array per shot, in cycles per meter.
        shot_mask : array of bool, shape (n_shots,)
            Which shots were retained. Discarded shots contribute nothing.
        fov_m : tuple of float
            Field of view per axis **in meters** (i.e. fov_mm / 1000).
        matrix : tuple of int
            Grid size per axis. Length determines the output rank (2-D or 3-D).
        like : array, optional
            Reference tensor fixing the backend and device of the result.

        Returns
        -------
        numpy.ndarray of bool, shape "matrix"
            True where at least one retained sample lands. Bin ordering matches
            "numpy.fft.fftshift(numpy.fft.fftn(x))", so the mask can be applied
            directly to a shifted spectrum.

        Notes
        -----
        Samples falling outside the grid (e.g. the +kmax edge of a trajectory that
        slightly overshoots) are dropped rather than wrapped.
        """
        to_cpu = ops.to_numpy
        matrix = tuple(int(n) for n in matrix)
        n_dims = len(matrix)

        grid = np.zeros(matrix, dtype=bool)
        keep = np.flatnonzero(np.asarray(to_cpu(shot_mask), dtype=bool).ravel())
        if keep.size == 0:
            return grid

        # Concatenate the retained shots and index once. Looping per shot costs
        # a Python round trip and a fancy-index write per shot, which is minutes
        # rather than seconds once a trajectory has thousands of them.
        blocks = []
        for s in keep:
            c = np.asarray(to_cpu(coords_per_shot[int(s)]), dtype=np.float64)
            blocks.append(c[None, :] if c.ndim == 1 else c)
        coords = np.concatenate(blocks, axis=0)

        idx, inside = [], np.ones(coords.shape[0], dtype=bool)
        for d in range(n_dims):
            i_d = np.rint(coords[:, d] * fov_m[d]).astype(np.int64) + matrix[d] // 2
            inside &= (i_d >= 0) & (i_d < matrix[d])
            idx.append(i_d)

        if inside.any():
            grid[tuple(i[inside] for i in idx)] = True

        return grid

    @staticmethod
    def cartesian_vd_mask(
        matrix: Tuple[int, ...],
        acceleration_factor: float,
        acs_frac: float = 0.06,
        vd_beta: float = 3.0,
        seed: Optional[int] = None,
        axes: Optional[Tuple[int, ...]] = None,
    ) -> np.ndarray:
        """
        Draw a variable-density Cartesian sampling mask directly on the grid.

        Sampling probability falls off with normalized distance from the k-space
        center as "exp(-vd_beta * r)", with a fully-sampled central region (the
        auto-calibration region) always retained.

        Unlike "_us_poisson_disc_cartesian", which reduces every shot to a single
        scalar along one axis and is therefore 1-D by construction, this draws a
        genuinely N-dimensional mask — the pattern used by compressed-sensing MRSI.

        Parameters
        ----------
        matrix : tuple of int
            Grid size, e.g. "(64, 64)" for an in-plane mask.
        acceleration_factor : float
            Target acceleration (>= 1). Retained bins ~ "prod(matrix) / af".
            Note "acs_frac" sets a floor: an ACS region larger than the target
            budget caps the achievable acceleration.
        acs_frac : float, default 0.06
            Fraction of the grid width forming the fully-sampled center, per axis.
        vd_beta : float, default 3.0
            Density exponent. 0 gives uniform random sampling; larger values
            concentrate samples nearer the k-space center.
        seed : int, optional
            RNG seed. Omit for a fresh draw per call — the right default when this
            is used as an augmentation.
        axes : tuple of int, optional
            Which axes to undersample. Axes not listed are fully sampled. Use e.g.
            "axes=(0, 1)" on a 3-D grid to undersample in-plane only.

        Returns
        -------
        numpy.ndarray of bool, shape "matrix"
            Bin ordering matches "numpy.fft.fftshift(numpy.fft.fftn(x))".
        """
        matrix = tuple(int(n) for n in matrix)
        if acceleration_factor < 1.0:
            raise ValueError(
                f"acceleration_factor must be >= 1.0 (got {acceleration_factor}); "
                "values below 1 would require more samples than fully sampled."
            )

        rng = np.random.default_rng(seed)
        undersampled = tuple(range(len(matrix))) if axes is None else tuple(sorted(axes))

        # The mask is drawn on the sub-grid spanned by the undersampled axes, then
        # broadcast along the rest. "Axis d is fully sampled" means every bin along d
        # is acquired for each retained position in the sub-grid — not that d gets its
        # own independent random draw.
        sub_shape = tuple(matrix[d] for d in undersampled)

        # Normalized radius in [0, 1] from the k-space center of the sub-grid.
        coords = np.meshgrid(
            *[(np.arange(n) - n // 2) / max(n / 2.0, 1.0) for n in sub_shape],
            indexing="ij",
        )
        r = np.sqrt(np.sum([c ** 2 for c in coords], axis=0))
        r = r / max(r.max(), 1e-12)

        # Fully-sampled center, sized per axis so anisotropic grids behave.
        acs = np.ones(sub_shape, dtype=bool)
        for i, n in enumerate(sub_shape):
            half = max(1, int(round(0.5 * acs_frac * n)))
            along = np.abs(np.arange(n) - n // 2) <= half
            acs &= along.reshape([-1 if k == i else 1 for k in range(len(sub_shape))])

        n_sub = int(np.prod(sub_shape))
        n_budget = int(math.ceil(n_sub / float(acceleration_factor)))
        n_acs = int(acs.sum())

        sub_mask = acs.copy()
        n_extra = n_budget - n_acs
        if n_extra > 0:
            cand = np.flatnonzero(~acs.ravel())
            prob = np.exp(-vd_beta * r.ravel()[cand])
            total = prob.sum()
            prob = prob / total if total > 0 else np.full(cand.size, 1.0 / max(cand.size, 1))
            chosen = rng.choice(cand, size=min(n_extra, cand.size), replace=False, p=prob)
            sub_mask.ravel()[chosen] = True
        elif n_acs > n_budget:
            warnings.warn(
                f"acs_frac={acs_frac} reserves {n_acs} bins but the acceleration budget "
                f"is {n_budget}; achieved acceleration will be "
                f"{n_sub / max(n_acs, 1):.2f}, not {acceleration_factor:.2f}.",
                RuntimeWarning,
                stacklevel=2,
            )

        if sub_shape == matrix:
            return sub_mask

        # Broadcast the sub-grid pattern across the fully-sampled axes.
        bcast_shape = [matrix[d] if d in undersampled else 1 for d in range(len(matrix))]
        return np.broadcast_to(sub_mask.reshape(bcast_shape), matrix).copy()


#**************************************************************************************************#
#                                     Class ShotPerturbations                                      #
#**************************************************************************************************#
#                                                                                                  #
# Simulation hooks for shot-to-shot acquisition imperfections, plus the density diagnostic used to #
# sanity-check a retained subset.                                                                  #
#                                                                                                  #
#**************************************************************************************************#
class ShotPerturbations:
    """
    Simulation hooks for shot-to-shot acquisition imperfections, plus the
    density diagnostic used to sanity-check a retained subset.
    """

    @staticmethod
    def apply_shot_phase_errors(
        coords_per_shot: List[Any],
        phase_std_rad: float,
        like=None,
        seed: int = 0,
    ) -> List[Any]:
        """
        Simulation hook: inject shot-to-shot phase errors (k-space modulation).

        This does NOT modify coordinates — it returns a list of scalar phase
        offset values (in radians) per shot that can be applied as complex
        exponential weights during NUFFT.

        Parameters
        ----------
        coords_per_shot : list of arrays
        phase_std_rad : float
            Standard deviation of normally distributed random phase offsets [rad].
        like : array, optional
        seed : int
            RNG seed.

        Returns
        -------
        list of float
            One phase offset (radians) per shot.
        """
        rng = np.random.default_rng(seed)
        return [float(rng.normal(0.0, phase_std_rad)) for _ in coords_per_shot]

    @staticmethod
    def apply_gradient_delay(
        coords_per_shot: List[Any],
        delay_samples: float,
        like=None,
    ) -> List[Any]:
        """
        Simulation hook: shift each shot's k-space trajectory by a gradient delay.

        The delay is applied along the readout dimension (axis 0 of each shot).

        k_shifted[n] = k[n + delay_samples] approximated by:
            k_shifted ≈ k + delay_samples * dk   where dk is the per-sample step.

        Parameters
        ----------
        coords_per_shot : list of arrays, each (Nsamples, Ndims)
        delay_samples : float
            Delay in samples (may be fractional).
        like : array, optional

        Returns
        -------
        list of arrays (same shapes), shifted along readout.
        """
        shifted = []
        for shot in coords_per_shot:
            cpu = ops.to_numpy(shot)
            if cpu.shape[0] < 2:
                shifted.append(shot)
                continue
            dk = cpu[1] - cpu[0]   # per-sample k-space increment (Ndims,)
            shift = dk * delay_samples
            shifted_cpu = cpu + shift[np.newaxis, :]
            shifted.append(ops.asarray_like(like, shifted_cpu))
        return shifted

    @staticmethod
    def apply_off_resonance_phase(
        coords_per_shot: List[Any],
        freq_offset_hz: float,
        dwell_time_s: float,
        like=None,
    ) -> List[float]:
        """
        Simulation hook: compute per-sample off-resonance phase for each shot.

        Returns modulation phases φ(t) = 2π * Δf * t_n for downstream application
        as complex weights. Does not modify coordinate arrays.

        Parameters
        ----------
        coords_per_shot : list of arrays, each (Nsamples, Ndims)
        freq_offset_hz : float
            Off-resonance frequency in Hz.
        dwell_time_s : float
            Sampling interval in seconds (== DwellTime from header).
        like : array, optional

        Returns
        -------
        list of arrays, each (Nsamples,) — phase in radians.
        """
        phases = []
        for shot in coords_per_shot:
            nsamples = ops.shape(shot)[0]
            t = ops.arange_like(like, nsamples) * dwell_time_s
            phi = 2.0 * math.pi * freq_offset_hz * t
            phases.append(phi)
        return phases

    @staticmethod
    def compute_density_estimate(
        coords_per_shot: List[Any],
        shot_mask: Any,
        meta: Dict[str, Any],
        like=None,
    ) -> dict:
        """
        Return a refreshed density estimate for a given (possibly undersampled) subset.

        Does NOT compute DCF — it returns an analytic placeholder dict with
        recomputed subset statistics for use by downstream pipe or iterative DCF code.

        Parameters
        ----------
        coords_per_shot : list of arrays (all shots, before masking)
        shot_mask : boolean array (N_shots,)
        meta : dict from generate_trajectory or get_kspace_shots_and_mask
        like : array, optional

        Returns
        -------
        dict
            Updated density estimate dict appropriate for the retained subset.
        """
        mask_np = ops.to_numpy(shot_mask).astype(bool)
        retained = [s for s, m in zip(coords_per_shot, mask_np) if m]
        n_retained = len(retained)
        sps = ops.shape(retained[0])[0] if retained else 0

        base = meta.get("density_estimate", {}).copy()
        base.update({
            "n_shots_subset":          n_retained,
            "samples_per_shot_subset": sps,
            "total_samples_subset":    n_retained * sps,
            "achieved_acceleration":             meta.get("n_shots", n_retained) / max(n_retained, 1),
            "recomputed_for_subset":   True,
        })
        return base


#**************************************************************************************************#
#                                           Class ShotIO                                           #
#**************************************************************************************************#
#                                                                                                  #
# Save and reload trajectories with their mask and metadata.                                       #
#                                                                                                  #
#**************************************************************************************************#
class ShotIO:
    """Save and reload trajectories with their mask and metadata."""

    @staticmethod
    def save_shots_npz(
        path: str,
        coords_per_shot: List[Any],
        shot_mask: Any,
        meta: Dict[str, Any],
        like=None,
    ) -> None:
        """
        Save trajectory and mask to a NumPy .npz archive.

        Saved arrays
        ------------
        shot_mask : (N_shots,) bool
        coords_N  : (Nsamples, Ndims) float32 for each shot N (0-padded index)
        meta keys serialised as individual 0-d string arrays.

        Parameters
        ----------
        path : str
            File path (should end in .npz).
        coords_per_shot : list of arrays
        shot_mask : boolean array
        meta : dict
        like : array, optional
        """
        payload: dict = {}
        payload["shot_mask"] = ops.to_numpy(shot_mask)
        for i, shot in enumerate(coords_per_shot):
            payload[f"coords_{i:06d}"] = ops.to_numpy(shot).astype(np.float32)
        # Save select meta fields as scalars
        for key in ["trajectory_type", "n_shots", "n_shots_retained",
                    "achieved_acceleration", "acceleration_requested", "undersampling_method"]:
            if key in meta:
                payload[f"meta_{key}"] = np.array(str(meta[key]))
        kmax = meta.get("kmax")
        if kmax:
            payload["meta_kmax"] = np.array(kmax, dtype=np.float32)
        np.savez_compressed(path, **payload)
        print(f"Saved {len(coords_per_shot)} shots + mask → '{path}'")

    @staticmethod
    def load_shots_npz(path: str, like=None) -> Tuple[List[Any], Any, dict]:
        """
        Load a .npz archive saved by save_shots_npz.

        Returns
        -------
        coords_per_shot, shot_mask, meta_partial
        """
        data = np.load(path, allow_pickle=False)
        shot_mask = ops.cast_bool(ops.asarray_like(like, data["shot_mask"].astype(np.uint8)))

        shot_keys = sorted(k for k in data.files if k.startswith("coords_"))
        coords_per_shot = [ops.asarray_like(like, data[k]) for k in shot_keys]

        meta_partial: dict = {}
        for k in data.files:
            if k.startswith("meta_"):
                field = k[len("meta_"):]
                arr = data[k]
                try:
                    meta_partial[field] = arr.item()
                except ValueError:
                    # Multi-element array (e.g. kmax tuple)
                    meta_partial[field] = arr.tolist()

        return coords_per_shot, shot_mask, meta_partial


#**************************************************************************************************#
#                                       Class KspaceSampler                                        #
#**************************************************************************************************#
#                                                                                                  #
# One call from geometry to an undersampled trajectory.                                            #
#                                                                                                  #
#**************************************************************************************************#
class KspaceSampler:
    """
    One call from geometry to an undersampled trajectory.

    >>> shots, mask, meta = KspaceSampler.get_kspace_shots_and_mask(
    ...     nifti, "golden_radial_2d", "prefix", acceleration_factor=4.0,
    ...     traj_params={"n_shots": 200}, us_params={}, like=None)
    """

    @staticmethod
    def get_kspace_shots_and_mask(
        header: Dict[str, Any],
        trajectory: str,
        undersampling: str,
        acceleration_factor: float,
        traj_params: Dict[str, Any],
        us_params: Dict[str, Any],
        like=None,
    ) -> Tuple[List[Any], Any, Dict[str, Any]]:
        """
        Full pipeline: generate trajectory → apply undersampling → return results.

        Parameters
        ----------
        header : dict
            Parsed NIfTI-MRS header dictionary.
        trajectory : str
            Trajectory name (see generate_trajectory for full list).
        undersampling : str
            Undersampling method name (see undersample_shots for full list).
        acceleration_factor : float
            Acceleration factor (≥ 1).
        traj_params : dict
            Passed verbatim to generate_trajectory.
        us_params : dict
            Passed verbatim to undersample_shots.
        like : array, optional
            Reference tensor fixing the backend and device of the result.

        Returns
        -------
        coords_per_shot : list of arrays, each shape (Nsamples, Ndims)
            Coordinates in cycles/m for **all** generated shots (before masking).
            Use shot_mask to select retained shots.
        shot_mask : boolean array, shape (N_shots,)
            True → shot is retained (acquired).
        meta : dict
            Extended metadata dict including:
                trajectory_type, kmax, fov_mm, matrix, n_shots, samples_per_shot,
                n_shots_retained, achieved_acceleration, density_estimate, params_used,
                undersampling_method, acceleration_requested.

        Example
        -------
        >>> from kspace_sampling import get_kspace_shots_and_mask
        >>> header = {
        ...     "dim": [4, 64, 64, 1, 1, 1, 1, 1],
        ...     "pixdim": [1.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0],
        ...     "DwellTime": 1e-4,
        ...     "SpectrometerFrequency": [127.74e6],
        ... }
        >>> shots, mask, meta = get_kspace_shots_and_mask(
        ...     header, "golden_radial_2d", "prefix", acceleration_factor=4.0,
        ...     traj_params={"n_shots": 200, "samples_per_shot": 64},
        ...     us_params={},
        ...     like=None,
        ... )
        >>> # shots: list of 200 arrays each (64, 2)
        >>> # mask: bool array (200,), ~50 True entries
        """
        
        coords_per_shot, meta = TrajectoryRegistry.generate(trajectory, header, traj_params, like)
        shot_mask, per_sample_masks = ShotUndersampler.undersample_shots(
            coords_per_shot, undersampling, acceleration_factor, us_params, like,
            trajectory_name=trajectory,
        )

        mask_np = ops.to_numpy(shot_mask).astype(bool)
        n_retained = int(mask_np.sum())
        achieved_af = len(coords_per_shot) / max(n_retained, 1)

        meta.update({
            "undersampling_method":  undersampling,
            "acceleration_requested":          acceleration_factor,
            "n_shots_retained":      n_retained,
            "achieved_acceleration":           achieved_af,
            "us_params_used":        us_params,
        })

        return coords_per_shot, shot_mask, meta


#**************************************************************************************************#
#                                    Class KspaceUndersampling                                     #
#**************************************************************************************************#
#                                                                                                  #
# Retrospectively undersample gridded MRSI data to simulate acceleration.                          #
#                                                                                                  #
#**************************************************************************************************#
class KspaceUndersampling(BaseModule):
    """
    Retrospectively undersample gridded MRSI data to simulate acceleration.

    Data layout follows the NIfTI-MRS convention — "(batch, X, Y, Z, T)" — with
    the spectral axis last, matching "SpatialAugmentations" and everything
    "NIfTI_MRS_Plus.get_data()" produces.

    Why this is a mask and not a NUFFT
    ----------------------------------
    In phase-encoded CSI one shot acquires the *whole* FID at one k-space
    position, so the sampling operator is identical for all spectral points. It
    can therefore be applied **once** as a transfer function rather than once per
    timepoint. Doing it literally — forward NUFFT, drop shots, adjoint NUFFT —
    costs "batch x Z x T" separate 2-D NUFFTs, roughly 240 GFLOP and tens of
    seconds per batch on CPU for a 64x64x32x384 volume. Masking on the Cartesian
    grid gets the coverage pattern, the aliasing behavior, and exact zeros in
    unacquired bins for about 1/100th of that, with no extra dependency.

    Modes
    -----
    "'cartesian'" (default)
        Draw a variable-density mask directly on the (X, Y) grid.
    "'gridded'"
        Generate any of the trajectories in "augmentrum.sampling.kspace_sampling"
        (radial, spiral, rosette, concentric rings, ...), undersample its shots,
        and rasterise the retained samples onto the grid. Nearest-neighbor, so
        the off-grid point-spread function is approximated.

        Careful with what "acceleration_factor" means here: it undersamples
        **shots**, and the fraction of grid bins that end up covered is a
        different number. A radial trajectory with many long shots can still
        touch most bins at nominal acceleration 3. Read the shot-level figure
        from "last_meta_['achieved_acceleration']" and the bin coverage from
        "last_masks_.mean()"; they are not interchangeable.
    "'off'"
        Identity — useful to keep a pipeline shape-identical across splits.

    Noise
    -----
    "noise_sigma_k" adds complex Gaussian noise **in k-space before masking**,
    which is where thermal noise actually is. Adding white noise in the image
    domain instead would fill the unacquired bins that a real accelerated
    acquisition leaves exactly zero — a shortcut a network learns immediately.
    Leave it None and use "~augmentrum.augmentation.gaussian_noise.GaussianNoise"
    if you want image-domain noise regardless.

    Examples
    --------
    >>> us = KspaceUndersampling(acceleration_factor=4.0)
    >>> out, _ = us.process_tensor(volume)             # (B, X, Y, Z, T)

    >>> # spiral coverage instead of a Cartesian mask
    >>> us = KspaceUndersampling(ksp_mode='gridded', trajectory='spiral_2d',
    ...                          undersampling='prefix', acceleration_factor=3.0)

    Through Augmentrum, ranges are sampled per batch:

    >>> Augmentrum(data=..., pipeline=['undersampling'],
    ...            acceleration_factor=(2.0, 6.0))
    """

    # FFT masking runs natively on every tensor backend. NIFTI_LIST is absent
    # because this module only implements process_tensor, so the pipeline routes
    # a NIfTI-list batch to a tensor backend instead. The nufft mode narrows
    # this further in __init__.
    SUPPORTED_BACKENDS = tuple(b for b in Backend if b is not Backend.NIFTI_LIST)

    MODES = ('cartesian', 'gridded', 'nufft', 'off')

    def __init__(self,
                 ksp_mode: str = 'cartesian',
                 nufft_osf: float = 2.0,
                 nufft_impl: str = 'gridding',
                 acceleration_factor: float = 3.0,
                 acs_frac: float = 0.06,
                 vd_beta: float = 3.0,
                 # 'gridded' mode: which trajectory / shot-undersampling scheme
                 trajectory: str = 'golden_radial_2d',
                 undersampling: str = 'prefix',
                 traj_params: Optional[Dict[str, Any]] = None,
                 us_params: Optional[Dict[str, Any]] = None,
                 # geometry / behavior
                 undersample_axes: Optional[Sequence[int]] = (0, 1),
                 per_sample_masks: bool = True,
                 noise_sigma_k: Optional[float] = None,
                 chunk_t: Optional[int] = 128,
                 us_seed: Optional[int] = None,
                 pixdim: Optional[Tuple[float, ...]] = None):
        """
        ksp_mode: 'cartesian', 'gridded', 'nufft' or 'off' (see class docstring).
        nufft_osf: NUFFT grid oversampling, used by ksp_mode='nufft' only.
        nufft_impl: which NUFFT to use for ksp_mode='nufft'.
                'gridding' (default) measures with the Kaiser-Bessel operator;
                'interp' measures by interpolating an oversampled Cartesian grid,
                which is cheaper and less accurate, and is the approximation
                whose error against nufft_osf is worth studying; both run on any
                backend with gradients intact. 'torchkbnufft' is the PyTorch
                reference, kept for comparison and restricted to PyTorch.
        acceleration_factor: target acceleration (>= 1). 4.0 keeps ~1/4 of k-space.
                Pass a (min, max) tuple to Augmentrum to sample it per batch.
        acs_frac: fraction of the grid width kept fully sampled at the k-space
                center. This is a floor on the retained samples, so a large ACS
                caps the achievable acceleration.
        vd_beta: variable-density exponent. 0 is uniform random; larger values
                concentrate samples near the k-space center.
        trajectory: trajectory name for 'gridded' mode — any key of
                "TrajectoryGenerator.GENERATORS".
        undersampling: shot-undersampling method for 'gridded' mode.
        traj_params, us_params: extra dicts forwarded to trajectory generation
                and shot undersampling respectively.
        undersample_axes: spatial axes to undersample, in (X, Y, Z) order.
                Defaults to in-plane only, (0, 1) — the usual MRSI case, where
                the slab direction is fully encoded. Axes left out are fully
                sampled and share one in-plane pattern.
        per_sample_masks: draw an independent mask per batch element. False
                applies one mask to the whole batch.
        noise_sigma_k: standard deviation per channel of complex Gaussian noise
                added in k-space before masking. None disables it.
        chunk_t: transform this many spectral points at a time to bound peak
                memory. None does all at once. Affects memory only, not results.
        us_seed: RNG seed. None (default) draws a fresh mask every call, which is
                what you want from an augmentation; set it to reproduce a mask.
        pixdim: voxel size in mm per spatial axis. Only used by 'gridded' mode to
                shape the trajectory. Normally left None — geometry is read off
                the NIfTI-MRS data by BaseModule and passed in automatically.
        """
        super().__init__()

        if ksp_mode not in self.MODES:
            raise ValueError(f"ksp_mode must be one of {self.MODES}, got {ksp_mode!r}")

        self.ksp_mode = ksp_mode
        if ksp_mode == 'nufft' and nufft_impl == 'torchkbnufft':
            # torchkbnufft is a PyTorch extension, so that implementation cannot
            # run anywhere else. The gridding implementation has no such limit.
            self.SUPPORTED_BACKENDS = (Backend.PYTORCH,)

        self.nufft_osf = float(nufft_osf)
        if nufft_impl not in ('gridding', 'interp', 'torchkbnufft'):
            raise ValueError(
                f"nufft_impl must be 'gridding', 'interp' or 'torchkbnufft', "
                f"got {nufft_impl!r}")
        self.nufft_impl = nufft_impl
        self.acceleration_factor = acceleration_factor
        self.acs_frac = acs_frac
        self.vd_beta = vd_beta
        self.trajectory = trajectory
        self.undersampling = undersampling
        self.traj_params = dict(traj_params or {})
        self.us_params = dict(us_params or {})
        self.undersample_axes = tuple(undersample_axes) if undersample_axes is not None else None
        self.per_sample_masks = bool(per_sample_masks)
        self.noise_sigma_k = noise_sigma_k
        self.chunk_t = chunk_t
        self.us_seed = us_seed
        self.pixdim = tuple(pixdim) if pixdim is not None else None

        # Populated after every call — what was actually applied, for provenance.
        self.last_masks_: Optional[np.ndarray] = None
        self.last_meta_: Optional[Dict[str, Any]] = None

    #**********************#
    #   nufft undersampling #
    #**********************#

    def _apply_nufft(self, data_array, matrix, geometry, rng):
        """
        Measure along the real trajectory and invert, instead of masking the grid.

        The honest counterpart to "gridded". That mode uses the trajectory only
        to choose which Cartesian bins to keep and then masks the data's own FFT,
        so it never leaves the grid and never sees an off-grid sample value. Here
        the volume is sampled at the coordinates the trajectory actually visits
        and reconstructed from there with a density-compensated adjoint.

        The spectral axis rides in the NUFFT's coil slot, so a volume with T
        spectral points costs ONE forward/adjoint pair rather than T of them —
        which is what makes this affordable at all. Memory scales as
        "batch x T x n_samples", so it is sized for MRSI matrices (32-64 in
        plane) rather than for imaging-resolution grids.

        A 2-D trajectory describes one plane, so kz joins the spectral axis in
        the coil slot and every slice is measured with the same in-plane pattern;
        a 3-D trajectory samples the volume directly.
        """
        header = self._header_for_trajectory(matrix, geometry)
        us_params = dict(self.us_params)
        us_params.setdefault('seed', int(rng.integers(0, 2 ** 31 - 1)))
        us_params.setdefault('vd_beta', self.vd_beta)
        us_params.setdefault('acs_radius_fraction', self.acs_frac)

        shots, shot_mask, meta = KspaceSampler.get_kspace_shots_and_mask(
            header, self.trajectory, self.undersampling,
            float(self.acceleration_factor),
            traj_params=dict(self.traj_params), us_params=us_params,
            like=None,
        )
        self.last_meta_ = meta
        self.last_masks_ = np.asarray(shot_mask, dtype=bool)

        # ShotUndersampler clamps to at least one retained shot at any
        # acceleration, so there is no empty case to guard against here.
        keep = np.flatnonzero(self.last_masks_)
        pts = np.concatenate([np.atleast_2d(np.asarray(shots[int(i)])) for i in keep])

        ndim = pts.shape[1]
        kmax = np.asarray(meta['kmax'][:ndim], dtype=np.float64)
        if self.nufft_impl == 'torchkbnufft':
            return self._nufft_torchkbnufft(data_array, matrix, pts, kmax, ndim)
        return self._nufft_gridding(data_array, matrix, pts, kmax, ndim)

    def _nufft_gridding(self, data_array, matrix, pts, kmax, ndim):
        """
        Measure and reconstruct with the backend-agnostic gridding NUFFT.

        Runs on whatever backend the data is on and keeps its gradients, so a
        non-Cartesian acquisition can sit inside a training loop.
        """
        from augmentrum.processing.interpolating import LinearInterpolator
        from augmentrum.sampling.kspace_reconstructor import GriddingNUFFT

        # Gridding takes the trajectory in [-0.5, 0.5) per axis
        coords = (pts / (2.0 * kmax[None, :])).astype(np.float32)

        nx, ny, nz = matrix
        vol = ops.cast(data_array, 'complex64')
        n_batch = int(ops.shape(vol)[0])
        n_t = int(ops.shape(vol)[4])

        if ndim == 2:                       # (B, X, Y, Z, T) -> (B, Z*T, X, Y)
            im_size = (nx, ny)
            stack = ops.reshape(ops.transpose(vol, (0, 3, 4, 1, 2)),
                                (n_batch, nz * n_t, nx, ny))
        else:                               # (B, X, Y, Z, T) -> (B, T, X, Y, Z)
            im_size = (nx, ny, nz)
            stack = ops.transpose(vol, (0, 4, 1, 2, 3))

        # 'interp' is the same operator read with a cheaper interpolator
        interpolator = LinearInterpolator() if self.nufft_impl == 'interp' else None
        nufft = GriddingNUFFT(im_size, self.nufft_osf, interpolator=interpolator)
        gridder = (GriddingNUFFT(im_size, self.nufft_osf)
                   if interpolator is not None else nufft)

        recon = []
        for b in range(n_batch):
            plane = ops.reshape(ops.take(stack, np.array([b]), axis=0),
                                (-1,) + im_size)
            # Measure with the chosen interpolator, always reconstruct with
            # Kaiser-Bessel: the point is to study the forward approximation.
            recon.append(gridder.adjoint(nufft.forward(plane, coords), coords))
        out = ops.stack(recon, axis=0)

        if ndim == 2:
            out = ops.transpose(ops.reshape(out, (n_batch, nz, n_t, nx, ny)),
                                (0, 3, 4, 1, 2))
        else:
            out = ops.transpose(out, (0, 2, 3, 4, 1))

        return self._match_scale(out, vol, data_array)

    def _nufft_torchkbnufft(self, data_array, matrix, pts, kmax, ndim):
        """Measure and reconstruct with torchkbnufft, the PyTorch reference."""
        import torch as _torch
        from augmentrum.sampling.kspace_reconstructor import KspaceReconstructor

        coords = _torch.from_numpy(pts.T).float()[None, None]          # [1, 1, D, K]
        coords = KspaceReconstructor.normalize_trajectory(coords, kmax)

        was_numpy = not is_torch(data_array)
        vol = _torch.as_tensor(np.asarray(data_array)) if was_numpy else data_array
        dtype = vol.dtype
        vol = vol.to(_torch.complex64)

        nx, ny, nz = matrix
        if ndim == 2:                       # (B, X, Y, Z, T) -> (B, Z*T, X, Y)
            im_size = (nx, ny)
            stack = vol.permute(0, 3, 4, 1, 2).reshape(vol.shape[0], -1, nx, ny)
        else:                               # (B, X, Y, Z, T) -> (B, T, X, Y, Z)
            im_size = (nx, ny, nz)
            stack = vol.permute(0, 4, 1, 2, 3)

        recon = KspaceReconstructor(im_size, self.nufft_osf)
        batch_coords = coords.expand(stack.shape[0], -1, -1, -1)
        ktraj, _ = KspaceReconstructor.flatten(batch_coords)

        # Forward: what the trajectory would measure. [B, C, K], C = the coil
        # slot carrying the spectral (and, in 2-D, the kz) axis.
        kdata = recon._tkbn().KbNufft(
            im_size=im_size,
            grid_size=tuple(int(self.nufft_osf * n) for n in im_size),
        )(stack, ktraj)

        # Adjoint wants [B, S, C, L]; every retained shot was concatenated into
        # one readout above, so S = 1 and L = K.
        out = recon.adjoint(kdata[:, None], batch_coords)

        if ndim == 2:
            out = out.reshape(vol.shape[0], nz, vol.shape[4], nx, ny).permute(0, 3, 4, 1, 2)
        else:
            out = out.permute(0, 2, 3, 4, 1)

        out = self._match_scale(out, vol, data_array)
        out = ops.cast_like(out, _torch.zeros((), dtype=dtype))
        return ops.to_numpy(out) if was_numpy else out

    @staticmethod
    def _match_scale(out, vol, like):
        """
        Restore the input's overall scale.

        A density-compensated adjoint recovers the image only up to a constant
        that depends on the trajectory, the DCF and the grid size. That is
        acceptable in an operator but not in a pipeline stage, which has to hand
        back data in the units it was given. Only the global factor is touched;
        the relative structure is whatever the reconstruction produced.
        """
        num = ops.sqrt(ops.sum(ops.abs(vol) ** 2))
        den = ops.sqrt(ops.sum(ops.abs(out) ** 2))
        scale = num / ops.where(den > 1e-12, den, den * 0 + 1e-12)
        return out * ops.cast_like(scale, out)

    #***********************#
    #   mask construction   #
    #***********************#

    def _draw_mask(self, matrix: Tuple[int, ...], geometry: Optional[dict],
                   rng: np.random.Generator) -> np.ndarray:
        """Draw one boolean sampling mask of shape *matrix* (in fftshift order)."""
        accel = float(self.acceleration_factor)

        if self.ksp_mode == 'cartesian':
            # np.random.Generator has no "give me an int seed" API, so derive one
            # from the generator to keep the whole module reproducible from us_seed.
            seed = int(rng.integers(0, 2 ** 31 - 1))
            return GridMask.cartesian_vd_mask(
                matrix, accel,
                acs_frac=self.acs_frac,
                vd_beta=self.vd_beta,
                seed=seed,
                axes=self.undersample_axes,
            )

        # --- 'gridded': trajectory -> shot mask -> raster onto the grid ---
        header = self._header_for_trajectory(matrix, geometry)
        us_params = dict(self.us_params)
        us_params.setdefault('seed', int(rng.integers(0, 2 ** 31 - 1)))
        us_params.setdefault('vd_beta', self.vd_beta)
        us_params.setdefault('acs_radius_fraction', self.acs_frac)

        coords, shot_mask, meta = KspaceSampler.get_kspace_shots_and_mask(
            header, self.trajectory, self.undersampling, accel,
            traj_params=dict(self.traj_params), us_params=us_params,
            like=None,
        )
        self.last_meta_ = meta

        traj_matrix = tuple(int(n) for n in meta['matrix'])
        fov_m = tuple(f / 1000.0 for f in meta['fov_mm'])
        grid = GridMask.rasterize_shots_to_grid(coords, shot_mask, fov_m, traj_matrix)

        # 2-D trajectories describe the in-plane pattern only; broadcast it along
        # any remaining axis so the slab direction stays fully encoded.
        while grid.ndim < len(matrix):
            grid = grid[..., None]
        return np.broadcast_to(grid, matrix).copy()

    def _header_for_trajectory(self, matrix: Tuple[int, ...],
                               geometry: Optional[dict]) -> dict:
        """Assemble the geometry header trajectory generation needs."""
        nx, ny = int(matrix[0]), int(matrix[1])
        nz = int(matrix[2]) if len(matrix) > 2 else 1

        if self.pixdim is not None:
            vx, vy, vz = (list(self.pixdim) + [1.0, 1.0, 1.0])[:3]
        elif geometry is not None and 'voxel_mm' in geometry:
            vx, vy, vz = geometry['voxel_mm']
        else:
            # Isotropic fallback. The index<->coordinate mapping is scale
            # invariant, so this only affects the *shape* of non-Cartesian
            # trajectories on anisotropic grids, never the raster mapping.
            vx = vy = vz = 1.0

        dwell = (geometry or {}).get('dwell_time') or 1e-3
        sf = (geometry or {}).get('spectrometer_frequency') or 127.732434e6
        return {
            "dim":    [4, nx, ny, nz, 1, 1, 1, 1],
            "pixdim": [1.0, float(vx), float(vy), float(vz), float(dwell), 1.0, 1.0, 1.0],
            "DwellTime": float(dwell),
            "SpectrometerFrequency": [float(sf)],
        }

    #**************************#
    #   basemodule interface   #
    #**************************#

    def process_tensor(self, data_array, water_array=None,
                       backend: Backend = Backend.PYTORCH, **kwargs):
        """
        Undersample a batch of MRSI volumes.

        Args:
            data_array: (batch, X, Y, Z, T) complex, or (batch, X, Y, Z, T, C)
                to undersample a receive array coil by coil.
            water_array: passed through unchanged.
            backend: Backend enum (unused; kept for the BaseModule signature).
            **kwargs: absorbs sw_hz / sf_mhz / geometry injected by BaseModule.

        Returns:
            (undersampled_data, water_unchanged), same shape and dtype in.
        """
        if self.ksp_mode == 'off':
            return data_array, water_array

        geometry = kwargs.get('geometry')

        if data_array.ndim not in (5, 6):
            raise ValueError(
                "KspaceUndersampling expects (batch, X, Y, Z, T) in the NIfTI "
                "layout, or (batch, X, Y, Z, T, C) with a receive array, got "
                f"shape {tuple(data_array.shape)}."
            )
        if data_array.ndim == 6:
            return self._apply_per_coil(data_array, **kwargs), water_array
        n_batch = int(data_array.shape[0])
        matrix = tuple(int(s) for s in data_array.shape[1:4])

        rng = np.random.default_rng(self.us_seed)
        if self.ksp_mode == 'nufft':
            return self._apply_nufft(data_array, matrix, geometry, rng), water_array

        n_masks = n_batch if self.per_sample_masks else 1
        masks = np.stack([self._draw_mask(matrix, geometry, rng) for _ in range(n_masks)])
        self.last_masks_ = masks

        return self._apply_masks(data_array, masks), water_array

    #***************#
    #   coils       #
    #***************#

    def _apply_per_coil(self, data_array, **kwargs):
        """
        Undersample a receive array, one element at a time.

        An acquisition visits one trajectory and every element of the array
        measures along it, so each coil must see the same pattern. Each pass
        redraws from "us_seed" rather than advancing a shared generator, which
        gives exactly that. Thermal noise does advance, since it is independent
        per element.

        Looping also keeps memory where it is: the NUFFT already carries the
        spectral axis in its channel slot, and multiplying that by the coil
        count would put tens of thousands of channels through one transform.

        Args:
            data_array: "(batch, X, Y, Z, T, C)" complex, on any backend.

        Returns:
            The same shape, each coil undersampled by the same acquisition.
        """
        n_coils = int(ops.shape(data_array)[5])
        shape = tuple(int(n) for n in ops.shape(data_array))[:5]

        per_coil = []
        for c in range(n_coils):
            volume = ops.reshape(ops.take(data_array, np.array([c]), axis=5), shape)
            sampled, _ = self.process_tensor(volume, **kwargs)
            per_coil.append(sampled)

        return ops.stack(per_coil, axis=5)

    #***************#
    #   masking     #
    #***************#

    def _spatial_axes(self, ndim: int = 5) -> Tuple[int, ...]:
        return (1, 2, 3)

    def _apply_masks(self, x, masks: np.ndarray):
        """
        FFT over the spatial axes, add k-space noise, zero the unacquired bins,
        transform back.

        The mask is in fftshift order, with the k-space center in the middle, so
        the spectrum is shifted to match rather than the mask being rolled.

        Runs on whatever backend *x* belongs to. Chunks are collected and joined
        rather than written into a preallocated output, because jax and
        tensorflow tensors are immutable.
        """
        axes = self._spatial_axes()

        if masks.shape[0] == 1 and x.shape[0] > 1:
            masks = np.broadcast_to(masks, (x.shape[0],) + masks.shape[1:])
        mask = ops.match_backend(masks[..., None], x)   # broadcast over T

        n_t = int(x.shape[-1])
        chunk = int(self.chunk_t) if self.chunk_t else n_t
        chunk = max(1, min(chunk, n_t))

        pieces = []
        for t0 in range(0, n_t, chunk):
            block = x[..., t0:t0 + chunk]
            k = ops.fftshift(ops.fftn(block, axes, norm='ortho'), axis=axes)
            if self.noise_sigma_k:
                real = self.rng.normal(tuple(k.shape), like=ops.real(k))
                imag = self.rng.normal(tuple(k.shape), like=ops.real(k))
                k = k + float(self.noise_sigma_k) * ops.cast_like(
                    ops.complex_from(real, imag), k)
            k = k * ops.cast_like(mask, k)
            pieces.append(ops.ifftn(ops.ifftshift(k, axis=axes), axes, norm='ortho'))

        return pieces[0] if len(pieces) == 1 else ops.concatenate(pieces, axis=-1)
