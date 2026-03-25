"""
kspace_sampling.py
==================
A backend-agnostic Python module for generating k-space sampling trajectories
and undersampling masks for MRI/MRS simulation pipelines.

Design Choices
--------------
Non-Cartesian trajectories default to center-out readout ordering, which places
the most signal-rich k-space center at the beginning of each shot and improves
compatibility with variable-density density-compensation functions. Golden-angle
and phyllotaxis-based orderings are used wherever angular distribution matters,
because any prefix of N shots retains near-uniform angular coverage — a critical
property for retrospective undersampling studies and for incrementally acquiring
datasets. Density estimates are recomputed analytically for every subset so that
downstream DCF pipelines receive a geometry-consistent starting point. No
gridding, reconstruction, or DCF computation is performed here; this module
exclusively returns coordinate arrays and boolean masks.

Interface Summary (plain text)
------------------------------
read_header_geometry(header: dict) -> dict
generate_trajectory(name: str, header: dict, params: dict, backend) -> (list[array], dict)
shots_from_continuous(coords: array, segmentation: dict, backend) -> list[array]
undersample_shots(coords_per_shot: list, method: str, AF: float, params: dict, backend)
    -> (array[bool], list[array[bool]] | None)
get_kspace_shots_and_mask(header, trajectory, undersampling, AF, traj_params, us_params, backend)
    -> (list[array], array[bool], dict)

Units: all coordinates in cycles per metre (1/m), k-space centre at 0.
Conversion: k [1/m] = k_index / FOV_m  where FOV_m = FOV_mm / 1000.

Backend must provide (see BackendInterface docstring below):
    array, zeros, ones, linspace, stack, concat, sin, cos, sqrt, pi,
    asarray, shape, to_cpu, floor, ceil, arange, randperm, meshgrid,
    cast_bool, cast_int, cast_float, full.

Dependencies: numpy (always available); torch (optional, only if TorchBackend used).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Array = Any          # numpy ndarray or torch.Tensor
Backend = Any        # NumpyBackend or TorchBackend instance
HeaderDict = Dict[str, Any]
MetaDict = Dict[str, Any]
ParamsDict = Dict[str, Any]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GOLDEN_ANGLE_RAD: float = math.pi * (3.0 - math.sqrt(5.0))   # ≈ 2.39996 rad
GOLDEN_RATIO: float = (1.0 + math.sqrt(5.0)) / 2.0           # ≈ 1.61803

# Trajectory → compatible undersampling strategies
_TRAJ_US_COMPAT: Dict[str, List[str]] = {
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
    "stack_of_cones":    ["prefix", "drop_every", "combined"],
    "stack_of_rosettes":  ["prefix", "drop_every", "combined"],
    "stack_of_ECCENTRIC":["prefix", "drop_every", "combined"],
    "3d_phyllotaxis":    ["prefix", "golden_prefix", "drop_every",
                          "random_vd", "shell_based", "combined"],
    "cones_3d":          ["prefix", "drop_every", "shell_based", "combined"],
    "cones_3d_rosette":  ["prefix", "drop_every", "shell_based", "combined"],
    "floret_3d":         ["prefix", "drop_every", "shell_based", "combined"],
    "3d_egg_rosette":    ["prefix", "golden_prefix", "drop_every",
                          "shell_based", "combined"],
}

# ---------------------------------------------------------------------------
# Backend Abstraction
# ---------------------------------------------------------------------------


class BackendInterface:
    """
    Required interface for any backend.

    Methods
    -------
    array(data, dtype=None)
        Create a backend array from a Python list or numpy array.
    zeros(shape, dtype=None)
        Array of zeros.
    ones(shape, dtype=None)
        Array of ones.
    full(shape, fill_value, dtype=None)
        Array filled with fill_value.
    linspace(start, stop, num, endpoint)
        Evenly-spaced values (like numpy.linspace).
    arange(start, stop, step=1)
        Evenly-spaced integers/floats (like numpy.arange).
    stack(arrays, axis=0)
        Stack sequence of arrays along a new axis.
    concat(arrays, axis=0)
        Concatenate arrays along an existing axis.
    sin(x) / cos(x) / sqrt(x)
        Element-wise trig / sqrt.
    pi()
        Return float pi.
    asarray(x)
        Convert to backend array (no-op if already correct type).
    shape(x) -> tuple
        Return shape as a plain Python tuple.
    to_cpu(x) -> numpy.ndarray
        Return numpy view/copy (for I/O and assertions).
    floor(x) / ceil(x)
        Element-wise floor / ceil.
    cast_bool(x) -> array
        Cast to boolean.
    cast_int(x) -> array
        Cast to int64.
    cast_float(x) -> array
        Cast to float32 or float64.
    randperm(n) -> 1-D int array
        Random permutation of range(n) — for random undersampling.
    meshgrid(*arrays, indexing='ij')
        N-D coordinate grids.
    """


class NumpyBackend:
    """Concrete backend using NumPy."""

    def array(self, data, dtype=None):
        return np.array(data, dtype=dtype)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype or np.float32)

    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype or np.float32)

    def full(self, shape, fill_value, dtype=None):
        return np.full(shape, fill_value, dtype=dtype or np.float32)

    def linspace(self, start, stop, num, endpoint=True):
        return np.linspace(start, stop, int(num), dtype=np.float32, endpoint=endpoint)

    def arange(self, start, stop=None, step=1):
        if stop is None:
            return np.arange(start, dtype=np.float32)
        return np.arange(start, stop, step, dtype=np.float32)

    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis=axis)

    def concat(self, arrays, axis=0):
        return np.concatenate(arrays, axis=axis)

    def sin(self, x):
        return np.sin(x)

    def cos(self, x):
        return np.cos(x)

    def sqrt(self, x):
        return np.sqrt(x)

    def pi(self):
        return math.pi

    def asarray(self, x):
        return np.asarray(x, dtype=np.float32)

    def shape(self, x):
        return tuple(x.shape)

    def to_cpu(self, x):
        if isinstance(x, np.ndarray):
            return x
        return np.array(x)

    def floor(self, x):
        return np.floor(x)

    def ceil(self, x):
        return np.ceil(x)

    def cast_bool(self, x):
        return x.astype(bool)

    def cast_int(self, x):
        return x.astype(np.int64)

    def cast_float(self, x):
        return x.astype(np.float32)

    def randperm(self, n):
        return np.random.permutation(n).astype(np.int64)

    def meshgrid(self, *arrays, indexing="ij"):
        return np.meshgrid(*arrays, indexing=indexing)

    def __repr__(self):
        return "NumpyBackend()"


class TorchBackend:
    """
    Concrete backend using PyTorch.

    Requires: import torch
    device keyword can be passed at construction, e.g. TorchBackend(device='cuda').
    """

    def __init__(self, device: str = "cpu", dtype=None):
        import torch  # noqa: F401 – deferred import so numpy-only usage works
        self._torch = torch
        self.device = torch.device(device)
        self._dtype = dtype or torch.float32

    def array(self, data, dtype=None):
        dt = dtype or self._dtype
        return self._torch.tensor(data, dtype=dt, device=self.device)

    def zeros(self, shape, dtype=None):
        return self._torch.zeros(shape, dtype=dtype or self._dtype, device=self.device)

    def ones(self, shape, dtype=None):
        return self._torch.ones(shape, dtype=dtype or self._dtype, device=self.device)

    def full(self, shape, fill_value, dtype=None):
        return self._torch.full(shape, fill_value, dtype=dtype or self._dtype,
                                device=self.device)

    def linspace(self, start, stop, num):
        return self._torch.linspace(start, stop, int(num), dtype=self._dtype,
                                    device=self.device)

    def arange(self, start, stop=None, step=1):
        if stop is None:
            return self._torch.arange(start, dtype=self._dtype, device=self.device)
        return self._torch.arange(start, stop, step, dtype=self._dtype,
                                  device=self.device)

    def stack(self, arrays, axis=0):
        return self._torch.stack(arrays, dim=axis)

    def concat(self, arrays, axis=0):
        return self._torch.cat(arrays, dim=axis)

    def sin(self, x):
        return self._torch.sin(x)

    def cos(self, x):
        return self._torch.cos(x)

    def sqrt(self, x):
        return self._torch.sqrt(x)

    def pi(self):
        return math.pi

    def asarray(self, x):
        if isinstance(x, self._torch.Tensor):
            return x.to(dtype=self._dtype, device=self.device)
        return self._torch.tensor(x, dtype=self._dtype, device=self.device)

    def shape(self, x):
        return tuple(x.shape)

    def to_cpu(self, x):
        if isinstance(x, self._torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def floor(self, x):
        return self._torch.floor(x)

    def ceil(self, x):
        return self._torch.ceil(x)

    def cast_bool(self, x):
        return x.bool()

    def cast_int(self, x):
        return x.long()

    def cast_float(self, x):
        return x.to(self._dtype)

    def randperm(self, n):
        return self._torch.randperm(n, device=self.device)

    def meshgrid(self, *arrays, indexing="ij"):
        return self._torch.meshgrid(*arrays, indexing=indexing)

    def __repr__(self):
        return f"TorchBackend(device={self.device}, dtype={self._dtype})"


def _validate_backend(backend: Backend) -> None:
    """Raise if backend is missing any required method."""
    required = [
        "array", "zeros", "ones", "full", "linspace", "arange",
        "stack", "concat", "sin", "cos", "sqrt", "pi",
        "asarray", "shape", "to_cpu", "floor", "ceil",
        "cast_bool", "cast_int", "cast_float", "randperm", "meshgrid",
    ]
    missing = [m for m in required if not hasattr(backend, m)]
    if missing:
        raise TypeError(
            f"Backend {backend!r} is missing required methods: {missing}. "
            "Implement a class with all methods listed in BackendInterface."
        )


# ---------------------------------------------------------------------------
# Header Geometry
# ---------------------------------------------------------------------------

# Standard NIfTI-MRS header field names (for documentation & parsing)
_NIFTI_MRS_FIELDS = {
    "pixdim":               "voxel sizes (mm) stored at pixdim[1:4]",
    "dim":                  "matrix dimensions stored at dim[1:4]",
    "qform_code":           "qform code (non-zero → use qform)",
    "sform_code":           "sform code (non-zero → use sform)",
    "SpectrometerFrequency":"transmitter centre frequency [Hz] list",
    "DwellTime":            "ADC dwell time [s] per spectral point",
    "ResonantNucleus":      "nucleus label e.g. '1H'",
}


def read_header_geometry(header: HeaderDict) -> dict:
    """
    Parse a NIfTI-MRS header dictionary and return imaging geometry fields.

    Parameters
    ----------
    header : dict
        Parsed NIfTI-MRS header. Recognised standard fields (see _NIFTI_MRS_FIELDS):
          - ``dim``    : array-like, dim[1..3] → matrix [nx, ny, nz]
          - ``pixdim`` : array-like, pixdim[1..3] → voxel size [vx, vy, vz] in mm
          - ``SpectrometerFrequency`` : list[float] → centre frequency in Hz
          - ``DwellTime``             : float → ADC dwell time in seconds
          Fallback keys also accepted: ``matrix``, ``voxel_size_mm``,
          ``fov_mm``, ``dwell_time``.

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


# ---------------------------------------------------------------------------
# Trajectory Helpers
# ---------------------------------------------------------------------------

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


def _density_estimate_placeholder(trajectory_name: str, n_shots: int,
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
    kmax = _kmax_from_geometry(geom)
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
            "For radial: density ∝ |k| (centre-heavy). "
            "For spirals: use gradient-delay-corrected NUFFT pipe density."
        ),
    }


# ---------------------------------------------------------------------------
# Trajectory Generators
# ---------------------------------------------------------------------------

def _traj_cartesian_2d(geom: dict, params: dict, backend: Backend) -> Tuple[List[Array], MetaDict]:
    """
    2D Cartesian trajectory.

    Each shot = one phase-encode line (ky) with full kx readout.

    Parameters (params)
    -------------------
    ordering : str
        'linear' (default) — ky from -N/2 to N/2-1
        'centric' — ky ordered from centre outwards
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

    kx_line = backend.linspace(-nx / 2, nx / 2 - 1, nx, True) / fov_x_m   # (nx,) in 1/m

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
        ky_line = backend.full((nx,), float(ky_val))
        shot = backend.stack([kx_line, ky_line], axis=1)   # (nx, 2)
        shots.append(shot)

    kmax_x, kmax_y, _ = _kmax_from_geometry(geom)
    meta = {
        "trajectory_type":    "cartesian_2d",
        "ordering":           ordering,
        "kmax":               (kmax_x, kmax_y),
        "fov_mm":             geom["fov_mm"][:2],
        "matrix":             (nx, ny),
        "n_shots":            n_shots,
        "samples_per_shot":   nx,
        "density_estimate":   _density_estimate_placeholder("cartesian_2d", n_shots, nx, geom),
        "params_used":        params,
    }
    return shots, meta


def _traj_cartesian_3d(geom: dict, params: dict, backend: Backend) -> Tuple[List[Array], MetaDict]:
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
    kx_line = backend.linspace(-nx / 2, nx / 2 - 1, nx) / fov_x_m

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
            ky_line = backend.full((nx,), float(ky_val))
            kz_line = backend.full((nx,), float(kz_val))
            shot = backend.stack([kx_line, ky_line, kz_line], axis=1)  # (nx, 3)
            shots.append(shot)
            count += 1

    kmax_x, kmax_y, kmax_z = _kmax_from_geometry(geom)
    n_shots = len(shots)
    meta = {
        "trajectory_type":    "cartesian_3d",
        "ordering":           ordering,
        "kmax":               (kmax_x, kmax_y, kmax_z),
        "fov_mm":             geom["fov_mm"],
        "matrix":             (nx, ny, nz),
        "n_shots":            n_shots,
        "samples_per_shot":   nx,
        "density_estimate":   _density_estimate_placeholder("cartesian_3d", n_shots, nx, geom),
        "params_used":        params,
    }
    return shots, meta


def _traj_radial_2d(geom: dict, params: dict, backend: Backend,
                    use_golden: bool = False) -> Tuple[List[Array], MetaDict]:
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
        If True (default), spoke goes from centre outward.
        If False, spoke is symmetric (–kmax → +kmax, centre at midpoint).
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
    center_out = bool(params.get("center_out", True))
    ga = float(params.get("golden_angle_rad", GOLDEN_ANGLE_RAD))

    name = "golden_radial_2d" if use_golden else "radial_2d"
    shots = []

    for i in range(n_shots):
        if use_golden:
            angle = i * ga
        else:
            angle = i * math.pi / n_shots

        if center_out:
            r = backend.linspace(0.0, 1.0, sps)
        else:
            r = backend.linspace(-1.0, 1.0, sps)

        kx = r * backend.cos(backend.array([angle]))[0] * kmax_x
        ky = r * backend.sin(backend.array([angle]))[0] * kmax_y
        shot = backend.stack([kx, ky], axis=1)
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
        "density_estimate":   _density_estimate_placeholder(name, n_shots, sps, geom),
        "params_used":        params,
    }
    return shots, meta


def _traj_spiral_2d(geom: dict, params: dict, backend: Backend) -> Tuple[List[Array], MetaDict]:
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
        Number of full turns per arm (default: n_shots * max(nx,ny)/4).
    vd_alpha : float
        Variable-density exponent ≥ 1 (1 = Archimedean, >1 = denser centre).
        r(t) ∝ t^(1/vd_alpha).
    rotate_first : bool
        If True (default), shot i is rotated by i * 2π / n_shots.
    """
    nx, ny, _ = geom["matrix"]
    fov_x_m = geom["fov_mm"][0] / 1000.0
    fov_y_m = geom["fov_mm"][1] / 1000.0
    kmax_x = (nx / 2.0) / fov_x_m
    kmax_y = (ny / 2.0) / fov_y_m

    n_shots = int(params.get("n_shots", 1))
    sps = int(params.get("samples_per_shot", max(nx * ny // max(n_shots, 1), 64)))
    turns = float(params.get("spiral_turns", n_shots * max(nx, ny) / 4.0))
    vd_alpha = float(params.get("vd_alpha", 1.0))
    rotate_first = bool(params.get("rotate_first", True))

    t = backend.linspace(0.0, 1.0, sps)           # normalised parameter
    # r(t) ∝ t^(1/vd_alpha), ensures r ∈ [0, 1]
    r = t ** (1.0 / vd_alpha)
    theta_base = r * (2.0 * math.pi * turns)

    shots = []
    for i in range(n_shots):
        rot = (2.0 * math.pi * i / n_shots) if rotate_first else 0.0
        theta = theta_base + rot
        kx = r * backend.cos(theta) * kmax_x
        ky = r * backend.sin(theta) * kmax_y
        shot = backend.stack([kx, ky], axis=1)
        shots.append(shot)

    meta = {
        "trajectory_type":    "spiral_2d",
        "spiral_turns":       turns,
        "vd_alpha":           vd_alpha,
        "kmax":               (kmax_x, kmax_y),
        "fov_mm":             geom["fov_mm"][:2],
        "matrix":             (nx, ny),
        "n_shots":            n_shots,
        "samples_per_shot":   sps,
        "density_estimate":   _density_estimate_placeholder("spiral_2d", n_shots, sps, geom),
        "params_used":        params,
    }
    return shots, meta


def _traj_rosette_2d(geom: dict, params: dict, backend: Backend) -> Tuple[List[Array], MetaDict]:
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
    sps = int(params.get("samples_per_shot", 512))
    omega1 = float(params.get("omega1", 5.0))
    omega2 = float(params.get("omega2", 4.0))
    rot_step = float(params.get("rotation_per_shot", GOLDEN_ANGLE_RAD))

    t = backend.linspace(0.0, 2.0 * math.pi, sps)
    shots = []
    for i in range(n_shots):
        rot = i * rot_step
        kx = backend.sin(t * omega1) * backend.cos(t * omega2 + rot) * kmax_x
        ky = backend.sin(t * omega1) * backend.sin(t * omega2 + rot) * kmax_y
        shot = backend.stack([kx, ky], axis=1)
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
        "density_estimate":   _density_estimate_placeholder("rosette_2d", n_shots, sps, geom),
        "params_used":        params,
    }
    return shots, meta


def _traj_rosette_2d_petals(geom: dict, params: dict, backend: Backend) -> Tuple[List[Array], MetaDict]:
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

    n_petals = int(params.get("n_petals", 7))
    sps = int(params.get("samples_per_shot", 512))
    start_angle = float(params.get("start_angle_rad", 0.0))

    t = backend.linspace(0.0, math.pi, sps)   # one petal: origin → tip → origin
    shots = []
    for k in range(n_petals):
        # -π/2 aligns petal 0 with +kx; 2π*k/n_petals rotates CCW
        phi = -math.pi / 2.0 + 2.0 * math.pi * k / n_petals + start_angle
        kx = backend.sin(t) * backend.cos(t + phi) * kmax_x
        ky = backend.sin(t) * backend.sin(t + phi) * kmax_y
        shot = backend.stack([kx, ky], axis=1)
        shots.append(shot)

    kmax_xv, kmax_yv, _ = _kmax_from_geometry(geom)
    meta = {
        "trajectory_type":    "rosette_2d_petals",
        "n_petals":           n_petals,
        "kmax":               (kmax_xv, kmax_yv),
        "fov_mm":             geom["fov_mm"][:2],
        "matrix":             (nx, ny),
        "n_shots":            n_petals,
        "samples_per_shot":   sps,
        "center_out_center":  True,
        "density_estimate":   _density_estimate_placeholder(
                                  "rosette_2d_petals", n_petals, sps, geom),
        "params_used":        params,
    }
    return shots, meta


def _traj_stack_of_x(geom: dict, params: dict, backend: Backend,
                     inplane_name: str) -> Tuple[List[Array], MetaDict]:
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
        "radial":        lambda: _traj_radial_2d(geom2d, params, backend, use_golden=False),
        "golden_radial": lambda: _traj_radial_2d(geom2d, params, backend, use_golden=True),
        "spiral":        lambda: _traj_spiral_2d(geom2d, params, backend),
        "cones":         lambda: _traj_spiral_2d(geom2d, params, backend),
        "ECCENTRIC":     lambda: _traj_eccentric(geom2d, params, backend),
        "rosette":       lambda: _traj_rosette_2d(geom2d, params, backend),
    }
    if inplane_name not in ip_map:
        raise ValueError(f"Unknown in-plane trajectory '{inplane_name}'. "
                         f"Valid: {list(ip_map.keys())}")
    inplane_shots, inplane_meta = ip_map[inplane_name]()

    all_shots = []
    for kz_val in kz_positions:
        kz_col = float(kz_val)
        for shot2d in inplane_shots:
            nrows = backend.shape(shot2d)[0]
            kz_col_arr = backend.full((nrows,), kz_col)
            shot3d = backend.concat([shot2d, backend.stack([kz_col_arr], axis=1)], axis=1)
            all_shots.append(shot3d)

    n_shots_total = len(all_shots)
    sps = backend.shape(all_shots[0])[0] if all_shots else 0
    traj_name = f"stack_of_{inplane_name}"
    kmax_x, kmax_y, _ = _kmax_from_geometry(geom)
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
        "density_estimate":   _density_estimate_placeholder(traj_name, n_shots_total, sps, geom),
        "params_used":        params,
    }
    return all_shots, meta


def _traj_eccentric(geom: dict, params: dict, backend: Backend) -> Tuple[List[Array], MetaDict]:
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
        Minimum ring radius in normalised units (default: 1e-3).
    """
    nx, ny, _ = geom["matrix"]
    fov_x_m = geom["fov_mm"][0] / 1000.0
    fov_y_m = geom["fov_mm"][1] / 1000.0
    kmax_x = (nx / 2.0) / fov_x_m
    kmax_y = (ny / 2.0) / fov_y_m

    n_rings = int(params.get("n_rings", ny // 2))
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
        theta_b = backend.asarray(theta)
        kx = backend.cos(theta_b) * r_norm * kmax_x
        ky = backend.sin(theta_b) * r_norm * kmax_y
        shot = backend.stack([kx, ky], axis=1)
        shots.append(shot)

    meta = {
        "trajectory_type":    "ECCENTRIC",
        "n_rings":            n_rings,
        "kmax":               (kmax_x, kmax_y),
        "fov_mm":             geom["fov_mm"][:2],
        "matrix":             (nx, ny),
        "n_shots":            n_rings,
        "samples_per_shot":   "variable (∝ ring circumference)",
        "density_estimate":   _density_estimate_placeholder("ECCENTRIC", n_rings,
                                                             ny // 2, geom),
        "params_used":        params,
    }
    return shots, meta


def _traj_concentric_rings_2d(geom: dict, params: dict, backend: Backend) -> Tuple[List[Array], MetaDict]:
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
    shots, meta_eccentric = _traj_eccentric(geom, params, backend)
    # Re-label trajectory type for clarity
    meta_eccentric["trajectory_type"] = "concentric_rings_2d"
    meta_eccentric["n_shots"] = len(shots)
    meta_eccentric["density_estimate"]["trajectory"] = "concentric_rings_2d"
    return shots, meta_eccentric


def _traj_3d_phyllotaxis(geom: dict, params: dict, backend: Backend) -> Tuple[List[Array], MetaDict]:
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
    r = backend.linspace(0.0, 1.0, sps)   # center-out
    for i in range(n_shots):
        cos_el = 1.0 - (2 * i + 1.0) / n_shots
        cos_el = float(np.clip(cos_el, -1.0, 1.0))
        sin_el = math.sqrt(max(0.0, 1.0 - cos_el ** 2))
        azimuth = i * GOLDEN_ANGLE_RAD    # 2D golden angle for azimuth

        dx = sin_el * math.cos(azimuth)
        dy = sin_el * math.sin(azimuth)
        dz = cos_el

        kx = r * dx * kmax_x
        ky = r * dy * kmax_y
        kz = r * dz * kmax_z
        shot = backend.stack([kx, ky, kz], axis=1)
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
        "density_estimate":   _density_estimate_placeholder("3d_phyllotaxis", n_shots, sps, geom),
        "params_used":        params,
    }
    return shots, meta


def _traj_cones_3d(geom: dict, params: dict, backend: Backend) -> Tuple[List[Array], MetaDict]:
    """
    3D cones trajectory: spiral arms arranged on cones at discrete polar angles.

    Each shot is a spiral arm on a cone surface. Cones are indexed by their
    polar angle θ measured from the +kz axis.

    By default the polar range is [0, π], covering both the upper (+kz) and
    lower (−kz) hemispheres symmetrically.  Set ``polar_max=π/2`` to restrict
    to the upper hemisphere only (legacy behaviour).

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

    n_cones       = int(params.get("n_cones", 14))
    arms_per_cone = int(params.get("arms_per_cone", 8))
    sps           = int(params.get("samples_per_shot", max(nx, ny, nz)))
    turns         = float(params.get("spiral_turns_per_arm", 1.0))
    polar_min     = float(params.get("polar_min", 0.15*math.pi))#0.0))
    polar_max     = float(params.get("polar_max", 0.85*math.pi))

    cone_angles = np.linspace(polar_min, polar_max, n_cones, True)
    t = backend.linspace(0.0, 1.0, sps)

    shots = []
    for ci, cone_ang in enumerate(cone_angles):
        sin_ca = math.sin(float(cone_ang))
        cos_ca = math.cos(float(cone_ang))   # negative for θ > π/2 → −kz cones
        for ai in range(arms_per_cone):
            az0 = ai * 2.0 * math.pi / arms_per_cone + ci * GOLDEN_ANGLE_RAD
            theta_spiral = t * (2.0 * math.pi * turns) + az0
            r = t  # centre-out

            kx = r * backend.cos(theta_spiral) * sin_ca * kmax_x
            ky = r * backend.sin(theta_spiral) * sin_ca * kmax_y
            kz = r * cos_ca * kmax_z          # naturally negative when θ > π/2
            shot = backend.stack([kx, ky, kz], axis=1)
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
        "density_estimate":   _density_estimate_placeholder("cones_3d", n_shots, sps, geom),
        "params_used":        params,
    }
    return shots, meta


def _traj_cones_3d_rosette(geom: dict, params: dict, backend: Backend) -> Tuple[List[Array], MetaDict]:
    """
    3D Cones Rosette trajectory.

    Replaces the spiral readout arms of ``cones_3d`` with rosette petals.
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

    n_cones   = int(params.get("n_cones", 14))
    ppc       = int(params.get("petals_per_cone", 9))
    sps       = int(params.get("samples_per_shot", 512))
    polar_min = float(params.get("polar_min", 0.15*math.pi))#0.0))
    polar_max = float(params.get("polar_max", 0.85*math.pi))

    cone_angles = np.linspace(polar_min, polar_max, n_cones, True)
    t     = backend.linspace(0.0, math.pi, sps)   # petal: 0 → π
    sin_t = backend.sin(t)                         # envelope — 0 at both ends

    shots = []
    for ci, theta_cone in enumerate(cone_angles):
        sin_ca = math.sin(float(theta_cone))
        cos_ca = math.cos(float(theta_cone))   # negative for θ > π/2 → −kz petals
        for pi_ in range(ppc):
            phi = pi_ * 2.0 * math.pi / ppc + ci * GOLDEN_ANGLE_RAD
            kx = sin_t * sin_ca * backend.cos(t + phi) * kmax_x
            ky = sin_t * sin_ca * backend.sin(t + phi) * kmax_y
            kz = sin_t * cos_ca * kmax_z       # sign follows cos(θ) naturally
            shot = backend.stack([kx, ky, kz], axis=1)
            shots.append(shot)

    n_shots = len(shots)
    kmax_xv, kmax_yv, kmax_zv = _kmax_from_geometry(geom)
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
        "density_estimate":   _density_estimate_placeholder(
                                  "cones_3d_rosette", n_shots, sps, geom),
        "params_used":        params,
    }
    return shots, meta


def _traj_floret_3d(geom: dict, params: dict, backend: Backend) -> Tuple[List[Array], MetaDict]:
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

    n_hubs = int(params.get("n_hubs", 5))
    aph = int(params.get("arms_per_hub", 15))
    sps = int(params.get("samples_per_shot", max(nx, ny, nz)))
    turns = float(params.get("spiral_turns", aph / 2.0))
    hub_ang_deg = float(params.get("hub_cone_angle_deg", 35.0))
    hub_ang_rad = math.radians(hub_ang_deg)

    t = backend.linspace(0.0, 1.0, sps)

    # Hub axis directions via phyllotaxis
    shots = []
    for h in range(n_hubs):
        cos_el_hub = 1.0 - (2 * h + 1.0) / n_hubs
        cos_el_hub = float(np.clip(cos_el_hub, -1.0, 1.0))
        sin_el_hub = math.sqrt(max(0.0, 1.0 - cos_el_hub ** 2))
        az_hub = h * GOLDEN_ANGLE_RAD

        # Hub rotation matrix columns (local x, y axes on hub cone)
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
            az0 = a * 2.0 * math.pi / aph
            theta = t * (2.0 * math.pi * turns) + az0
            r = t
            # Arm direction sweeps within the hub cone
            arm_vec_x = r * backend.cos(theta) * math.sin(hub_ang_rad)
            arm_vec_y = r * backend.sin(theta) * math.sin(hub_ang_rad)
            arm_vec_z = r * math.cos(hub_ang_rad)

            # Rotate from hub local frame to global
            kx = (arm_vec_x * local_x[0] + arm_vec_y * local_y[0]
                  + arm_vec_z * hub_axis[0]) * kmax_x
            ky = (arm_vec_x * local_x[1] + arm_vec_y * local_y[1]
                  + arm_vec_z * hub_axis[1]) * kmax_y
            kz = (arm_vec_x * local_x[2] + arm_vec_y * local_y[2]
                  + arm_vec_z * hub_axis[2]) * kmax_z
            shot = backend.stack([kx, ky, kz], axis=1)
            shots.append(shot)

    n_shots = len(shots)
    meta = {
        "trajectory_type":    "floret_3d",
        "n_hubs":             n_hubs,
        "arms_per_hub":       aph,
        "hub_cone_angle_deg": hub_ang_deg,
        "spiral_turns":       turns,
        "kmax":               (kmax_x, kmax_y, kmax_z),
        "fov_mm":             geom["fov_mm"],
        "matrix":             (nx, ny, nz),
        "n_shots":            n_shots,
        "samples_per_shot":   sps,
        "density_estimate":   _density_estimate_placeholder("floret_3d", n_shots, sps, geom),
        "params_used":        params,
    }
    return shots, meta


def _traj_3d_egg_rosette(geom: dict, params: dict, backend: Backend) -> Tuple[List[Array], MetaDict]:
    """
    3D Egg-Shaped Rosette trajectory (Strasser, Medical University of Vienna).

    Reference: Strasser B et al., "Ultra-high resolution whole-brain MR spectroscopic
    imaging using a 3D egg-shaped rosette k-space trajectory", MRM 2017 / ISMRM abstracts.

    Each shot traces a closed egg-shaped loop on the surface of the k-space sphere:

        kx(t) = kmax_x · sin(ω₁·t) · cos(ω₂·t + φₙ)
        ky(t) = kmax_y · sin(ω₁·t) · sin(ω₂·t + φₙ)
        kz(t) = kmax_z · cos(ω₁·t)

    where t ∈ [0, 2π] traces one complete egg-shaped loop:
      - ω₁ = 1 (polar oscillation; kz sweeps north → south → north once per shot)
      - ω₂ = azimuthal wrap frequency (number of azimuthal lobes per shot)
      - φₙ = n · golden_angle_rad — per-shot azimuthal phase offset ensuring that
        any prefix of N shots gives near-uniform sphere coverage.

    The "egg" geometry arises from the interplay of sin(ω₁t) (which vanishes at both
    poles) and cos(ω₁t) (which drives kz); the result is a lemniscate-like curve on
    the sphere whose half-width at the equator equals kmax and whose lobes are
    distributed uniformly in azimuth via the golden-angle increment between shots.

    Parameters (params)
    -------------------
    n_shots : int
        Number of egg-rosette shots (default: 20).
        Any prefix gives uniform angular coverage thanks to golden-angle ordering.
    samples_per_shot : int
        Number of samples per shot (default: 4 * max(nx, ny, nz)).
    omega1 : float
        Polar frequency (default 1 → one polar oscillation per shot).
        Increase to add multiple north–south loops per trajectory.
    omega2 : int
        Azimuthal frequency — number of lobes per egg (default 5).
        Best chosen as an odd integer coprime with omega1 for closed trajectories.
    rotation_per_shot : float
        Azimuthal phase increment between shots in radians
        (default: GOLDEN_ANGLE_RAD ≈ 2.3999 rad).

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

    n_shots = int(params.get("n_shots", 20))
    sps = int(params.get("samples_per_shot", 4 * max(nx, ny, nz)))
    omega1 = float(params.get("omega1", 1.0))
    omega2 = float(params.get("omega2", 5.0))
    rot_per_shot = float(params.get("rotation_per_shot", GOLDEN_ANGLE_RAD))

    # t ∈ [0, 2π] → one complete polar oscillation per shot
    t = backend.linspace(0.0, 2.0 * math.pi, sps)

    shots = []
    for n in range(n_shots):
        phi_n = n * rot_per_shot                                   # golden-angle azimuth offset
        kx = backend.sin(t * omega1) * backend.cos(t * omega2 + phi_n) * kmax_x
        ky = backend.sin(t * omega1) * backend.sin(t * omega2 + phi_n) * kmax_y
        kz = backend.cos(t * omega1) * kmax_z                     # kz independent of phi_n
        shot = backend.stack([kx, ky, kz], axis=1)                # (sps, 3)
        shots.append(shot)

    meta = {
        "trajectory_type":    "3d_egg_rosette",
        "omega1":             omega1,
        "omega2":             omega2,
        "rotation_per_shot":  rot_per_shot,
        "kmax":               (kmax_x, kmax_y, kmax_z),
        "fov_mm":             geom["fov_mm"],
        "matrix":             (nx, ny, nz),
        "n_shots":            n_shots,
        "samples_per_shot":   sps,
        "reference":          "Strasser et al., MedUni Vienna",
        "density_estimate":   _density_estimate_placeholder(
                                  "3d_egg_rosette", n_shots, sps, geom),
        "params_used":        params,
    }
    return shots, meta

_TRAJECTORY_GENERATORS = {
    "cartesian_2d":         lambda g, p, b: _traj_cartesian_2d(g, p, b),
    "cartesian_3d":         lambda g, p, b: _traj_cartesian_3d(g, p, b),
    "radial_2d":            lambda g, p, b: _traj_radial_2d(g, p, b, use_golden=False),
    "golden_radial_2d":     lambda g, p, b: _traj_radial_2d(g, p, b, use_golden=True),
    "spiral_2d":            lambda g, p, b: _traj_spiral_2d(g, p, b),
    "rosette_2d":           lambda g, p, b: _traj_rosette_2d(g, p, b),
    "rosette_2d_petals":    lambda g, p, b: _traj_rosette_2d_petals(g, p, b),
    "concentric_rings_2d":  lambda g, p, b: _traj_concentric_rings_2d(g, p, b),
    "stack_of_stars":       lambda g, p, b: _traj_stack_of_x(g, p, b, "golden_radial"),
    "stack_of_spirals":     lambda g, p, b: _traj_stack_of_x(g, p, b, "spiral"),
    "stack_of_cones":       lambda g, p, b: _traj_stack_of_x(g, p, b, "cones"),
    "stack_of_rosettes":    lambda g, p, b: _traj_stack_of_x(g, p, b, "rosette"),
    "stack_of_ECCENTRIC":   lambda g, p, b: _traj_stack_of_x(g, p, b, "ECCENTRIC"),
    "3d_phyllotaxis":       lambda g, p, b: _traj_3d_phyllotaxis(g, p, b),
    "cones_3d":             lambda g, p, b: _traj_cones_3d(g, p, b),
    "cones_3d_rosette":     lambda g, p, b: _traj_cones_3d_rosette(g, p, b),
    "floret_3d":            lambda g, p, b: _traj_floret_3d(g, p, b),
    "3d_egg_rosette":       lambda g, p, b: _traj_3d_egg_rosette(g, p, b),
}


# ---------------------------------------------------------------------------
# Public API: generate_trajectory
# ---------------------------------------------------------------------------

def generate_trajectory(
    name: str,
    header: HeaderDict,
    params: ParamsDict,
    backend: Backend,
) -> Tuple[List[Array], MetaDict]:
    """
    Generate a k-space trajectory and return shot-organised coordinates.

    Parameters
    ----------
    name : str
        Trajectory name. One of:
        'cartesian_2d', 'cartesian_3d', 'radial_2d', 'golden_radial_2d',
        'spiral_2d', 'rosette_2d', 'stack_of_stars', 'stack_of_spirals',
        'stack_of_cones', 'stack_of_ECCENTRIC',
        '3d_phyllotaxis', 'cones_3d', 'floret_3d'.
    header : dict
        Parsed NIfTI-MRS header (passed to read_header_geometry internally).
    params : dict
        Trajectory-specific parameters (see individual trajectory docstrings).
    backend : Backend
        NumpyBackend() or TorchBackend() instance.

    Returns
    -------
    coords_per_shot : list of arrays, each shape (Nsamples, Ndims)
        Ndims = 2 for 2D trajectories, 3 for 3D. All values in cycles/m.
    meta : dict
        trajectory_type, kmax (per-axis tuple), fov_mm, matrix, n_shots,
        samples_per_shot, density_estimate placeholder, params_used.

    Raises
    ------
    KeyError
        If required header fields are missing.
    ValueError
        If trajectory name is unknown or geometry is incompatible.
    TypeError
        If backend is missing required methods.
    """
    _validate_backend(backend)
    if name not in _TRAJECTORY_GENERATORS:
        raise ValueError(
            f"Unknown trajectory '{name}'. Valid trajectories: "
            f"{sorted(_TRAJECTORY_GENERATORS.keys())}."
        )
    geom = read_header_geometry(header)
    if geom["inferred"]:
        warnings.warn(
            f"Some geometry fields were inferred: {geom['inferred']}. "
            "Verify header completeness for accurate k-space scaling.",
            UserWarning,
            stacklevel=2,
        )
    return _TRAJECTORY_GENERATORS[name](geom, params, backend)


# ---------------------------------------------------------------------------
# Public API: shots_from_continuous
# ---------------------------------------------------------------------------

def shots_from_continuous(
    coords: Array,
    segmentation: Dict[str, Any],
    backend: Backend,
) -> List[Array]:
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
    backend : Backend

    Returns
    -------
    list of arrays, each shape (Nsamples_i, Ndims)

    Raises
    ------
    ValueError
        If segmentation dict lacks required keys or lengths are inconsistent.
    """
    _validate_backend(backend)
    total = backend.shape(coords)[0]
    cpu_coords = backend.to_cpu(coords)

    if "n_shots" in segmentation:
        n = int(segmentation["n_shots"])
        if total % n != 0:
            raise ValueError(
                f"'n_shots'={n} does not evenly divide total samples={total}. "
                "Use 'shot_lengths' or 'shot_boundaries' for unequal shots."
            )
        sps = total // n
        shots = [backend.asarray(cpu_coords[i * sps: (i + 1) * sps]) for i in range(n)]

    elif "shot_lengths" in segmentation:
        lengths = list(segmentation["shot_lengths"])
        if sum(lengths) != total:
            raise ValueError(
                f"Sum of shot_lengths ({sum(lengths)}) != total samples ({total})."
            )
        shots = []
        start = 0
        for length in lengths:
            shots.append(backend.asarray(cpu_coords[start: start + length]))
            start += length

    elif "shot_boundaries" in segmentation:
        bounds = list(segmentation["shot_boundaries"]) + [total]
        shots = []
        for s, e in zip(bounds[:-1], bounds[1:]):
            shots.append(backend.asarray(cpu_coords[s:e]))

    else:
        raise ValueError(
            "segmentation must contain 'n_shots', 'shot_lengths', or 'shot_boundaries'. "
            f"Got keys: {list(segmentation.keys())}."
        )
    return shots


# ---------------------------------------------------------------------------
# Undersampling Strategies (internal helpers)
# ---------------------------------------------------------------------------

def _us_prefix(n_shots: int, af: float, params: dict) -> np.ndarray:
    """Keep first ceil(n_shots / AF) shots."""
    n_keep = math.ceil(n_shots / af)
    mask = np.zeros(n_shots, dtype=bool)
    mask[:n_keep] = True
    return mask


def _us_golden_prefix(n_shots: int, af: float, params: dict) -> np.ndarray:
    """
    Equivalent to prefix for golden-angle sorted trajectories.
    Alias of prefix; golden-angle ordering guarantees spatial uniformity.
    """
    return _us_prefix(n_shots, af, params)


def _us_drop_every(n_shots: int, af: float, params: dict) -> np.ndarray:
    """Drop every M-th shot. M = round(AF) by default or params['drop_every_m']."""
    m = int(params.get("drop_every_m", max(1, round(af))))
    mask = np.ones(n_shots, dtype=bool)
    mask[::m] = False
    return mask


def _us_random_vd(n_shots: int, af: float, params: dict,
                  coords_per_shot: List[Array], backend: Backend) -> np.ndarray:
    """
    Variable-density random undersampling.
    Centre shots (lower norm) are kept with higher probability.

    params
    ------
    vd_beta : float
        Density decay exponent. 0 = uniform random, >0 = centre-weighted (default 2).
    acs_fraction : float
        Fraction of shots nearest to k-centre always kept (default 0.05).
    """
    n_keep = math.ceil(n_shots / af)
    vd_beta = float(params.get("vd_beta", 2.0))
    acs_frac = float(params.get("acs_fraction", 0.05))

    # Compute shot centre norms (use numpy via to_cpu)
    norms = []
    for shot in coords_per_shot:
        cpu = backend.to_cpu(shot)
        norms.append(float(np.linalg.norm(cpu.mean(axis=0))))
    norms = np.array(norms, dtype=np.float32)

    # Normalise to [0,1]
    if norms.max() > 0:
        norms_n = norms / norms.max()
    else:
        norms_n = norms

    # Probability: centre gets high prob
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


def _us_keep_acs(n_shots: int, af: float, params: dict,
                 coords_per_shot: List[Array], backend: Backend,
                 traj_name: str) -> np.ndarray:
    """
    Keep shots whose centre lies within the ACS region (innermost fraction of k-space).

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
        cpu = backend.to_cpu(shot)
        norms.append(float(np.linalg.norm(cpu.mean(axis=0))))
    norms = np.array(norms, dtype=np.float32)

    if norms.max() > 0:
        norms_n = norms / norms.max()
    else:
        norms_n = np.zeros(n_shots, dtype=np.float32)

    if "acs_n_lines" in params and "cartesian" in traj_name:
        # Centre the selection
        acs_n = int(params["acs_n_lines"])
        centre_idx = np.argsort(norms_n)[:acs_n]
        mask[centre_idx] = True
    else:
        mask[norms_n <= acs_frac] = True

    if not mask.any():
        warnings.warn(
            "keep_acs found no shots inside ACS region. "
            f"Increase acs_radius_fraction (currently {acs_frac}) or acs_n_lines.",
            UserWarning, stacklevel=3,
        )
    return mask


def _us_poisson_disc_cartesian(n_shots: int, af: float, params: dict,
                               coords_per_shot: List[Array],
                               backend: Backend) -> np.ndarray:
    """
    Variable-density Poisson-disc undersampling for Cartesian trajectories.

    Full Poisson-disc is expensive; this uses a simplified VD-random approach
    with a centre-dense probability map as a practical stand-in compatible
    with MRI pipelines.

    params
    ------
    acs_lines : int
        Number of central lines always included (default: 24).
    vd_beta : float
        Density exponent (default: 3.0).
    seed : int
        RNG seed (default: 42).
    """
    acs_lines = int(params.get("acs_lines", 24))
    vd_beta = float(params.get("vd_beta", 3.0))
    seed = int(params.get("seed", 42))
    rng = np.random.default_rng(seed)

    # Phase encode index from shot centre
    norms = []
    for shot in coords_per_shot:
        cpu = backend.to_cpu(shot)
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


def _us_combined(n_shots: int, af: float, params: dict,
                 coords_per_shot: List[Array],
                 backend: Backend, traj_name: str) -> np.ndarray:
    """
    Combined strategy: keep ACS + prefix/random_vd on remaining shots.

    params
    ------
    acs_radius_fraction : float (default 0.15)
    outer_method : str  'prefix' | 'random_vd' | 'drop_every' (default 'prefix')
    outer_af : float  (default = af, applied to outer shots only)
    """
    acs_mask = _us_keep_acs(n_shots, af, params, coords_per_shot, backend, traj_name)
    outer_idx = np.where(~acs_mask)[0]
    outer_method = str(params.get("outer_method", "prefix"))
    outer_af = float(params.get("outer_af", af))

    outer_coords = [coords_per_shot[i] for i in outer_idx]
    n_outer = len(outer_idx)

    if outer_method == "prefix":
        outer_mask = _us_prefix(n_outer, outer_af, params)
    elif outer_method == "random_vd":
        outer_mask = _us_random_vd(n_outer, outer_af, params, outer_coords, backend)
    elif outer_method == "drop_every":
        outer_mask = _us_drop_every(n_outer, outer_af, params)
    else:
        raise ValueError(
            f"combined outer_method '{outer_method}' not recognised. "
            "Use 'prefix', 'random_vd', or 'drop_every'."
        )

    mask = acs_mask.copy()
    mask[outer_idx[outer_mask]] = True
    return mask


def _us_variable_density_spiral(n_shots: int, af: float, params: dict) -> np.ndarray:
    """
    Spiral undersampling by removing outer arms.
    Arms are assumed ordered centre-to-outer; keep first ceil(n/AF).
    """
    return _us_prefix(n_shots, af, params)


def _us_shell_based(n_shots: int, af: float, params: dict,
                    coords_per_shot: List[Array], backend: Backend) -> np.ndarray:
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

    norms = np.array([float(np.linalg.norm(backend.to_cpu(s).mean(axis=0)))
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


# ---------------------------------------------------------------------------
# Public API: undersample_shots
# ---------------------------------------------------------------------------

def undersample_shots(
    coords_per_shot: List[Array],
    method: str,
    AF: float,
    params: ParamsDict,
    backend: Backend,
    trajectory_name: str = "",
) -> Tuple[Array, Optional[List[Array]]]:
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
    AF : float
        Acceleration factor (≥ 1). Number of retained shots ≈ N / AF.
        Exact: N_keep = ceil(N_total / AF).
    params : dict
        Method-specific parameters (see each _us_* helper for keys).
    backend : Backend
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
        If method is not supported, AF < 1, or trajectory/method incompatible.
    TypeError
        If backend is missing required methods.
    """
    _validate_backend(backend)

    if AF < 1.0:
        raise ValueError(
            f"AF must be ≥ 1.0 (got {AF}). "
            "AF < 1 would require more shots than fully sampled — not valid."
        )

    n_shots = len(coords_per_shot)
    if n_shots == 0:
        raise ValueError("coords_per_shot is empty — no shots to undersample.")

    # Compatibility check
    if trajectory_name:
        compat = _TRAJ_US_COMPAT.get(trajectory_name, None)
        if compat is not None and method not in compat:
            raise ValueError(
                f"Undersampling method '{method}' is not compatible with "
                f"trajectory '{trajectory_name}'. "
                f"Supported methods for this trajectory: {compat}. "
                "Pass trajectory_name='' to bypass compatibility checking."
            )

    valid_methods = [
        "prefix", "golden_prefix", "drop_every", "poisson_disc_cartesian",
        "random_vd", "keep_acs", "variable_density_spiral", "shell_based", "combined",
    ]
    if method not in valid_methods:
        raise ValueError(
            f"Unknown undersampling method '{method}'. Valid: {valid_methods}."
        )

    traj_name = trajectory_name or ""

    if method == "prefix":
        mask_np = _us_prefix(n_shots, AF, params)
    elif method == "golden_prefix":
        mask_np = _us_golden_prefix(n_shots, AF, params)
    elif method == "drop_every":
        mask_np = _us_drop_every(n_shots, AF, params)
    elif method == "poisson_disc_cartesian":
        mask_np = _us_poisson_disc_cartesian(n_shots, AF, params, coords_per_shot, backend)
    elif method == "random_vd":
        mask_np = _us_random_vd(n_shots, AF, params, coords_per_shot, backend)
    elif method == "keep_acs":
        mask_np = _us_keep_acs(n_shots, AF, params, coords_per_shot, backend, traj_name)
    elif method == "variable_density_spiral":
        mask_np = _us_variable_density_spiral(n_shots, AF, params)
    elif method == "shell_based":
        mask_np = _us_shell_based(n_shots, AF, params, coords_per_shot, backend)
    elif method == "combined":
        mask_np = _us_combined(n_shots, AF, params, coords_per_shot, backend, traj_name)

    achieved_af = n_shots / max(mask_np.sum(), 1)
    if abs(achieved_af - AF) > 0.5 * AF:
        warnings.warn(
            f"Achieved AF={achieved_af:.2f} differs substantially from requested AF={AF:.2f}. "
            "Consider adjusting 'drop_every_m', 'acs_lines', or 'acs_radius_fraction'.",
            UserWarning,
            stacklevel=2,
        )

    shot_mask = backend.cast_bool(backend.asarray(mask_np.astype(np.uint8)))
    return shot_mask, None   # per_sample_masks deferred → None


# ---------------------------------------------------------------------------
# Public API: get_kspace_shots_and_mask  (full pipeline wrapper)
# ---------------------------------------------------------------------------

def get_kspace_shots_and_mask(
    header: HeaderDict,
    trajectory: str,
    undersampling: str,
    AF: float,
    traj_params: ParamsDict,
    us_params: ParamsDict,
    backend: Backend,
) -> Tuple[List[Array], Array, MetaDict]:
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
    AF : float
        Acceleration factor (≥ 1).
    traj_params : dict
        Passed verbatim to generate_trajectory.
    us_params : dict
        Passed verbatim to undersample_shots.
    backend : Backend
        NumpyBackend() or TorchBackend() instance.

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
            n_shots_retained, achieved_AF, density_estimate, params_used,
            undersampling_method, AF_requested.

    Example
    -------
    >>> from kspace_sampling import get_kspace_shots_and_mask, NumpyBackend
    >>> header = {
    ...     "dim": [4, 64, 64, 1, 1, 1, 1, 1],
    ...     "pixdim": [1.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0],
    ...     "DwellTime": 1e-4,
    ...     "SpectrometerFrequency": [127.74e6],
    ... }
    >>> shots, mask, meta = get_kspace_shots_and_mask(
    ...     header, "golden_radial_2d", "prefix", AF=4.0,
    ...     traj_params={"n_shots": 200, "samples_per_shot": 64},
    ...     us_params={},
    ...     backend=NumpyBackend(),
    ... )
    >>> # shots: list of 200 arrays each (64, 2)
    >>> # mask: bool array (200,), ~50 True entries
    """
    _validate_backend(backend)

    coords_per_shot, meta = generate_trajectory(trajectory, header, traj_params, backend)
    shot_mask, per_sample_masks = undersample_shots(
        coords_per_shot, undersampling, AF, us_params, backend,
        trajectory_name=trajectory,
    )

    mask_np = backend.to_cpu(shot_mask).astype(bool)
    n_retained = int(mask_np.sum())
    achieved_af = len(coords_per_shot) / max(n_retained, 1)

    meta.update({
        "undersampling_method":  undersampling,
        "AF_requested":          AF,
        "n_shots_retained":      n_retained,
        "achieved_AF":           achieved_af,
        "us_params_used":        us_params,
    })

    return coords_per_shot, shot_mask, meta


# ---------------------------------------------------------------------------
# Simulation Hooks (signatures only; brief/optional implementations)
# ---------------------------------------------------------------------------

def apply_shot_phase_errors(
    coords_per_shot: List[Array],
    phase_std_rad: float,
    backend: Backend,
    seed: int = 0,
) -> List[Array]:
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
    backend : Backend
    seed : int
        RNG seed.

    Returns
    -------
    list of float
        One phase offset (radians) per shot.
    """
    rng = np.random.default_rng(seed)
    return [float(rng.normal(0.0, phase_std_rad)) for _ in coords_per_shot]


def apply_gradient_delay(
    coords_per_shot: List[Array],
    delay_samples: float,
    backend: Backend,
) -> List[Array]:
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
    backend : Backend

    Returns
    -------
    list of arrays (same shapes), shifted along readout.
    """
    shifted = []
    for shot in coords_per_shot:
        cpu = backend.to_cpu(shot)
        if cpu.shape[0] < 2:
            shifted.append(shot)
            continue
        dk = cpu[1] - cpu[0]   # per-sample k-space increment (Ndims,)
        shift = dk * delay_samples
        shifted_cpu = cpu + shift[np.newaxis, :]
        shifted.append(backend.asarray(shifted_cpu))
    return shifted


def apply_off_resonance_phase(
    coords_per_shot: List[Array],
    freq_offset_hz: float,
    dwell_time_s: float,
    backend: Backend,
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
    backend : Backend

    Returns
    -------
    list of arrays, each (Nsamples,) — phase in radians.
    """
    phases = []
    for shot in coords_per_shot:
        nsamples = backend.shape(shot)[0]
        t = backend.arange(0, nsamples) * dwell_time_s
        phi = 2.0 * math.pi * freq_offset_hz * t
        phases.append(phi)
    return phases


# ---------------------------------------------------------------------------
# Density Estimate (public wrapper)
# ---------------------------------------------------------------------------

def compute_density_estimate(
    coords_per_shot: List[Array],
    shot_mask: Array,
    meta: MetaDict,
    backend: Backend,
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
    backend : Backend

    Returns
    -------
    dict
        Updated density estimate dict appropriate for the retained subset.
    """
    mask_np = backend.to_cpu(shot_mask).astype(bool)
    retained = [s for s, m in zip(coords_per_shot, mask_np) if m]
    n_retained = len(retained)
    sps = backend.shape(retained[0])[0] if retained else 0

    base = meta.get("density_estimate", {}).copy()
    base.update({
        "n_shots_subset":          n_retained,
        "samples_per_shot_subset": sps,
        "total_samples_subset":    n_retained * sps,
        "achieved_AF":             meta.get("n_shots", n_retained) / max(n_retained, 1),
        "recomputed_for_subset":   True,
    })
    return base


# ---------------------------------------------------------------------------
# Utility: export shots and mask to .npz
# ---------------------------------------------------------------------------

def save_shots_npz(
    path: str,
    coords_per_shot: List[Array],
    shot_mask: Array,
    meta: MetaDict,
    backend: Backend,
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
    backend : Backend
    """
    payload: dict = {}
    payload["shot_mask"] = backend.to_cpu(shot_mask)
    for i, shot in enumerate(coords_per_shot):
        payload[f"coords_{i:06d}"] = backend.to_cpu(shot).astype(np.float32)
    # Save select meta fields as scalars
    for key in ["trajectory_type", "n_shots", "n_shots_retained",
                "achieved_AF", "AF_requested", "undersampling_method"]:
        if key in meta:
            payload[f"meta_{key}"] = np.array(str(meta[key]))
    kmax = meta.get("kmax")
    if kmax:
        payload["meta_kmax"] = np.array(kmax, dtype=np.float32)
    np.savez_compressed(path, **payload)
    print(f"Saved {len(coords_per_shot)} shots + mask → '{path}'")


def load_shots_npz(path: str, backend: Backend) -> Tuple[List[Array], Array, dict]:
    """
    Load a .npz archive saved by save_shots_npz.

    Returns
    -------
    coords_per_shot, shot_mask, meta_partial
    """
    data = np.load(path, allow_pickle=False)
    shot_mask = backend.cast_bool(backend.asarray(data["shot_mask"].astype(np.uint8)))

    shot_keys = sorted(k for k in data.files if k.startswith("coords_"))
    coords_per_shot = [backend.asarray(data[k]) for k in shot_keys]

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


# ---------------------------------------------------------------------------
# Self-test / example block
# ---------------------------------------------------------------------------

def _run_selftest() -> None:
    """
    Self-test: exercise the main pipeline with a synthetic 2D golden-radial
    trajectory and prefix undersampling (AF=4).

    Test description (also serves as usage example):
    ------------------------------------------------
    1. Create a synthetic NIfTI-MRS-style header dict with keys:
       'dim', 'pixdim', 'DwellTime', 'SpectrometerFrequency'.
    2. Instantiate NumpyBackend.
    3. Call get_kspace_shots_and_mask with trajectory='golden_radial_2d',
       undersampling='prefix', AF=4.0, n_shots=200, samples_per_shot=64.
    4. Assert: len(coords_per_shot)==200, each shape (64,2), shot_mask shape==(200,),
       sum(shot_mask)==ceil(200/4)==50.
    5. Save to /tmp/kspace_test.npz and reload — verify round-trip shape.
    6. (Optional) Swap to TorchBackend: backend = TorchBackend(device='cpu')
       and rerun; all returned objects will be torch.Tensor.

    To swap to TorchBackend verbally:
        Replace NumpyBackend() with TorchBackend() (or TorchBackend(device='cuda')).
        All coordinate arrays and the shot_mask will be torch.Tensor objects
        on the specified device, fully compatible with NUFFT libraries like
        torchkbnufft or sigpy.
    """
    print("=" * 60)
    print("kspace_sampling self-test")
    print("=" * 60)

    # ----- Synthetic NIfTI-MRS header -----
    header = {
        "dim":    [4, 64, 64, 1, 1, 1, 1, 1],   # dim[1..3] = 64×64×1
        "pixdim": [1.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0],  # pixdim[1..3] = 3×3×3 mm
        "DwellTime": 1e-4,                         # 100 µs dwell
        "SpectrometerFrequency": [127.74e6],        # 3T 1H
    }

    backend = NumpyBackend()
    print(f"Backend: {backend}")

    # ----- Geometry parsing -----
    geom = read_header_geometry(header)
    print(f"  Geometry: fov_mm={geom['fov_mm']}, matrix={geom['matrix']}, "
          f"voxel_mm={geom['voxel_mm']}, ndim={geom['ndim']}")

    # ----- Trajectory generation -----
    traj_params = {"n_shots": 200, "samples_per_shot": 64, "center_out": True}
    shots, mask, meta = get_kspace_shots_and_mask(
        header, "golden_radial_2d", "prefix", AF=4.0,
        traj_params=traj_params, us_params={}, backend=backend,
    )

    n_shots = len(shots)
    sps = backend.shape(shots[0])[0]
    ndims = backend.shape(shots[0])[1]
    mask_np = backend.to_cpu(mask).astype(bool)
    n_retained = int(mask_np.sum())
    expected_retained = math.ceil(n_shots / 4.0)

    print(f"  n_shots: {n_shots} (expected 200)")
    print(f"  samples_per_shot: {sps} (expected 64)")
    print(f"  Ndims: {ndims} (expected 2)")
    print(f"  shot_mask shape: {mask_np.shape} (expected (200,))")
    print(f"  n_retained: {n_retained} (expected {expected_retained})")
    print(f"  kmax: {meta['kmax']}")
    print(f"  trajectory_type: {meta['trajectory_type']}")

    assert n_shots == 200,             f"Expected 200 shots, got {n_shots}"
    assert sps == 64,                  f"Expected 64 samples/shot, got {sps}"
    assert ndims == 2,                 f"Expected 2D, got Ndims={ndims}"
    assert mask_np.shape == (200,),    f"shot_mask shape mismatch: {mask_np.shape}"
    assert n_retained == expected_retained, \
        f"n_retained={n_retained} != expected {expected_retained}"

    # ----- Round-trip .npz -----
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "kspace_test.npz")
        save_shots_npz(fpath, shots, mask, meta, backend)
        shots2, mask2, meta2 = load_shots_npz(fpath, backend)
        assert len(shots2) == 200
        assert backend.shape(shots2[0]) == (64, 2)
        assert backend.to_cpu(mask2).astype(bool).sum() == n_retained
        print(f"  Round-trip .npz: OK (loaded {len(shots2)} shots)")

    # ----- 3D phyllotaxis quick smoke test -----
    header3d = {
        "dim":    [4, 32, 32, 32, 1, 1, 1, 1],
        "pixdim": [1.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0],
        "DwellTime": 1e-4,
    }
    shots3d, mask3d, meta3d = get_kspace_shots_and_mask(
        header3d, "3d_phyllotaxis", "prefix", AF=2.0,
        traj_params={"n_shots": 100, "samples_per_shot": 32},
        us_params={}, backend=backend,
    )
    assert backend.shape(shots3d[0])[1] == 3, "Expected 3D coordinates"
    print(f"  3D phyllotaxis: {len(shots3d)} shots, "
          f"shape per shot: {backend.shape(shots3d[0])}, OK")

    print("All self-tests PASSED.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Usage example (plain text at module end)
# ---------------------------------------------------------------------------
#
# USAGE EXAMPLE
# =============
#
# Expected header dict keys (NIfTI-MRS standard + common fallbacks):
# ------------------------------------------------------------------
#   Required (standard):
#     "dim"    : list[int], dim[1..3] = [nx, ny, nz]
#     "pixdim" : list[float], pixdim[1..3] = [vx, vy, vz] in mm
#   Optional standard fields:
#     "SpectrometerFrequency": [float]   — transmitter frequency in Hz
#     "DwellTime"             : float    — ADC dwell time in seconds
#     "qform_code"            : int      — qform validity flag
#     "sform_code"            : int      — sform validity flag
#   Fallback (non-standard, accepted if standard absent):
#     "matrix"       : [nx, ny, nz]
#     "voxel_size_mm": [vx, vy, vz]
#     "fov_mm"       : [fx, fy, fz]
#     "dwell_time"   : float
#
# Calling get_kspace_shots_and_mask:
# -----------------------------------
#   from kspace_sampling import get_kspace_shots_and_mask, NumpyBackend, TorchBackend
#
#   header = {
#       "dim":    [4, 128, 128, 1, 1, 1, 1, 1],
#       "pixdim": [1.0, 2.5, 2.5, 2.5, 1.0, 1.0, 1.0, 1.0],
#       "DwellTime": 4e-6,
#       "SpectrometerFrequency": [297.2e6],
#   }
#
#   # NumPy backend
#   backend = NumpyBackend()
#
#   coords_per_shot, shot_mask, meta = get_kspace_shots_and_mask(
#       header          = header,
#       trajectory      = "golden_radial_2d",
#       undersampling   = "prefix",
#       AF              = 4.0,
#       traj_params     = {"n_shots": 400, "samples_per_shot": 128},
#       us_params       = {},
#       backend         = backend,
#   )
#
#   # Outputs:
#   #   coords_per_shot : list of 400 numpy arrays, each shape (128, 2)
#   #                     coordinates in cycles/m, kx and ky columns
#   #   shot_mask       : numpy bool array shape (400,), 100 True entries (AF=4)
#   #   meta            : dict with keys:
#   #       trajectory_type="golden_radial_2d", kmax=(kmax_x, kmax_y),
#   #       fov_mm=(320.0, 320.0), matrix=(128,128),
#   #       n_shots=400, n_shots_retained=100, achieved_AF≈4.0,
#   #       density_estimate={...placeholder...}, params_used={...}
#
#   # Select retained shots:
#   retained = [s for s, m in zip(coords_per_shot, shot_mask) if m]
#
#   # Switch to PyTorch (CUDA):
#   backend_gpu = TorchBackend(device="cuda")
#   coords_pt, mask_pt, meta_pt = get_kspace_shots_and_mask(
#       header, "golden_radial_2d", "prefix", AF=4.0,
#       traj_params={"n_shots": 400, "samples_per_shot": 128},
#       us_params={}, backend=backend_gpu,
#   )
#   # coords_pt[0] is a torch.Tensor on cuda, shape (128, 2)
#   # mask_pt      is a torch.BoolTensor on cuda, shape (400,)
#
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _run_selftest()
