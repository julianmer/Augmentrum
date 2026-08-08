"""
tests/test_readme_table.py  — Backend Support Auto-Discovery + Speed Table
===========================================================================
Tries to run every module on every backend and records:

    ✓  native  — runs natively (SUPPORTED_BACKENDS ∋ backend or SUPPORTED_BACKENDS=[])
    ~  auto    — runs via base-class NIfTI-list routing / implicit conversion
    ✗  fail    — crashed (note says why; marked xfail, not a hard failure)
    —  n/a     — framework not installed / explicitly unsupported

At the END two summary tables are printed (use -s flag to see them):
    1. Support table   (✓ / ~ / ✗ / —)
    2. Speed table     (mean ms per call over N_REPS runs)

Adding a new module:  append one ModuleEntry to _REGISTRY below.  Done.

Run:
    mamba activate mine
    pytest tests/test_readme_table.py -v -s

Skip the final summary:
    pytest tests/test_readme_table.py -v -s -k "not summary"
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

from nifti_mrs_plus import NIfTI_MRS_Plus, Backend
from augmentrum.core.base_module import BaseModule
from tests.module_specs import SPECS

# ── augmentation imports ──────────────────────────────────────────────────────
from augmentrum.augmentation import (
    AmplitudeScaling, Apodization, ArtificialPeaks, BaselineAugmentation,
    EddyCurrent, FrequencyShift, GaussianNoise, LineBroadening, PhaseShift,
    ResidualWater, SpuriousEchoes, SpatialAugmentations, ZeroFill,
)
from augmentrum.processing.nifti_raw_processor import NIfTI_RawProcessor
from augmentrum.sampling.coil_sampling import CoilSampler

# ── optional frameworks ───────────────────────────────────────────────────────
try:
    import torch;          TORCH_AVAILABLE = True
except ImportError:        TORCH_AVAILABLE = False

try:
    import tensorflow as tf; TF_AVAILABLE = True
except ImportError:          TF_AVAILABLE = False

try:
    import jax;            JAX_AVAILABLE = True
except ImportError:        JAX_AVAILABLE = False

try:
    import keras;          KERAS_AVAILABLE = True
except ImportError:        KERAS_AVAILABLE = False

try:
    import torchkbnufft;   TORCHKBNUFFT_AVAILABLE = True
except ImportError:        TORCHKBNUFFT_AVAILABLE = False

# ── outcome symbols ───────────────────────────────────────────────────────────
NATIVE = "✓"
AUTO   = "~"
FAIL   = "✗"
NA     = "—"

# ── backends under test ───────────────────────────────────────────────────────
NIFTI   = Backend.NIFTI_LIST
NUMPY   = Backend.NUMPY
PYTORCH = Backend.PYTORCH
TF      = Backend.TENSORFLOW
JAX     = Backend.JAX
KERAS   = Backend.KERAS

_ALL_BACKENDS = [NIFTI, NUMPY, PYTORCH, TF, JAX, KERAS]

_FRAMEWORK_GUARD: dict[Backend, tuple[bool, str]] = {
    PYTORCH: (TORCH_AVAILABLE,  "torch"),
    TF:      (TF_AVAILABLE,     "tensorflow"),
    JAX:     (JAX_AVAILABLE,    "jax"),
    KERAS:   (KERAS_AVAILABLE,  "keras"),
}

# ── data parameters ───────────────────────────────────────────────────────────
N_SUBJ    = 3
N_PTS     = 512
DWELLTIME = 1 / 2000.0
SPEC_FREQ = 123.0
N_COILS   = 4
N_AVG     = 8
N_REPS    = 3        # repetitions for mean speed measurement


#**************************************************************************************************#
#                                        Class ModuleEntry                                         #
#**************************************************************************************************#
#                                                                                                  #
# One row in the module registry.                                                                  #
#                                                                                                  #
#**************************************************************************************************#
@dataclass
class ModuleEntry:
    """One row in the module registry.

    Args:
        name:             Display name / pytest ID.
        factory:          Zero-arg callable returning a fresh module instance.
        note:             Free-text note shown in the summary for known issues.
        needs_multicoil:  Use multi-coil NIfTI fixture (needs DIM_COIL + DIM_DYN).
        spatial:          Special-case: 2-D image-tensor module (SpatialAugmentations).
        volume:           Special-case: 5-D volume module (KspaceUndersampling).
    """
    name:            str
    factory:         Callable
    note:            str  = ""
    needs_multicoil: bool = False
    spatial:         bool = False
    volume:          bool = False


#****************************************************#
#   ★  module registry  ★   ← add new modules here   #
#****************************************************#
_REGISTRY: list[ModuleEntry] = [
    ModuleEntry(spec.label, spec.factory, note=spec.note,
                needs_multicoil=spec.needs_multicoil, spatial=spec.spatial,
                volume=spec.volume)
    for spec in SPECS
]


#**************************************************************************************************#
#                                         Class CellResult                                         #
#**************************************************************************************************#
@dataclass
class CellResult:
    outcome:  str   = NA
    time_ms:  float = float("nan")
    note:     str   = ""
    error:    str   = ""


_RESULTS: dict[tuple[str, str], CellResult] = {}


#**************#
#   fixtures   #
#**************#
@pytest.fixture(scope="module")
def single_coil_nifti_list():
    rng = np.random.default_rng(42)
    out = []
    for _ in range(N_SUBJ):
        data = (rng.standard_normal((1, 1, 1, N_PTS))
                + 1j * rng.standard_normal((1, 1, 1, N_PTS))).astype(np.complex64)
        out.append(gen_nifti_mrs(data, dwelltime=DWELLTIME, spec_freq=SPEC_FREQ))
    return out


@pytest.fixture(scope="module")
def multi_coil_nifti_list():
    rng = np.random.default_rng(7)
    out = []
    for _ in range(N_SUBJ):
        data = (rng.standard_normal((1, 1, 1, N_PTS, N_COILS, N_AVG))
                + 1j * rng.standard_normal((1, 1, 1, N_PTS, N_COILS, N_AVG))).astype(np.complex64)
        n = gen_nifti_mrs(data, dwelltime=DWELLTIME, spec_freq=SPEC_FREQ)
        n.set_dim_tag(4, "DIM_COIL")
        n.set_dim_tag(5, "DIM_DYN")
        out.append(n)
    return out


#*************#
#   helpers   #
#*************#
def _infer_outcome(module: BaseModule, backend_enum: Backend) -> str:
    sb = module.SUPPORTED_BACKENDS
    return NATIVE if (not sb or backend_enum in sb) else AUTO


def _classify_error(exc: Exception) -> str:
    msg = str(exc).lower()
    tb  = traceback.format_exc().lower()
    combined = msg + " " + tb
    if "broadcast" in combined or ("shape" in combined and "cannot" in combined):
        return "Shape mismatch — N_PTS varies per call; tensor batch must be uniform (TODO)"
    if "expected to be a" in combined or ("dtype" in combined and "tensor" in combined):
        return "Dtype mismatch — numpy intermediate not cast to framework dtype (TODO)"
    if any(k in combined for k in ("nan", "overflow", "divide by zero", "invalid value")):
        return "Numerical instability at float32 precision (TODO)"
    if "no module named" in combined:
        return "Optional dependency not installed"
    return str(exc)[:120]


def _run_mrs_module(module, nifti_list, backend_enum) -> float:
    nplus = NIfTI_MRS_Plus(nifti_list=nifti_list, backend=backend_enum, volatile=True)
    t0 = time.perf_counter()
    module(nplus, None)
    return (time.perf_counter() - t0) * 1000.0


def _run_volume_module(module, backend_enum) -> float:
    """
    Modules that take a 5-D MRSI volume "(batch, X, Y, Z, T)".

    The spectra fixtures cannot exercise these — "KspaceUndersampling" masks
    k-space across the spatial axes, so handing it a "(1, 1, 1, N_PTS)" spectrum
    raises on the layout check rather than telling you anything about backend
    support.
    """
    x_np = (np.random.randn(1, 8, 8, 4, 16)
            + 1j * np.random.randn(1, 8, 8, 4, 16)).astype(np.complex64)
    t0 = time.perf_counter()
    if backend_enum == PYTORCH and TORCH_AVAILABLE:
        module.process_tensor(torch.from_numpy(x_np))
    else:
        module.process_tensor(x_np)
    return (time.perf_counter() - t0) * 1000.0


def _run_spatial_module(module, backend_enum) -> float:
    x_np = np.random.randn(2, 1, 32, 32).astype(np.float32)
    t0 = time.perf_counter()
    if backend_enum == PYTORCH and TORCH_AVAILABLE:
        x = torch.from_numpy(x_np)
        module.apply(x, pipeline="data")
    else:
        # NIFTI_LIST, NUMPY, TF, JAX, KERAS — all go through process_tensor
        module.process_tensor(x_np)
    return (time.perf_counter() - t0) * 1000.0


#****************************#
#   main parametrised test   #
#****************************#
@pytest.mark.parametrize("backend_enum", _ALL_BACKENDS,
                         ids=[b.value for b in _ALL_BACKENDS])
@pytest.mark.parametrize("entry", _REGISTRY, ids=[e.name for e in _REGISTRY])
def test_discover_backend_support(
    entry: ModuleEntry,
    backend_enum: Backend,
    single_coil_nifti_list,
    multi_coil_nifti_list,
):
    """
    Auto-discovers whether (module, backend) works.
    • PASS  → NATIVE or AUTO (works fine)
    • XFAIL → FAIL (module error, recorded with note, not a hard failure)
    • SKIP  → n/a  (framework not installed)
    Results accumulate in _RESULTS for the final summary tables.
    """
    key = (entry.name, backend_enum.value)

    # ── framework guard ───────────────────────────────────────────────────────
    if backend_enum in _FRAMEWORK_GUARD:
        avail, pkg = _FRAMEWORK_GUARD[backend_enum]
        if not avail:
            _RESULTS[key] = CellResult(outcome=NA, note=f"{pkg} not installed")
            pytest.skip(f"{pkg} not installed")

    # ── choose runner ─────────────────────────────────────────────────────────
    if entry.volume:
        runner = _run_volume_module
    elif entry.spatial:
        runner = _run_spatial_module
    else:
        runner = _run_mrs_module
    run_args = (entry.factory(),) + (
        (backend_enum,) if (entry.spatial or entry.volume)
        else (multi_coil_nifti_list if entry.needs_multicoil else single_coil_nifti_list,
              backend_enum)
    )

    # ── timed runs ────────────────────────────────────────────────────────────
    # run_args = (module, nifti_list, backend_enum)  or  (module, backend_enum)
    # runner signature matches exactly: _run_mrs_module(module, nifti_list, backend_enum)
    #                                   _run_spatial_module(module, backend_enum)
    module = run_args[0]
    times: list[float] = []
    try:
        for _ in range(N_REPS):
            times.append(runner(*run_args))
    except Exception as exc:
        note  = entry.note or _classify_error(exc)
        error = str(exc)[:120]
        _RESULTS[key] = CellResult(
            outcome=FAIL,
            time_ms=float(np.mean(times)) if times else float("nan"),
            note=note, error=error,
        )
        pytest.xfail(f"{entry.name}/{backend_enum.value}: {str(exc)[:80]}")
        return

    outcome = _infer_outcome(module, backend_enum)
    _RESULTS[key] = CellResult(
        outcome=outcome,
        time_ms=float(np.mean(times)),
        note=entry.note,
    )


#**************************************************************************************************#
#                                Class TestKspaceReconstructorTable                                #
#**************************************************************************************************#
#                                                                                                  #
# KspaceReconstructor: standalone PyTorch-only class (needs torchkbnufft to run).                  #
#                                                                                                  #
#**************************************************************************************************#
class TestKspaceReconstructorTable:
    """
    KspaceReconstructor: standalone PyTorch-only class (needs torchkbnufft to run).
    README: NIfTI=—, NumPy=—, PyTorch=✓, TF=—, JAX=—, Keras=—
    """

    def test_importable_without_torchkbnufft(self):
        """torchkbnufft is imported lazily, so importing the module must not need it."""
        from augmentrum.sampling.kspace_reconstructor import KspaceReconstructor
        assert isinstance(KspaceReconstructor, type)
        assert not issubclass(KspaceReconstructor, BaseModule)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_constructs_and_validates_geometry(self):
        from augmentrum.sampling.kspace_reconstructor import KspaceReconstructor

        recon = KspaceReconstructor(image_size=(64, 64), oversampling_factor=1.5)
        assert recon.image_size == [64, 64]
        assert recon.grid_size == [96, 96]

        # a scalar factor is broadcast; a per-axis one is taken as given
        assert KspaceReconstructor((8, 8, 8), (1.0, 2.0, 1.0)).grid_size == [8, 16, 8]

        with pytest.raises(ValueError):
            KspaceReconstructor(image_size=(8, 8, 8, 8))
        with pytest.raises(ValueError):
            KspaceReconstructor(image_size=(8, 8), oversampling_factor=0.5)
        with pytest.raises(ValueError):
            KspaceReconstructor(image_size=(8, 8), oversampling_factor=(1.5, 1.5, 1.5))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_flatten_and_normalize_are_backend_free(self):
        """The coordinate plumbing is plain torch and must work without torchkbnufft."""
        import torch
        from augmentrum.sampling.kspace_reconstructor import KspaceReconstructor

        B, S, D, L, C = 2, 3, 2, 5, 1
        coords = torch.rand(B, S, D, L) * 2 - 1
        kdata = torch.randn(B, S, C, L, dtype=torch.complex64)

        k_traj, kdata_flat = KspaceReconstructor.flatten(coords, kdata)
        assert tuple(k_traj.shape) == (B, D, S * L)
        assert tuple(kdata_flat.shape) == (B, C, S * L)

        # [-1, 1] in, [-pi, pi] out
        assert torch.allclose(k_traj[0, :, 0], coords[0, 0, :, 0] * torch.pi)

        # cycles/m -> [-1, 1]
        k_max = 250.0
        assert torch.allclose(
            KspaceReconstructor.normalize_trajectory(coords * k_max, k_max), coords
        )

        with pytest.raises(ValueError):
            KspaceReconstructor.flatten(coords[0])
        with pytest.raises(ValueError):
            KspaceReconstructor.flatten(coords, kdata[:, :-1])

    @pytest.mark.skipif(TORCHKBNUFFT_AVAILABLE,
                        reason="only meaningful when torchkbnufft is absent")
    def test_missing_dependency_is_actionable(self):
        from augmentrum.sampling.kspace_reconstructor import KspaceReconstructor
        with pytest.raises(ImportError, match="torchkbnufft"):
            KspaceReconstructor._tkbn()

    def test_not_exported_as_augmentation(self):
        import augmentrum.augmentation as pkg
        assert not hasattr(pkg, "KspaceReconstructor"), (
            "KspaceReconstructor is a standalone class, not an augmentation, and "
            "should not be exported from augmentrum.augmentation."
        )


#*******************************************************************************#
#   summary tables  (zzz → sorts last, always runs after all discovery tests)   #
#*******************************************************************************#
def test_zzz_print_summary():
    """
    Prints two tables to stdout (requires -s / --capture=no):

        Table 1 — Backend Support
            ✓ native  |  ~ auto (NIfTI routing)  |  ✗ fail  |  — n/a

        Table 2 — Speed (mean ms per call, N_REPS repeats)

    This test always PASSES. It is informational only.
    """
    col_w   = 11
    name_w  = 34
    backends = _ALL_BACKENDS
    labels   = ["NIfTI", "NumPy", "PyTorch", "TF", "JAX", "Keras"]

    sep = "─" * (name_w + col_w * len(backends))
    hdr = f"{'Module':<{name_w}}" + "".join(f"{l:^{col_w}}" for l in labels)

    note_index: list[str] = []

    # ── Table 1: support ──────────────────────────────────────────────────────
    print("\n\n" + "═" * len(sep))
    print("  README TABLE — BACKEND SUPPORT DISCOVERY")
    print(f"  N_SUBJ={N_SUBJ}  N_PTS={N_PTS}  N_REPS={N_REPS}")
    print("  ✓ native | ~ auto (NIfTI routing) | ✗ fail[n] (see notes) | — n/a")
    print("═" * len(sep))
    print(hdr)
    print(sep)

    for entry in _REGISTRY:
        cells = []
        for b in backends:
            r = _RESULTS.get((entry.name, b.value))
            if r is None:
                cells.append(f"{'?':^{col_w}}")
                continue
            sym = r.outcome
            if sym == FAIL and r.note:
                note_index.append(f"  [{len(note_index)+1}] {entry.name} / {b.value}: {r.note}")
                sym = f"✗[{len(note_index)}]"
            cells.append(f"{sym:^{col_w}}")
        print(f"{entry.name[:name_w-1]:<{name_w}}" + "".join(cells))

    print(sep)
    print("  KspaceReconstructor" + " " * (name_w - 20) +
          "".join(f"{'—':^{col_w}}" * 2) +
          f"{'✓(fn)':^{col_w}}" +
          "".join(f"{'—':^{col_w}}" * 3))
    print(f"  (standalone PyTorch function, not a BaseModule; needs torchkbnufft)")

    if note_index:
        print("\n  FAIL NOTES:")
        for n in note_index:
            print(n)

    # ── Table 2: speed ────────────────────────────────────────────────────────
    print("\n\n" + "═" * len(sep))
    print(f"  SPEED TABLE  (mean ms / call, N_REPS={N_REPS},  — = n/a or not run)")
    print("═" * len(sep))
    print(hdr)
    print(sep)

    for entry in _REGISTRY:
        cells = []
        for b in backends:
            r = _RESULTS.get((entry.name, b.value))
            if r is None or r.outcome == NA:
                cells.append(f"{'—':^{col_w}}")
            elif np.isnan(r.time_ms):
                cells.append(f"{'err':^{col_w}}")
            else:
                cells.append(f"{r.time_ms:^{col_w}.2f}")
        print(f"{entry.name[:name_w-1]:<{name_w}}" + "".join(cells))

    print(sep)
    print()
    assert True

