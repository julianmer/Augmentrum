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

from augmentrum.core.nifti_mrs_plus import NIfTI_MRS_Plus, Backend
from augmentrum.core.base_module import BaseModule

# ── augmentation imports ──────────────────────────────────────────────────────
from augmentrum.augmentation import (
    AmplitudeScaling, Apodization, ArtificialPeaks, BaselineAugmentation,
    EddyCurrent, FrequencyShift, GaussianNoise, LineBroadening, PhaseShift,
    ResidualWater, SpuriousEchoes, SpatialAugmentations, ZeroFill,
)
from augmentrum.processing.nifti_raw_processor import NIfTI_RawProcessor
from augmentrum.sampling.coil_average_sampler import CoilAverageSampler

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


# ─────────────────────────────────────────────────────────────────────────────
# Registry entry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModuleEntry:
    """One row in the module registry.

    Args:
        name:             Display name / pytest ID.
        factory:          Zero-arg callable returning a fresh module instance.
        note:             Free-text note shown in the summary for known issues.
        needs_multicoil:  Use multi-coil NIfTI fixture (needs DIM_COIL + DIM_DYN).
        spatial:          Special-case: image-tensor module (SpatialAugmentations).
    """
    name:            str
    factory:         Callable
    note:            str  = ""
    needs_multicoil: bool = False
    spatial:         bool = False


# ─────────────────────────────────────────────────────────────────────────────
# ★  MODULE REGISTRY  ★   ← add new modules here
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: list[ModuleEntry] = [

    # ── amplitude ─────────────────────────────────────────────────────────────
    ModuleEntry("AmplitudeScaling[uniform]",
                lambda: AmplitudeScaling(scale_factor=0.8)),
    ModuleEntry("AmplitudeScaling[range]",
                lambda: AmplitudeScaling(scale_factor=(0.7, 1.3))),

    # ── apodization ───────────────────────────────────────────────────────────
    ModuleEntry("Apodization[exponential]",
                lambda: Apodization(mode="exponential", lb_hz=5.0)),
    ModuleEntry("Apodization[truncate]",
                lambda: Apodization(mode="truncate", n_pts=256)),

    # ── artificial peaks ──────────────────────────────────────────────────────
    ModuleEntry("ArtificialPeaks",     lambda: ArtificialPeaks()),

    # ── baseline ──────────────────────────────────────────────────────────────
    ModuleEntry("BaselineAugmentation[random_walk]",
                lambda: BaselineAugmentation(mode="random_walk")),
    ModuleEntry("BaselineAugmentation[bspline]",
                lambda: BaselineAugmentation(mode="bspline"),
                note="float32 precision → bspline solver NaN/overflow (TODO)"),
    ModuleEntry("BaselineAugmentation[polynomial]",
                lambda: BaselineAugmentation(mode="polynomial")),

    # ── eddy currents ─────────────────────────────────────────────────────────
    ModuleEntry("EddyCurrent[synthetic]",
                lambda: EddyCurrent(mode="synthetic")),

    # ── frequency / phase ─────────────────────────────────────────────────────
    ModuleEntry("FrequencyShift",
                lambda: FrequencyShift(shift_hz=5.0)),
    ModuleEntry("PhaseShift[zero_order]",
                lambda: PhaseShift(zero_order_deg=30.0)),
    ModuleEntry("PhaseShift[first_order]",
                lambda: PhaseShift(first_order_deg=45.0)),

    # ── noise ─────────────────────────────────────────────────────────────────
    ModuleEntry("GaussianNoise[sigma]",
                lambda: GaussianNoise(sigma=0.01)),
    ModuleEntry("GaussianNoise[snr_db]",
                lambda: GaussianNoise(snr_db=20.0)),
    ModuleEntry("GaussianNoise[sigma_frac]",
                lambda: GaussianNoise(sigma_frac=0.02)),

    # ── line broadening ───────────────────────────────────────────────────────
    ModuleEntry("LineBroadening[lorentzian]",
                lambda: LineBroadening(lb_hz=5.0, mode="lorentzian")),
    ModuleEntry("LineBroadening[gaussian]",
                lambda: LineBroadening(gb_hz=5.0, mode="gaussian")),
    ModuleEntry("LineBroadening[voigt]",
                lambda: LineBroadening(lb_hz=3.0, gb_hz=2.0, mode="voigt")),

    # ── residual water ────────────────────────────────────────────────────────
    ModuleEntry("ResidualWater",       lambda: ResidualWater()),

    # ── spurious echoes ───────────────────────────────────────────────────────
    ModuleEntry("SpuriousEchoes[replica]",
                lambda: SpuriousEchoes(mode="replica")),
    ModuleEntry("SpuriousEchoes[hybrid]",
                lambda: SpuriousEchoes(mode="hybrid")),

    # ── zero fill ─────────────────────────────────────────────────────────────
    ModuleEntry("ZeroFill[pad]",
                lambda: ZeroFill(target_pts=1024)),   # 512 → 1024 (double)
    ModuleEntry("ZeroFill[crop]",
                lambda: ZeroFill(target_pts=256)),    # 512 → 256  (half)

    # ── spatial ───────────────────────────────────────────────────────────────
    ModuleEntry("SpatialAugmentations",
                lambda: SpatialAugmentations(dim=2, prob=1.0),
                note="Image module — NIfTI/NumPy convert to PyTorch (F.grid_sample) internally",
                spatial=True),

    # ── processing ────────────────────────────────────────────────────────────
    ModuleEntry("NIfTI_RawProcessor",
                lambda: NIfTI_RawProcessor(
                    conj=False, coil=False, align=False, remove_outliers=False,
                    average=False, ecc=False, truncate=False, remove_water=False,
                    shift_ref=False, phase_correct=False,
                ),
                note="NIfTI-native; non-NIfTI auto-route; all steps disabled for speed",
                needs_multicoil=True),

    # ── sampling ──────────────────────────────────────────────────────────────
    ModuleEntry("CoilAverageSampler",
                lambda: CoilAverageSampler(mode="random", n_coils=2, n_averages=4),
                note="NIfTI-native; non-NIfTI auto-route; needs DIM_COIL data",
                needs_multicoil=True),

    # ── KspaceReconstructor: standalone function — tested in class below ───────
    # (not a BaseModule, PyTorch-only, needs torchkbnufft)
]


# ─────────────────────────────────────────────────────────────────────────────
# Result collection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CellResult:
    outcome:  str   = NA
    time_ms:  float = float("nan")
    note:     str   = ""
    error:    str   = ""


_RESULTS: dict[tuple[str, str], CellResult] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Main parametrised test
# ─────────────────────────────────────────────────────────────────────────────

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
    runner   = _run_spatial_module if entry.spatial else _run_mrs_module
    run_args = (entry.factory(),) + (
        (backend_enum,) if entry.spatial
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


# ─────────────────────────────────────────────────────────────────────────────
# KspaceReconstructor — standalone function, not a BaseModule
# ─────────────────────────────────────────────────────────────────────────────

class TestKspaceReconstructorTable:
    """
    KspaceReconstructor: standalone PyTorch-only function (needs torchkbnufft).
    README: NIfTI=—, NumPy=—, PyTorch=✓(fn), TF=—, JAX=—, Keras=—
    """

    @pytest.mark.skipif(not TORCHKBNUFFT_AVAILABLE,
                        reason="torchkbnufft not installed")
    def test_is_standalone_callable(self):
        from augmentrum.sampling.kspace_reconstructor import reconstruct_with_masking
        assert callable(reconstruct_with_masking)
        assert not (isinstance(reconstruct_with_masking, type)
                    and issubclass(reconstruct_with_masking, BaseModule))

    @pytest.mark.skipif(not (TORCH_AVAILABLE and TORCHKBNUFFT_AVAILABLE),
                        reason="torch + torchkbnufft required")
    def test_pytorch_entry_importable(self):
        from augmentrum.sampling.kspace_reconstructor import reconstruct_with_masking
        assert callable(reconstruct_with_masking)

    def test_not_exported_as_augmentation(self):
        import augmentrum.augmentation as pkg
        assert not hasattr(pkg, "KspaceReconstructor"), (
            "KspaceReconstructor is a standalone function and should not be "
            "exported from augmentrum.augmentation."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Summary tables  (zzz → sorts last, always runs after all discovery tests)
# ─────────────────────────────────────────────────────────────────────────────

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

