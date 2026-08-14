"""
Backend compatibility & timing tests for all augmentation modules.

For every (module × backend) combination this file:
  1. Verifies the module runs without error.
  2. Checks that the output shape is preserved (truncation has a special case).
  3. Checks that the data was actually modified.
  4. Records wall-clock time and prints a one-line summary.

Uses single-coil NIfTI data (shape 1,1,1,N_PTS) so the spectral dimension
is always the *last* axis — this is the shape convention that process_tensor
implementations rely on.

Run with:
    pytest tests/augmentation/test_backend_compatibility.py -v -s
"""

import time
import pytest
import numpy as np
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

from nifti_mrs_plus import NIfTI_MRS_Plus, Backend, ops
from tests.module_specs import SPECS
from augmentrum.augmentation import (
    Noise,
    LineBroadening,
    PhaseShift,
    FrequencyShift,
    AmplitudeScaling,
    BaselineAugmentation,
    EddyCurrent,
    SpuriousEchoes,
    ArtificialPeaks,
    ResidualWater,
    Apodization,
    SpatialAugmentations,
)

# ── Optional framework availability ──────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

# ── Backend list (only include installed frameworks) ──────────────────────────
ALL_BACKENDS: list = [
    (Backend.NIFTI_LIST, "NIFTI_LIST"),
    (Backend.NUMPY,      "NUMPY"),
]
if TORCH_AVAILABLE:
    ALL_BACKENDS.append((Backend.PYTORCH, "PYTORCH"))
if TF_AVAILABLE:
    ALL_BACKENDS.append((Backend.TENSORFLOW, "TENSORFLOW"))
if JAX_AVAILABLE:
    ALL_BACKENDS.append((Backend.JAX, "JAX"))
if KERAS_AVAILABLE:
    ALL_BACKENDS.append((Backend.KERAS, "KERAS"))

# ── Dataset parameters ────────────────────────────────────────────────────────
N_SUBJECTS = 4      # number of synthetic subjects per run
N_PTS      = 2048   # spectral points
DWELLTIME  = 1 / 2000   # → sw_hz = 2000 Hz
SPEC_FREQ  = 123.0  # MHz (3T ¹H)

# Derived from the shared spec table, filtered to the variants this file's
# fixtures can actually exercise: single- or multi-coil SPECTRA. Modules taking
# 2-D images or 5-D volumes are swept in test_readme_table instead, and the
# guard in test_module_registry is what stops a new module escaping both.
MRS_MODULES = [
    (spec.label, spec.cls, spec.kwargs)
    for spec in SPECS
    if not (spec.spatial or spec.volume or spec.needs_multicoil or spec.coiled)
]

# Variants that rewrite the spectral length, so the shape-preservation assertion
# below does not apply to them. Keyed off the spec flag rather than matched on the
# module name, which previously exempted anything with 'trunc' in its label.
LENGTH_CHANGING = {spec.label for spec in SPECS if spec.changes_length}

# Variants that pass data through unchanged by design, so the data-was-modified
# assertion below does not apply to them.
IDENTITY = {spec.label for spec in SPECS if spec.identity}

# Variants that append a new acquisition axis, so the assertion is growth by
# exactly one trailing axis rather than shape preservation.
ADDS_DIM = {spec.label for spec in SPECS if spec.adds_dim}


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def single_coil_nifti_list():
    """
    N_SUBJECTS single-coil NIfTI-MRS objects, each shape (1,1,1,N_PTS).
    Spectral dim is the last axis — correct for all process_tensor calls.
    """
    niftis = []
    rng = np.random.default_rng(0)
    for _ in range(N_SUBJECTS):
        data = (rng.standard_normal((1, 1, 1, N_PTS))
                + 1j * rng.standard_normal((1, 1, 1, N_PTS))).astype(np.complex64)
        n = gen_nifti_mrs(data, dwelltime=DWELLTIME, spec_freq=SPEC_FREQ)
        niftis.append(n)
    return niftis


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_native(module, backend_enum: Backend) -> bool:
    """Return True when backend_enum is natively supported (no conversion)."""
    sb = module.SUPPORTED_BACKENDS
    if not sb:       # empty → supports ALL
        return True
    return backend_enum in sb


def _run_module(module, nifti_list, backend_enum) -> tuple:
    """Instantiate NIfTI_MRS_Plus, call module, return (result, elapsed_s).

    Deep-copies the nifti_list so the shared module-scoped fixture is never
    mutated in-place by the augmentation (modules do "nifti[:] = processed"
    which would otherwise corrupt subsequent tests in this session).
    """
    from copy import deepcopy
    nplus = NIfTI_MRS_Plus(nifti_list=deepcopy(nifti_list), backend=backend_enum, volatile=True)
    t0 = time.perf_counter()
    result, _ = module(nplus, None)
    elapsed = time.perf_counter() - t0
    return result, elapsed


# ── Main parametrized test ────────────────────────────────────────────────────

@pytest.mark.parametrize("module_name,module_cls,module_kwargs", MRS_MODULES)
@pytest.mark.parametrize("backend_enum,backend_name", ALL_BACKENDS)
def test_module_on_backend(
    module_name, module_cls, module_kwargs,
    backend_enum, backend_name,
    single_coil_nifti_list,
):
    """
    Every MRS augmentation module must run on every available backend
    (via native support or the base-class fallback conversion) and must
    actually change the data.
    """
    module = module_cls(**module_kwargs)
    native = _is_native(module, backend_enum)
    native_tag = "native" if native else "conv."

    # Capture BEFORE running — modules mutate NIfTI objects in-place
    original_data = single_coil_nifti_list[0][:].copy()

    result, elapsed_ms = _run_module(module, single_coil_nifti_list, backend_enum)
    elapsed_ms *= 1000

    print(f"\n  {module_name:<35s} | {backend_name:<12s} | {native_tag:<7s} | {elapsed_ms:6.1f} ms")

    assert result is not None, f"{module_name} returned None on {backend_name}"

    # Length-changing modules: the spectral axis is meant to differ, so only the
    # subject count and the fact that the length actually moved are checked.
    if module_name in LENGTH_CHANGING:
        assert len(result) == N_SUBJECTS, (
            f"{module_name}/{backend_name}: lost subjects after resizing"
        )
        assert result[0][:].shape[-1] != N_PTS, (
            f"{module_name}/{backend_name}: spectral length unchanged at {N_PTS}"
        )
        return

    # Dimension-adding modules: the input shape must survive as the leading
    # axes, with exactly one new acquisition axis behind it.
    if module_name in ADDS_DIM:
        grown = result[0][:]
        assert grown.shape[:original_data.ndim] == original_data.shape, (
            f"{module_name}/{backend_name}: leading axes changed "
            f"{original_data.shape} → {grown.shape}"
        )
        assert grown.ndim == original_data.ndim + 1, (
            f"{module_name}/{backend_name}: expected one new axis, got "
            f"{original_data.shape} → {grown.shape}"
        )
        return

    # All other modules: shape must be identical and data must have changed —
    # except identity modules, which must leave the data exactly as it was.
    result_data = result[0][:]

    assert result_data.shape == original_data.shape, (
        f"{module_name}/{backend_name}: shape changed "
        f"{original_data.shape} → {result_data.shape}"
    )
    if module_name in IDENTITY:
        assert np.allclose(result_data, original_data, atol=1e-6), (
            f"{module_name}/{backend_name}: identity module modified the data"
        )
        return
    assert not np.allclose(result_data, original_data, atol=1e-8), (
        f"{module_name}/{backend_name}: data was not modified"
    )


# ── Backend-specific Noise mode tests ─────────────────────────────────

# ── Backend-specific Noise mode tests ─────────────────────────────────

def _run_noise(mode_kwargs, nifti_list, backend_enum):
    module = Noise(**mode_kwargs)
    nplus  = NIfTI_MRS_Plus(nifti_list=nifti_list, backend=backend_enum, volatile=True)
    orig   = nplus[0][:].copy()
    result, _ = module(nplus, None)
    return result, orig


@pytest.mark.parametrize("backend_enum,backend_name", ALL_BACKENDS)
def test_noise_sigma_mode(backend_enum, backend_name, single_coil_nifti_list):
    """Noise(sigma=…) runs on every backend."""
    result, orig = _run_noise({"sigma": 0.05}, single_coil_nifti_list, backend_enum)
    assert not np.allclose(result[0][:], orig, atol=1e-8)


@pytest.mark.parametrize("backend_enum,backend_name", ALL_BACKENDS)
def test_noise_sigma_frac_mode(backend_enum, backend_name, single_coil_nifti_list):
    """Noise(sigma_frac=…) runs on every backend."""
    result, orig = _run_noise({"sigma_frac": 0.05}, single_coil_nifti_list, backend_enum)
    assert not np.allclose(result[0][:], orig, atol=1e-8)


@pytest.mark.parametrize("backend_enum,backend_name", ALL_BACKENDS)
def test_noise_snr_db_mode(backend_enum, backend_name, single_coil_nifti_list):
    """Noise(snr_db=…) runs on every backend."""
    result, orig = _run_noise({"snr_db": 15.0}, single_coil_nifti_list, backend_enum)
    assert not np.allclose(result[0][:], orig, atol=1e-8)


@pytest.mark.parametrize("backend_enum,backend_name", ALL_BACKENDS)
def test_noise_snr_mode(backend_enum, backend_name, single_coil_nifti_list):
    """Noise(snr=…) runs on every backend."""
    result, orig = _run_noise({"snr": 20.0}, single_coil_nifti_list, backend_enum)
    assert not np.allclose(result[0][:], orig, atol=1e-8)


# ── SpatialAugmentations (uses 4-D/5-D image tensors, not MRS FIDs) ──────────


#**************************************************************************************************#
#                              Class TestSpatialAugmentationsBackends                              #
#**************************************************************************************************#
#                                                                                                  #
# SpatialAugmentations — native on NIFTI_LIST / NUMPY / PYTORCH only.                              #
#                                                                                                  #
#**************************************************************************************************#
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
class TestSpatialAugmentationsBackends:
    """SpatialAugmentations — native on NIFTI_LIST / NUMPY / PYTORCH only."""

    def _make_2d_batch(self, B=2, C=1, H=32, W=32):
        return torch.randn(B, C, H, W, dtype=torch.float32)

    def test_pytorch_native(self):
        aug = SpatialAugmentations(dim=2, prob=1.0)
        x   = self._make_2d_batch()
        x_aug, _ = aug.apply(x, pipeline="data")
        assert x_aug.shape == x.shape

    def test_numpy_converts_to_pytorch(self):
        aug   = SpatialAugmentations(dim=2, prob=1.0)
        x_np  = np.random.randn(2, 1, 32, 32).astype(np.float32)
        x_aug, _ = aug.process_tensor(x_np)
        assert isinstance(x_aug, np.ndarray)
        assert x_aug.shape == x_np.shape

    def test_supported_backends_are_correct(self):
        aug = SpatialAugmentations(dim=2, prob=1.0)
        for backend in Backend:
            assert backend in aug.SUPPORTED_BACKENDS, (
                f"SpatialAugmentations no longer claims {backend.value}"
            )

    @pytest.mark.skipif(not TF_AVAILABLE, reason="TF not installed")
    def test_tensorflow_runs_via_fallback(self, single_coil_nifti_list):
        """TF backend: routed through NIfTI fallback — should still succeed."""
        aug  = SpatialAugmentations(dim=2, prob=0.5)
        # Use a simple 2-D tensor processed through process_tensor for TF
        x_np = np.random.randn(2, 1, 32, 32).astype(np.float32)
        x_aug, _ = aug.process_tensor(x_np)
        assert x_aug.shape == x_np.shape


# ── Invariant: modules with SUPPORTED_BACKENDS=[] must have process_tensor ────

_CLAIM_ALL_BACKENDS_FACTORIES = [
    ("Noise",        lambda: Noise(sigma=0.01)),
    ("LineBroadening",       lambda: LineBroadening(lb_hz=5.0)),
    ("PhaseShift",           lambda: PhaseShift(zero_order_deg=30.0)),
    ("FrequencyShift",       lambda: FrequencyShift(shift_hz=10.0)),
    ("AmplitudeScaling",     lambda: AmplitudeScaling(scale_factor=0.9)),
    ("BaselineAugmentation", lambda: BaselineAugmentation(mode="random_walk")),
    ("EddyCurrent",          lambda: EddyCurrent(mode="synthetic")),
    ("SpuriousEchoes",       lambda: SpuriousEchoes(mode="replica")),
    ("ArtificialPeaks",      lambda: ArtificialPeaks()),
    ("ResidualWater",        lambda: ResidualWater()),
    ("Apodization",          lambda: Apodization(mode="exponential", lb_hz=5.0)),
]


@pytest.mark.parametrize(
    "name,factory",
    _CLAIM_ALL_BACKENDS_FACTORIES,
    ids=[x[0] for x in _CLAIM_ALL_BACKENDS_FACTORIES],
)
def test_all_backends_claim_requires_process_tensor(name, factory):
    """
    Any module with SUPPORTED_BACKENDS=[] (i.e. claims ALL backends) must
    override process_tensor(); otherwise tensor backends silently fall back
    to the NIfTI-list path while falsely claiming native support.
    """
    from augmentrum.core.base_module import BaseModule
    mod = factory()
    if not mod.SUPPORTED_BACKENDS:
        has_pt = type(mod).process_tensor is not BaseModule.process_tensor
        assert has_pt, (
            f"{name} sets SUPPORTED_BACKENDS=[] (all backends) "
            f"but does not override process_tensor(). "
            f"Either add process_tensor() or restrict SUPPORTED_BACKENDS."
        )


def test_spatial_augmentations_is_native_on_every_backend():
    """
    The resampling must run on the tensor's own backend and give the same answer.

    Claiming a backend is not enough: the same augmentation spec is replayed on
    each one and the results are compared, so a module that silently converted
    would still be caught by the type check.
    """
    rng = np.random.default_rng(0)
    arr = (rng.standard_normal((1, 12, 12, 2))
           + 1j * rng.standard_normal((1, 12, 12, 2))).astype(np.complex64)

    # one real spec, replayed everywhere
    _, spec = SpatialAugmentations(dim=2, prob=1.0, pixdim=(1.0, 1.0)).apply(arr.copy())

    cases = [("numpy", lambda a: a, lambda v: isinstance(v, np.ndarray))]
    if TORCH_AVAILABLE:
        import torch
        cases.append(("torch", torch.from_numpy, ops.is_torch))
    if JAX_AVAILABLE:
        import jax.numpy as jnp
        cases.append(("jax", jnp.array, ops.is_jax))
    if TF_AVAILABLE:
        import tensorflow as tf
        cases.append(("tensorflow", tf.constant, ops.is_tf))

    reference = None
    for name, convert, is_native in cases:
        out, _ = SpatialAugmentations(dim=2, prob=1.0, pixdim=(1.0, 1.0)).apply(
            convert(arr), aug_spec_list=spec)
        assert is_native(out), f"{name}: result left its backend"

        result = ops.to_numpy(out)
        if reference is None:
            reference = result
        else:
            assert np.allclose(result, reference, atol=1e-5), (
                f"{name}: resampling disagrees with the NumPy result"
            )


# ── Timing benchmark (separate class so it can be run selectively) ────────────

TIMING_REPS = 3   # repetitions for mean ± std

@pytest.mark.parametrize("module_name,module_cls,module_kwargs", MRS_MODULES)
@pytest.mark.parametrize("backend_enum,backend_name", ALL_BACKENDS)
def test_timing_benchmark(
    module_name, module_cls, module_kwargs,
    backend_enum, backend_name,
    single_coil_nifti_list,
):
    """
    Measure mean ± std wall-clock time for every module × backend.
    Always passes — this test is purely informational.  Run with -s to see output.
    """
    module = module_cls(**module_kwargs)
    times  = []

    for _ in range(TIMING_REPS):
        _, elapsed = _run_module(module, single_coil_nifti_list, backend_enum)
        times.append(elapsed * 1000)

    mean_ms = np.mean(times)
    std_ms  = np.std(times)
    native  = "native" if _is_native(module, backend_enum) else "conv. "

    print(
        f"\n  {module_name:<35s} | {backend_name:<12s} | {native} | "
        f"{mean_ms:6.1f} ± {std_ms:.1f} ms  (n={TIMING_REPS}, {N_SUBJECTS} subjects)"
    )

    # Soft guard: no single call should take more than 30 seconds
    assert mean_ms < 30_000, (
        f"{module_name} on {backend_name} averaged {mean_ms:.0f} ms — too slow"
    )

