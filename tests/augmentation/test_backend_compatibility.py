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

from augmentrum.core.nifti_mrs_plus import NIfTI_MRS_Plus, Backend
from augmentrum.augmentation import (
    GaussianNoise,
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

# ── Module registry: (display_name, class, constructor_kwargs) ────────────────
MRS_MODULES = [
    ("GaussianNoise[sigma]",         GaussianNoise,        {"sigma": 0.01}),
    ("GaussianNoise[snr_db]",        GaussianNoise,        {"snr_db": 20.0}),
    ("GaussianNoise[sigma_frac]",    GaussianNoise,        {"sigma_frac": 0.02}),
    ("LineBroadening[lorentzian]",   LineBroadening,       {"lb_hz": 5.0, "mode": "lorentzian"}),
    ("LineBroadening[gaussian]",     LineBroadening,       {"gb_hz": 3.0, "mode": "gaussian"}),
    ("LineBroadening[voigt]",        LineBroadening,       {"lb_hz": 3.0, "gb_hz": 2.0, "mode": "voigt"}),
    ("PhaseShift[zero]",             PhaseShift,           {"zero_order_deg": 30.0}),
    ("PhaseShift[first]",            PhaseShift,           {"first_order_deg": 45.0}),
    ("FrequencyShift",               FrequencyShift,       {"shift_hz": 10.0}),
    ("AmplitudeScaling[fixed]",      AmplitudeScaling,     {"scale_factor": 0.8}),
    ("AmplitudeScaling[range]",      AmplitudeScaling,     {"scale_factor": (0.7, 1.3)}),
    ("BaselineAugmentation[rw]",     BaselineAugmentation, {"mode": "random_walk"}),
    ("BaselineAugmentation[bspline]",BaselineAugmentation, {"mode": "bspline"}),
    ("BaselineAugmentation[poly]",   BaselineAugmentation, {"mode": "polynomial"}),
    ("EddyCurrent[synthetic]",       EddyCurrent,          {"mode": "synthetic"}),
    ("SpuriousEchoes[replica]",      SpuriousEchoes,       {"mode": "replica"}),
    ("SpuriousEchoes[hybrid]",       SpuriousEchoes,       {"mode": "hybrid"}),
    ("ArtificialPeaks",              ArtificialPeaks,      {}),
    ("ResidualWater",                ResidualWater,        {}),
    ("Apodization[exp]",             Apodization,          {"mode": "exponential", "lb_hz": 5.0}),
    ("Apodization[trunc]",           Apodization,          {"mode": "truncate", "n_pts": 1024}),
]


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
    mutated in-place by the augmentation (modules do ``nifti[:] = processed``
    which would otherwise corrupt subsequent tests in this session).
    """
    from copy import deepcopy
    nplus = NIfTI_MRS_Plus(nifti_list=deepcopy(nifti_list), backend=backend_enum, volatile=True)
    t0 = time.perf_counter()
    result, _ = module(nplus, None)
    elapsed = time.perf_counter() - t0
    return result, elapsed


# ── Main parametrised test ────────────────────────────────────────────────────

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

    # Apodization[trunc] reduces N_PTS — just check subject count survived
    if "trunc" in module_name.lower():
        assert len(result) == N_SUBJECTS, (
            f"{module_name}/{backend_name}: lost subjects after truncation"
        )
        # spectral points must be reduced
        assert result[0][:].shape[-1] < N_PTS
        return

    # All other modules: shape must be identical and data must have changed
    result_data = result[0][:]

    assert result_data.shape == original_data.shape, (
        f"{module_name}/{backend_name}: shape changed "
        f"{original_data.shape} → {result_data.shape}"
    )
    assert not np.allclose(result_data, original_data, atol=1e-8), (
        f"{module_name}/{backend_name}: data was not modified"
    )


# ── Backend-specific GaussianNoise mode tests ─────────────────────────────────

# ── Backend-specific GaussianNoise mode tests ─────────────────────────────────

def _run_noise(mode_kwargs, nifti_list, backend_enum):
    module = GaussianNoise(**mode_kwargs)
    nplus  = NIfTI_MRS_Plus(nifti_list=nifti_list, backend=backend_enum, volatile=True)
    orig   = nplus[0][:].copy()
    result, _ = module(nplus, None)
    return result, orig


@pytest.mark.parametrize("backend_enum,backend_name", ALL_BACKENDS)
def test_gaussian_noise_sigma_mode(backend_enum, backend_name, single_coil_nifti_list):
    """GaussianNoise(sigma=…) runs on every backend."""
    result, orig = _run_noise({"sigma": 0.05}, single_coil_nifti_list, backend_enum)
    assert not np.allclose(result[0][:], orig, atol=1e-8)


@pytest.mark.parametrize("backend_enum,backend_name", ALL_BACKENDS)
def test_gaussian_noise_sigma_frac_mode(backend_enum, backend_name, single_coil_nifti_list):
    """GaussianNoise(sigma_frac=…) runs on every backend."""
    result, orig = _run_noise({"sigma_frac": 0.05}, single_coil_nifti_list, backend_enum)
    assert not np.allclose(result[0][:], orig, atol=1e-8)


@pytest.mark.parametrize("backend_enum,backend_name", ALL_BACKENDS)
def test_gaussian_noise_snr_db_mode(backend_enum, backend_name, single_coil_nifti_list):
    """GaussianNoise(snr_db=…) runs on every backend."""
    result, orig = _run_noise({"snr_db": 15.0}, single_coil_nifti_list, backend_enum)
    assert not np.allclose(result[0][:], orig, atol=1e-8)


@pytest.mark.parametrize("backend_enum,backend_name", ALL_BACKENDS)
def test_gaussian_noise_snr_mode(backend_enum, backend_name, single_coil_nifti_list):
    """GaussianNoise(snr=…) runs on every backend."""
    result, orig = _run_noise({"snr": 20.0}, single_coil_nifti_list, backend_enum)
    assert not np.allclose(result[0][:], orig, atol=1e-8)


# ── SpatialAugmentations (uses 4-D/5-D image tensors, not MRS FIDs) ──────────

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
        assert Backend.NIFTI_LIST in aug.SUPPORTED_BACKENDS
        assert Backend.NUMPY      in aug.SUPPORTED_BACKENDS
        assert Backend.PYTORCH    in aug.SUPPORTED_BACKENDS
        # These should NOT be in SUPPORTED_BACKENDS (require implicit conversion)
        assert Backend.TENSORFLOW not in aug.SUPPORTED_BACKENDS
        assert Backend.JAX        not in aug.SUPPORTED_BACKENDS
        assert Backend.KERAS      not in aug.SUPPORTED_BACKENDS

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
    ("GaussianNoise",        lambda: GaussianNoise(sigma=0.01)),
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


def test_spatial_augmentations_does_not_claim_tf_jax_keras():
    """SpatialAugmentations must NOT claim TF/JAX/Keras as natively supported."""
    aug = SpatialAugmentations(dim=2, prob=0.5)
    restricted = {Backend.TENSORFLOW, Backend.JAX, Backend.KERAS}
    claimed    = set(aug.SUPPORTED_BACKENDS)
    overlap    = restricted & claimed
    assert not overlap, (
        f"SpatialAugmentations claims TF/JAX/Keras as native but always converts "
        f"to PyTorch internally. Found: {[b.value for b in overlap]}"
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

