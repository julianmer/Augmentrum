"""
Tests for the generic severity-segmentation scheme (augmentrum/reconstruction/segmented_phase.py)
against a synthetic, analytically-known phase term - independent of GIRF-Sim/RepoAdapter so these
stay fast and self-contained. A smooth, analytically known B0-like phase term lets us check both
"does segmentation approximate the exact phase reasonably" and "does more segments mean a
strictly better approximation" without needing any real trajectory/map data.
"""
import torch

from augmentrum.reconstruction.segmented_phase import SegmentedPhaseTerm


class _SyntheticB0Phase:
    """phi(r, t) = 2*pi*f(r)*t, f(r) a smooth function of the first position coordinate -
    mimics StaticOffResonancePhase's own exact formula, with a KNOWN, smooth severity landscape."""

    def __init__(self, freq_scale: float = 50.0):
        self.freq_scale = freq_scale

    def freq_map(self, positions: torch.Tensor) -> torch.Tensor:
        return self.freq_scale * torch.sin(2 * torch.pi * positions[:, 0])

    def phase(self, positions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        f = self.freq_map(positions)
        return 2 * torch.pi * f.reshape(-1, 1) * t.reshape(1, -1)


def _make_positions_and_t(n=400, T=200):
    torch.manual_seed(0)
    positions = torch.rand(n, 3) - 0.5   # arbitrary unit-ish positions, like a normalized FOV
    t = torch.linspace(0, 0.02, T)       # 20 ms readout, comparable scale to the real figure
    return positions, t


def test_segmentation_error_shrinks_with_more_segments():
    term = _SyntheticB0Phase()
    positions, t = _make_positions_and_t()

    errors = []
    for n_segments in (2, 4, 8, 16, 32):
        seg = SegmentedPhaseTerm(n_segments=n_segments, binning="quantile")
        seg.prepare(term, positions, t)
        errors.append(seg.max_abs_representation_error(term, positions, t))

    # Monotonic non-increase (allow tiny numerical slack) as segmentation gets finer.
    for a, b in zip(errors, errors[1:]):
        assert b <= a + 1e-6, f"error did not shrink: {errors}"
    # With 32 segments over a smooth, bounded-severity field, error should be well below the
    # unsegmented worst case (~pi) - not exact, since piecewise-constant-in-r can never be exact
    # for a continuously-varying field, but a clear, large improvement.
    assert errors[-1] < 1.0, f"32-segment error unexpectedly large: {errors[-1]}"


def test_bin_masks_partition_all_voxels_exactly_once():
    term = _SyntheticB0Phase()
    positions, t = _make_positions_and_t()
    seg = SegmentedPhaseTerm(n_segments=6, binning="quantile").prepare(term, positions, t)

    coverage = seg.bin_masks.sum(dim=0)   # (N,) - should be exactly 1 for every voxel
    assert torch.allclose(coverage, torch.ones_like(coverage))


def test_representative_curve_matches_an_actual_voxel_in_its_bin():
    """Each bin's phase curve must be the EXACT phase at a real representative voxel (not some
    fitted/averaged curve) - check the chosen voxel is really in its own bin and its own exact
    phase curve matches the stored one exactly."""
    term = _SyntheticB0Phase()
    positions, t = _make_positions_and_t()
    seg = SegmentedPhaseTerm(n_segments=5, binning="quantile").prepare(term, positions, t)

    for l in range(5):
        v = seg.representative_voxel[l].item()
        if seg.bin_masks[l].sum() == 0:
            continue   # empty bin - nothing to check
        assert seg.bin_masks[l, v] == 1.0
        exact_curve = term.phase(positions[v:v + 1], t)[0]
        stored_phase = torch.angle(seg.bin_phase_exp[l])
        torch.testing.assert_close(torch.remainder(exact_curve + torch.pi, 2 * torch.pi) - torch.pi,
                                   stored_phase, atol=1e-4, rtol=1e-4)


def test_uniform_binning_handles_empty_bins_without_error():
    # A severity distribution with a huge outlier makes uniform-width bins mostly empty - must
    # not raise.
    term = _SyntheticB0Phase(freq_scale=1.0)
    positions = torch.zeros(20, 3)
    positions[0, 0] = 100.0   # one huge outlier severity, rest clustered near 0
    positions[1:, 0] = torch.linspace(0, 0.01, 19)
    t = torch.linspace(0, 0.01, 50)
    seg = SegmentedPhaseTerm(n_segments=8, binning="uniform").prepare(term, positions, t)
    assert seg.bin_masks.shape == (8, 20)
    assert torch.all(seg.bin_masks.sum(dim=0) <= 1.0)   # no double-assignment
