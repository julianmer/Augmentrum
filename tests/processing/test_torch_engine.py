####################################################################################################
#                                     test_torch_engine.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-17                                                                              #
#                                                                                                  #
# Purpose: Holds the batched torch engine of RawProcessor to the FSL-MRS reference - step by       #
#          step, end to end, and on masked subsets exactly as on the gathered ones - and keeps     #
#          it on the device it was given.                                                          #
#                                                                                                  #
####################################################################################################

"""
Tests for registration_method='torch' and its building blocks.

The engine exists to be fast, so the thing worth guarding is that it is still
the FSL-MRS pipeline: every estimate against its reference, and a mask over
coils or transients against the subset it stands for.
"""

#*************#
#   imports   #
#*************#
import warnings

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from fsl_mrs.utils import preproc
from nifti_mrs_plus import Backend, NIfTI_MRS_Plus

from augmentrum.processing import torch_engine as engine
from augmentrum.processing.raw_processing import RawProcessor
from augmentrum.processing.utils import fid_to_spec, ppm_window
from tests.processing.test_raw_processing import (ALL_OFF, N_B, N_C, N_D, N_T, SF, SW, TAGS,
                                                  _few_averages, _rel, _synth_batch,
                                                  _synth_niftis, _water_with)

CUDA = torch.cuda.is_available()
FULL = dict(conj=False, coil=True, align=True, remove_outliers=True, average=True, ecc=True,
            truncate=False, remove_water=False, shift_ref=True, phase_correct=True)
WATER_TAGS = ['DIM_COIL', None, None]


#*************#
#   helpers   #
#*************#
def _against_list_engine(mets, wats, **settings):
    """
    The list engine's output next to the torch engine's, through the dispatch.

    Returns ((ref, ref_water), (got, got_water)).
    """
    outs = []
    for backend, extra in ((Backend.NIFTI_LIST, {}),
                           (Backend.PYTORCH, {'registration_method': 'torch'})):
        data = NIfTI_MRS_Plus([m.copy() for m in mets], backend=backend)
        water = (NIfTI_MRS_Plus([w.copy() for w in wats], backend=backend)
                 if wats is not None else None)
        outs.append(RawProcessor(**settings, **extra)(data, water))
    return outs


def _tensor_error(flags, **methods):
    """Max relative error of the torch engine against the list engine on raw tensors."""
    mets, wats, met_t, wat_t = _synth_batch()
    ref, _ = RawProcessor(volatile=True, **flags, **methods).process_nifti_list(mets, wats)
    got, _ = RawProcessor(volatile=True, registration_method='torch', **flags,
                          **methods).process_tensor(met_t, wat_t, sw_hz=SW, sf_mhz=SF,
                                                    dim_tags=TAGS)
    ref = np.stack([np.squeeze(n[:]) for n in ref])
    got = np.squeeze(np.asarray(got))
    if ref.shape != got.shape:
        ref = np.moveaxis(ref, 1, -1)
    return np.abs(ref - got).max() / np.abs(ref).max()


def _masked_and_gathered(settings, coils, dyns):
    """
    The torch engine on masked full batches and on the gathered subsets.

    Args:
        settings: RawProcessor flags.
        coils: Per sample, the coil indices kept (None: all).
        dyns: Per sample, the transient indices kept (None: all).

    Returns:
        (masked, gathered) outputs as NumPy arrays.
    """
    _, _, met_t, wat_t = _synth_batch()
    met = torch.from_numpy(met_t)
    wat = torch.from_numpy(wat_t[..., 0])                                  # one transient
    kw = dict(sw_hz=SW, sf_mhz=SF, dim_tags=TAGS, water_dim_tags=WATER_TAGS)

    masks = {}
    if coils is not None:
        masks['DIM_COIL'] = torch.zeros(N_B, N_C, dtype=torch.bool)
        for b, keep in enumerate(coils):
            masks['DIM_COIL'][b, keep] = True
    if dyns is not None:
        masks['DIM_DYN'] = torch.zeros(N_B, N_D, dtype=torch.bool)
        for b, keep in enumerate(dyns):
            masks['DIM_DYN'][b, keep] = True
    masked, _ = RawProcessor(registration_method='torch', volatile=True, **settings) \
        .process_tensor(met, wat, dim_masks=masks, **kw)

    gathered = []
    for b in range(N_B):
        sample, water = met[b:b + 1], wat[b:b + 1]
        if coils is not None:
            sample = sample[:, :, :, :, coils[b]]
            water = water[:, :, :, :, :, coils[b]]
        if dyns is not None:
            sample = sample[:, :, :, :, :, dyns[b]]
        out, _ = RawProcessor(registration_method='torch', volatile=True, **settings) \
            .process_tensor(sample, water, **kw)
        gathered.append(out)
    return masked.numpy(), torch.cat(gathered).numpy()


#**************************************************************************************************#
#                                   Class TestEngineParity                                         #
#**************************************************************************************************#
#                                                                                                  #
# Every step, and the whole pipeline, against the FSL-MRS list engine.                             #
#                                                                                                  #
#**************************************************************************************************#
class TestEngineParity:
    """Every step, and the whole pipeline, against the FSL-MRS list engine."""

    @pytest.mark.parametrize('flags', [
        {'coil': True},
        {'coil': True, 'average': True},
        {'coil': True, 'remove_outliers': True, 'average': True},
        {'coil': True, 'average': True, 'ecc': True},
        {'coil': True, 'average': True, 'shift_ref': True},
        {'coil': True, 'average': True, 'phase_correct': True},
        {'coil': True, 'average': True, 'conj': True},
        {'coil': True, 'average': True, 'truncate': True},
    ], ids=lambda f: '+'.join(sorted(f)))
    def test_each_step(self, flags):
        assert _tensor_error({**ALL_OFF, **flags}) < 1e-10

    def test_coil_combination_without_a_reference(self):
        mets, _ = _synth_niftis(2)
        (ref, _), (got, _) = _against_list_engine(mets, None, **{**ALL_OFF, 'coil': True})
        assert _rel(got.get_data(Backend.NUMPY), ref.get_data(Backend.NUMPY)) < 1e-10

    @pytest.mark.parametrize('method', ['smoothed', 'fsl-mrs'])
    def test_full_pipeline(self, method):
        assert _tensor_error(FULL, ecc_method=method) < 1e-5

    def test_full_pipeline_with_water_removal(self):
        assert _tensor_error({**FULL, 'remove_water': True}) < 1e-5

    @pytest.mark.parametrize('transients', [1, 2])
    def test_water_layouts_through_the_dispatch(self, transients):
        mets, wats = _synth_niftis(2)
        wats = [_water_with(w, transients) for w in wats]
        (ref, ref_w), (got, got_w) = _against_list_engine(mets, wats)

        assert got.get_data(Backend.NUMPY).shape == (2, 1, 1, 1, N_T)
        assert got.dim_tags == [None, None, None] and got_w.dim_tags == [None, None, None]
        assert _rel(got.get_data(Backend.NUMPY), ref.get_data(Backend.NUMPY)) < 1e-5
        assert _rel(got_w.get_data(Backend.NUMPY), ref_w.get_data(Backend.NUMPY)) < 1e-5

    def test_adaptive_combination(self):
        mets, wats = _synth_niftis(2)
        (ref, _), (got, _) = _against_list_engine(mets, wats, coil_method='adaptive')
        assert _rel(got.get_data(Backend.NUMPY), ref.get_data(Backend.NUMPY)) < 1e-5

    def test_too_few_noise_samples_drop_prewhitening(self):
        mets, wats = _few_averages(*_synth_niftis(2))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            (ref, _), (got, _) = _against_list_engine(mets, wats)
        assert any('prewhitening' in str(w.message) for w in caught)
        assert _rel(got.get_data(Backend.NUMPY), ref.get_data(Backend.NUMPY)) < 1e-10

    def test_dimensions_kept_without_averaging(self):
        mets, wats = _synth_niftis(2)
        wats = [_water_with(w, 1, keep_dyn=True) for w in wats]
        (ref, ref_w), (got, got_w) = _against_list_engine(mets, wats, **{**ALL_OFF, 'coil': True})

        assert got.get_data(Backend.NUMPY).shape == (2, 1, 1, 1, N_T, N_D)
        assert got.dim_tags == ['DIM_DYN', None, None]
        assert got_w.dim_tags == [None, None, None]
        assert _rel(got.get_data(Backend.NUMPY), ref.get_data(Backend.NUMPY)) < 1e-10

    def test_conftest_fixtures(self, dummy_nifti_mrs, dummy_nifti_water):
        """The shared noise fixtures (8 coils, 16 averages); no peaks to align on."""
        (ref, ref_w), (got, got_w) = _against_list_engine(
            [dummy_nifti_mrs], [dummy_nifti_water], align=False)

        assert got.get_data(Backend.NUMPY).shape == (1, 1, 1, 1, 2048)
        assert _rel(got.get_data(Backend.NUMPY), ref.get_data(Backend.NUMPY)) < 1e-5
        assert _rel(got_w.get_data(Backend.NUMPY), ref_w.get_data(Backend.NUMPY)) < 1e-5

    def test_numpy_in_numpy_out(self):
        _, _, met_t, wat_t = _synth_batch()
        got, got_w = RawProcessor(registration_method='torch', volatile=True).process_tensor(
            met_t, wat_t, sw_hz=SW, sf_mhz=SF, dim_tags=TAGS)
        assert isinstance(got, np.ndarray) and isinstance(got_w, np.ndarray)
        assert got.dtype == met_t.dtype

    def test_unknown_dimensions_are_refused(self):
        _, _, met_t, _ = _synth_batch()
        with pytest.raises(ValueError, match="DIM_EDIT"):
            RawProcessor(registration_method='torch').process_tensor(
                met_t, sw_hz=SW, sf_mhz=SF, dim_tags=['DIM_COIL', 'DIM_EDIT', None])

    def test_the_list_backend_is_routed_to_tensors(self):
        assert Backend.NIFTI_LIST not in RawProcessor(registration_method='torch') \
            .SUPPORTED_BACKENDS


#**************************************************************************************************#
#                                  Class TestEstimates                                             #
#**************************************************************************************************#
#                                                                                                  #
# The building blocks against the reference numerics they replace.                                #
#                                                                                                  #
#**************************************************************************************************#
class TestEstimates:
    """The building blocks against the reference numerics they replace."""

    def test_registration_matches_fsl_estimates(self):
        """Per transient, the shift within 0.05 Hz and the phase within a degree."""
        mets, wats = _synth_niftis(2)
        combined, _ = RawProcessor(volatile=True, **{**ALL_OFF, 'coil': True}) \
            .process_nifti_list(mets, wats)
        fids = np.stack([np.squeeze(n[:]).T for n in combined])            # (B, D, T)
        phi, eps = engine.align(torch.from_numpy(fids), torch.ones(fids.shape[:2], dtype=bool),
                                SW, SF, (0.2, 4.2))
        for b in range(fids.shape[0]):
            _, phi_ref, eps_ref = preproc.phase_freq_align(
                fids[b].copy(), SW, SF, nucleus='1H', ppmlim=(0.2, 4.2), niter=2,
                verbose=False, target=None)
            assert np.abs(eps[b].numpy() - eps_ref).max() < 0.05
            turn = np.angle(np.exp(1j * (phi[b].numpy() - phi_ref)))
            assert np.degrees(np.abs(turn)).max() < 1.0

    def test_registration_reaches_the_cost_minimum(self):
        """The shift cost of FSL-MRS, evaluated directly, is no higher than Powell's."""
        mets, wats = _synth_niftis(1)
        combined, _ = RawProcessor(volatile=True, **{**ALL_OFF, 'coil': True}) \
            .process_nifti_list(mets, wats)
        fids = np.squeeze(combined[0][:]).T.astype(np.complex128)           # (D, T)
        phi, eps = engine.align(torch.from_numpy(fids)[None], torch.ones(1, N_D, dtype=bool),
                                SW, SF, (0.2, 4.2))
        _, phi_ref, eps_ref = preproc.phase_freq_align(
            fids.copy(), SW, SF, nucleus='1H', ppmlim=(0.2, 4.2), niter=2, target=None)

        t = np.linspace(1 / SW, N_T / SW, N_T)
        first, last = ppm_window(N_T, SW, SF, (0.2, 4.2))
        target = fids[np.argmin(np.linalg.norm(fids - fids.mean(0), axis=-1))]
        window = fid_to_spec(target)[first:last]

        def cost(p, e):
            shifted = np.exp(-1j * p[:, None]) * fids * np.exp(-2j * np.pi * t * e[:, None])
            return np.linalg.norm(fid_to_spec(shifted)[:, first:last] - window, axis=-1)

        ours = cost(phi[0].numpy(), eps[0].numpy())
        assert np.all(ours <= cost(phi_ref, eps_ref) * (1 + 1e-6) + 1e-12)

    def test_masked_registration_ignores_the_masked(self):
        """Masked transients stay put, and the rest align as their subset would."""
        _, _, met_t, _ = _synth_batch()
        fids = torch.from_numpy(met_t[:, 0, 0, 0, 0])                       # (B, D, T)
        keep = torch.tensor([[True, False, True, True, False, True]] * N_B)
        phi, eps = engine.align(fids, keep, SW, SF, (0.2, 4.2))
        sub_phi, sub_eps = engine.align(fids[:, keep[0]], torch.ones(N_B, 4, dtype=bool),
                                        SW, SF, (0.2, 4.2))
        assert torch.all(phi[~keep] == 0) and torch.all(eps[~keep] == 0)
        assert torch.allclose(eps[keep].reshape(N_B, 4), sub_eps, atol=1e-9)
        assert torch.allclose(phi[keep].reshape(N_B, 4), sub_phi, atol=1e-9)

    def test_single_transient_is_not_aligned(self):
        fids = torch.randn(2, 3, 64, dtype=torch.complex128)
        keep = torch.tensor([[True, False, False], [True, True, True]])
        phi, eps = engine.align(fids, keep, SW, SF, (0.2, 4.2))
        assert phi[0].abs().sum() == 0 and eps[0].abs().sum() == 0

    def test_outlier_mask_matches_the_reference(self):
        _, _, met_t, _ = _synth_batch()
        fids = RawProcessor(volatile=True, **{**ALL_OFF, 'coil': True}).process_tensor(
            met_t, sw_hz=SW, sf_mhz=SF, dim_tags=TAGS)[0][:, 0, 0, 0]       # (B, D, T)
        ref = RawProcessor._unlike_mask(fids)
        got = engine.unlike_mask(torch.from_numpy(fids), torch.ones(N_B, N_D, dtype=bool))
        assert np.array_equal(got.numpy(), ref)
        assert not ref.all(), "the synthetic outlier must be caught"

    def test_masked_median_follows_numpy(self):
        rng = np.random.default_rng(0)
        values = rng.standard_normal((3, 7, 5)) + 1j * rng.standard_normal((3, 7, 5))
        keep = np.array([[1, 1, 0, 1, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0, 0]],
                        dtype=bool)
        got = engine.masked_median(torch.from_numpy(values), torch.from_numpy(keep)).numpy()
        for b in range(3):
            rows = values[b][keep[b]]
            ref = np.median(rows.real, axis=0) + 1j * np.median(rows.imag, axis=0)
            assert np.allclose(got[b], ref, atol=1e-14)

    def test_ecc_phase_and_unwrap(self):
        rng = np.random.default_rng(1)
        refs = rng.standard_normal((4, 300)) + 1j * rng.standard_normal((4, 300))
        assert np.allclose(engine.ecc_phase(torch.from_numpy(refs)).numpy(),
                           RawProcessor._ecc_phase(refs), atol=1e-10)
        phase = np.cumsum(rng.uniform(-3.5, 3.5, (4, 300)), axis=-1)
        assert np.array_equal(engine.unwrap(torch.from_numpy(phase)).numpy(),
                              np.unwrap(phase, axis=-1))

    @pytest.mark.parametrize('ratio', [1e-3, 0.5, 0.97])
    def test_principal_vector(self, ratio):
        generator = torch.Generator().manual_seed(0)
        q, _ = torch.linalg.qr(torch.randn(8, 16, 16, dtype=torch.complex128,
                                           generator=generator))
        spectrum = torch.cat([torch.ones(8, 1), ratio * torch.ones(8, 1),
                              0.5 * ratio * torch.rand(8, 14, generator=generator)], dim=-1)
        gram = (q * spectrum[:, None, :].to(q.dtype)) @ q.mH
        got = engine.principal_vector(gram)
        ref = torch.linalg.eigh(gram).eigenvectors[..., -1]
        assert torch.allclose((got.conj() * ref).sum(-1).abs(), torch.ones(8, dtype=torch.float64),
                              atol=1e-10)

    def test_noise_covariance_from_moments(self):
        """The subset covariance from cached per-transient moments is np.cov of the subset."""
        rng = np.random.default_rng(2)
        tails = rng.standard_normal((1, 1, 4, 6, 50)) + 1j * rng.standard_normal((1, 1, 4, 6, 50))
        second, first = engine.noise_moments(torch.from_numpy(tails))
        keep = torch.tensor([[True, False, True, True, False, True]])
        cov, n = engine.noise_covariance(second, first, 50, keep)
        samples = tails[0, 0][:, keep[0].numpy()].transpose(1, 2, 0).reshape(-1, 4)
        assert n.item() == samples.shape[0]
        assert np.allclose(cov[0].numpy(), np.cov(samples, rowvar=False), atol=1e-12)

    def test_masked_coil_weights_equal_the_gathered(self):
        rng = np.random.default_rng(3)
        reference = rng.standard_normal((2, 1, 40, 6)) + 1j * rng.standard_normal((2, 1, 40, 6))
        noise = rng.standard_normal((2, 6, 6)) + 1j * rng.standard_normal((2, 6, 6))
        cov = torch.from_numpy(noise @ noise.conj().transpose(0, 2, 1) + 6 * np.eye(6))
        gram = engine.reference_gram(torch.from_numpy(reference))
        keep = [0, 2, 3, 5]
        mask = torch.zeros(2, 6, dtype=torch.bool)
        mask[:, keep] = True
        whiten = torch.ones(2, dtype=torch.bool)
        masked = engine.wsvd_weights(gram, cov, mask, whiten, True)
        gathered = engine.wsvd_weights(gram[..., keep, :][..., keep], cov[:, keep][:, :, keep],
                                       torch.ones(2, 4, dtype=torch.bool), whiten, True)
        assert torch.allclose(masked[..., keep], gathered, rtol=1e-10, atol=0)
        assert torch.all(masked[..., [1, 4]] == 0)


#**************************************************************************************************#
#                                    Class TestMaskedSubsets                                       #
#**************************************************************************************************#
#                                                                                                  #
# A mask over coils or transients processes exactly as the subset it stands for.                   #
#                                                                                                  #
#**************************************************************************************************#
class TestMaskedSubsets:
    """A mask over coils or transients processes exactly as the subset it stands for."""

    COILS = [[0, 2], [1]]
    DYNS = [[0, 1, 3, 5], [2, 4, 5]]

    @pytest.mark.parametrize('coils, dyns', [(COILS, None), (None, DYNS), (COILS, DYNS)],
                             ids=['coils', 'transients', 'both'])
    def test_full_pipeline(self, coils, dyns):
        masked, gathered = _masked_and_gathered(FULL, coils, dyns)
        assert _rel(masked, gathered) < 1e-6

    def test_combination_and_average_only(self):
        settings = {**ALL_OFF, 'coil': True, 'average': True}
        masked, gathered = _masked_and_gathered(settings, self.COILS, self.DYNS)
        assert _rel(masked, gathered) < 1e-6

    def test_outliers_among_the_drawn(self):
        settings = {**ALL_OFF, 'coil': True, 'remove_outliers': True, 'average': True}
        masked, gathered = _masked_and_gathered(settings, None, [[0, 1, 2, 3, 5], [1, 2, 3, 4]])
        assert _rel(masked, gathered) < 1e-6

    def test_unconsumed_masks_become_zeros(self):
        """Without averaging the transient axis stays: dropped ones are zeros, and reported."""
        _, _, met_t, _ = _synth_batch()
        keep = torch.tensor([[True, True, False, True, True, False]] * N_B)
        processor = RawProcessor(registration_method='torch', volatile=True,
                                 **{**ALL_OFF, 'coil': True})
        out, _ = processor.process_tensor(torch.from_numpy(met_t), sw_hz=SW, sf_mhz=SF,
                                          dim_tags=TAGS, dim_masks={'DIM_DYN': keep})
        assert torch.all(out[..., ~keep[0], :] == 0)
        assert torch.all(out[..., keep[0], :].abs().amax(-1) > 0)
        assert torch.equal(processor.dim_masks_['DIM_DYN'], keep)

    def test_reference_engine_processes_masks_per_sample(self):
        """The NumPy engines take masks too, one gathered sample at a time."""
        mets, wats = _synth_niftis(2)
        wats = [_water_with(w, 1) for w in wats]
        met_t = np.moveaxis(np.stack([n[:] for n in mets]), 4, -1)
        wat_t = np.stack([n[:] for n in wats])
        keep = np.array([[True, False, True, True, True, True],
                         [True, True, True, False, True, False]])
        got, _ = RawProcessor(volatile=True).process_tensor(
            met_t, wat_t, sw_hz=SW, sf_mhz=SF, dim_tags=TAGS, water_dim_tags=WATER_TAGS,
            dim_masks={'DIM_DYN': keep})
        for b in range(2):
            sample = np.ascontiguousarray(met_t[b:b + 1][:, :, :, :, :, keep[b]])
            ref, _ = RawProcessor(volatile=True).process_tensor(
                sample, wat_t[b:b + 1], sw_hz=SW, sf_mhz=SF, dim_tags=TAGS,
                water_dim_tags=WATER_TAGS)
            assert _rel(got[b:b + 1], ref) < 1e-12


#**************************************************************************************************#
#                                     Class TestDevices                                            #
#**************************************************************************************************#
#                                                                                                  #
# The engine stays on the device it was given.                                                     #
#                                                                                                  #
#**************************************************************************************************#
class TestDevices:
    """The engine stays on the device it was given."""

    def _run(self, device):
        _, _, met_t, wat_t = _synth_batch()
        keep = torch.tensor([[True, False, True]] * N_B, device=device)
        return RawProcessor(registration_method='torch', volatile=True).process_tensor(
            torch.from_numpy(met_t).to(device), torch.from_numpy(wat_t).to(device),
            backend=Backend.PYTORCH, sw_hz=SW, sf_mhz=SF, dim_tags=TAGS,
            dim_masks={'DIM_COIL': keep})

    def test_cpu(self):
        met, wat = self._run('cpu')
        assert met.device.type == 'cpu' and wat.device.type == 'cpu'

    @pytest.mark.skipif(not CUDA, reason="CUDA not available")
    def test_cuda_matches_cpu(self):
        met, wat = self._run('cuda')
        assert met.device.type == 'cuda' and wat.device.type == 'cuda'
        cpu, _ = self._run('cpu')
        assert _rel(met.cpu().numpy(), cpu.numpy()) < 1e-8
