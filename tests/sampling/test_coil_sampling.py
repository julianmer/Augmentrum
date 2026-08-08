####################################################################################################
#                                      test_coil_sampling.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-07                                                                              #
#                                                                                                  #
# Purpose: Holds the receive array to the properties that make it worth having: the maps are       #
#          physical, applying them survives every backend and gradients, and the resulting         #
#          acquisition genuinely encodes position.                                                 #
#                                                                                                  #
####################################################################################################

"""
Tests for synthetic sensitivity maps and the module that applies them.

The decisive test is the last one. Undersampling coil-combined data throws
information away and nothing can bring it back; undersampling a receive array
does not, because the elements encode position. That distinction is the whole
reason this exists, so it is asserted directly on the encoding rather than
inferred from an image looking better.
"""

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from nifti_mrs_plus import ops

from augmentrum.core import Backend, NIfTI_MRS_Plus
from augmentrum.sampling import (
    Birdcage,
    CoilSampler,
    KspaceUndersampling,
    MapSource,
    Supplied,
    T2starMove,
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


#**************#
#   fixtures   #
#**************#

@pytest.fixture
def volume():
    """One small MRSI volume, "(batch, X, Y, Z, T)"."""
    rng = np.random.default_rng(0)
    return (rng.standard_normal((2, 8, 8, 4, 6))
            + 1j * rng.standard_normal((2, 8, 8, 4, 6))).astype(np.complex64)


#*******************#
#   map synthesis   #
#*******************#

@pytest.mark.parametrize("matrix", [(16, 16), (16, 16, 4)])
def test_maps_are_unit_sensitivity(matrix):
    """Root-sum-of-squares across coils is 1, so the array does not rescale."""
    maps = Birdcage(n_coils=8).maps(matrix)

    assert maps.shape == tuple(matrix) + (8,)
    assert np.allclose(np.sqrt((np.abs(maps) ** 2).sum(-1)), 1.0, atol=1e-5)


def test_maps_are_smooth():
    """A sensitivity map is low-frequency; that is what makes it estimable."""
    maps = Birdcage(n_coils=8).maps((32, 32, 8))

    energy = np.abs(np.fft.fftshift(np.fft.fftn(maps[..., 0], axes=(0, 1, 2)),
                                    axes=(0, 1, 2))) ** 2
    cx, cy = energy.shape[0] // 2, energy.shape[1] // 2
    core = energy[cx - 4:cx + 5, cy - 4:cy + 5, :].sum() / energy.sum()

    assert core > 0.9, f"only {core:.1%} of the map's energy is low-frequency"


def test_more_coils_each_cover_less():
    """Unit total sensitivity shared over more elements means less each."""
    means = [np.abs(Birdcage(n_coils=n).maps((16, 16, 4))).mean()
             for n in (2, 4, 8, 16)]

    assert all(a > b for a, b in zip(means, means[1:])), means


def test_radius_flattens_the_maps():
    """Elements further out see the volume more evenly."""
    spread = []
    for radius in (1.1, 1.5, 3.0):
        one = np.abs(Birdcage(n_coils=8, radius=radius).maps((16, 16, 4))[..., 0])
        spread.append(one.max() / one.min())

    assert all(a > b for a, b in zip(spread, spread[1:])), spread


def test_rings_encode_z():
    """One ring is blind along z; stacking rings is what gives it structure."""
    def variation(n_rings):
        return np.abs(Birdcage(n_coils=8, n_rings=n_rings)
                      .maps((16, 16, 8))).std(axis=2).mean()

    assert variation(2) > variation(1)


@pytest.mark.parametrize("matrix", [(16,), (8, 8, 8, 8)])
def test_matrix_rank_is_checked(matrix):
    """Only 2-D and 3-D grids describe something an array can sit around."""
    with pytest.raises(ValueError, match="2-D or 3-D"):
        Birdcage(n_coils=4).maps(matrix)


#***********************#
#   applying the maps   #
#***********************#

def test_coil_axis_follows_the_nifti_convention(volume):
    """Coils land after the spectral axis, where NIfTI-MRS puts them."""
    out, _ = CoilSampler(mode='synthesize', n_coils=5, seed=0).process_tensor(volume)

    assert out.shape == volume.shape + (5,)
    assert out.dtype == np.complex64


def test_combining_the_array_recovers_the_input(volume):
    """Unit total sensitivity means an RSS combination gives the volume back."""
    out, _ = CoilSampler(mode='synthesize', n_coils=6, seed=0).process_tensor(volume)

    assert np.allclose(np.sqrt((np.abs(out) ** 2).sum(-1)), np.abs(volume), atol=1e-4)


def test_supplied_maps_are_used(volume):
    """A caller with real maps can hand them over instead of synthesizing."""
    maps = Birdcage(n_coils=3).maps((8, 8, 4))
    out, _ = CoilSampler(mode='synthesize', source=Supplied(maps), seed=0).process_tensor(volume)

    assert np.allclose(out, volume[..., None] * maps[None, :, :, :, None, :], atol=1e-5)


def test_supplied_maps_must_match_the_matrix(volume):
    """Maps covering a different grid are a mistake, not something to broadcast."""
    with pytest.raises(ValueError, match="must match the spatial matrix"):
        CoilSampler(mode='synthesize', source=Supplied(Birdcage(n_coils=3).maps((4, 4, 2))),
                    seed=0).process_tensor(volume)


def test_data_that_already_has_coils_is_rejected(volume):
    """Applying an array to data that has one would be meaningless."""
    with_coils, _ = CoilSampler(mode='synthesize', n_coils=2, seed=0).process_tensor(volume)

    with pytest.raises(ValueError, match="already carries a coil axis"):
        CoilSampler(mode='synthesize', n_coils=2, seed=0).process_tensor(with_coils)


def test_a_seed_replays_exactly(volume):
    """Reproducible on demand: same seed, same array placement."""
    first, _ = CoilSampler(mode='synthesize', n_coils=4, seed=7).process_tensor(volume)
    again, _ = CoilSampler(mode='synthesize', n_coils=4, seed=7).process_tensor(volume)
    other, _ = CoilSampler(mode='synthesize', n_coils=4, seed=8).process_tensor(volume)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


#**************#
#   backends   #
#**************#

def test_every_backend_agrees_and_stays_native(volume):
    """The same maps on any backend give the same answer, without converting."""
    maps = Birdcage(n_coils=3).maps((8, 8, 4))
    reference = None

    for backend in Backend:
        if backend is Backend.NIFTI_LIST:
            continue
        try:
            native = ops.match_backend(volume, _probe(backend))
        except (ImportError, RuntimeError):
            pytest.skip(f"{backend.value} not installed")

        out, _ = CoilSampler(mode='synthesize', source=Supplied(maps), seed=0).process_tensor(native)

        assert type(out).__module__.split('.')[0] == type(native).__module__.split('.')[0], (
            f"{backend.value} left its own backend"
        )
        result = ops.to_numpy(out)
        if reference is None:
            reference = result
        assert np.abs(result - reference).max() < 1e-5


def _probe(backend: Backend):
    """A one-element tensor on *backend*, to promote against."""
    if backend is Backend.NUMPY:
        return np.zeros(1, np.complex64)
    if backend is Backend.PYTORCH:
        import torch
        return torch.zeros(1, dtype=torch.complex64)
    if backend is Backend.JAX:
        import jax.numpy as jnp
        return jnp.zeros(1, jnp.complex64)
    if backend is Backend.TENSORFLOW:
        import tensorflow as tf
        return tf.zeros(1, tf.complex64)
    if backend is Backend.KERAS:
        import keras
        return keras.ops.zeros(1, 'complex64')
    raise RuntimeError(f"no probe for {backend}")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_gradients_reach_the_data_through_the_pipeline(volume):
    """Applying an array is a multiply, so a training loop can sit around it."""
    from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

    plus = NIfTI_MRS_Plus(
        nifti_list=[gen_nifti_mrs(volume[0], 1 / 2000.0, 123.0)],
        backend=Backend.PYTORCH, volatile=True)

    leaf = torch.tensor(2.0, requires_grad=True)
    plus.set_data(plus.get_data(Backend.PYTORCH)
                  * torch.complex(leaf, torch.zeros_like(leaf)), Backend.PYTORCH)

    out, _ = CoilSampler(mode='synthesize', n_coils=4, seed=0)(plus)
    result = out.get_data(Backend.PYTORCH)

    assert result.grad_fn is not None, "the coil axis severed the autograd graph"
    torch.abs(result).sum().backward()
    assert leaf.grad is not None and float(leaf.grad.abs()) > 0


def test_the_result_is_a_real_nifti_coil_dimension(volume):
    """DIM_COIL after the spectral axis, so coil combination works on it."""
    from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

    plus = NIfTI_MRS_Plus(
        nifti_list=[gen_nifti_mrs(volume[0], 1 / 2000.0, 123.0)],
        backend=Backend.NUMPY)

    out, _ = CoilSampler(mode='synthesize', n_coils=6, seed=0)(plus)
    obj = out.list()[0]

    assert obj.dim_tags[0] == 'DIM_COIL'
    assert obj.shape == volume.shape[1:] + (6,)
    assert np.allclose(np.sqrt((np.abs(obj[:]) ** 2).sum(-1)), np.abs(volume[0]), atol=1e-4)


#***********************#
#   per-coil sampling   #
#***********************#

@pytest.mark.parametrize("ksp_mode,kwargs", [
    ("cartesian", {}),
    ("gridded", {"trajectory": "radial_2d"}),
])
def test_undersampling_accepts_a_receive_array(volume, ksp_mode, kwargs):
    """Six dimensions go in, six come out, one coil at a time."""
    coils, _ = CoilSampler(mode='synthesize', n_coils=3, seed=0).process_tensor(volume)
    out, _ = KspaceUndersampling(ksp_mode=ksp_mode, acceleration_factor=2.0,
                                 us_seed=0, **kwargs).process_tensor(coils)

    assert out.shape == coils.shape


def test_every_coil_meets_the_same_acquisition(volume):
    """A receive array measures one trajectory; all its elements share it."""
    coils, _ = CoilSampler(mode='synthesize', n_coils=4, seed=0).process_tensor(volume)

    masks = []
    for c in range(coils.shape[-1]):
        module = KspaceUndersampling(ksp_mode='cartesian', acceleration_factor=2.0,
                                     us_seed=0)
        module.process_tensor(coils[..., c])
        masks.append(np.asarray(module.last_masks_))

    assert all(np.array_equal(masks[0], m) for m in masks[1:])


#***************************#
#   the encoding property   #
#***************************#

def test_coils_make_an_accelerated_acquisition_invertible():
    """
    The point of the whole exercise, asserted on the encoding itself.

    At acceleration R a single coil measures N/R of the N unknowns, so the
    system is rank-deficient and the discarded positions are gone for good.
    Sensitivities add independent equations without adding samples, and once
    there are enough of them the volume is recoverable exactly.

    This is stated as a least-squares inverse of the explicit forward operator
    rather than as image quality, because the module produces a zero-filled
    reconstruction: combining aliased coil images leaves them aliased. What the
    receive array changes is whether the information is there to be recovered,
    which is what is measured here.
    """
    n, acceleration = 8, 2
    rng = np.random.default_rng(0)
    image = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))

    mask = np.zeros((n, n), bool)
    mask[::acceleration, :] = True

    def encoding(n_coils):
        """Forward operator: weight by each coil, transform, keep sampled bins."""
        maps = Birdcage(n_coils=n_coils).maps((n, n)).astype(np.complex128)
        columns = []
        for i in range(n * n):
            pixel = np.zeros(n * n, np.complex128)
            pixel[i] = 1.0
            columns.append(np.concatenate(
                [np.fft.fft2(pixel.reshape(n, n) * maps[..., c])[mask]
                 for c in range(n_coils)]))
        return np.array(columns).T

    errors = {}
    for n_coils in (1, 2, 8):
        operator = encoding(n_coils)
        estimate, *_ = np.linalg.lstsq(operator, operator @ image.ravel(), rcond=None)
        errors[n_coils] = (np.linalg.norm(estimate - image.ravel())
                           / np.linalg.norm(image))
        rank = np.linalg.matrix_rank(operator)

        if n_coils >= acceleration:
            assert rank == n * n, (
                f"{n_coils} coils at R={acceleration} should fully determine the "
                f"volume, but the operator has rank {rank}/{n * n}"
            )

    assert errors[1] > 0.1, "one coil should not be able to recover the volume"
    assert errors[2] < 1e-6, "two coils at R=2 should recover it exactly"
    assert errors[8] < 1e-6, "more coils should stay exact"


#*******************#
#   drawing         #
#*******************#

def test_drawing_is_native_and_keeps_gradients(volume):
    """A draw is a gather along one axis, so it need not leave the backend."""
    array, _ = CoilSampler(mode='synthesize', n_coils=8, seed=0).process_tensor(volume)
    tensor = torch.from_numpy(array).requires_grad_(True)

    drawn, _ = CoilSampler(mode='random', n_coils=3, seed=0).process_tensor(
        tensor, dim_tags=['DIM_COIL', None, None])

    assert drawn.shape[-1] == 3
    assert torch.is_tensor(drawn), "drawing left the tensor backend"
    drawn.abs().sum().backward()
    assert tensor.grad is not None and float(tensor.grad.abs().sum()) > 0


def test_drawing_without_tags_is_a_no_op(volume):
    """Nothing is named, so there is no axis to draw along."""
    array, _ = CoilSampler(mode='synthesize', n_coils=4, seed=0).process_tensor(volume)
    drawn, _ = CoilSampler(mode='random', n_coils=2, seed=0).process_tensor(array)

    assert drawn.shape == array.shape


def test_deterministic_keeps_everything(volume):
    """Nothing is discarded unless a draw asks for it."""
    array, _ = CoilSampler(mode='synthesize', n_coils=5, seed=0).process_tensor(volume)
    kept, _ = CoilSampler(mode='deterministic').process_tensor(
        array, dim_tags=['DIM_COIL', None, None])

    assert np.array_equal(kept, array)


def test_only_the_coil_axis_is_touched(volume):
    """Averages belong to their own sampler; this one must leave them alone."""
    batch = np.ones(volume.shape + (6, 10), np.complex64)

    drawn, _ = CoilSampler(mode='random', n_coils=3, seed=0).process_tensor(
        batch, dim_tags=['DIM_COIL', 'DIM_DYN', None])

    assert drawn.shape[-2:] == (3, 10), "the average axis must be left as it was"


def test_an_unknown_mode_is_refused():
    """A typo should not silently become a mode that does nothing."""
    with pytest.raises(ValueError, match="mode must be one of"):
        CoilSampler(mode='sythesize')


#***********************#
#   the whole chain     #
#***********************#

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_synthesize_draw_undersample_stays_differentiable():
    """
    The reason the two halves live in one module.

    Synthesizing an array, drawing from it and undersampling it is the natural
    order for simulating an accelerated multi-coil acquisition, and every step
    has to stay on the tensor for the whole thing to sit in a training loop.
    """
    from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

    plus = NIfTI_MRS_Plus(
        nifti_list=[gen_nifti_mrs(np.ones((8, 8, 2, 16), np.complex64), 1 / 2000.0, 123.0)],
        backend=Backend.PYTORCH, volatile=True)

    leaf = torch.tensor(2.0, requires_grad=True)
    plus.set_data(plus.get_data(Backend.PYTORCH)
                  * torch.complex(leaf, torch.zeros_like(leaf)), Backend.PYTORCH)

    for module in (CoilSampler(mode='synthesize', n_coils=8, seed=0),
                   CoilSampler(mode='random', n_coils=4, seed=0),
                   KspaceUndersampling(ksp_mode='cartesian',
                                       acceleration_factor=2.0, us_seed=0)):
        plus, _ = module(plus)
        assert plus.dim_tags[0] == 'DIM_COIL', f"{type(module).__name__} lost the coil tag"
        assert plus.get_data(Backend.PYTORCH).grad_fn is not None, (
            f"{type(module).__name__} severed the autograd graph"
        )

    result = plus.get_data(Backend.PYTORCH)
    assert tuple(result.shape) == (1, 8, 8, 2, 16, 4)
    torch.abs(result).sum().backward()
    assert leaf.grad is not None and float(leaf.grad.abs()) > 0

    # and it still materializes into a valid NIfTI-MRS object at the end
    obj = plus.list()[0]
    assert obj.shape == (8, 8, 2, 16, 4)
    assert obj.dim_tags[0] == 'DIM_COIL'


#***********************#
#   measured maps       #
#***********************#
# These never reach the network. Downloading itself is tested next to the code
# that does it, in tests/utils/test_download.py; what is tested here is what is
# specific to this dataset - the layout it stores maps in, the cache it leaves,
# and that the result is interchangeable with a synthetic array.

h5py = pytest.importorskip("h5py")


@pytest.fixture
def fake_archive(tmp_path, monkeypatch):
    """A stand-in T2*-MOVE archive, in the layout the dataset documents."""
    from augmentrum.sampling import coil_sampling
    from augmentrum.utils import download

    monkeypatch.setenv('AUGMENTRUM_CACHE', str(tmp_path))

    slices, echoes, coils, phase, readout = 4, 3, 8, 10, 12
    rng = np.random.default_rng(0)
    sens = (rng.standard_normal((slices, echoes, coils, phase, readout))
            + 1j * rng.standard_normal((slices, echoes, coils, phase, readout))
            ).astype(np.complex64)

    raw = tmp_path / 'raw.hf'
    with h5py.File(raw, 'w') as handle:
        handle.create_dataset('sens_maps', data=sens)

    cache = download.cache_root() / 'csm'
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / coil_sampling.T2starMove.ARCHIVE

    import zipfile
    with zipfile.ZipFile(archive, 'w') as bundle:
        for subject in ('sub-02', 'sub-04'):
            bundle.write(raw, f'mr_data/val_recon/{subject}/t2s_gre_fr.hf')
            bundle.write(raw, f'mr_data/val_recon/{subject}/t2s_gre_fr_recon.hf')
    return archive


def test_measured_maps_arrive_in_this_packages_layout(fake_archive):
    """The dataset stores coils third; this package wants them last."""
    maps = T2starMove.fetch(progress=False, keep_archive=True)

    assert maps.shape == (12, 10, 4, 8)       # (X, Y, Z, C) from (S, E, C, PE, RO)
    assert maps.dtype == np.complex64
    assert np.allclose(np.sqrt((np.abs(maps) ** 2).sum(-1)), 1.0, atol=1e-5)


def test_the_archive_is_read_once(fake_archive):
    """A 5.4 GB fetch is not repeated: later calls read the cache."""
    first = T2starMove.fetch(progress=False)
    assert not fake_archive.exists(), "the archive should be dropped once extracted"

    again = T2starMove.fetch(progress=False)
    assert np.array_equal(first, again)


def test_a_subject_can_be_chosen(fake_archive):
    """Different volunteers wore the array differently, so the choice matters."""
    from augmentrum.sampling import coil_sampling
    import zipfile

    with zipfile.ZipFile(fake_archive) as bundle:
        assert '/sub-04/' in coil_sampling.T2starMove._member(bundle, 'sub-04')

        with pytest.raises(ValueError, match="not in this archive"):
            coil_sampling.T2starMove._member(bundle, 'sub-99')


def test_measured_maps_are_interchangeable_with_synthetic(fake_archive):
    """Same convention, so the sampler cannot tell them apart."""
    maps = T2starMove.fetch(progress=False)
    volume = np.ones((1,) + maps.shape[:3] + (5,), np.complex64)

    out, _ = CoilSampler(mode='synthesize', source=Supplied(maps), seed=0).process_tensor(volume)

    assert out.shape == volume.shape + (maps.shape[-1],)
    assert np.allclose(np.sqrt((np.abs(out) ** 2).sum(-1)), np.abs(volume), atol=1e-4)


def test_measured_maps_are_normalized_where_there_is_signal(fake_archive):
    """
    Unit sensitivity inside the object, nothing outside it.

    A measured map has no sensitivity in air, so those voxels normalize to zero
    rather than to one. That is the honest answer and it is what makes measured
    maps carry the object's extent, which a synthetic array does not.
    """
    from augmentrum.sampling.coil_sampling import T2starMove
    import h5py

    # a map set that is zero outside a small block, as a real one is outside the head
    coils, shape = 4, (3, 1, 4, 6, 8)
    sens = np.zeros(shape, np.complex64)
    sens[:, :, :, 2:4, 2:6] = 1.0 + 1j

    raw = fake_archive.parent / 'block.hf'
    with h5py.File(raw, 'w') as handle:
        handle.create_dataset('sens_maps', data=sens[:, :, :coils])

    maps = T2starMove._read(raw)
    rss = np.sqrt((np.abs(maps) ** 2).sum(-1))

    assert np.allclose(rss[rss > 0], 1.0, atol=1e-5), "signal voxels are not unit sensitivity"
    assert (rss == 0).any(), "background should stay zero, not be normalized to one"


#***********************#
#   resizing an array   #
#***********************#
# What an array of maps can and cannot be turned into. The distinction that
# matters is between more elements and more information: displaced copies of a
# measured element are new functions and do add independent channels, whereas
# any recombination of what is already there cannot.

def _rank(maps):
    """Independent channels among *maps*, counting only voxels it saw."""
    flat = maps.reshape(-1, maps.shape[-1])
    return np.linalg.matrix_rank(flat[np.abs(flat).sum(1) > 0])


@pytest.fixture
def array():
    """An array of overlapping elements, as a real one is."""
    return Birdcage(n_coils=12).maps((12, 12, 4))


def test_compression_keeps_the_count_it_was_asked_for(array):
    """Fewer virtual coils, still unit sensitivity."""
    fewer = MapSource.compress(array, 4)

    assert fewer.shape == array.shape[:3] + (4,)
    rss = np.sqrt((np.abs(fewer) ** 2).sum(-1))
    assert np.allclose(rss[rss > 1e-6], 1.0, atol=1e-4)


def test_compression_keeps_the_strongest_directions(array):
    """Four virtual coils out of twelve should still see most of the array."""
    flat = array.reshape(-1, array.shape[-1])
    spectrum = np.linalg.svd(flat[np.abs(flat).sum(1) > 0], compute_uv=False)
    kept = (spectrum[:4] ** 2).sum() / (spectrum ** 2).sum()

    assert kept > 0.5, f"the leading four directions hold only {kept:.1%}"
    assert _rank(MapSource.compress(array, 4)) == 4


def test_extending_adds_independent_channels(array):
    """Displaced elements are new functions, so the rank actually grows."""
    rng = np.random.default_rng(0)
    bigger = MapSource.extend(array, 20, rng)

    assert bigger.shape[-1] == 20
    assert _rank(bigger) > _rank(array), "displaced copies added no new channels"


def test_recombination_cannot_add_information(array):
    """
    The limit worth stating out loud.

    Mixing the elements of an array, however many mixtures are made, spans no
    more than the array itself. Asking for more coils buys extra elements to
    sample with, never extra knowledge about what was scanned.
    """
    rng = np.random.default_rng(0)
    mixed = np.tensordot(array, rng.standard_normal((array.shape[-1], 40)), axes=([-1], [0]))

    assert mixed.shape[-1] == 40
    assert _rank(mixed) <= _rank(array)


@pytest.mark.parametrize("wanted", [4, 12, 20])
def test_the_sampler_resizes_supplied_maps(volume, wanted):
    """n_coils applies to supplied maps too, in either direction."""
    supplied = Birdcage(n_coils=12).maps(volume.shape[1:4])
    out, _ = CoilSampler(mode='synthesize', source=Supplied(supplied), n_coils=wanted,
                         seed=0).process_tensor(volume)

    assert out.shape[-1] == wanted


def test_supplied_maps_are_used_as_they_are_by_default(volume):
    """Maps already say how many elements there are; do not silently resize."""
    supplied = Birdcage(n_coils=7).maps(volume.shape[1:4])
    out, _ = CoilSampler(mode='synthesize', source=Supplied(supplied), seed=0).process_tensor(volume)

    assert out.shape[-1] == 7


#*******************#
#   map sources     #
#*******************#
# Three ways to answer the same question, so a caller can swap between them
# without anything else changing. Each keeps its own settings; resizing to a
# coil count happens once, in the contract, not in each of them.

def test_every_source_answers_the_same_contract():
    """Whatever the origin, the result is maps over the grid asked for."""
    matrix = (8, 8, 4)
    supplied = Birdcage(n_coils=5).maps(matrix)

    for source in (Birdcage(n_coils=6), Supplied(supplied)):
        maps = source.maps(matrix)

        assert isinstance(source, MapSource)
        assert maps.shape[:3] == matrix
        assert maps.dtype == np.complex64


def test_a_source_carries_its_own_settings():
    """Ring radius belongs to the birdcage, not to whatever applies it."""
    matrix = (12, 12, 4)
    tight = np.abs(Birdcage(n_coils=8, radius=1.1).maps(matrix)[..., 0])
    loose = np.abs(Birdcage(n_coils=8, radius=3.0).maps(matrix)[..., 0])

    assert tight.max() / tight.min() > loose.max() / loose.min()


@pytest.mark.parametrize("wanted", [3, 5, 11])
def test_the_contract_resizes_whatever_the_source_gives(wanted):
    """One place decides the coil count, so no source has to think about it."""
    supplied = Supplied(Birdcage(n_coils=5).maps((8, 8, 4)))

    assert supplied.maps((8, 8, 4), n_coils=wanted).shape[-1] == wanted


def test_a_source_that_cannot_cover_the_grid_says_so():
    """Maps for a different matrix are a mistake, not something to broadcast."""
    source = Supplied(Birdcage(n_coils=4).maps((4, 4, 2)))

    with pytest.raises(ValueError, match="must match the spatial matrix"):
        source.maps((8, 8, 4))


def test_the_default_source_needs_nothing(volume):
    """Every dataset can have a receive array, with no download and no maps."""
    out, _ = CoilSampler(mode='synthesize').process_tensor(volume)

    assert out.shape == volume.shape + (8,)
