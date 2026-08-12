####################################################################################################
#                                  test_kspace_reconstructor.py                                    #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-07-31                                                                              #
#                                                                                                  #
# Purpose: Verifies the NUFFT reconstruction path end to end: coordinate normalization, the        #
#          forward/adjoint operator pair, and gridding reconstruction from real trajectories.      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from augmentrum.sampling import KspaceReconstructor, TrajectoryRegistry

torch = pytest.importorskip("torch")
tkbn = pytest.importorskip("torchkbnufft",
                           reason="torchkbnufft is an optional dependency")


N = 64
HEADER = {"dim": [4, N, N, 1, 1, 1, 1, 1],
          "pixdim": [1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
          "DwellTime": 0.83e-3}


#*************#
#   helpers   #
#*************#
def _trajectory(name, **params):
    """Trajectory as normalized "[1, S, D, L]" coordinates, plus its metadata."""
    shots, meta = TrajectoryRegistry.generate(name, HEADER, params)
    coords = np.stack([np.asarray(s) for s in shots]).transpose(0, 2, 1)   # [S, D, L]
    coords = torch.from_numpy(coords).float().unsqueeze(0)
    return KspaceReconstructor.normalize_trajectory(coords, meta["kmax"]), meta


def _phantom():
    """A small off-center disc — band-limited enough to survive a disc trajectory."""
    y, x = np.mgrid[0:N, 0:N]
    img = ((x - N / 2 + 6) ** 2 + (y - N / 2 - 4) ** 2 < (N / 5) ** 2).astype(np.float32)
    return img


def _sample(img, coords):
    """Forward NUFFT: what a scanner would measure along *coords*. "[1, S, C, L]"."""
    S, L = coords.shape[1], coords.shape[3]
    ktraj, _ = KspaceReconstructor.flatten(coords)
    flat = tkbn.KbNufft(im_size=(N, N), grid_size=(2 * N, 2 * N))(
        torch.from_numpy(img).to(torch.complex64)[None, None], ktraj)
    return flat.reshape(1, 1, S, L).permute(0, 2, 1, 3)


def _ls_scale(rec, ref):
    """Least-squares scale factor; gridding fixes the image only up to a constant."""
    return rec * (np.vdot(rec.ravel(), ref.ravel()).real
                  / np.vdot(rec.ravel(), rec.ravel()).real)


def _nrmse(ref, test):
    return float(100.0 * np.sqrt(np.mean((test - ref) ** 2)) / np.ptp(ref))


#******************************#
#   coordinate normalization   #
#******************************#
def test_normalize_trajectory_is_per_axis():
    """Anisotropic voxels give a per-axis kmax, and each axis must use its own."""
    coords = torch.ones(1, 2, 3, 4)
    out = KspaceReconstructor.normalize_trajectory(coords, (250.0, 125.0, 500.0))
    assert np.allclose(out[0, 0, :, 0].tolist(), [1 / 250.0, 1 / 125.0, 1 / 500.0])

    scalar = KspaceReconstructor.normalize_trajectory(coords, 250.0)
    assert np.allclose(scalar.numpy(), (coords / 250.0).numpy())

    with pytest.raises(ValueError, match="one value per axis"):
        KspaceReconstructor.normalize_trajectory(coords, (1.0, 2.0))
    with pytest.raises(ValueError, match="positive"):
        KspaceReconstructor.normalize_trajectory(coords, (1.0, 0.0, 2.0))


def test_real_trajectory_fills_the_unit_box():
    """A trajectory normalized by its own kmax reaches the edge but never leaves."""
    for name, params in [("radial_2d", {"n_shots": 33}),
                         ("spiral_2d", {"n_shots": 8}),
                         ("cartesian_2d", {})]:
        coords, _ = _trajectory(name, **params)
        peak = float(coords.abs().max())
        assert peak <= 1.0 + 1e-5, f"{name} leaves the unit box at {peak}"
        assert peak > 0.9, f"{name} only reaches {peak}, so kmax is mis-scaled"


#***********************#
#   operator identity   #
#***********************#
def test_forward_adjoint_are_a_true_pair():
    """Dot-product test: <Ax, y> == <x, A^H y> to within float32 precision."""
    coords, _ = _trajectory("radial_2d", n_shots=33)
    ktraj, _ = KspaceReconstructor.flatten(coords)
    n_samples = ktraj.shape[-1]

    fwd = tkbn.KbNufft(im_size=(N, N), grid_size=(2 * N, 2 * N))
    adj = tkbn.KbNufftAdjoint(im_size=(N, N), grid_size=(2 * N, 2 * N))

    torch.manual_seed(0)
    x = torch.randn(1, 1, N, N, dtype=torch.complex64)
    y = torch.randn(1, 1, n_samples, dtype=torch.complex64)

    lhs = torch.vdot(fwd(x, ktraj).ravel(), y.ravel())
    rhs = torch.vdot(x.ravel(), adj(y, ktraj).ravel())
    assert abs(lhs - rhs) / abs(lhs) < 1e-4


#********************#
#   reconstruction   #
#********************#
def test_cartesian_roundtrip_is_near_exact():
    """
    Cartesian samples land on the grid, so the NUFFT reduces to an FFT.

    This is the reference case: it pins down the units bridge, the shot/sample
    flattening and the [-pi, pi] convention all at once. Any error here is a
    plumbing bug, not a gridding approximation.
    """
    img = _phantom()
    coords, _ = _trajectory("cartesian_2d")
    rec = np.abs(KspaceReconstructor((N, N), 2.0)(coords, _sample(img, coords))[0, 0].numpy())
    assert _nrmse(img, _ls_scale(rec, img)) < 1.0


def test_point_source_is_not_shifted():
    """A delta must reconstruct at its own pixel — catches fftshift and flip errors."""
    delta = np.zeros((N, N), np.float32)
    delta[N // 2 + 5, N // 2 - 7] = 1.0

    for name, params in [("cartesian_2d", {}), ("radial_2d", {"n_shots": 65})]:
        coords, _ = _trajectory(name, **params)
        rec = np.abs(KspaceReconstructor((N, N), 2.0)(coords, _sample(delta, coords))[0, 0].numpy())
        assert np.unravel_index(rec.argmax(), rec.shape) == (N // 2 + 5, N // 2 - 7), name


@pytest.mark.parametrize("name,params", [("radial_2d", {"n_shots": 129}),
                                         ("spiral_2d", {"n_shots": 32})])
def test_non_cartesian_reconstructs_the_phantom(name, params):
    """
    Density-compensated gridding recovers the phantom to a few percent.

    It is not exact and cannot be: a single DCF-weighted adjoint approximates
    the inverse, and a disc trajectory never reaches the corners of k-space.
    The bar is 'recognizably the phantom', which the undersampled comparison
    below turns into a strict statement.
    """
    img = _phantom()
    coords, _ = _trajectory(name, **params)
    rec = np.abs(KspaceReconstructor((N, N), 2.0)(coords, _sample(img, coords))[0, 0].numpy())
    assert _nrmse(img, _ls_scale(rec, img)) < 12.0


def test_undersampling_degrades_the_reconstruction():
    """
    Dropping shots must make the reconstruction worse, monotonically.

    This is what ties the reconstructor to the sampling side: the mask is the
    same [B, S] shot mask ShotUndersampler produces.
    """
    img = _phantom()
    coords, _ = _trajectory("radial_2d", n_shots=129)
    kdata = _sample(img, coords)
    recon = KspaceReconstructor((N, N), 2.0)
    n_shots = coords.shape[1]

    errors = []
    for accel in (1, 2, 4, 8):
        mask = torch.zeros(1, n_shots, dtype=torch.bool)
        mask[0, ::accel] = True
        out = np.abs(recon(coords, kdata, mask=mask)[0, 0].numpy())
        errors.append(_nrmse(img, _ls_scale(out, img)))

    assert errors == sorted(errors), f"error should grow with acceleration, got {errors}"
    assert errors[-1] > errors[0] * 1.5, "8x undersampling should be clearly worse"


def test_mask_shape_is_validated():
    img = _phantom()
    coords, _ = _trajectory("radial_2d", n_shots=33)
    kdata = _sample(img, coords)
    with pytest.raises(ValueError, match=r"mask must be"):
        KspaceReconstructor((N, N), 2.0)(coords, kdata, mask=torch.ones(1, 5, dtype=torch.bool))


#*****************************#
#   the nufft undersampling   #
#*****************************#
class TestKspaceUndersamplingNufft:
    """
    "ksp_mode='nufft'": the honest counterpart to "gridded".

    "gridded" uses the trajectory only to pick Cartesian bins and then masks
    the data's own FFT, so it never leaves the grid. This mode measures at the
    coordinates the trajectory actually visits and inverts from there.
    """

    B, X, Y, Z, T = 1, 32, 32, 8, 48

    def _volume(self):
        yy, xx = np.mgrid[0:self.X, 0:self.Y]
        disc = ((xx - self.X / 2 + 3) ** 2
                + (yy - self.Y / 2 - 2) ** 2 < (self.X / 4) ** 2).astype(np.float32)
        vol = np.zeros((self.B, self.X, self.Y, self.Z, self.T), np.complex64)
        vol[0] = (disc[:, :, None, None]
                  * np.exp(-np.arange(self.T) / 20.0)[None, None, None, :])
        return vol

    def _run(self, accel=2.0, trajectory='radial_2d', array=None):
        from augmentrum.sampling import KspaceUndersampling
        us = KspaceUndersampling(ksp_mode='nufft', trajectory=trajectory,
                                 acceleration_factor=accel, us_seed=0,
                                 traj_params={'n_shots': 64})
        out, water = us.process_tensor(self._volume() if array is None else array)
        return out, us

    def test_shape_dtype_and_backend_are_preserved(self):
        vol = self._volume()
        out, _ = self._run(array=vol.copy())
        assert isinstance(out, np.ndarray)
        assert out.shape == vol.shape and out.dtype == vol.dtype

        out_t, _ = self._run(array=torch.from_numpy(vol.copy()))
        assert isinstance(out_t, torch.Tensor)
        assert tuple(out_t.shape) == vol.shape

    def test_output_keeps_the_input_scale(self):
        """
        A pipeline stage must hand back data in the units it was given.

        A density-compensated adjoint recovers the image only up to a constant,
        so without rescaling this returned values ~10^4 times the input.
        """
        vol = self._volume()
        out, _ = self._run(array=vol.copy())
        ratio = np.abs(out).max() / np.abs(vol).max()
        assert 0.2 < ratio < 5.0, f"scale drifted by {ratio:.1f}x"

    def test_error_grows_with_acceleration(self):
        vol = self._volume()
        ref = np.abs(vol)
        errs = []
        for accel in (1.0, 2.0, 4.0, 8.0):
            out, _ = self._run(accel=accel, array=vol.copy())
            errs.append(float(np.sqrt(np.mean((np.abs(out) - ref) ** 2)) / np.ptp(ref)))
        assert errs == sorted(errs), f"error should grow with acceleration, got {errs}"

    def test_works_for_a_3d_trajectory(self):
        """kz is sampled by the trajectory rather than joining the coil slot."""
        out, us = self._run(trajectory='cones_3d')
        assert out.shape == (self.B, self.X, self.Y, self.Z, self.T)
        assert us.last_meta_ is not None and us.last_masks_ is not None

    def test_extreme_acceleration_still_reconstructs(self):
        """
        ShotUndersampler clamps to one retained shot however hard it is pushed,
        so the mode has no empty-trajectory case — it degrades instead of failing.
        """
        out, us = self._run(accel=1e6)
        assert out.shape == (self.B, self.X, self.Y, self.Z, self.T)
        assert int(us.last_masks_.sum()) >= 1
        assert np.isfinite(np.abs(out)).all()

    def test_nufft_is_a_registered_mode(self):
        from augmentrum.sampling import KspaceUndersampling
        assert 'nufft' in KspaceUndersampling.MODES
        with pytest.raises(ValueError, match='ksp_mode must be one of'):
            KspaceUndersampling(ksp_mode='nonsense')


#****************************#
#   trajectory as geometry   #
#****************************#
# Moving the sample locations is the cheap, exact half of spatial augmentation:
# no interpolation anywhere. Scaling and rotating do different things, and the
# difference is easy to state wrongly, so both are pinned here.

def _reconstructed(accel=1.0, **kwargs):
    """A small object put through the nufft path."""
    from augmentrum.sampling import KspaceUndersampling

    vol = np.zeros((1, 32, 32, 1, 4), np.complex64)
    vol[0, 10:22, 12:20, 0, :] = 1.0
    module = KspaceUndersampling(ksp_mode='nufft', trajectory='radial_2d',
                                 acceleration_factor=accel, us_seed=0, **kwargs)
    return np.abs(np.asarray(module.process_tensor(vol)[0])[0, :, :, 0, 0])


def test_scaling_the_trajectory_lowers_the_resolution():
    """Keeping only the center of k-space is what a coarser acquisition measures."""
    sharpness = []
    for scale in (1.0, 0.5, 0.25):
        image = _reconstructed(traj_scale=scale)
        image = image / image.max()
        sharpness.append(np.abs(np.diff(image[:, 16])).max())

    assert sharpness[0] > sharpness[1] > sharpness[2], sharpness


def test_rotating_the_trajectory_leaves_a_fully_sampled_image_alone():
    """
    It turns the sampling pattern, not the object.

    Reconstructing on the same coordinates it was sampled on gives the object
    back exactly, which is worth asserting because the opposite is a natural
    thing to assume.
    """
    plain = _reconstructed(accel=1.0)
    turned = _reconstructed(accel=1.0, traj_rotation_deg=30.0)

    assert np.corrcoef(plain.ravel(), turned.ravel())[0, 1] > 0.999


def test_rotating_the_trajectory_changes_the_aliasing_when_accelerated():
    """
    And this is what it is for.

    Under acceleration a turned trajectory visits different spokes, so the
    artifact is a different realization from the same acquisition scheme.
    """
    plain = _reconstructed(accel=4.0)
    turned = _reconstructed(accel=4.0, traj_rotation_deg=30.0)

    assert np.corrcoef(plain.ravel(), turned.ravel())[0, 1] < 0.9
