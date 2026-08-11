####################################################################################################
#                                       train_deep_er.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-11                                                                              #
#                                                                                                  #
# Purpose: Train a Deep-ER-style joint-domain reconstruction network on MRSI Challenge simulated   #
#          data, with Augmentrum supplying the whole forward model on the fly: coil synthesis,     #
#          macromolecules, spectral augmentation, undersampling (ECCENTRIC stack, true-3D cones    #
#          or shells, 3-D Cartesian) and k-space noise. Dataset-size ablation via --n-train.       #
#                                                                                                  #
# The network is an original implementation of the published architecture — Weiser et al.,         #
# "Deep-ER: Deep Learning ECCENTRIC Reconstruction for fast high-resolution neurometabolic         #
# imaging", NeuroImage 309:121045 (2025), building on the Interlacer of Singh et al. (2022):       #
# recurrent layers holding a multi-coil k-space branch and a coil-combined image branch, joined    #
# by learnable mixing and coil-sensitivity-aware domain transfer, trained per FID timepoint        #
# with an MSE + (1 - SSIM) image loss. No upstream code is used or required.                       #
#                                                                                                  #
# Examples:                                                                                        #
#     python scripts/train_deep_er.py --dry-run                                                    #
#     python scripts/train_deep_er.py --arm augmentrum --n-train 12 --epochs 100                   #
#     python scripts/train_deep_er.py --arm none --trajectory cartesian-3d --n-train 6             #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from augmentrum.augmentation import (Macromolecules, LineBroadening,
                                     FrequencyShift, PhaseShift, ZeroFill)
from augmentrum.core.base_module import Tap
from augmentrum.core.pipeline import AugmentationPipeline
from augmentrum.dataset.mrsi_challenge import MRSIChallengeData, MRSIChallengeDataModule
from augmentrum.sampling.coil_sampling import CoilSampler
from augmentrum.sampling.kspace_sampling import KspaceUndersampling


####################################################################################################
#                                     experiment configuration                                     #
####################################################################################################
#                                                                                                  #
# Everything an experiment defines lives in this block — nothing below it needs touching:          #
#                                                                                                  #
#   1. TRAJECTORIES      how k-space is sampled                        (picked with --trajectory)  #
#   2. RANGES            per-batch sampling ranges for module parameters                           #
#   3. build_pipeline    the COMPLETE chain, every step visible        (--arm picks augmentations) #
#                                                                                                  #
# To change something: edit here, then look before training with                                   #
#   python scripts/train_deep_er.py --preview --arm augmentrum --trajectory eccentric-stack        #
#                                                                                                  #
####################################################################################################

#: How k-space is sampled — the --trajectory axis. Values are
#: KspaceUndersampling arguments; shot counts are measured Nyquist points.
TRAJECTORIES = {
    # the published Deep-ER/ECCENTRIC acquisition: 2-D eccentric circles per
    # phase-encoded kz partition, prospective center-crossing acceleration
    'eccentric-stack': dict(trajectory='stack_of_eccentric',
                            undersampling='center_crossing',
                            traj_params={'n_shots': 128}),
    # true 3-D readouts, undersampled across all three axes
    'cones-3d':  dict(trajectory='cones_3d_rosette', undersampling='shell_based',
                      traj_params={'n_shots': 2048}, undersample_axes=(0, 1, 2)),
    'shells-3d': dict(trajectory='concentric_shells_3d', undersampling='stride',
                      traj_params={'n_shells': 32}, undersample_axes=(0, 1, 2)),
    # variable-density random mask over all three axes
    'cartesian-3d': dict(undersample_axes=(0, 1, 2)),
}

#: Per-batch sampling ranges — keys match module constructor arguments;
#: parameters not listed keep their constructor values.
RANGES = dict(
    zero_order_deg=(-180.0, 180.0),   # global receiver phase
    lb_hz=(0.0, 4.0),                 # broadening ON TOP of the sim's linewidth
    shift_hz=(-10.0, 10.0),           # B0 drift
    mm_scale=(0.05, 0.25),            # macromolecule amplitude vs signal max
)


def build_pipeline(args) -> AugmentationPipeline:
    """The complete pipeline, every step visible; --arm picks the augmentations."""
    voxel_mm = MRSIChallengeDataModule.VOXEL_MM

    augmentations = {                     # object-level: in input AND target
        'none':       [],
        'native':     [PhaseShift()],
        'augmentrum': [Macromolecules(seed=args.seed), LineBroadening(),
                       FrequencyShift(), PhaseShift()],
        # spatial warps disabled for now; to re-enable, add to the arms:
        #   SpatialAugmentations(dim=3, prob=0.5, pixdim=voxel_mm, allow_rot90=False)
    }[args.arm]

    trajectory = TRAJECTORIES[args.trajectory]
    steps = [
        *augmentations,                                           # 1. augment the object
        CoilSampler(mode='synthesize', n_coils=args.n_coils,      # 2. receive array
                    seed=args.seed),
        Tap(name='clean'),                                        # 3. TARGET frozen here
        KspaceUndersampling(                                      # 4. degrade INPUT only
            ksp_mode='cartesian' if 'trajectory' not in trajectory else args.ksp_mode,
            acceleration_factor=(args.acc_low + args.acc_high) / 2.0,
            noise_sigma_k=args.noise_sigma_k,
            pixdim=voxel_mm,
            **trajectory,
        ),
    ]
    if args.crop_t:                                               # debug: shorter FID
        steps.insert(0, ZeroFill(target_pts=args.crop_t))

    ranges = dict(RANGES, acceleration_factor=(args.acc_low, args.acc_high))
    return AugmentationPipeline(steps, user_kwargs=ranges)


#**************************#
#   domain transfer ops    #
#**************************#
def img_to_k(img, sense):
    """
    Coil-combined image channels to multi-coil k-space channels, batched.

    Args:
        img: "(B, 2, X, Y, Z)" real/imag channels.
        sense: "(X, Y, Z, C)" complex sensitivity maps, shared over the batch.

    Returns:
        "(B, 2C, X, Y, Z)" — real coils then imaginary coils.
    """
    cc = torch.complex(img[:, 0], img[:, 1])                       # (B, X, Y, Z)
    coils = cc.unsqueeze(-1) * sense                               # (B, X, Y, Z, C)
    k = torch.fft.fftshift(torch.fft.fftn(
        torch.fft.ifftshift(coils, dim=(1, 2, 3)), dim=(1, 2, 3)), dim=(1, 2, 3))
    k = torch.moveaxis(k, -1, 1)                                   # (B, C, X, Y, Z)
    return torch.cat((k.real, k.imag), dim=1)


def k_to_img(k, sense):
    """
    Multi-coil k-space channels to a coil-combined image, batched.

    The inverse companion of "img_to_k": inverse FFT per coil, then the
    conjugate-sensitivity combination.
    """
    n_coils = k.shape[1] // 2
    kc = torch.complex(k[:, :n_coils], k[:, n_coils:])             # (B, C, X, Y, Z)
    kc = torch.moveaxis(kc, 1, -1)                                 # (B, X, Y, Z, C)
    coils = torch.fft.fftshift(torch.fft.ifftn(
        torch.fft.ifftshift(kc, dim=(1, 2, 3)), dim=(1, 2, 3)), dim=(1, 2, 3))
    cc = torch.sum(torch.conj(sense) * coils, dim=-1)              # (B, X, Y, Z)
    return torch.stack((cc.real, cc.imag), dim=1)


def piecewise_k_activation(x):
    """The paper's three-piece k-space nonlinearity: linear near zero, steeper tails."""
    return x + torch.relu((x - 1) / 2) + torch.relu((-1 - x) / 2)


#**************************************************************************************************#
#                                            Class Mix                                             #
#**************************************************************************************************#
class Mix(nn.Module):
    """Learnable convex combination of two same-shaped feature maps."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, a, b):
        w = torch.sigmoid(self.weight)
        return w * a + (1 - w) * b


#**************************************************************************************************#
#                                      Class InterlacerLayer                                       #
#**************************************************************************************************#
#                                                                                                  #
# One joint-domain layer: mix each branch with the other's domain transfer, then convolve.         #
#                                                                                                  #
#**************************************************************************************************#
class InterlacerLayer(nn.Module):
    """
    One joint-domain layer: mix each branch with the other's domain transfer,
    then convolve.

    Per the paper: the image branch runs three 3x3x3 conv blocks (2-64-2
    features, InstanceNorm, ReLU); the k-space branch one block of 64 filters
    with the three-piece activation. Each branch also sees the network inputs
    again, concatenated before the convolutions.
    """

    def __init__(self, n_coils: int, features: int = 64):
        super().__init__()
        k_ch = 2 * n_coils

        self.mix_img = Mix()
        self.mix_k = Mix()

        self.img_convs = nn.Sequential(
            nn.Conv3d(4, features, 3, padding='same'), nn.InstanceNorm3d(features),
            nn.ReLU(),
            nn.Conv3d(features, features, 3, padding='same'), nn.InstanceNorm3d(features),
            nn.ReLU(),
            nn.Conv3d(features, 2, 3, padding='same'),
        )
        self.k_conv_in = nn.Conv3d(2 * k_ch, features, 3, padding='same')
        self.k_norm = nn.InstanceNorm3d(features)
        self.k_conv_out = nn.Conv3d(features, k_ch, 3, padding='same')

    def forward(self, img_in, k_in, img0, k0, sense):
        img_mixed = self.mix_img(img_in, k_to_img(k_in, sense))
        k_mixed = self.mix_k(k_in, img_to_k(img_in, sense))

        img_out = self.img_convs(torch.cat((img_mixed, img0), dim=1))

        k_feat = piecewise_k_activation(self.k_norm(
            self.k_conv_in(torch.cat((k_mixed, k0), dim=1))))
        k_out = self.k_conv_out(k_feat)

        return img_out, k_out


#**************************************************************************************************#
#                                          Class DeepER                                            #
#**************************************************************************************************#
#                                                                                                  #
# Deep-ER: recurrent Interlacer layers with residual add-back and final 1x1x1 projections.         #
#                                                                                                  #
#**************************************************************************************************#
class DeepER(nn.Module):
    """
    Deep-ER: recurrent Interlacer layers with residual add-back and final
    1x1x1 projections, reconstructing one FID timepoint as a 3-D volume.

    Original implementation of Weiser et al. (NeuroImage 2025); batched over
    items and agnostic to the coil count.
    """

    def __init__(self, n_coils: int, n_layers: int = 10, features: int = 64):
        super().__init__()
        k_ch = 2 * n_coils
        self.layers = nn.ModuleList(
            InterlacerLayer(n_coils, features) for _ in range(n_layers))
        self.out_img = nn.Conv3d(4, 2, 1)
        self.out_k = nn.Conv3d(2 * k_ch, k_ch, 1)

    def forward(self, img0, k0, sense):
        img, k = img0, k0
        for layer in self.layers:
            img_delta, k_delta = layer(img, k, img0, k0, sense)
            img = img + img_delta
            k = k + k_delta
        img = self.out_img(torch.cat((img, img0), dim=1))
        k = self.out_k(torch.cat((k, k0), dim=1))
        return img, k


#**********#
#   loss   #
#**********#
def make_loss(device, use_ssim: bool = True):
    """
    The paper's image loss: MSE + (1 - SSIM), on brain-masked volumes
    normalized by the target's maximum.
    """
    ssim = None
    if use_ssim:
        try:
            from torchmetrics.image import StructuralSimilarityIndexMeasure
            ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        except ImportError:
            print("torchmetrics not installed — using plain MSE.")

    def loss_func(reco, target, mask):
        scale = torch.amax(target.abs(), dim=(1, 2, 3, 4), keepdim=True).clamp(min=1e-12)
        reco = reco / scale * mask[:, None]
        target = target / scale * mask[:, None]
        loss = nn.functional.mse_loss(reco, target)
        if ssim is not None:
            loss = loss + (1 - ssim(reco, target))
        return loss

    return loss_func


def build_data(args):
    """The MRSI Challenge wrapped in Augmentrum, yielding (input, target) pairs."""
    pipelines = {
        'train': build_pipeline(args),
        'val':   build_pipeline(args),
        'test_track1': [],
        'test_track2': [],
    }
    outputs = {'train': ('data', 'clean'), 'val': ('data', 'clean'),
               'test_track1': None, 'test_track2': None}

    return MRSIChallengeData(
        args.data_dir,
        signal='clean',
        n_train=args.n_train,
        n_val=args.n_val,
        batch_size=1,                      # one volume per pull; batching is over timepoints
        pipelines=pipelines,
        outputs=outputs,
        backend='pytorch',
        volatile=True,
    )


#**************************************************************************************************#
#                                       Class DeepERAdapter                                        #
#**************************************************************************************************#
#                                                                                                  #
# Maps Augmentrum (input, target) volume pairs onto per-timepoint training batches.                #
#                                                                                                  #
#**************************************************************************************************#
class DeepERAdapter:
    """
    Maps Augmentrum (input, target) volume pairs onto per-timepoint batches.

    The network reconstructs each FID timepoint as an independent 3-D volume,
    so a training batch is a set of timepoints drawn from one augmented volume
    — late, low-SNR timepoints with a decaying probability (full weight over
    the first quarter of the FID, tapering after).
    """

    def __init__(self, aug, split: str, n_timepoints: int, device: torch.device,
                 seed: int = None):
        self.aug = aug
        self.split = split
        self.n_timepoints = n_timepoints
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.coil_module = next(s for s in aug.pipelines[split].steps
                                if isinstance(s, CoilSampler))

    @staticmethod
    def _combine(coils, sense):
        """(X, Y, Z, C) complex -> (2, X, Y, Z) float via conjugate-CSM combination."""
        cc = torch.sum(torch.conj(sense) * coils, dim=3)
        return torch.stack((cc.real, cc.imag), dim=0)

    @staticmethod
    def _to_k(coils):
        """(X, Y, Z, C) complex -> (2C, X, Y, Z) float, matching "img_to_k"."""
        k = torch.fft.fftshift(torch.fft.fftn(
            torch.fft.ifftshift(coils, dim=(0, 1, 2)), dim=(0, 1, 2)), dim=(0, 1, 2))
        k = torch.moveaxis(k, -1, 0)
        return torch.cat((k.real, k.imag), dim=0)

    def _draw_timepoints(self, n_total: int) -> np.ndarray:
        i = np.arange(n_total)
        prob = np.clip(i / n_total * (-2.0 / 3.0) + 7.0 / 6.0, 0.0, 1.0)
        prob /= prob.sum()
        return self.rng.choice(n_total, size=min(self.n_timepoints, n_total),
                               replace=False, p=prob)

    def batches(self):
        """Yield per-timepoint batches: one augmented volume -> n_timepoints items."""
        loader = self.aug.dataloader(split=self.split, framework='pytorch')

        for x, y in loader:
            # (B, X, Y, Z, T, C) complex; one volume per pull (B == 1)
            x = x.to(self.device).to(torch.cfloat)[0]
            y = y.to(self.device).to(torch.cfloat)[0]

            sense = torch.as_tensor(self.coil_module.last_maps_,
                                    dtype=torch.cfloat, device=self.device)

            # The simulated metabolite signal is exactly zero outside the head,
            # so the brain mask falls out of the (augmented) target itself and
            # stays consistent with any spatial warp.
            mask3d = (y.abs().sum(dim=(3, 4)) > 0).float()

            items = {key: [] for key in ('img', 'ksp', 'img_gt', 'mask')}
            for t in self._draw_timepoints(x.shape[3]):
                xt, yt = x[..., t, :], y[..., t, :]

                img_under = self._combine(xt, sense)
                scale = torch.quantile(img_under.square().sum(0).sqrt(), 0.95)
                scale = torch.clamp(scale, min=1e-12)

                items['img'].append(img_under / scale)
                items['ksp'].append(self._to_k(xt) / scale)
                items['img_gt'].append(self._combine(yt, sense) / scale)
                items['mask'].append(mask3d)

            yield {
                'inputs_img':    torch.stack(items['img']),
                'inputs_kspace': torch.stack(items['ksp']),
                'img_gt':        torch.stack(items['img_gt']),
                'mask':          torch.stack(items['mask']),
                'sense':         sense,
            }


#*************#
#   preview   #
#*************#
def preview(args, aug):
    """
    Render the configured pipeline on subject 0 for hand-tuning: raw vs
    target vs input images, the trajectory's kept/dropped shots, the k-space
    mask, centre-voxel spectra and NAA-band metabolite maps. Saved to
    <out-dir>/preview_<arm>_<trajectory>.png.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from augmentrum.core import Backend, NIfTI_MRS_Plus
    from augmentrum.sampling.kspace_sampling import (ShotUndersampler,
                                                     TrajectoryRegistry)

    # One deterministic subject through the exact training pipeline.
    data, _ = aug.splits['train']
    subject = data.nifti_list[0]
    plus = NIfTI_MRS_Plus(nifti_list=[subject], backend=Backend.PYTORCH,
                          volatile=True)
    pipe = aug.pipelines['train']
    batch_params = pipe.sample_batch_parameters(1)
    print("drawn augmentation parameters for this preview:")
    for step_idx, params in batch_params.items():
        name = pipe.steps[step_idx].__class__.__name__
        values = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in params.items())
        print(f"  {name}: {values}")
    if any(type(s).__name__ == 'SpatialAugmentations' for s in pipe.steps):
        print("  (SpatialAugmentations draws its warp internally, prob=0.5 per pull)")
    out, _, taps = pipe(plus, None, batch_params=batch_params)

    x = out.get_data(Backend.PYTORCH)[0].to(torch.cfloat)         # (X, Y, Z, T, C)
    y = taps['clean'][0].get_data(Backend.PYTORCH)[0].to(torch.cfloat)
    raw = torch.as_tensor(np.asarray(subject[:]), dtype=torch.cfloat)  # (X, Y, Z, T)

    coil = next(s for s in pipe.steps if isinstance(s, CoilSampler))
    sense = torch.as_tensor(coil.last_maps_, dtype=torch.cfloat)
    cc = lambda vol: torch.sum(torch.conj(sense)[:, :, :, None, :] * vol, dim=-1)
    x_cc, y_cc = cc(x), cc(y)                                     # (X, Y, Z, T)

    sw = 1.0 / subject.dwelltime
    sf = subject.spectrometer_frequency[0]

    def spectra(vol):
        spec = torch.fft.fftshift(torch.fft.ifft(vol, dim=-1), dim=-1)
        freq = np.fft.fftshift(np.fft.fftfreq(vol.shape[-1], d=1.0 / sw))
        return spec, 4.7 - freq / sf

    def naa_map(vol):
        spec, ppm = spectra(vol)
        band = torch.as_tensor((ppm > 1.8) & (ppm < 2.2))
        return spec[..., band].abs().sum(dim=-1)

    z_mid = x.shape[2] // 2
    vx, vy = x.shape[0] // 2, x.shape[1] // 2
    ink, kept_col, drop_col = '#374151', '#2563eb', '#d1d5db'
    in_col, gt_col, raw_col = '#d97706', '#2563eb', '#9ca3af'

    fig, axes = plt.subplots(3, 3, figsize=(14, 13))

    # row 1: images at t=0
    vmax = y_cc[:, :, z_mid, 0].abs().max()
    for ax, img, label in [
            (axes[0, 0], raw[:, :, z_mid, 0].abs(), 'raw challenge |image| (unaugmented)'),
            (axes[0, 1], y_cc[:, :, z_mid, 0].abs(), 'target |image| (augmented, tapped)'),
            (axes[0, 2], x_cc[:, :, z_mid, 0].abs(), 'input |image| (undersampled + noise)')]:
        scale = raw[:, :, z_mid, 0].abs().max() if 'raw' in label else vmax
        ax.imshow(img.numpy().T, cmap='gray', origin='lower', vmax=float(scale))
        ax.set_xlabel(label, color=ink)
        ax.set_xticks([]), ax.set_yticks([])

    # row 2: trajectory shots, k-space mask, centre-voxel spectra
    ax = axes[1, 0]
    traj_cfg = TRAJECTORIES[args.trajectory]
    if 'trajectory' in traj_cfg:
        matrix = tuple(int(n) for n in x.shape[:3])
        voxel = MRSIChallengeDataModule.VOXEL_MM
        geom = {"matrix": matrix, "ndim": 3, "inferred": [],
                "fov_mm": tuple(v * n for v, n in zip(voxel, matrix))}
        params = {**traj_cfg.get('traj_params', {}), 'seed': args.seed}
        shots, meta = TrajectoryRegistry.create(
            traj_cfg['trajectory'], **params).generate(geom)
        kept, _ = ShotUndersampler.undersample_shots(
            shots, traj_cfg['undersampling'], (args.acc_low + args.acc_high) / 2,
            {'seed': args.seed}, trajectory_name=traj_cfg['trajectory'])
        kept = np.asarray(kept).astype(bool)
        # For stacks: the first kz partition. For true 3-D readouts there is
        # no partition — draw the in-plane projection of a legible subset.
        n_show = int(meta.get('n_inplane_shots', min(len(shots), 150)))
        for i in range(n_show):
            pts = np.asarray(shots[i])
            ax.plot(pts[:, 0], pts[:, 1], lw=0.6,
                    color=kept_col if kept[i] else drop_col,
                    alpha=0.8 if kept[i] else 0.5)
        where = ('first kz partition' if 'n_inplane_shots' in meta
                 else f'kx-ky projection, first {n_show} shots')
        ax.set_xlabel(f"{args.trajectory}: kept (blue) vs dropped shots,\n{where}",
                      color=ink, fontsize=9)
        ax.set_aspect('equal')
    else:
        ax.text(0.5, 0.5, 'Cartesian:\nphase-encoded grid,\nsee mask →',
                ha='center', va='center', color=ink, transform=ax.transAxes)
    ax.set_xticks([]), ax.set_yticks([])

    # Full 3-D FFT so per-partition structure is not unioned away, then the
    # SAME view for every trajectory — sampling density averaged over kz,
    # fixed [0, 1] scale — so the four trajectory previews are comparable.
    k3 = torch.fft.fftshift(torch.fft.fftn(x[..., 0, 0], dim=(0, 1, 2)),
                            dim=(0, 1, 2)).abs().numpy()
    covered = (k3 > 1e-6 * k3.max())
    axes[1, 1].imshow(covered.mean(axis=2).T, cmap='gray', origin='lower',
                      vmin=0.0, vmax=1.0)
    axes[1, 1].set_xlabel(f"sampling density, mean over kz — "
                          f"{covered.mean():.0%} of 3-D bins", color=ink)
    axes[1, 1].set_xticks([]), axes[1, 1].set_yticks([])

    ax = axes[1, 2]
    s_raw, ppm = spectra(raw[vx, vy, z_mid])
    s_gt, _ = spectra(y_cc[vx, vy, z_mid])
    s_in, _ = spectra(x_cc[vx, vy, z_mid])
    ax.plot(ppm, s_raw.abs().numpy(), color=raw_col, lw=1.0, label='raw (clean sim)')
    ax.plot(ppm, s_gt.abs().numpy(), color=gt_col, lw=1.2, label='target')
    ax.plot(ppm, s_in.abs().numpy(), color=in_col, lw=0.9, alpha=0.85, label='input')
    ax.set_xlim(4.5, 0.5)
    ax.set_xlabel('ppm', color=ink)
    ax.set_ylabel('|spectrum| (a.u.)', color=ink)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    # row 3: NAA-band (1.8-2.2 ppm) metabolite maps
    maps = [(naa_map(raw), 'NAA-band map, raw'),
            (naa_map(y_cc), 'NAA-band map, target'),
            (naa_map(x_cc), 'NAA-band map, input')]
    vmax_map = float(maps[1][0][:, :, z_mid].max())
    for ax, (mmap, label) in zip(axes[2], maps):
        ax.imshow(mmap[:, :, z_mid].numpy().T, cmap='viridis', origin='lower',
                  vmax=vmax_map)
        ax.set_xlabel(label, color=ink)
        ax.set_xticks([]), ax.set_yticks([])

    fig.tight_layout()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"preview_{args.arm}_{args.trajectory}.png"
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


#**************#
#   training   #
#**************#
def run_batch(model, loss_func, batch):
    """One forward pass + loss on a batch."""
    reco_img, _ = model(batch['inputs_img'], batch['inputs_kspace'], batch['sense'])
    return loss_func(reco_img, batch['img_gt'], batch['mask']), reco_img


def train(args):
    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))

    aug = build_data(args)

    if args.preview:
        preview(args, aug)
        return

    train_adapter = DeepERAdapter(aug, 'train', args.n_timepoints, device, seed=args.seed)
    val_adapter = DeepERAdapter(aug, 'val', args.n_timepoints, device, seed=args.seed)

    model = DeepER(n_coils=args.n_coils, n_layers=args.n_layers).to(device)
    loss_func = make_loss(device)

    if args.dry_run:
        batch = next(train_adapter.batches())
        for key, value in batch.items():
            print(f"  {key:14s} {tuple(value.shape)!s:24s} {value.dtype}")
        loss, reco = run_batch(model, loss_func, batch)
        loss.backward()
        grads = sum(p.grad.abs().sum().item() for p in model.parameters()
                    if p.grad is not None)
        print(f"  reco_img       {tuple(reco.shape)}")
        print(f"  loss           {loss.item():.6f}   (grad magnitude {grads:.3g})")
        print("Dry run OK — data, model, loss and gradients are wired.")
        return

    run_name = f"{args.arm}_{args.trajectory}_n{args.n_train or 'all'}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    run = None
    try:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=run_name, config=vars(args))
    except Exception as error:
        print(f"wandb unavailable ({error}) — logging to CSV only.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float('inf')
    log_path = out_dir / 'log.csv'
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'seconds'])

    for epoch in range(args.epochs):
        t0 = time.time()

        model.train()
        train_losses = []
        batches = train_adapter.batches()
        for _ in range(args.steps_per_epoch):
            batch = next(batches)
            optimizer.zero_grad()
            loss, _ = run_batch(model, loss_func, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_adapter.batches():          # fixed mode: one pass
                loss, _ = run_batch(model, loss_func, batch)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses)) if val_losses else float('nan')
        seconds = time.time() - t0
        print(f"epoch {epoch + 1:4d}/{args.epochs}  train {train_loss:.5f}  "
              f"val {val_loss:.5f}  ({seconds:.0f}s)")

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, train_loss, val_loss, round(seconds, 1)])
        if run:
            run.log({'loss/train': train_loss, 'loss/val': val_loss,
                     'epoch': epoch + 1})

        torch.save(model.state_dict(), out_dir / 'model_last.pt')
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / 'model_best.pt')

    if run:
        run.finish()
    print(f"Done. Checkpoints and logs in {out_dir}")


#**********#
#   main   #
#**********#
def main():
    parser = argparse.ArgumentParser(
        description="Train a Deep-ER-style network on MRSI Challenge data with Augmentrum.")
    parser.add_argument('--data-dir', default='data/MRSI_Challenge')
    parser.add_argument('--out-dir', default='results/deep_er')
    parser.add_argument('--arm', choices=['none', 'native', 'augmentrum'],
                        default='augmentrum')
    parser.add_argument('--trajectory', choices=sorted(TRAJECTORIES),
                        default='eccentric-stack')
    parser.add_argument('--ksp-mode', choices=['nufft', 'gridded'], default='nufft',
                        help="Faithful NUFFT round trip, or cheaper gridded rasterization.")
    parser.add_argument('--n-train', type=int, default=None,
                        help="Contest subjects for train+val — the ablation axis.")
    parser.add_argument('--n-val', type=int, default=5)
    parser.add_argument('--n-coils', type=int, default=32)
    parser.add_argument('--n-layers', type=int, default=10,
                        help="Interlacer layers (paper: 10).")
    parser.add_argument('--n-timepoints', type=int, default=4,
                        help="FID timepoints per batch (one volume each pull).")
    parser.add_argument('--crop-t', type=int, default=None,
                        help="Debug: truncate the FID to this many points first.")
    parser.add_argument('--acc-low', type=float, default=1.0)
    parser.add_argument('--acc-high', type=float, default=6.0)
    parser.add_argument('--noise-sigma-k', type=float, default=0.002)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--steps-per-epoch', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None)
    parser.add_argument('--wandb-project', default='deep-er')
    parser.add_argument('--wandb-entity', default='augmentrum')
    parser.add_argument('--dry-run', action='store_true',
                        help="One batch through data, model, loss and backward; no training.")
    parser.add_argument('--preview', action='store_true',
                        help="Render the configured pipeline on subject 0 (images, "
                             "trajectory, mask, spectra, NAA maps) and exit — for "
                             "hand-tuning modules and RANGES.")
    train(parser.parse_args())


if __name__ == '__main__':
    main()
