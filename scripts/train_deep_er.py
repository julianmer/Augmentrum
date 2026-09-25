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
#          macromolecules, spectral and spatial augmentation, undersampling (ECCENTRIC stack,      #
#          true-3D cones or shells, 3-D Cartesian) and noise. The ablation layers each spatial     #
#          transform, all of them, and extra noise on the three spectral arms.                     #
#                                                                                                  #
# The network is an original implementation of the published architecture — Weiser et al.,         #
# "Deep-ER: Deep Learning ECCENTRIC Reconstruction for fast high-resolution neurometabolic         #
# imaging", NeuroImage 309:121045 (2025), building on the Interlacer of Singh et al. (2022):       #
# recurrent layers holding a multi-coil k-space branch and a coil-combined image branch, joined    #
# by learnable mixing and coil-sensitivity-aware domain transfer, trained per FID timepoint        #
# with an MSE + (1 - SSIM) image loss. No upstream code is used or required.                       #
#                                                                                                  #
# The whole ablation, from the repository root — the default, no arguments needed:               #
#     python scripts/train_deep_er.py                                                              #
# It downloads the data once (~170 GB), trains all 30 conditions on every visible GPU, one run     #
# per GPU at a time, resumes interrupted runs, skips finished ones, and collects every test        #
# result in <out-dir>/summary.csv. Defaults are set for an A100 (bf16, subjects on the GPU).       #
# Not converged (see each run's log.csv)? Run it again with a larger --steps: every run resumes    #
# from its checkpoint up to the new budget and is tested again.                                    #
#                                                                                                  #
# One run, or a look before training:                                                              #
#     python scripts/train_deep_er.py --arm augmentrum --spatial all --noise                       #
#     python scripts/train_deep_er.py --dry-run                                                    #
#     python scripts/train_deep_er.py --bench 50          (or --tune: compare the speed options)   #
# Needs torchmetrics (the SSIM loss) and wandb for --wandb, beside augmentrum's torch extra.       #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from augmentrum.augmentation import (Macromolecules, LineBroadening,
                                     FrequencyShift, PhaseShift, ZeroFill)
from augmentrum.augmentation.noise import Noise
from augmentrum.augmentation.spatial_augmentations import SpatialAugmentations
from augmentrum.core import Backend
from augmentrum.core.base_module import BaseModule, Tap
from augmentrum.core.pipeline import AugmentationPipeline
from augmentrum.dataset.mrsi_challenge import MRSIChallengeData, MRSIChallengeDataModule
from augmentrum.sampling.coil_sampling import Birdcage, CoilSampler
from augmentrum.sampling.kspace_sampling import KspaceUndersampling


####################################################################################################
#                                     experiment configuration                                     #
####################################################################################################
#                                                                                                  #
# Everything an experiment defines lives in this block — nothing below it needs touching:          #
#                                                                                                  #
#   1. TRAJECTORIES      how k-space is sampled                        (picked with --trajectory)  #
#   2. ARMS              the spectral augmentations                    (picked with --arm)         #
#   3. SPATIAL           sizes of the spatial transforms               (picked with --spatial)     #
#   4. RANGES            per-batch sampling ranges for module parameters                           #
#   5. EVALUATION        the fixed validation and test inputs, the same for every run              #
#   6. build_pipeline    the COMPLETE chain, every step visible                                    #
#                                                                                                  #
# To change something: edit here, then look before training with                                   #
#   python scripts/train_deep_er.py --preview --arm augmentrum --spatial all --noise               #
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

#: The spectral arms the spatial ablation is layered on. Macromolecules are
#: part of every arm: the test truth ('meta+mm') carries them.
ARMS = {
    'none':       [],
    'native':     [PhaseShift],
    'augmentrum': [LineBroadening, FrequencyShift, PhaseShift],
}

#: Spatial transforms, each an ablation condition on its own; --spatial all
#: draws every one. Sizes are what a head plausibly does in the scanner:
#: in-plane (axial) rotation only, left-right flips only.
SPATIAL_KINDS = ('translation', 'rotation', 'zoom', 'anisotropic', 'shear', 'flip')
SPATIAL = dict(
    prob=0.5,                 # each transform, independently, per pull
    translation_frac=0.05,    # of the axis length (~9 mm in x)
    max_z_angle_deg=15.0,     # in-plane rotation about z
    zoom_min=0.9, zoom_max=1.1,
    shear_max=0.05,
)

#: Per-batch sampling ranges — keys match module constructor arguments;
#: parameters not listed keep their constructor values.
RANGES = dict(
    zero_order_deg=(-180.0, 180.0),   # global receiver phase
    lb_hz=(0.0, 4.0),                 # broadening ON TOP of the sim's linewidth
    shift_hz=(-10.0, 10.0),           # B0 drift
    mm_scale=(0.05, 0.25),            # macromolecule amplitude vs signal max
    sigma=(0.0, 2.0e-3),              # --noise: extra noise per coil, SD per point
)

#: Fixed validation and test inputs: one root seed, one acceleration each and
#: evenly spaced timepoints, independent of the run's own seed and condition.
EVALUATION = dict(
    seed=0,
    val_acc=4.0, val_timepoints=8,
    test_acc=(2.0, 4.0, 6.0), test_timepoints=32,
    val_mm_scale=0.15,              # the val subjects ship no MM: add the midpoint
)


#: One subject's metabolite signal as complex64: 64 x 64 x 32 x 384 x 8 bytes.
POOL_GB_PER_SUBJECT = 0.403


def spectral_steps(arm: str, seed: int):
    """Macromolecules plus the arm's spectral modules; object-level, in input AND target."""
    return [Macromolecules(seed=seed)] + [module() for module in ARMS[arm]]


def build_pipeline(args, device, keep_all: bool = False) -> AugmentationPipeline:
    """
    The complete training chain, every step visible.

    The timepoints are drawn right after the spectral modules, which need the
    whole FID. Every later step acts on each timepoint alike — the warp per
    channel, the coils voxel by voxel, the k-space mask the same for every
    spectral point — so only the drawn ones go through them. keep_all
    passes the whole FID on (the preview).
    """
    kinds = SPATIAL_KINDS if args.spatial == 'all' else \
        (() if args.spatial == 'none' else (args.spatial,))

    steps = [
        ToDevice(device),                                         # 0. one volume to the GPU
        *spectral_steps(args.arm, args.seed),                     # 1. augment the spectra
        TimepointSampler(None if keep_all else args.n_timepoints),# 2. keep what is trained on
        *([SpatialTransforms(kinds, **SPATIAL)] if kinds else []),# 3. augment the object
        CoilSampler(mode='synthesize', n_coils=args.n_coils,      # 4. receive array
                    source=CachedBirdcage(), seed=args.seed),
        Tap(name='clean'),                                        # 5. TARGET frozen here
        *([Noise(sigma=RANGES['sigma'][1])] if args.noise else []),  # 6. noise, input only
        kspace_step(args, (args.acc_low + args.acc_high) / 2.0),  # 7. degrade INPUT only
    ]
    if args.crop_t:                                               # debug: shorter FID
        steps.insert(1, ZeroFill(target_pts=args.crop_t))

    ranges = dict(RANGES, acceleration_factor=(args.acc_low, args.acc_high))
    if not args.noise:
        ranges.pop('sigma')
    return AugmentationPipeline(steps, user_kwargs=ranges)


def eval_pipeline(args, device, n_timepoints: int, acc: float, add_mm: bool) -> AugmentationPipeline:
    """The fixed chain for validation and test: no augmentation, one acceleration."""
    steps = [
        ToDevice(device),
        *([Macromolecules(mm_scale=EVALUATION['val_mm_scale'])] if add_mm else []),
        TimepointSampler(n_timepoints, evenly=True),
        CoilSampler(mode='synthesize', n_coils=args.n_coils, source=CachedBirdcage()),
        Tap(name='clean'),
        kspace_step(args, acc, traj_seed=EVALUATION['seed']),
    ]
    return AugmentationPipeline(steps)


def kspace_step(args, acc: float, traj_seed=None) -> KspaceUndersampling:
    """
    The undersampling step. us_seed does not reach the trajectory, whose
    shots are placed anew on every call unless it gets a seed of its own:
    training leaves it free, the fixed sets pin it.
    """
    trajectory = dict(TRAJECTORIES[args.trajectory])
    if traj_seed is not None and 'traj_params' in trajectory:
        trajectory['traj_params'] = dict(trajectory['traj_params'], seed=traj_seed)
    return KspaceUndersampling(
        ksp_mode='cartesian' if 'trajectory' not in trajectory else args.ksp_mode,
        acceleration_factor=acc,
        noise_sigma_k=args.noise_sigma_k,
        pixdim=MRSIChallengeDataModule.VOXEL_MM,
        **trajectory,
    )


#**************************************************************************************************#
#                                        Class ToDevice                                            #
#**************************************************************************************************#
#                                                                                                  #
# Moves the batch to the training device, so the subject pool stays in host memory.                #
#                                                                                                  #
#**************************************************************************************************#
class ToDevice(BaseModule):
    """Moves the batch to the training device; the subject pool stays in host memory."""

    SUPPORTED_BACKENDS = (Backend.PYTORCH,)
    MASKS = 'pass'

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = torch.device(device)

    def process_tensor(self, data_array, water_array=None, backend=None, **kwargs):
        move = lambda t: None if t is None else t.to(self.device)
        return move(data_array), move(water_array)


#**************************************************************************************************#
#                                     Class TimepointSampler                                       #
#**************************************************************************************************#
#                                                                                                  #
# Keeps the FID timepoints a batch trains on.                                                      #
#                                                                                                  #
#**************************************************************************************************#
class TimepointSampler(BaseModule):
    """
    Keeps the FID timepoints a batch trains on.

    Drawn: late, low-SNR timepoints with a decaying probability (full weight
    over the first quarter of the FID, tapering after), without replacement.
    evenly: n_timepoints fixed ones, evenly spaced over the whole FID.
    n_timepoints None keeps the whole FID.
    """

    SUPPORTED_BACKENDS = (Backend.PYTORCH,)

    def __init__(self, n_timepoints=None, evenly: bool = False, seed=None):
        super().__init__(seed=seed)
        self.n_timepoints = n_timepoints
        self.evenly = evenly
        self.last_timepoints_ = None

    def draw(self, n_total: int) -> np.ndarray:
        i = np.arange(n_total)
        prob = np.clip(i / n_total * (-2.0 / 3.0) + 7.0 / 6.0, 0.0, 1.0)
        prob /= prob.sum()
        return self.rng.numpy_rng().choice(
            n_total, size=min(self.n_timepoints, n_total), replace=False, p=prob)

    def process_tensor(self, data_array, water_array=None, backend=None, **kwargs):
        n_total = int(data_array.shape[-1])
        if self.n_timepoints is None:
            self.last_timepoints_ = np.arange(n_total)
            return data_array, water_array
        if self.evenly:
            keep = np.unique(np.round(np.linspace(0, n_total - 1, self.n_timepoints)).astype(int))
        else:
            keep = self.draw(n_total)
        self.last_timepoints_ = keep
        index = torch.as_tensor(keep, device=data_array.device)
        return torch.index_select(data_array, -1, index), water_array


#**************************************************************************************************#
#                                     Class SpatialTransforms                                      #
#**************************************************************************************************#
#                                                                                                  #
# SpatialAugmentations restricted to the chosen transforms, with left-right flips only.            #
#                                                                                                  #
#**************************************************************************************************#
class SpatialTransforms(SpatialAugmentations):
    """
    SpatialAugmentations restricted to the chosen transforms.

    The draw is the parent's; a transform not chosen is set to identity in
    each spec, since the affine is built from the values, not the flags. The
    free 3-D rotation and rot90 are always off (no head turns that way, and
    the in-plane field of view is not square), and a flip is left-right only.
    """

    def __init__(self, kinds=SPATIAL_KINDS, prob: float = 0.5,
                 translation_frac: float = 0.05, max_z_angle_deg: float = 15.0,
                 zoom_min: float = 0.9, zoom_max: float = 1.1, shear_max: float = 0.05):
        super().__init__(dim=3, prob=prob, translation_frac=translation_frac,
                         max_z_angle_deg=max_z_angle_deg, max_random_angle_deg=0.0,
                         zoom_min=zoom_min, zoom_max=zoom_max, shear_max=shear_max,
                         pixdim=MRSIChallengeDataModule.VOXEL_MM, allow_rot90=False)
        unknown = set(kinds) - set(SPATIAL_KINDS)
        if unknown:
            raise ValueError(f"Unknown spatial transforms {sorted(unknown)}.")
        self.kinds = tuple(kinds)

    def sample_augmentations(self, batch_size, pipeline='data', rng=None):
        specs = super().sample_augmentations(batch_size, pipeline=pipeline, rng=rng)
        on = lambda kind, spec, flag: kind in self.kinds and spec[flag]
        for s in specs:
            if not on('translation', s, 'do_translate'):
                s.update(do_translate=False, tx=0.0, ty=0.0, tz=0.0)
            if not on('rotation', s, 'do_z_rot'):
                s.update(do_z_rot=False, z_angle_deg=0.0)
            if not on('shear', s, 'do_shear'):
                s.update(do_shear=False, shear_xy=(0.0, 0.0), shear_z=0.0)
            flip = on('flip', s, 'do_flip')
            s.update(do_flip=flip, flip_x=flip, flip_y=False, flip_z=False)

            aniso, iso = on('anisotropic', s, 'do_anisotropic'), on('zoom', s, 'do_zoom')
            if not aniso:
                # the parent drew per axis when its anisotropic flag fired;
                # one axis of that draw is an isotropic draw from the same range
                s['zoom_xyz'] = (s['zoom_xyz'][0],) * 3 if iso else (1.0, 1.0, 1.0)
            s.update(do_anisotropic=aniso, do_zoom=iso and not aniso)

            s.update(do_rot90=False, k_rot90=0, do_random_csm_rot=False,
                     random_rot_deg=0.0, do_coil_sub=False, coil_keep=None)
        return specs


#**************************************************************************************************#
#                                     Class CachedBirdcage                                         #
#**************************************************************************************************#
#                                                                                                  #
# Birdcage maps built once per grid; each call only applies its random turn.                       #
#                                                                                                  #
#**************************************************************************************************#
class CachedBirdcage(Birdcage):
    """
    Birdcage maps built once per grid; each call only applies its random turn.

    The parent's per-call turn is one phase added to every element, and its
    normalization only divides by the root-sum-of-squares, so the maps of a
    call are the unturned maps times exp(i * turn), drawn the same way. That
    saves rebuilding 32 coils over the whole grid in NumPy on every pull.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._unturned = {}

    def build(self, matrix, n_coils, rng):
        key = (tuple(matrix), n_coils)
        if key not in self._unturned:
            self._unturned[key] = super().build(matrix, n_coils, None)
        maps = self._unturned[key]
        if rng is None:
            return maps
        return maps * np.complex64(np.exp(1j * rng.uniform(0.0, 2.0 * np.pi)))


#*************************#
#   domain transfer ops   #
#*************************#
_CHECKERBOARDS = {}


def checkerboard(shape, device, signed: bool = False):
    """
    The centered FFT without its shifts. For even sizes,
    fftshift(fftn(ifftshift(x))) = s * M * fftn(M * x) with M = (-1)^(x+y+z)
    and s = prod((-1)^(N/2)): a sign multiply instead of two full copies
    (torch's shifts are rolls). The same holds for the inverse. Returns M,
    or s * M when *signed* — which goes on exactly one side of the FFT.
    """
    key = (tuple(shape), device, signed)
    if key not in _CHECKERBOARDS:
        if any(n % 2 for n in shape):
            raise ValueError(f"checkerboard FFT needs even sizes, got {tuple(shape)}")
        grids = torch.meshgrid(*[torch.arange(n, device=device) for n in shape],
                               indexing='ij')
        board = (1 - 2 * (sum(grids) % 2)).to(torch.float32)
        if signed and sum(n // 2 for n in shape) % 2:
            board = -board
        _CHECKERBOARDS[key] = board
    return _CHECKERBOARDS[key]


def coil_first(sense):
    """
    "(X, Y, Z, C)" maps to the layout the transfers use: "(C, X, Y, Z)",
    with the checkerboard folded in, so each transfer does one sign multiply.
    """
    sense = sense.permute(3, 0, 1, 2)
    return (sense * checkerboard(sense.shape[1:], sense.device)).contiguous()



def img_to_k(img, sense):
    """
    Coil-combined image channels to multi-coil k-space channels, batched.

    Args:
        img: "(B, 2, X, Y, Z)" real/imag channels.
        sense: "(C, X, Y, Z)" maps from "coil_first", shared over the batch.

    Returns:
        "(B, 2C, X, Y, Z)" — real coils then imaginary coils, centered k-space.
    """
    cc = torch.complex(img[:, 0], img[:, 1])                       # (B, X, Y, Z)
    k = torch.fft.fftn(cc.unsqueeze(1) * sense, dim=(-3, -2, -1))  # (B, C, X, Y, Z)
    k = k * checkerboard(k.shape[-3:], k.device, signed=True)
    return torch.cat((k.real, k.imag), dim=1)


def k_to_img(k, sense):
    """
    Multi-coil k-space channels to a coil-combined image, batched.

    The inverse companion of "img_to_k": inverse FFT per coil, then the
    conjugate-sensitivity combination.
    """
    n_coils = k.shape[1] // 2
    kc = torch.complex(k[:, :n_coils], k[:, n_coils:])             # (B, C, X, Y, Z)
    coils = torch.fft.ifftn(kc * checkerboard(kc.shape[-3:], kc.device, signed=True),
                            dim=(-3, -2, -1))
    cc = torch.sum(torch.conj(sense) * coils, dim=1)               # (B, X, Y, Z)
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

    Under autocast the convolutions run in reduced precision; their outputs
    come back in float32, so the domain transfers (FFTs) and the residual
    sums stay float32.
    """

    def __init__(self, n_coils: int, features: int = 64, channels_last: bool = False):
        super().__init__()
        k_ch = 2 * n_coils
        self.channels_last = channels_last

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

        layout = (torch.channels_last_3d if self.channels_last
                  else torch.contiguous_format)
        img_out = self.img_convs(
            torch.cat((img_mixed, img0), dim=1).contiguous(memory_format=layout))

        k_feat = piecewise_k_activation(self.k_norm(self.k_conv_in(
            torch.cat((k_mixed, k0), dim=1).contiguous(memory_format=layout))))
        k_out = self.k_conv_out(k_feat)

        return img_out.float(), k_out.float()


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

    def __init__(self, n_coils: int, n_layers: int = 10, features: int = 64,
                 channels_last: bool = False):
        super().__init__()
        k_ch = 2 * n_coils
        self.layers = nn.ModuleList(
            InterlacerLayer(n_coils, features, channels_last) for _ in range(n_layers))
        self.out_img = nn.Conv3d(4, 2, 1)
        self.out_k = nn.Conv3d(2 * k_ch, k_ch, 1)

    def forward(self, img0, k0, sense):
        """sense: "(X, Y, Z, C)" complex maps, shared over the batch."""
        sense = coil_first(sense)
        img, k = img0, k0
        for layer in self.layers:
            img_delta, k_delta = layer(img, k, img0, k0, sense)
            img = img + img_delta
            k = k + k_delta
        img = self.out_img(torch.cat((img, img0), dim=1))
        k = self.out_k(torch.cat((k, k0), dim=1))
        return img.float(), k.float()


#**********#
#   loss   #
#**********#
def make_loss(device, use_ssim: bool = True):
    """
    The paper's image loss, MSE + (1 - SSIM), on brain-masked volumes.

    The volumes arrive on each timepoint's input scale (make_batch). They are
    not renormalized by the target's maximum: late in the FID the target is
    a small fraction of the noise, and dividing by it blew single timepoints
    up to losses of ~1e5.
    """
    ssim = None
    if use_ssim:
        # a missing package must not quietly turn the loss into plain MSE
        from torchmetrics.image import StructuralSimilarityIndexMeasure  # pip install torchmetrics
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def loss_func(reco, target, mask):
        reco = reco * mask[:, None]
        target = target * mask[:, None]
        loss = nn.functional.mse_loss(reco, target)
        if ssim is not None:
            loss = loss + (1 - ssim(reco, target))
        return loss

    return loss_func


def metrics(reco, target, mask):
    """
    Per item, inside the brain mask and on the loss's scale: the error and
    target energies (for an NRMSE pooled over a subject's timepoints) and SSIM.

    NRMSE is pooled, not averaged per timepoint: late in the FID the target
    is nearly zero, so a per-timepoint ratio runs into the thousands and
    would decide the mean on its own.
    """
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    m = mask[:, None]
    err = (((reco - target) * m) ** 2).sum(dim=(1, 2, 3, 4))
    energy = ((target * m) ** 2).sum(dim=(1, 2, 3, 4))
    s = ssim(reco * m, target * m, data_range=1.0, reduction='none')
    return err, energy, s


#**********#
#   data   #
#**********#
def make_batch(x, y, sense):
    """
    One pull, "(X, Y, Z, n, C)" complex input and target over its kept
    timepoints, into a batch of n per-timepoint volumes: the coil-combined
    image, the multi-coil k-space and the target image, each timepoint
    scaled by the 95th percentile of its input image magnitude.

    The simulated metabolite signal is exactly zero outside the head, so the
    brain mask falls out of the (augmented) target itself and stays
    consistent with any spatial warp.
    """
    xs = x.permute(3, 0, 1, 2, 4)                                  # (n, X, Y, Z, C)
    ys = y.permute(3, 0, 1, 2, 4)
    combine = lambda v: torch.sum(torch.conj(sense) * v, dim=-1)   # (n, X, Y, Z)
    as_channels = lambda c: torch.stack((c.real, c.imag), dim=1)   # (n, 2, X, Y, Z)

    img = combine(xs)
    scale = torch.quantile(img.abs().flatten(1), 0.95, dim=1).clamp(min=1e-12)
    scale = scale.view(-1, 1, 1, 1, 1)

    grid, dev = xs.shape[1:4], xs.device                           # centered FFT, see checkerboard
    k = torch.fft.fftn(xs.permute(0, 4, 1, 2, 3) * checkerboard(grid, dev), dim=(-3, -2, -1))
    k = k * checkerboard(grid, dev, signed=True)                   # (n, C, X, Y, Z)

    mask = (ys.abs().sum(dim=(0, 4)) > 0).float()                  # (X, Y, Z)
    return {
        'inputs_img':    as_channels(img) / scale,
        'inputs_kspace': torch.cat((k.real, k.imag), dim=1) / scale,
        'img_gt':        as_channels(combine(ys)) / scale,
        'mask':          mask.expand(xs.shape[0], -1, -1, -1).contiguous(),
        'sense':         sense,
    }


def pulls(aug, split: str, device):
    """Yield (batch, timepoints) per pull of a split, on the device."""
    pipe = aug.pipelines[split]
    coil = next(s for s in pipe.steps if isinstance(s, CoilSampler))
    sampler = next(s for s in pipe.steps if isinstance(s, TimepointSampler))
    for x, y in aug.dataloader(split=split, framework='pytorch'):
        sense = torch.as_tensor(coil.last_maps_, dtype=torch.cfloat, device=device)
        x = x[0].to(device, torch.cfloat)                          # (X, Y, Z, n, C)
        y = y[0].to(device, torch.cfloat)
        yield make_batch(x, y, sense), sampler.last_timepoints_


def build_train_data(args, device):
    """The training subjects wrapped in Augmentrum, one volume per pull."""
    return MRSIChallengeData(
        args.data_dir, signal='clean',
        splits={'train': subjects(args)['train']},
        batch_size=1,                      # one volume per pull; batching is over timepoints
        pipelines={'train': build_pipeline(args, device)},
        outputs={'train': ('data', 'clean')},
        modes={'train': 'on-the-fly'},
        backend='pytorch', volatile=True, seed=args.seed,
        # the whole pool on the GPU skips the copy per pull
        device=device if pool_on_gpu(args, device) else None,
    )


def pool_on_gpu(args, device) -> bool:
    """--pool-on-gpu, or by default: on when the GPU has room for it and 16 GB more."""
    if device.type != 'cuda':
        return False
    if args.pool_on_gpu is not None:
        return args.pool_on_gpu
    free, _ = torch.cuda.mem_get_info(device)
    return free > args.train * POOL_GB_PER_SUBJECT * 1e9 + 16e9


def subjects(args):
    return MRSIChallengeDataModule.resolve({'train': args.train, 'val': args.val})


def fixed_set(args, device, split: str, names, acc: float, n_timepoints: int):
    """
    The fixed inputs of a validation or test split, one pull per subject.

    Built from EVALUATION's root seed, so every run — whatever its own seed
    or condition — sees the same coils, masks, noise and timepoints. The
    validation subjects ship only the metabolites, so they get the midpoint
    macromolecule level; the test truth ('meta+mm') has its own.
    """
    is_test = split.startswith('test')
    aug = MRSIChallengeData(
        args.data_dir, signal={split: 'meta+mm' if is_test else 'clean'},
        splits={split: tuple(names)},
        batch_size=1,
        pipelines={split: eval_pipeline(args, device, n_timepoints, acc, add_mm=not is_test)},
        outputs={split: ('data', 'clean')},
        modes={split: 'fixed'},
        backend='pytorch', volatile=True, seed=EVALUATION['seed'],
    )
    return aug, split


#**************#
#   batching   #
#**************#
class Prefetcher:
    """
    Builds the next batches in a background thread, on its own CUDA stream,
    while the current one trains. The pipeline runs almost entirely in torch
    kernels, which release the GIL, so the two overlap.
    """

    def __init__(self, make_iter, device, depth: int = 2):
        self.queue = queue.Queue(maxsize=depth)
        self.cuda = device.type == 'cuda'
        self.stream = torch.cuda.Stream(device) if self.cuda else None
        self.thread = threading.Thread(target=self._run, args=(make_iter,), daemon=True)
        self.thread.start()

    def _run(self, make_iter):
        try:
            if self.cuda:
                torch.cuda.set_device(self.stream.device)
                with torch.cuda.stream(self.stream):
                    for item in make_iter():
                        self.stream.synchronize()
                        self.queue.put(item)
            else:
                for item in make_iter():
                    self.queue.put(item)
        except BaseException as error:              # surfaced in the training thread
            self.queue.put(error)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if isinstance(item, BaseException):
            raise item
        batch, timepoints = item
        if self.cuda:
            # made on the side stream, used and freed on this one
            for value in batch.values():
                value.record_stream(torch.cuda.current_stream())
        return batch, timepoints


#*************#
#   preview   #
#*************#
def preview(args):
    """
    Render the configured pipeline on subject 0 for hand-tuning: raw vs
    target vs input images, the trajectory's kept/dropped shots, the k-space
    mask, centre-voxel spectra and NAA-band metabolite maps. Saved to
    <out-dir>/preview_<run name>.png. Runs on the CPU over the whole FID.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from augmentrum.core import NIfTI_MRS_Plus
    from augmentrum.sampling.kspace_sampling import (ShotUndersampler,
                                                     TrajectoryRegistry)

    # One deterministic subject through the exact training chain.
    aug = MRSIChallengeData(
        args.data_dir, signal='clean', splits={'train': subjects(args)['train'][:1]},
        batch_size=1, pipelines={'train': build_pipeline(args, 'cpu', keep_all=True)},
        outputs={'train': ('data', 'clean')}, modes={'train': 'on-the-fly'},
        backend='pytorch', volatile=True, seed=args.seed,
    )
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
    out, _, taps = pipe(plus, None, batch_params=batch_params)
    spatial = next((s for s in pipe.steps if isinstance(s, SpatialTransforms)), None)
    if spatial is not None:
        print(f"  SpatialTransforms {spatial.kinds}: {spatial.aug_specs_[0]}")

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
    path = out_dir / f"preview_{run_name(args)}.png"
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


#**************#
#   training   #
#**************#
def run_name(args) -> str:
    return (f"{args.arm}_{args.spatial}{'_noise' if args.noise else ''}"
            f"_{args.trajectory}_n{args.train}_s{args.seed}")


def run_batch(model, loss_func, batch, amp: bool):
    """One forward pass + loss on a batch."""
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
        reco_img, _ = model(batch['inputs_img'], batch['inputs_kspace'], batch['sense'])
    return loss_func(reco_img, batch['img_gt'], batch['mask']), reco_img


#: Timepoints the model sees at once when evaluating; a test subject has 32.
EVAL_CHUNK = 8


@torch.no_grad()
def evaluate(model, loss_func, batches, amp: bool):
    """Mean loss, NRMSE and SSIM over fixed batches, and the same per batch."""
    model.eval()
    rows = []
    for batch in batches:
        n = batch['inputs_img'].shape[0]
        losses, errs, energies, ssims = [], [], [], []
        for lo in range(0, n, EVAL_CHUNK):
            part = {k: (v if k == 'sense' else v[lo:lo + EVAL_CHUNK]) for k, v in batch.items()}
            loss, reco = run_batch(model, loss_func, part, amp)
            err, energy, ssim = metrics(reco, part['img_gt'], part['mask'])
            losses.append(loss.item() * len(err))
            errs.append(err)
            energies.append(energy)
            ssims.append(ssim)
        nrmse = (torch.cat(errs).sum() / torch.cat(energies).sum().clamp(min=1e-12)).sqrt()
        rows.append({'loss': sum(losses) / n, 'nrmse': nrmse.item(),
                     'ssim': torch.cat(ssims).mean().item()})
    model.train()
    return {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}, rows


def load_fixed(args, device, split, names, acc, n_timepoints, keep_on):
    """Every pull of a fixed split, as batches on *keep_on*."""
    aug, split = fixed_set(args, device, split, names, acc, n_timepoints)
    out = []
    for batch, _ in pulls(aug, split, device):
        out.append({k: v.to(keep_on) for k, v in batch.items()})
    return out


def to_device(batches, device):
    for batch in batches:
        yield {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def test(args, model, loss_func, device, amp):
    """Every test subject at every test acceleration, per subject."""
    results = {}
    for acc in EVALUATION['test_acc']:
        for track in args.test:
            split = f'test_{track}'
            names = getattr(MRSIChallengeDataModule, f'{track.upper()}_SUBJECTS')
            aug, split = fixed_set(args, device, split, names, acc,
                                   EVALUATION['test_timepoints'])
            batches = (batch for batch, _ in pulls(aug, split, device))
            mean, rows = evaluate(model, loss_func, batches, amp)
            results[f'acc{acc:g}/{track}'] = {'mean': mean,
                                              'subjects': dict(zip(names, rows))}
            print(f"  test acc {acc:g} {track}: loss {mean['loss']:.4f}  "
                  f"nrmse {mean['nrmse']:.4f}  ssim {mean['ssim']:.4f}")
    return results


def train(args):
    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    amp = args.amp and device.type == 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if args.preview:
        preview(args)
        return

    aug = build_train_data(args, device)
    model = DeepER(n_coils=args.n_coils, n_layers=args.n_layers,
                   channels_last=args.channels_last).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
    loss_func = make_loss(device)
    step_model = torch.compile(model) if args.compile else model

    if args.dry_run:
        batch, timepoints = next(pulls(aug, 'train', device))
        print(f"  timepoints     {sorted(int(t) for t in timepoints)}")
        for key, value in batch.items():
            print(f"  {key:14s} {tuple(value.shape)!s:24s} {value.dtype}")
        loss, reco = run_batch(step_model, loss_func, batch, amp)
        loss.backward()
        grads = sum(p.grad.abs().sum().item() for p in model.parameters()
                    if p.grad is not None)
        print(f"  reco_img       {tuple(reco.shape)}")
        print(f"  loss           {loss.item():.6f}   (grad magnitude {grads:.3g})")
        print("Dry run OK — data, model, loss and gradients are wired.")
        return

    if args.bench:
        bench(args, aug, step_model, loss_func, device, amp)
        return

    out_dir = Path(args.out_dir) / run_name(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'config.json').write_text(json.dumps(
        {'args': vars(args), 'spatial': SPATIAL, 'ranges': RANGES,
         'evaluation': EVALUATION}, indent=2, default=str))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 fused=device.type == 'cuda')
    step, best_val = 0, float('inf')
    last_path = out_dir / 'last.pt'
    log_path = out_dir / 'log.csv'
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        step, best_val = state['step'], state['best_val']
        # a fresh stream, so the resumed run does not replay its first batches
        aug.reseed(args.seed + step)
        print(f"resumed at step {step}")
    else:
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'train_loss', 'val_loss', 'val_nrmse',
                                    'val_ssim', 'ms_per_step'])

    val_names = subjects(args)['val']
    val_set = load_fixed(args, device, 'val', val_names, EVALUATION['val_acc'],
                         EVALUATION['val_timepoints'],
                         keep_on='cpu' if args.val_on_cpu else device)
    print(f"validation: {len(val_set)} subjects x {EVALUATION['val_timepoints']} "
          f"timepoints at acceleration {EVALUATION['val_acc']:g}")

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         group=Path(args.out_dir).name, name=run_name(args),
                         config=vars(args), resume='allow',
                         id=f"{Path(args.out_dir).name}-{run_name(args)}")

    batches = Prefetcher(lambda: pulls(aug, 'train', device), device, args.prefetch)
    model.train()
    running, t0 = [], time.time()
    while step < args.steps:
        batch, _ = next(batches)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = run_batch(step_model, loss_func, batch, amp)
        loss.backward()
        optimizer.step()
        running.append(loss.detach())
        step += 1

        if step % args.eval_every and step != args.steps:
            continue
        ms = (time.time() - t0) / len(running) * 1e3
        train_loss = torch.stack(running).mean().item()
        val, _ = evaluate(step_model, loss_func, to_device(val_set, device), amp)
        print(f"step {step:7d}/{args.steps}  train {train_loss:.5f}  val {val['loss']:.5f}  "
              f"nrmse {val['nrmse']:.4f}  ssim {val['ssim']:.4f}  ({ms:.0f} ms/step)",
              flush=True)
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([step, train_loss, val['loss'], val['nrmse'],
                                    val['ssim'], round(ms, 1)])
        if run:
            run.log({'loss/train': train_loss, 'loss/val': val['loss'],
                     'val/nrmse': val['nrmse'], 'val/ssim': val['ssim'],
                     'ms_per_step': ms}, step=step)
        if val['loss'] < best_val:
            best_val = val['loss']
            torch.save(model.state_dict(), out_dir / 'best.pt')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'step': step, 'best_val': best_val}, last_path)
        running, t0 = [], time.time()

    del batches, val_set
    results = {}
    for weights in ('last', 'best'):
        if weights == 'best':
            model.load_state_dict(torch.load(out_dir / 'best.pt', map_location=device))
        print(f"test, {weights} weights:")
        results[weights] = test(args, model, loss_func, device, amp)
    (out_dir / 'test.json').write_text(json.dumps(results, indent=2))
    if run:
        run.summary.update({f'test/{w}/{k}/{m}': v['mean'][m]
                            for w, r in results.items() for k, v in r.items()
                            for m in v['mean']})
        run.finish()
    print(f"Done. Checkpoints, logs and test.json in {out_dir}")


def bench(args, aug, model, loss_func, device, amp):
    """Time data only, model only, and the full prefetched step."""
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else (lambda: None)
    gen = pulls(aug, 'train', device)
    batch, _ = next(gen)
    sync()
    t0 = time.time()
    for _ in range(args.bench):
        next(gen)
    sync()
    data_ms = (time.time() - t0) / args.bench * 1e3

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 fused=device.type == 'cuda')

    def step(b):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = run_batch(model, loss_func, b, amp)
        loss.backward()
        optimizer.step()

    for _ in range(3):                             # warm-up (and compile)
        step(batch)
    sync()
    t0 = time.time()
    for _ in range(args.bench):
        step(batch)
    sync()
    model_ms = (time.time() - t0) / args.bench * 1e3

    batches = Prefetcher(lambda: pulls(aug, 'train', device), device, args.prefetch)
    step(next(batches)[0])
    sync()
    t0 = time.time()
    for _ in range(args.bench):
        step(next(batches)[0])
    sync()
    full_ms = (time.time() - t0) / args.bench * 1e3
    peak = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == 'cuda' else 0
    print(f"data {data_ms:.0f} ms/pull   model {model_ms:.0f} ms/step   "
          f"prefetched step {full_ms:.0f} ms   peak GPU {peak:.1f} GB   "
          f"pool on {'GPU' if pool_on_gpu(args, device) else 'CPU'}")


#************#
#   ablation   #
#************#
def conditions():
    """(arm, spatial, noise) of every ablation run: each arm alone, with each
    spatial transform, with all of them, with extra noise, with all plus noise."""
    extras = ([(s, False) for s in ('none',) + SPATIAL_KINDS + ('all',)]
              + [('none', True), ('all', True)])
    return [(arm, spatial, noise) for arm in ARMS for spatial, noise in extras]


def condition_argv(arm, spatial, noise):
    return ['--arm', arm, '--spatial', spatial] + (['--noise'] if noise else [])


#: Options that pick a run or drive the grid; everything else is passed on.
GRID_OWN = {'arm', 'spatial', 'noise', 'seed', 'device', 'gpus', 'per_gpu',
            'seeds', 'stagger', 'tune', 'fetch', 'resume', 'bench', 'dry_run', 'preview'}


def passed_on(parser, args):
    """The command-line options of *args* that differ from their defaults."""
    argv = []
    for action in parser._actions:
        dest = action.dest
        if dest in GRID_OWN or not action.option_strings or dest == 'help':
            continue
        value = getattr(args, dest)
        if value == action.default:
            continue
        flag = action.option_strings[0]
        if isinstance(action, argparse.BooleanOptionalAction):
            argv.append(flag if value else f"--no-{flag[2:]}")
        elif isinstance(action, argparse._StoreTrueAction):
            argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv += [flag, *map(str, value)]
        else:
            argv += [flag, str(value)]
    return argv


def fetch_all(args):
    """Every subject the ablation touches, downloaded once before any run starts."""
    M = MRSIChallengeDataModule
    names = list(M.TRAIN_SUBJECTS) + list(M.TRACK1_SUBJECTS) + list(M.TRACK2_SUBJECTS)
    M.fetch(names, root=args.data_dir)
    print(f"all {len(names)} subjects in {args.data_dir}")


#: Speed options --tune compares on the heaviest condition.
TUNE_VARIANTS = {
    'fp32':                       ['--no-amp'],
    'bf16':                       [],
    'bf16+channels-last':         ['--channels-last'],
    'bf16+compile':               ['--compile'],
    'bf16+channels-last+compile': ['--channels-last', '--compile'],
}


def tune(parser, args):
    """Time each speed option on this GPU; print the fastest flags."""
    base = [sys.executable, __file__, '--bench', str(args.bench or 30),
            *passed_on(parser, args), *condition_argv('augmentrum', 'all', True)]
    rows = []
    for name, flags in TUNE_VARIANTS.items():
        print(f"{name:28s}", end=' ', flush=True)
        proc = subprocess.run(base + flags, capture_output=True, text=True)
        line = next((l for l in proc.stdout.splitlines() if 'ms/pull' in l), None)
        if line is None:
            print("failed:", (proc.stderr.strip().splitlines() or ['?'])[-1])
            continue
        print(line)
        rows.append((float(line.split('prefetched step')[1].split()[0]), name, flags))
    if not rows:
        sys.exit("no variant ran")
    step, name, flags = min(rows)
    print(f"\nfastest: {name}, {step:.0f} ms/step. Add to the ablation command: {' '.join(flags) or '(nothing)'}")


def grid(parser, args):
    """
    Every condition x seed, --per-gpu runs at a time on each of --gpus,
    started --stagger seconds apart. Finished runs (test.json) are skipped,
    interrupted ones resume; each run logs to <run>/run.log.
    """
    gpus = args.gpus if args.gpus else list(range(torch.cuda.device_count()))
    if not gpus:
        sys.exit("no GPU visible: the ablation needs at least one (or --device cpu for one run)")
    fetch_all(args)                     # once, before any run: no races between downloads

    common = passed_on(parser, args)
    jobs = []
    for seed in args.seeds:
        for condition in conditions():
            run_args = parser.parse_args(condition_argv(*condition) + ['--seed', str(seed)])
            out = Path(args.out_dir) / run_name(argparse.Namespace(
                **{**vars(args), **{k: getattr(run_args, k)
                                    for k in ('arm', 'spatial', 'noise', 'seed')}}))
            if not finished(out, args.steps):
                jobs.append((condition, seed, out))
    slots = [gpu for gpu in gpus for _ in range(args.per_gpu)]
    print(f"{len(jobs)} runs to do on GPUs {gpus}, {args.per_gpu} per GPU; "
          f"passed on: {' '.join(common) or '(defaults)'}")

    running, failed = {}, []
    while jobs or running:
        for slot, (proc, name) in list(running.items()):
            if proc.poll() is not None:
                del running[slot]
                ok = proc.returncode == 0
                print(f"{time.strftime('%H:%M')}  {'done' if ok else 'FAILED':7s} {name}",
                      flush=True)
                if not ok:
                    failed.append(name)
        free = [i for i in range(len(slots)) if i not in running]
        if jobs and free:
            condition, seed, out = jobs.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, __file__, *condition_argv(*condition), '--seed', str(seed),
                   '--device', 'cuda', '--resume', *common]
            # each run sees only its own GPU, so nothing of it can land on another
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slots[free[0]]))
            running[free[0]] = (subprocess.Popen(cmd, stdout=open(out / 'run.log', 'a'),
                                                 stderr=subprocess.STDOUT, env=env), out.name)
            print(f"{time.strftime('%H:%M')}  started {out.name} on GPU {slots[free[0]]}",
                  flush=True)
            time.sleep(args.stagger)
            continue
        time.sleep(10)
    print(f"grid finished, {len(failed)} failed" + (f": {failed}" if failed else ""))
    summarize(args.out_dir)


def finished(out: Path, steps: int) -> bool:
    """Tested, at *steps* or more: a larger budget later resumes and retests the run."""
    log = out / 'log.csv'
    if not (out / 'test.json').exists() or not log.exists():
        return False
    last = log.read_text().strip().splitlines()[-1].split(',')[0]
    return last.isdigit() and int(last) >= steps


def summarize(out_dir):
    """
    Every finished run's test results in one table, <out_dir>/summary.csv:
    one row per run, weights (last/best), acceleration and test track.
    """
    rows = []
    for path in sorted(Path(out_dir).glob('*/test.json')):
        config = json.loads((path.parent / 'config.json').read_text())['args']
        for weights, results in json.loads(path.read_text()).items():
            for key, result in results.items():
                acc, track = key.split('/')
                rows.append({'run': path.parent.name, 'arm': config['arm'],
                             'spatial': config['spatial'], 'noise': config['noise'],
                             'seed': config['seed'], 'weights': weights,
                             'acceleration': acc[3:], 'track': track, **result['mean']})
    if not rows:
        print("no finished runs to summarize")
        return
    path = Path(out_dir) / 'summary.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} results of {len({r['run'] for r in rows})} runs in {path}")
    print("best weights, mean over tracks and accelerations:")
    runs = sorted({r['run'] for r in rows})
    for run in runs:
        mine = [r for r in rows if r['run'] == run and r['weights'] == 'best']
        print(f"  {run:48s} nrmse {np.mean([r['nrmse'] for r in mine]):.4f}   "
              f"ssim {np.mean([r['ssim'] for r in mine]):.4f}")


#**********#
#   main   #
#**********#
def main():
    parser = argparse.ArgumentParser(
        description="Train a Deep-ER-style network on MRSI Challenge data with Augmentrum.")
    parser.add_argument('--data-dir', default='data/mrsi_challenge',
                        help="Release root; missing subjects are fetched from Zenodo.")
    parser.add_argument('--out-dir', default='results/deep_er/ablation')
    parser.add_argument('--arm', choices=sorted(ARMS), default=None,
                        help="Train this one condition. Without it, the whole ablation runs.")
    parser.add_argument('--spatial', choices=('none',) + SPATIAL_KINDS + ('all',),
                        default=None, help="Spatial transform(s) on top of the arm (default none).")
    parser.add_argument('--noise', action='store_true',
                        help="Extra noise per coil on the input, level drawn per batch.")
    parser.add_argument('--trajectory', choices=sorted(TRAJECTORIES),
                        default='eccentric-stack')
    parser.add_argument('--ksp-mode', choices=['nufft', 'gridded'], default='nufft',
                        help="Faithful NUFFT round trip, or cheaper gridded rasterization.")
    parser.add_argument('--train', type=int, default=19,
                        help="Contest subjects for training.")
    parser.add_argument('--val', type=int, default=5)
    parser.add_argument('--test', nargs='*', choices=['track1', 'track2'],
                        default=['track1', 'track2'],
                        help="Test sets to evaluate at the end; pass --test alone for none.")
    parser.add_argument('--n-coils', type=int, default=32)
    parser.add_argument('--n-layers', type=int, default=10,
                        help="Interlacer layers (paper: 10).")
    parser.add_argument('--n-timepoints', type=int, default=4,
                        help="FID timepoints per batch (one volume each pull).")
    parser.add_argument('--crop-t', type=int, default=None,
                        help="Debug: truncate the FID to this many points first.")
    parser.add_argument('--acc-low', type=float, default=1.0)
    parser.add_argument('--acc-high', type=float, default=6.0)
    parser.add_argument('--noise-sigma-k', type=float, default=4.4e-3,
                        help="k-space noise; 4.4e-3 gives the coil-combined input the "
                             "challenge's own noise, SD 1.0e-3 per point (Sub1, outside "
                             "the brain, measured at acceleration 1).")
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--steps', type=int, default=20_000,
                        help="Training steps per run (one pull of 4 timepoints each).")
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True,
                        help="bfloat16 convolutions; FFTs and the loss stay float32.")
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--pool-on-gpu', action=argparse.BooleanOptionalAction, default=None,
                        help="Keep the training subjects on the GPU (0.4 GB each). "
                             "Default: when it has room for them and 16 GB more.")
    parser.add_argument('--channels-last', action='store_true',
                        help="Channels-last 3-D layout for the convolutions.")
    parser.add_argument('--prefetch', type=int, default=2,
                        help="Batches built ahead in a background thread.")
    parser.add_argument('--val-on-cpu', action='store_true',
                        help="Keep the fixed validation batches in host memory.")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', default='deep-er')
    parser.add_argument('--wandb-entity', default='augmentrum')
    parser.add_argument('--dry-run', action='store_true',
                        help="One batch through data, model, loss and backward; no training.")
    parser.add_argument('--bench', type=int, default=0,
                        help="Time this many data pulls and steps, then exit.")
    parser.add_argument('--preview', action='store_true',
                        help="Render the configured pipeline on subject 0 (images, "
                             "trajectory, mask, spectra, NAA maps) and exit — for "
                             "hand-tuning modules and RANGES.")
    parser.add_argument('--fetch', action='store_true',
                        help="Download every subject of the ablation once, then exit.")
    parser.add_argument('--tune', action='store_true',
                        help="Time the speed options on this GPU and print the fastest flags.")
    parser.add_argument('--gpus', type=int, nargs='+', default=None,
                        help="GPUs for the ablation (default: every visible one).")
    parser.add_argument('--per-gpu', type=int, default=1)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--stagger', type=int, default=60,
                        help="Seconds between ablation run starts.")

    args = parser.parse_args()
    one_run = args.arm is not None or args.dry_run or args.bench or args.preview
    if args.fetch:
        fetch_all(args)
    elif args.tune:
        tune(parser, args)
    elif not one_run:
        grid(parser, args)
    else:
        args.arm = args.arm or 'augmentrum'
        args.spatial = args.spatial or 'none'
        train(args)


if __name__ == '__main__':
    main()
