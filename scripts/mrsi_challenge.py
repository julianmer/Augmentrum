####################################################################################################
#                                      mrsi_challenge.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-07-29                                                                              #
#                                                                                                  #
# Purpose: Trains-ready augmentation pipeline for the MRSI Challenge dataset. Loads the clean      #
#          metabolite signal, splits train/val from the contest subjects, pins the challenge's     #
#          own two test sets, and lets Augmentrum supply the degradation (spatial augmentation,    #
#          k-space undersampling, baseline, noise).                                                #
#                                                                                                  #
# Pipeline:                                                                                        #
#   1. Load MRSI Challenge volumes (cached as memory-mapped NIfTI-MRS on first run)                #
#   2. Build an Augmentrum with pinned track-1 / track-2 test splits                               #
#   3. Pull a batch from every split and report shapes / memory                                    #
#   4. Render figures: maps, sampling masks, voxel spectra, track comparison                       #
#   5. Optionally run a short PyTorch loop to prove the loader feeds a model                       #
#                                                                                                  #
# Output (results/mrsi_challenge/):                                                                #
#   clean_map.png              metabolite band map of an unaugmented volume                        #
#   augmented_map.png          the same volume after the training pipeline                         #
#   sampling_masks.png         k-space masks across an acceleration sweep                          #
#   voxel_spectra.png          clean vs augmented spectrum at one voxel                            #
#   track1_vs_track2.png       the nuisance difference between the two test sets                   #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import argparse
import os
import sys
import time
import warnings

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # Ensure relative paths like 'data/...' resolve correctly

import matplotlib
matplotlib.use("Agg")   # figures are written to disk, never shown

from augmentrum.dataset.mrsi_challenge import MRSIChallengeData, MRSIChallengeDataModule
from augmentrum.utils.plotting import (
    plot_kspace_mask,
    plot_mrsi_map,
    plot_mrsi_voxel_spectra,
)
from augmentrum.sampling import GridMask


#************#
#   config   #
#************#
DEFAULT_CONFIG = {
    # Data
    'data_dir':      'data/MRSI_Challenge',   # release root
    'signal':        'clean',                 # xtMeta: metabolites only, noiseless
    'source':        'mat',                   # authoritative, internally consistent
    'n_train':       8,                       # contest subjects to load (train + val)
    'n_val':         2,                       # of those, held out for validation

    # Output
    'save_dir':      'results/mrsi_challenge',

    # Batching
    'batch_size':    2,                       # a full volume is ~400 MB as complex64
    'seed':          42,

    # Augmentation
    'acceleration_factor': (2.0, 6.0),        # keep ~1/2 to ~1/6 of k-space
    'sigma':               1.0e-3,            # the challenge's own training sigma
    'baseline':            False,   # per-voxel, ~400s/volume — see --baseline
    'baseline_frac':       0.05,
    'prob':                0.5,               # per-augmentation activation
    'ksp_mode':            'cartesian',
    'trajectory':          'golden_radial_2d',   # only used by ksp_mode='gridded'

    # Figures
    'ppm_band':      (1.8, 2.2),              # brackets NAA at 2.008 ppm
    'ppm_lim':       (0.2, 4.2),
}


#**************#
#   pipeline   #
#**************#
def run_pipeline(config):
    """Load the challenge, build the augmenter, pull batches, and render figures."""
    warnings.filterwarnings("ignore")
    start = time.time()
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  Step 1: Loading MRSI Challenge data                             ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  data_dir : {config['data_dir']}")
    print(f"  signal   : {config['signal']}  (source={config['source']})")
    print(f"  subjects : {config['n_train']} contest ({config['n_val']} for val) "
          f"+ 5 track-1 + 3 track-2")
    print("  First run extracts each volume from its .mat and caches it as an")
    print("  uncompressed NIfTI-MRS; later runs memory-map that cache instead.")

    augmenter = MRSIChallengeData(
        data_dir=config['data_dir'],
        signal=config['signal'],
        source=config['source'],
        n_train=config['n_train'],
        n_val=config['n_val'],
        batch_size=config['batch_size'],
        seed=config['seed'],
        with_aux=True,
        baseline=config['baseline'],
        backend='pytorch',
        acceleration_factor=config['acceleration_factor'],
        sigma=config['sigma'],
        baseline_frac=config['baseline_frac'],
        prob=config['prob'],
        ksp_mode=config['ksp_mode'],
        trajectory=config['trajectory'],
    )
    module = augmenter.data_module
    ppm = module.ppm_axis()
    print(f"\n  ✓ loaded in {time.time() - start:.1f}s")
    for split, names in augmenter.subject_names.items():
        print(f"      {split:12s} {len(names):2d}  {', '.join(names) if names else '-'}")
    print(f"  ppm axis : {ppm.min():.2f} .. {ppm.max():.2f}  "
          f"({module.acquisition['n_points']} points, "
          f"{module.acquisition['spectrometer_frequency_mhz']:.6f} MHz)")

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  Step 2: Augmentation pipeline                                   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    augmenter.show_pipeline(detailed=True)

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  Step 3: One batch from every split                              ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    batches = {}
    for split in ('train', 'val', 'test_track1', 'test_track2'):
        t0 = time.time()
        batch, _ = next(augmenter.dataloader(split=split))
        batches[split] = batch
        mb = batch.numel() * batch.element_size() / 1e6
        print(f"  {split:12s} {tuple(batch.shape)}  {str(batch.dtype).replace('torch.',''):10s} "
              f"{mb:6.0f} MB  in {time.time() - t0:5.1f}s")

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  Step 4: Figures                                                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    figures = render_figures(augmenter, module, batches, config, save_dir)
    for label, path in figures.items():
        print(f"  {label:22s} {path}")

    print(f"\n  ✓ finished in {time.time() - start:.1f}s")
    return augmenter, figures


def render_figures(augmenter, module, batches, config, save_dir):
    """Render the figure set. Returns a dict of label -> saved path."""
    ppm = module.ppm_axis()
    band, ppm_lim = config['ppm_band'], config['ppm_lim']
    figures = {}

    # Before/after on the SAME subject. Taking the "before" from a test split
    # would compare two different people and attribute the difference to the
    # augmentation, so the training pipeline is run explicitly on one training
    # subject and shown against that subject's own clean volume.
    from nifti_mrs_plus import Backend, NIfTI_MRS_Plus

    train_plus, _ = augmenter.splits['train']
    subject = NIfTI_MRS_Plus(nifti_list=[train_plus.list()[0]],
                             backend=augmenter.backend, volatile=True)
    clean_vol = subject.get_data(Backend.PYTORCH).numpy()[0]

    augmented, _ = augmenter.pipelines['train'](
        NIfTI_MRS_Plus(nifti_list=[train_plus.list()[0].copy()],
                       backend=augmenter.backend, volatile=True))
    aug_vol = augmented.get_data(Backend.PYTORCH).numpy()[0]
    subject_name = augmenter.subject_names['train'][0]

    clean_spec = module.to_spectrum(clean_vol)
    aug_spec = module.to_spectrum(aug_vol)

    aux = augmenter.aux['train'][0]
    mask = aux.get('brainMask')
    mask = mask.astype(bool) if mask is not None else None

    figures['clean map'] = plot_mrsi_map(
        clean_spec, ppm, band=band, mask=mask,
        title=f"{subject_name}, unaugmented ({min(band)}-{max(band)} ppm)",
        save_path=save_dir, name='clean_map')

    figures['augmented map'] = plot_mrsi_map(
        aug_spec, ppm, band=band,
        title=f"{subject_name}, after the training pipeline",
        save_path=save_dir, name='augmented_map')

    # Acceleration sweep, drawn with the same machinery the pipeline uses.
    nx, ny = clean_vol.shape[0], clean_vol.shape[1]
    accels = (1.5, 2.0, 4.0, 8.0)
    masks = np.stack([GridMask.cartesian_vd_mask((nx, ny), a, acs_frac=0.06, seed=config['seed'])
                      for a in accels])
    figures['sampling masks'] = plot_kspace_mask(
        masks, title="k-space sampling masks",
        subtitles=[f"acceleration {a:g}x" for a in accels],
        save_path=save_dir, name='sampling_masks')

    # A brain voxel, chosen from the mask so the spectrum is not background.
    voxel = _pick_voxel(clean_vol, mask)
    figures['voxel spectra'] = plot_mrsi_voxel_spectra(
        [clean_spec, aug_spec], ppm, voxel,
        labels=[f'{subject_name} clean', f'{subject_name} augmented'],
        ppm_lim=ppm_lim, normalize=True,
        title=f"voxel {voxel} — augmentation effect on the spectrum",
        save_path=save_dir, name='voxel_spectra')

    # The two held-out test sets, both unaugmented. On signal='clean' both are
    # metabolite-only, so this shows the subjects rather than the nuisance
    # difference the tracks are named for — load signal='composite' to see that.
    t1_spec = module.to_spectrum(batches['test_track1'][0].numpy())
    t2_spec = module.to_spectrum(batches['test_track2'][0].numpy())
    figures['track1 vs track2'] = plot_mrsi_voxel_spectra(
        [t1_spec, t2_spec], ppm, voxel,
        labels=['track 1', 'track 2'], ppm_lim=ppm_lim, normalize=True,
        title=f"voxel {voxel} — track-1 vs track-2 test subject",
        save_path=save_dir, name='track1_vs_track2')

    return figures


def _pick_voxel(volume, mask=None):
    """A voxel with real signal — the median-index brain voxel, or the brightest."""
    if mask is not None and mask.any():
        idx = np.argwhere(mask)
        return tuple(int(v) for v in idx[len(idx) // 2])
    energy = np.abs(volume).sum(axis=-1)
    return tuple(int(v) for v in np.unravel_index(np.argmax(energy), energy.shape))


#*******************#
#   training demo   #
#*******************#
def run_training_demo(augmenter, n_batches=3):
    """
    Feed a few batches through a small network, to show the loader is trainable.

    Complex data is fed as two real channels via "torch.view_as_real" rather
    than cast with ".float()", which silently discards the imaginary part.
    """
    import torch
    import torch.nn as nn

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  Step 5: Short training loop                                     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    class VoxelEncoder(nn.Module):
        """Per-voxel 1-D encoder over the spectral axis; 2 channels = (real, imag)."""

        def __init__(self, n_out=16):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(2, 16, kernel_size=7, padding=3), nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=7, padding=3), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(32, n_out)

        def forward(self, x):                      # x: (n_voxels, n_points) complex
            x = torch.view_as_real(x).permute(0, 2, 1)     # -> (n_voxels, 2, n_points)
            return self.head(self.net(x).squeeze(-1))

    model = VoxelEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    loader = augmenter.train_dataloader()
    for i in range(n_batches):
        batch, _ = next(loader)
        b, nx, ny, nz, nt = batch.shape

        # Train on a random subset of voxels: a full volume is 131k spectra.
        voxels = batch.reshape(-1, nt)
        keep = torch.randperm(voxels.shape[0])[:512]
        voxels = voxels[keep]

        target = torch.zeros(voxels.shape[0], 16)
        loss = criterion(model(voxels), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"  batch {i + 1}/{n_batches}: {b} volumes {(nx, ny, nz, nt)}, "
              f"512 voxels sampled, loss {loss.item():.5f}")

    print("  ✓ gradients flow end to end from the augmented loader")


#*********#
#   cli   #
#*********#
def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment the MRSI Challenge dataset with Augmentrum.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # quick run on four subjects
  python scripts/mrsi_challenge.py --n-train 4 --n-val 1

  # the full contest set, heavier acceleration
  python scripts/mrsi_challenge.py --n-train 24 --n-val 5 --acceleration 8

  # keep the challenge's macromolecules and baseline instead of synthesizing them
  python scripts/mrsi_challenge.py --signal nuisance_free

  # non-Cartesian coverage from a spiral trajectory
  python scripts/mrsi_challenge.py --ksp-mode gridded --trajectory spiral_2d

  # also run the short training loop
  python scripts/mrsi_challenge.py --train
        """)
    d = DEFAULT_CONFIG
    parser.add_argument('--data-dir', default=d['data_dir'], help='release root')
    parser.add_argument('--signal', default=d['signal'],
                        choices=sorted(MRSIChallengeDataModule.SIGNALS),
                        help="component to load (default: clean = metabolites only)")
    parser.add_argument('--source', default=d['source'], choices=['mat', 'nifti', 'auto'])
    parser.add_argument('--n-train', type=int, default=d['n_train'])
    parser.add_argument('--n-val', type=int, default=d['n_val'])
    parser.add_argument('--batch-size', type=int, default=d['batch_size'])
    parser.add_argument('--seed', type=int, default=d['seed'])
    parser.add_argument('--acceleration', type=float, nargs='+', default=None,
                        help='one value for a fixed acceleration, two for a sampled range')
    parser.add_argument('--sigma', type=float, default=d['sigma'],
                        help='absolute noise sigma; the challenge trains at 1e-3')
    parser.add_argument('--ksp-mode', default=d['ksp_mode'],
                        choices=['cartesian', 'gridded', 'off'])
    parser.add_argument('--trajectory', default=d['trajectory'],
                        help="trajectory for --ksp-mode gridded")
    parser.add_argument('--save', default=d['save_dir'], help='figure output directory')
    parser.add_argument('--no-cache', action='store_true',
                        help='re-read the source files instead of the volume cache')
    parser.add_argument('--baseline', action='store_true',
                        help='add the spectral baseline augmentation; it is per-voxel '
                             'and costs ~400s per volume against ~25s for the rest')
    parser.add_argument('--train', action='store_true',
                        help='also run a short PyTorch training loop')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({
        'data_dir':   args.data_dir,
        'signal':     args.signal,
        'source':     args.source,
        'n_train':    args.n_train,
        'n_val':      args.n_val,
        'batch_size': args.batch_size,
        'seed':       args.seed,
        'sigma':      args.sigma,
        'ksp_mode':   args.ksp_mode,
        'trajectory': args.trajectory,
        'save_dir':   args.save,
        'baseline':   args.baseline,
    })
    if args.acceleration is not None:
        config['acceleration_factor'] = (tuple(args.acceleration)
                                         if len(args.acceleration) > 1
                                         else float(args.acceleration[0]))

    augmenter, figures = run_pipeline(config)

    if args.train:
        run_training_demo(augmenter)
