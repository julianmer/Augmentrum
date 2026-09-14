####################################################################################################
#                                  04_mrsi_recon_training.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-11                                                                              #
#                                                                                                  #
# Purpose: Compact MRSI reconstruction training demo: a pipeline with a supervised tap turns       #
#          clean volumes into (undersampled input, clean target) pairs, and a toy network          #
#          takes a few gradient steps. Uses the real MRSI Challenge data when it is present        #
#          under data/mrsi_challenge, and falls back to synthetic volumes otherwise.               #
#                                                                                                  #
# The real thing lives in scripts/train_deep_er.py: a Deep-ER-style joint-domain network           #
# (Weiser et al., NeuroImage 2025) with 32 synthesized coils and faithful ECCENTRIC sampling.      #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from augmentrum import Augmentrum
from augmentrum.dataset.mrsi_challenge import MRSIChallengeData, MRSIChallengeDataModule

# The release lives on Zenodo, one 4.4 GB zip per subject. This example uses
# it when the subjects it needs are already on disk and otherwise builds
# synthetic volumes, so that running an example never starts a download.
DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'mrsi_challenge'
SPLITS = {'train': 6, 'val': 1}


#******************#
#   data loading   #
#******************#
# The MRSI Challenge's own training recipe — spatial warp, undersampling,
# noise — with a supervised tap inserted: the warp sits BEFORE the tap (a
# property of the object, in both input and target), undersampling and noise
# after it degrade only the input. The outputs spec decides what the
# dataloader yields. Full volumes take a few seconds per step on CPU.
PIPELINE = ['spatial', 'tap:clean', 'undersampling', 'noise']
RANGES = dict(acceleration_factor=(2.0, 4.0), sigma=(0.5e-3, 1.5e-3))

module = MRSIChallengeDataModule(DATA_DIR, download=False)
subjects = [s for members in module.resolve(SPLITS).values() for s in members]

if all(map(module.available, subjects)):
    print(f"Loading the MRSI Challenge from {DATA_DIR} ...")
    aug = MRSIChallengeData(
        DATA_DIR,
        signal='clean',                     # metabolites only — Augmentrum adds the rest
        splits=SPLITS,
        pipelines={'train': PIPELINE, 'val': []},
        outputs={'train': ('data', 'clean'), 'val': None},
        batch_size=1,
        **RANGES,
    )
else:
    print("MRSI Challenge data not found — building synthetic volumes instead.")
    print(f"To use the real thing, fetch {len(subjects)} subjects once with\n"
          f"  MRSIChallengeDataModule.fetch({subjects}, {str(DATA_DIR)!r})\n")
    from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

    rng = np.random.default_rng(0)
    X, Y, Z, T = 16, 16, 4, 64

    xx, yy = np.meshgrid(np.linspace(-1, 1, X), np.linspace(-1, 1, Y), indexing='ij')
    head = (xx ** 2 + yy ** 2 < 0.8 ** 2).astype(np.float32)[..., None]   # (X, Y, 1)

    t = np.arange(T) / 2000.0
    fid = np.exp(2j * np.pi * (-2.0 * 123.2) * t) * np.exp(-t * 12 * np.pi)

    volumes = []
    for _ in range(6):
        amplitude = head * (0.5 + rng.random((X, Y, Z)).astype(np.float32) * head)
        volume = amplitude[..., None] * fid[None, None, None, :]
        volumes.append(gen_nifti_mrs(volume.astype(np.complex64), 1 / 2000, 123.2))

    aug = Augmentrum(
        data=volumes,
        pipeline=PIPELINE,
        outputs=('data', 'clean'),
        batch_size=2,
        backend='pytorch',
        volatile=True,
        **RANGES,
    )

print(aug.visualize_pipeline('train'))


#*****************#
#   toy network   #
#*****************#
# 2-channel (real/imag) 3-D conv net over (X, Y, Z), one FID timepoint at a
# time — the same per-timepoint framing Deep-ER uses, minus everything else.
net = nn.Sequential(
    nn.Conv3d(2, 16, 3, padding='same'), nn.ReLU(),
    nn.Conv3d(16, 16, 3, padding='same'), nn.ReLU(),
    nn.Conv3d(16, 2, 3, padding='same'),
)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


def to_channels(batch, timepoint=0):
    """(B, X, Y, Z, T) complex -> (B, 2, X, Y, Z) float at one timepoint."""
    volume = batch[..., timepoint]
    return torch.stack((volume.real, volume.imag), dim=1)


#**************#
#   training   #
#**************#
print("\nTraining the toy network for a few steps ...")
loader = aug.dataloader()
for step in range(10):
    x, y = next(loader)                      # (input, target) — the outputs spec
    inputs = to_channels(x.to(torch.cfloat))
    target = to_channels(y.to(torch.cfloat))

    optimizer.zero_grad()
    loss = nn.functional.mse_loss(net(inputs), target)
    loss.backward()
    optimizer.step()
    print(f"  step {step + 1:2d}   loss {loss.item():.6f}")

print("""
Done. What happened:
  - each batch drew a fresh spatial warp, acceleration factor and noise level,
  - the tap froze the target BEFORE undersampling and noise,
  - the network learned to undo the degradation it saw.

Scale this up with scripts/train_deep_er.py: the same forward model with 32
synthesized coils, faithful ECCENTRIC sampling and a Deep-ER-style network
on the full MRSI Challenge volumes.
""")
