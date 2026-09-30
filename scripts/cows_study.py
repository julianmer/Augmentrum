####################################################################################################
#                                          cows_study.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-23                                                                              #
#                                                                                                  #
# Purpose: The COWS study: does Augmentrum help a self-supervised quantification network when      #
#          in-vivo data are scarce? Trains the network on the fly on the raw COWS transients of    #
#          1-8 subjects (OpenNeuro ds006812), with and without Augmentrum's modules, and scores   #
#          it on simulated spectra with known concentrations.                                      #
#                                                                                                  #
####################################################################################################

"""
The COWS study: data, signal models, network, training, test sets, scores.

Pieces
------
- Folds: 10 subjects, 5 folds of 2 validation subjects; a fold's 8 training
  subjects are added in a fixed order, so 1..8 subjects are nested.
- Conditions: the Augmentrum pipeline of a run ("CONDITIONS"): coil and
  transient sampling on the raw acquisition, the processing, the modules, then
  the 'clean' tap. Every module sits before the tap, so the augmented spectrum is
  both the network input and the target of the self-supervised loss.
- Signal models: 'A', one Voigt shared by all basis spectra (FSL-MRS' default
  model), and 'PB', that Voigt plus an extra Lorentzian per basis spectrum within
  Osprey's bounds (0-10 s^-1, macromolecules 0-100). Both with a complex poly-2
  baseline and zero- and first-order phase, fitted on 0.5-4.2 ppm.
- Network: the OoD paper's MLP (512 x 3), its head bounded per parameter kind,
  trained with the spectral loss normalised per spectrum, Adam, batch 16.
- Test sets: simulated spectra that copy one of the 90 processed scans as one
  tool fitted it (FSL-MRS PB, LCModel or Osprey; see "generate_testset"). The
  selection set, on which checkpoints are chosen, is the same construction with
  another seed.
- Score: MOSAE, the mean absolute concentration error after each spectrum's
  optimal scale (macromolecules left out), and Lin's CCC on that scale.

Usage
-----
    # the 90 processed scans (the benchmark's input, cows_benchmark.py)
    python scripts/cows_study.py process --out results/cows/processed_scans.npz

    # a test set (seed 0) and the selection set (seed 1) from the benchmark's fits
    python scripts/cows_study.py testset --fits results/cows/invivo --seed 0 \\
        --out results/cows/testsets
    # one run
    python scripts/cows_study.py train --condition noise --n-subjects 1 --fold 0 \\
        --testset results/cows/testsets/test_n1000_s0.npz \\
        --selection-set results/cows/testsets/test_n1000_s1.npz --out results/cows/runs_x
    # a finished run's checkpoints on a test set
    python scripts/cows_study.py score results/cows/runs_x/runs/<run id> --testset ...

Run from the Augmentrum root; everything sits under data/ (git-ignored):
data/openneuro_ds006812 (the dataset, the repo's submodule path),
data/BasisSets/TE26_basis_summed (the FSL-MRS JSON basis, one Cr and one PCr) and
data/cows_cache (Augmentrum's uncompressed NIfTI cache of the raw scans, built on
first use). --data-dir, --basis-dir and --cache-dir override them.
"""

#*************#
#   imports   #
#*************#
import os
import sys

# uncapped BLAS threads stall numpy on a shared machine
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')

import argparse
import copy
import csv
import fcntl
import signal
import hashlib
import json
import math
import re
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#**************#
#   settings   #
#**************#
DATA_DIR = os.environ.get('COWS_DATA_DIR', 'data/openneuro_ds006812')
BASIS_DIR = os.environ.get('COWS_BASIS_DIR', 'data/BasisSets/TE26_basis_summed')
CACHE_DIR = os.environ.get('COWS_CACHE_DIR', 'data/cows_cache')

BW, POINTS = 4000.0, 2048                   # COWS after oversampling removal; the basis to match
PPM_WINDOW = (0.5, 4.2)                     # the network's input and loss, every tool's fit window
PPM_NAA = (1.9, 2.1)                        # the SNR's reference peak
PPM_NOISE = (-2.0, 0.0)                     # a signal-free region for the noise SD
MM_NAMES = ('MM', 'MM09')
TOTALS = {'tNAA': ('NAA', 'NAAG'), 'tCr': ('Cr', 'PCr'), 'tCho': ('Cho', 'GPC'),
          'Glx': ('Glu', 'Gln'), 'mI': ('mI',), 'GSH': ('GSH',), 'Tau': ('Tau',)}

SUBJECTS = tuple(f'sub-{i:02d}' for i in range(1, 11))
FOLDS = {k: SUBJECTS[2 * k:2 * k + 2] for k in range(5)}   # fold k validates 2k+1, 2k+2
SUBSET_SEED = 2026                          # the order in which a fold's subjects are added
#: the scans the test set (seed 0) and the selection set (seed 1) copy: disjoint subjects, no leakage
TESTSET_SUBJECTS = {0: SUBJECTS[0::2], 1: SUBJECTS[1::2]}

TRAIN = dict(variant='A', activation='elu', dropout=0.0, width=512, depth=3, batch=16, lr=1e-4,
             weight_decay=0.0, max_steps=2_000_000, eval_every=5000)
PROCESSING = dict(conj=False, registration_method='torch')

#: Osprey's upper bounds on a basis spectrum's Lorentzian (fit_OspreyPrelimStep2.m), applied as
#: exp(-lorentzLB t), the convention of gamma here: model PB's extra Lorentzian per basis spectrum
PB_DELTA_MAX, PB_DELTA_MAX_MM = 10.0, 100.0


#****************#
#   conditions   #
#****************#
# One pipeline step per module: {Augmentrum registry name: step kwargs}. "ref" below is a module's
# amplitude reference, the real maximum of the spectrum (NAA or the residual water). The ranges
# were set on 2026-09-18 from the 90 processed scans and fits of the decomposed basis; they are
# re-derived from the summed-basis fits before the main experiment.
MODULES = {
    # FWHM added: NAA Lorentzian 0.8-4.0 Hz, Gaussian 1.8-4.1 Hz in vivo (p5-p95)
    'line_broadening': {'line_broadening': {'mode': 'voigt', 'lb_hz': (0.0, 3.3),
                                            'gb_hz': (0.0, 4.1)}},
    # the fitted shift spans -0.70 to 0.61 Hz (p5-p95)
    'frequency_shift': {'frequency_shift': {'shift_hz': (-0.69, 0.69)}},
    # phi0 spans 25.5 deg, phi1 about 180 deg (p5-p95)
    'phase_shift': {'phase_shift': {'zero_order_deg': (-14.0, 14.0),
                                    'first_order_deg': (-90.0, 90.0)}},
    # the fitted MM + MM09 peak is 0.13-0.29 of ref
    'macromolecules': {'macromolecules': {'mm_source': 'semi_parametrized',
                                          'mm_scale': (0.0, 0.15)}},
    # the residual water is 0.23-1.06 of ref, at any phase
    'residual_water': {'residual_water': {'amplitude_scale': (0.0, 0.83),
                                          'phase_deg': (-180.0, 180.0)}},
    # the fitted poly-2 baseline peaks at 0.016-0.12 of ref, at any phase
    'baseline': {'baseline': {'mode': 'bspline', 'baseline_frac': (0.0, 0.10),
                              'phase_deg': (-180.0, 180.0)}},
    # the largest unexplained residual: <= 0.05 of ref at 0.5-1.75 ppm (lipid / MM region)
    'artificial_peaks': {'artificial_peaks': {'peaks': [
        {'ppm': (0.5, 1.75), 'amp': (0.0, 0.05), 'lb_hz': (5.0, 20.0), 'gb_hz': 0.0,
         'phase_deg': 0.0}]}},
    # eddy-current phase of the uncorrected waters: 0.04-0.48 rad RMS, i.e. strength 0-2
    'eddy_current': {'eddy_current': {'mode': 'synthetic', 'strength': (0.0, 2.0)}},
    # no echo is visible in vivo: up to 1 % of max |FID|, the rest as SMART-MRS
    'spurious_echoes': {'spurious_echoes': {'mode': 'echo', 'echoes': [
        {'t_echo_frac': (0.1, 0.9), 'T2': (0.01, 0.05), 'ppm': (0.0, 8.0),
         'phase_deg': (0.0, 360.0), 'amp': (0.0, 0.01)}]}},
    # max |spectrum| / added-noise SD: from half the lowest in-vivo SNR to little added noise
    'noise': {'noise': {'snr': (30.0, 330.0)}},
}
# coil / transient subsets drawn per sample: 16-32 of the 32 each
SAMPLERS = {
    'coil_sampling': {'coil_sampling': {'per_sample': True, 'n_coils': (16, 32)}},
    'average_sampling': {'average_sampling': {'per_sample': True, 'n_averages': (16, 32)}},
}
# name -> (samplers, modules)
CONDITIONS = {
    'none': ((), ()),
    'coil_sampling': (('coil_sampling',), ()),
    'average_sampling': (('average_sampling',), ()),
    'sampling': (('coil_sampling', 'average_sampling'), ()),
    **{name: ((), (name,)) for name in MODULES},
    'all': (tuple(SAMPLERS), tuple(MODULES)),
}


def pipeline_spec(condition, processed=False):
    """
    A condition's training pipeline as an Augmentrum entry list: samplers on the raw acquisition,
    the processing, the modules, and the 'clean' tap (the loss target, the augmented spectrum).

    Args:
        processed: The data are already processed (a condition without samplers processes every
            scan the same way each time): no samplers, no processing.
    """
    samplers, modules = CONDITIONS[condition]
    if processed and samplers:
        raise ValueError(f'{condition!r} draws coils / transients: it must process on the fly')
    entry = lambda table, e: copy.deepcopy(table[e] if isinstance(e, str) else e)
    spec = [] if processed else ([entry(SAMPLERS, s) for s in samplers]
                                 + [{'processing': dict(PROCESSING)}])
    return spec + [entry(MODULES, m) for m in modules] + ['tap:clean']


def _ranges(obj):
    """JSON back into Augmentrum's conventions: a pair of numbers is a (lo, hi) range."""
    if isinstance(obj, dict):
        return {k: _ranges(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) == 2 and all(isinstance(v, (int, float)) for v in obj):
            return tuple(obj)
        return [_ranges(v) for v in obj]
    return obj


def register_augment(spec):
    """
    A condition given as data (cows_screen.py): {'name', 'samplers': [...], 'modules': [...]},
    each item a name from SAMPLERS / MODULES or an Augmentrum entry ({module: kwargs}). Adds it
    to CONDITIONS and returns its name.
    """
    if isinstance(spec, str):
        with open(spec) as f:
            spec = json.load(f)
    name = spec['name']
    if '__' in name or not re.fullmatch(r'[A-Za-z0-9_.+=-]+', name):
        raise ValueError(f'bad condition name {name!r}')
    CONDITIONS[name] = (tuple(_ranges(spec.get('samplers', []))),
                        tuple(_ranges(spec.get('modules', []))))
    return name


#***********#
#   basis   #
#***********#
@dataclass
class BasisSet:
    """
    The basis as the network uses it: fids (T, n) complex, conjugated so that fft of basis and
    data put NAA at the same bin; t (T,) starting at one dwell time; ppm (T,) of each unshifted
    FFT bin.
    """
    fids: np.ndarray
    names: list
    t: np.ndarray
    ppm: np.ndarray
    cf: float
    bw: float

    def window(self, ppmlim=PPM_WINDOW):
        """Nearest indices of the ppm limits: (first, last)."""
        return (int(np.abs(self.ppm - ppmlim[0]).argmin()),
                int(np.abs(self.ppm - ppmlim[1]).argmin()))


def load_basis(basis_dir=BASIS_DIR, bw=BW, points=POINTS):
    """Read the FSL-MRS JSON basis and format it to the COWS grid (4000 Hz, 2048 points)."""
    from fsl_mrs.core.basis import Basis
    from fsl_mrs.utils.mrs_io.fsl_io import readFSLBasisFiles

    basis = Basis(*readFSLBasisFiles(basis_dir))
    # formatted in place, so that FSL-MRS' axes (ppm, dwell) describe the formatted basis
    basis._raw_fids = basis.get_formatted_basis(bw, points)
    basis._dt = 1.0 / bw
    basis._names = basis.get_formatted_names()
    fids = basis._raw_fids
    spec = np.abs(np.fft.fft(fids[:, 0]))
    if spec[:points // 2].max() > spec[points // 2:].max():
        fids = np.conjugate(fids)
    dt = float(basis.original_dwell)
    return BasisSet(fids=fids, names=[n.split('.')[0] for n in basis.names],
                    t=np.arange(dt, dt * (points + 1), dt)[:points],
                    ppm=np.fft.ifftshift(basis.original_ppm_shift_axis), cf=float(basis.cf),
                    bw=float(1.0 / dt))


def poly_baseline(n_points, first, last, order=2):
    """
    FSL-MRS' complex polynomial baseline on the window: x^i on linspace(-1, 1), each orthogonalised
    against the previous ones, a real and an imaginary column per order, zero outside.
    """
    x = np.zeros(n_points, complex)
    x[first:last] = np.linspace(-1, 1, last - first)
    cols = []
    for i in range(order + 1):
        reg = x ** i
        if i > 0:
            conf = np.squeeze(np.asarray(cols)).T
            reg = reg - conf @ (np.linalg.pinv(conf) @ reg)
        cols += [reg.flatten(), 1j * reg.flatten()]
    B = np.asarray(cols).T
    out = np.zeros_like(B)
    out[first:last] = B[first:last]
    return out


def to_network(fid):
    """Batched FIDs (B, ..., T) complex -> (B, 2, T) real: fft without fftshift, re / im stacked."""
    spec = torch.fft.fft(fid.reshape(fid.shape[0], fid.shape[-1]), dim=-1)
    return torch.stack((spec.real, spec.imag), dim=1)


#**************************************************************************************************#
#                                        Class SignalModel                                         #
#**************************************************************************************************#
#                                                                                                  #
# fft( sum_k c_k b_k(t) exp(-(i eps + gamma_k + sigma^2 t) t) ) exp(-i (phi0 + phi1 f)) + B beta, #
# f the true frequency of each unshifted FFT bin. 'A': gamma_k = gamma for all k. 'PB': a gamma_k  #
# per basis spectrum (the network bounds how far they spread, see QuantNet).                       #
#                                                                                                  #
#**************************************************************************************************#
class SignalModel(nn.Module):

    VARIANTS = ('A', 'PB')

    def __init__(self, basis, variant='A', first=None, last=None, order=2, dtype=torch.float32):
        """
        Args:
            basis: A "BasisSet".
            variant: 'A' or 'PB'.
            first, last: The window (default "PPM_WINDOW").
            dtype: Real dtype of the buffers; complex ones are stored as real (..., 2) views, so
                .to(dtype) keeps their imaginary parts.
        """
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f'variant must be one of {self.VARIANTS}, got {variant!r}')
        if first is None or last is None:
            first, last = basis.window()
        self.variant, self.first, self.last = variant, int(first), int(last)
        self.names = list(basis.names)
        n, T = len(self.names), basis.fids.shape[0]
        baseline = poly_baseline(T, first, last, order)

        sizes = OrderedDict(con=n, gamma=n if variant == 'PB' else 1, sigma=1, eps=1, phi0=1,
                            phi1=1, baseline=baseline.shape[1])
        self.layout, start = OrderedDict(), 0
        for key, size in sizes.items():
            self.layout[key] = slice(start, start + size)
            start += size
        self.n_params = start
        # the parameters that scale with the spectrum: concentrations and baseline
        linear = np.r_[np.arange(n), np.arange(self.n_params)[self.layout['baseline']]]

        dt = float(basis.t[1] - basis.t[0])
        real = lambda a: torch.as_tensor(np.stack((a.real, a.imag), axis=-1), dtype=dtype)
        self.register_buffer('basis_ri', real(basis.fids))
        self.register_buffer('baseline_ri', real(baseline))
        self.register_buffer('t', torch.as_tensor(basis.t, dtype=dtype))
        self.register_buffer('f', torch.as_tensor(np.fft.fftfreq(T, d=dt), dtype=dtype))
        self.register_buffer('linear', torch.as_tensor(linear, dtype=torch.long))
        # 1 / window norm of each basis spectrum: c_k * c_scale_k is component k's size in a
        # unit-norm spectrum
        spec = np.fft.fft(basis.fids, axis=0)[first:last]
        self.register_buffer('c_scale', torch.as_tensor(1.0 / np.sqrt((np.abs(spec) ** 2).sum(0)),
                                                         dtype=dtype))

    @property
    def basis(self):
        return torch.view_as_complex(self.basis_ri)

    @property
    def baseline(self):
        return torch.view_as_complex(self.baseline_ri)

    def split(self, theta):
        """theta (B, P) -> {name: (B, size)} in layout order."""
        return OrderedDict((k, theta[:, s]) for k, s in self.layout.items())

    def join(self, parts):
        return torch.cat([parts[k].reshape(parts[k].shape[0], -1) for k in self.layout], dim=1)

    def describe(self):
        return {k: [s.start, s.stop] for k, s in self.layout.items()}

    def metab_fid(self, theta):
        """The broadened, shifted metabolite FID (B, T), before phase and baseline."""
        p = self.split(theta)
        t = self.t
        if self.variant == 'PB':
            # the shared factor is complex, the per-basis one real: the (B, T, n) tensor stays real
            shared = torch.exp(-(1j * p['eps'] + p['sigma'] ** 2 * t) * t)
            w = torch.exp(-p['gamma'][:, None, :] * t[None, :, None]) * p['con'][:, None, :]
            return shared * torch.complex(torch.einsum('btn,tn->bt', w, self.basis.real),
                                          torch.einsum('btn,tn->bt', w, self.basis.imag))
        lin = torch.exp(-(1j * p['eps'] + p['gamma'] + p['sigma'] ** 2 * t) * t)
        return lin * (p['con'].to(self.basis.dtype) @ self.basis.T)

    def forward(self, theta, baseline_out=False):
        """Complex spectrum (B, T) in unshifted FFT order, as fft(fid) of the data."""
        p = self.split(theta)
        S = torch.fft.fft(self.metab_fid(theta), dim=-1)
        ex = torch.exp(-1j * (p['phi0'] + p['phi1'] * self.f))
        ba = p['baseline'].to(self.baseline.dtype) @ self.baseline.T
        spec = ex * S + ba
        return (spec, ba) if baseline_out else spec

    def loss(self, theta, target):
        """Mean squared real and imaginary error over the window, per spectrum (B,)."""
        w = slice(self.first, self.last)
        spec = self.forward(theta)[:, w]
        return ((spec.real - target[:, 0, w]) ** 2 + (spec.imag - target[:, 1, w]) ** 2).mean(-1)


#**************************************************************************************************#
#                                            the network                                           #
#**************************************************************************************************#
class MLP(nn.Module):
    """BatchNorm over the two channels, flatten, fc layers halving the width, a linear output."""

    def __init__(self, shape_x, n_out, width=512, depth=3, activation='elu', dropout=0.0):
        super().__init__()
        self.act = {'elu': F.elu, 'relu': F.relu}[activation]
        self.bn = nn.BatchNorm1d(shape_x[0])
        self.drop = nn.Dropout(dropout)
        widths = [shape_x[0] * shape_x[1]] + [width // 2 ** i for i in range(depth)]
        self.fcs = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:]))
        self.out = nn.Linear(widths[-1], n_out)

    def forward(self, x):
        x = self.bn(x).flatten(1)
        for fc in self.fcs:
            x = self.drop(self.act(fc(x)))
        return self.out(x)


class QuantNet(nn.Module):
    """
    The network: the window of the unit-norm input, the MLP, and a head that bounds every
    parameter kind; "rescale" gives the parameters of the input itself.

    Head: concentrations softplus x c_scale (so every output starts at O(1)); widths softplus of
    10 z; shifts 10 z; phi0 z; phi1 3e-3 tanh(z); baseline 0.1 z. PB: gamma_k = gamma0 + delta_k,
    gamma0 softplus (one extra raw output), delta_k = bound_k sigmoid(z_k) ("PB_DELTA_MAX").
    """

    HEAD = {'con': F.softplus, 'gamma': F.softplus, 'sigma': F.softplus,
            'eps': lambda z: z, 'phi0': lambda z: z,
            'phi1': lambda z: 3e-3 * torch.tanh(z), 'baseline': lambda z: z}
    SCALE = {'gamma': 10.0, 'sigma': 10.0, 'eps': 10.0, 'baseline': 0.1}

    def __init__(self, model, **kwargs):
        super().__init__()
        self.first, self.last = model.first, model.last
        self.layout = OrderedDict(model.layout)
        self.register_buffer('linear', model.linear.clone())
        self.register_buffer('c_scale', model.c_scale.clone())
        self.pb = model.variant == 'PB'
        n_out = model.n_params
        if self.pb:
            self.g0 = self.layout['gamma'].start          # raw: [..., gamma0, delta_1..n, ...]
            n_out += 1
            self.register_buffer('delta_max', torch.as_tensor(
                [PB_DELTA_MAX_MM if nm in MM_NAMES else PB_DELTA_MAX for nm in model.names],
                dtype=model.c_scale.dtype))
        self.net = MLP((2, self.last - self.first), n_out, **kwargs)

    def head(self, z):
        if self.pb:
            g0 = z[:, self.g0:self.g0 + 1]
            z = torch.cat((z[:, :self.g0], z[:, self.g0 + 1:]), dim=1)
        keys = list(self.layout)
        out = [self.HEAD[k](self.SCALE.get(k, 1.0) * z[:, s]) for k, s in self.layout.items()]
        out[keys.index('con')] = out[keys.index('con')] * self.c_scale
        if self.pb:
            out[keys.index('gamma')] = (F.softplus(self.SCALE['gamma'] * g0)
                                        + self.delta_max * torch.sigmoid(z[:, self.layout['gamma']]))
        return torch.cat(out, dim=1)

    def forward(self, x):
        """(B, 2, T) -> (theta_n, norm): parameters of x / its window norm, and that norm (B, 1)."""
        xs = x[:, :, self.first:self.last]
        norm = torch.clamp(torch.sqrt((xs ** 2).sum(dim=(1, 2)))[:, None], min=1e-8)
        return self.head(self.net(xs / norm[:, :, None])), norm

    def rescale(self, theta_n, norm):
        """Concentrations and baseline scale with the spectrum."""
        scale = torch.ones_like(theta_n)
        scale[:, self.linear] = norm.to(theta_n.dtype)
        return theta_n * scale

    def predict(self, x):
        return self.rescale(*self.forward(x))


def spectral_loss(model, theta_n, target, norm):
    """The self-supervised loss per spectrum, target and reconstruction over the input's norm."""
    return model.loss(theta_n, target / norm[:, :, None])


#**************************************************************************************************#
#                                              the data                                            #
#**************************************************************************************************#
def fold_subjects(fold):
    """(the fold's 8 training subjects in their fixed order of addition, its 2 validation ones)."""
    val = FOLDS[fold]
    rest = [s for s in SUBJECTS if s not in val]
    order = np.random.default_rng([SUBSET_SEED, fold]).permutation(len(rest))
    return [rest[i] for i in order], list(val)


def load_scans(subjects=None, data_dir=DATA_DIR, cache_dir=CACHE_DIR):
    """Raw metabolite scans and waters of *subjects* (None: all), with names and subject ids."""
    from augmentrum.dataset.cows import COWSDataModule, scan_info

    loader = COWSDataModule(data_dir, cache_dir=cache_dir, strict=True, subjects=subjects)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data, water, _, _, names = loader.load_twix()
    return data, water, names, [scan_info(d)['SubjectID'] for d in data]


def process_scans(data, water, spec=None, device='cuda', seed=0, precision=None):
    """
    Every scan through a pipeline, one at a time, all coils and transients (Augmentrum's torch
    engine reads a batch's spectrometer frequency from its first scan, so scans are not batched).

    Returns:
        (N, T) complex tensor of the output FIDs.
    """
    from augmentrum import Augmentrum

    spec = spec or [{'processing': dict(PROCESSING)}]
    aug = Augmentrum(data=data, water=water, split_indices={'all': list(range(len(data)))},
                     pipelines={'all': spec}, modes={'all': 'fixed'}, outputs={'all': ('data',)},
                     batch_size=1, device=str(device), seed=seed, volatile=True,
                     **({'precision': precision} if precision else {}))
    return torch.cat([b[0].reshape(b[0].shape[0], -1) for b in aug.dataloader(split='all')])


def processed_niftis(data, fids):
    """Processed FIDs as NIfTI-MRS (1, 1, 1, T) with the header fields the modules read."""
    from augmentrum.dataset.cows import HEADER_FIELDS, scan_info
    from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

    out = []
    for fid, raw in zip(fids.cpu().numpy(), data):
        nii = gen_nifti_mrs(fid.reshape(1, 1, 1, -1), dwelltime=raw.dwelltime,
                            spec_freq=raw.spectrometer_frequency[0], nucleus=raw.nucleus[0])
        for key in ('EchoTime', 'RepetitionTime'):
            if key in raw.hdr_ext:
                nii.add_hdr_field(key, raw.hdr_ext[key])
        nii.add_hdr_field('SubjectID', scan_info(raw)['SubjectID'], doc=HEADER_FIELDS['SubjectID'])
        out.append(nii)
    return out


#**************************************************************************************************#
#                                         Class COWSStream                                         #
#**************************************************************************************************#
class COWSStream:
    """
    One run's data: an infinite training stream and the fixed validation set.

    Only the run's subjects are read, pooled once on the device by Augmentrum and drawn from by
    indexing. A condition without samplers processes every scan the same way, so its scans are
    processed once up front and only its modules run per batch; the samplers' conditions process
    the drawn coil / transient subsets on the fly. The validation subjects are processed once.

    Args:
        seed: Augmentrum's root seed; runs sharing it draw the same scans whatever the condition.
    """

    def __init__(self, condition, fold, n_subjects, seed=0, batch_size=16, device='cuda',
                 precision=None, data_dir=DATA_DIR, cache_dir=CACHE_DIR):
        from augmentrum import Augmentrum

        self.condition, self.fold, self.n_subjects = condition, fold, n_subjects
        self.precision = precision
        train, self.val_subjects = fold_subjects(fold)
        if not 1 <= n_subjects <= len(train):
            raise ValueError(f'n_subjects must be in 1..{len(train)}, got {n_subjects}')
        self.train_subjects = train[:n_subjects]
        self.device = torch.device(device)
        self.cached = not CONDITIONS[condition][0]
        self.spec = pipeline_spec(condition, processed=self.cached)

        data, water, names, groups = load_scans(self.train_subjects + self.val_subjects, data_dir,
                                                cache_dir)
        split = {'train': [i for i, g in enumerate(groups) if g in self.train_subjects],
                 'val': [i for i, g in enumerate(groups) if g in self.val_subjects]}
        self.scans = {k: [names[i] for i in v] for k, v in split.items()}
        pick = lambda items, key: [items[i] for i in split[key]]
        self._val_raw = (pick(data, 'val'), pick(water, 'val'))
        data, water = pick(data, 'train'), pick(water, 'train')
        self._train_raw = (data, water)
        self._train_fids = None
        if self.cached:
            self._train_fids = process_scans(data, water, device=self.device, precision=precision)
            data, water = processed_niftis(data, self._train_fids), None
        self.aug = Augmentrum(
            data=data, water=water, split_indices={'train': list(range(len(data)))},
            groups=pick(groups, 'train'), pipelines={'train': self.spec},
            modes={'train': 'on-the-fly'}, outputs={'train': ('data', 'clean')},
            batch_size=batch_size, device=str(self.device), seed=seed, volatile=True,
            **({'precision': precision} if precision else {}))
        self._train, self._val, self._train_full = None, None, None

    def next_batch(self):
        """(x, y): network input and loss target, (B, 2, T) on the device."""
        if self._train is None:
            self._train = ((to_network(d), to_network(c))
                           for d, c in self.aug.dataloader(split='train'))
        return next(self._train)

    def reseed(self, seed):
        """Restart the training stream from a new root seed (when resuming)."""
        self.aug.reseed(seed)
        self._train = None

    def validation(self):
        """(x (V, 2, T), stems) of the validation scans, processed once."""
        if self._val is None:
            self._val = to_network(process_scans(*self._val_raw, device=self.device,
                                                 precision=self.precision))
        return self._val, self.scans['val']

    def training_scans(self):
        """(x (N, 2, T), stems) of the training scans with all coils and transients."""
        if self._train_full is None:
            fids = self._train_fids
            if fids is None:
                fids = process_scans(*self._train_raw, device=self.device,
                                     precision=self.precision)
            self._train_full = to_network(fids)
        return self._train_full, self.scans['train']

    def describe(self):
        return dict(condition=self.condition, fold=self.fold, n_subjects=self.n_subjects,
                    train_subjects=self.train_subjects, val_subjects=self.val_subjects,
                    n_train_scans=len(self.scans['train']), n_val_scans=len(self.scans['val']),
                    pipeline=repr(self.spec), processing_cached=self.cached)


def process_invivo(out, device='cuda', data_dir=DATA_DIR, cache_dir=CACHE_DIR):
    """
    The 90 metabolite scans with all coils and transients, processed as the network's validation
    scans are: the input of every in-vivo fit (cows_benchmark.py). Saves fids, stems, subjects,
    regions, cf, bw.
    """
    from augmentrum.dataset.cows import scan_info

    data, water, names, subjects = load_scans(None, data_dir, cache_dir)
    fids = process_scans(data, water, device=device).cpu().numpy()
    info = [scan_info(d) for d in data]
    np.savez(out, fids=fids, stems=np.array(names), subjects=np.array(subjects),
             regions=np.array([i.get('Region', '') for i in info]),
             cf=float(data[0].spectrometer_frequency[0]), bw=float(1.0 / data[0].dwelltime))
    return out


#**************************************************************************************************#
#                                              scoring                                             #
#**************************************************************************************************#
def optimal_scale(y, y_hat):
    """
    Per spectrum, the w >= 0 minimising mean |y - w y_hat| (the OoD paper's optimal reference):
    the median of y_i / y_hat_i weighted by |y_hat_i|. (N, 1); 1 where y_hat is all zero.
    """
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    weight = np.abs(y_hat)
    ratio = np.where(weight > 0, y / np.where(weight > 0, y_hat, 1.0), 0.0)
    order = np.argsort(ratio, axis=1)
    r = np.take_along_axis(ratio, order, axis=1)
    cw = np.cumsum(np.take_along_axis(weight, order, axis=1), axis=1)
    total = cw[:, -1:]
    k = np.argmax(cw >= 0.5 * total, axis=1)
    w = np.where(total[:, 0] > 0, r[np.arange(len(r)), k], 1.0)
    return np.clip(w, 0, None)[:, None]


def lin_ccc(x, y):
    """Lin's concordance correlation coefficient (population moments)."""
    mx, my = x.mean(), y.mean()
    cov = ((x - mx) * (y - my)).mean()
    return float(2 * cov / (x.var() + y.var() + (mx - my) ** 2))


def totals(con, names, table=TOTALS):
    return {k: con[:, [names.index(m) for m in v]].sum(1) for k, v in table.items()
            if all(m in names for m in v)}


def concentration_metrics(pred, true, names, mm_names=MM_NAMES):
    """
    Scores of predicted concentrations: mosae (macromolecules zeroed in both, each spectrum on its
    optimal scale), mosae_<metabolite>, ccc_<metabolite> and ccc_<total> on that scale, ccc_mean
    over the metabolites; and without scaling mae_<x> and nmae_<x> (MAE / mean truth) per
    metabolite and total.
    """
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    out = {}
    for i, nm in enumerate(names):
        err = np.abs(pred[:, i] - true[:, i])
        out[f'mae_{nm}'] = float(err.mean())
        out[f'nmae_{nm}'] = float(err.mean() / max(true[:, i].mean(), 1e-12))
    tp, tt = totals(pred, names), totals(true, names)
    for k in tt:
        err = np.abs(tp[k] - tt[k])
        out[f'mae_{k}'] = float(err.mean())
        out[f'nmae_{k}'] = float(err.mean() / max(tt[k].mean(), 1e-12))
    mm = [i for i, nm in enumerate(names) if nm in mm_names]
    metab = [i for i in range(len(names)) if i not in mm]
    y, y_hat = true.copy(), pred.copy()
    y[:, mm] = 0
    y_hat[:, mm] = 0
    est = optimal_scale(y, y_hat) * y_hat
    scaled = np.abs(y - est)
    out['mosae'] = float(scaled.mean())
    for i in metab:
        out[f'mosae_{names[i]}'] = float(scaled[:, i].mean())
        out[f'ccc_{names[i]}'] = lin_ccc(est[:, i], y[:, i])
    out['ccc_mean'] = float(np.mean([out[f'ccc_{names[i]}'] for i in metab]))
    te, tt = totals(est, names), totals(y, names)
    for k in tt:
        out[f'ccc_{k}'] = lin_ccc(te[k], tt[k])
    return out


#**************************************************************************************************#
#                                             test sets                                            #
#**************************************************************************************************#
def spec_noise_sd(spec, ppm, lim=PPM_NOISE):
    """Noise SD per real component in a signal-free region, a quadratic trend removed first."""
    sel = (ppm >= lim[0]) & (ppm <= lim[1])
    u = ppm[sel] - ppm[sel].mean()
    parts = [part - np.polyval(np.polyfit(u, part, 2), u)
             for part in (spec[sel].real, spec[sel].imag)]
    return float(np.std(np.concatenate(parts), ddof=1))


def scan_snr(fids, ppm):
    """NAA SNR of processed scans: max |spectrum| on 1.9-2.1 ppm / noise SD per component."""
    specs = np.fft.fft(fids, axis=-1)
    naa = (ppm >= PPM_NAA[0]) & (ppm <= PPM_NAA[1])
    return np.array([np.abs(s[naa]).max() / spec_noise_sd(s, ppm) for s in specs])


def _spread(x):
    """Robust SD across rows (per column)."""
    return 1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0)


def pb_theta(fits_dir, model):
    """FSL-MRS PB's in-vivo fits (cows_benchmark.py invivo) in *model*'s layout: (theta, stems)."""
    pb = np.load(os.path.join(fits_dir, 'invivo_fsl_pb.npz'))
    stems = [str(s) for s in pb['stems']]
    if [str(s) for s in pb['names']] != model.names:
        raise ValueError('the PB fits and the basis name their spectra differently')
    L = model.layout
    theta = np.zeros((len(stems), model.n_params))
    theta[:, L['con']] = pb['con']
    theta[:, L['gamma']] = pb['gamma']
    theta[:, L['sigma']] = np.asarray(pb['sigma'], float).reshape(len(stems), -1)[:, :1]
    theta[:, L['eps']] = np.asarray(pb['eps'], float).reshape(len(stems), -1)[:, :1]
    theta[:, L['phi0']] = pb['phi0'][:, None]
    theta[:, L['phi1']] = pb['phi1'][:, None]
    theta[:, L['baseline']] = pb['baseline']
    return theta, stems


#: a test-set row: model PB with a shift per basis spectrum and a baseline curve ("simulate_rows")
ROW_KEYS = ('con', 'gamma', 'sigma', 'eps', 'phi0', 'phi1', 'baseline')
#: Osprey's lineshape kernel, optional in a row: 'kernel' (N, KERNEL_TAPS) taps on Osprey's
#: zero-filled grid (a step of bw / 2T), centred, convolved with the metabolites (not the
#: macromolecules); Osprey's own are 5-15 taps on the COWS scans, zero-padded
KERNEL_TAPS = 15


def simulate_rows(rows, basis, chunk=100):
    """
    Noiseless spectra (N, T) complex, in the data's unshifted FFT order, of test-set rows:

        exp(-i (phi0 + phi1 f)) fft( sum_k c_k b_k(t) exp(-(i eps_k + gamma_k + sigma^2 t) t) )
        + baseline(f),

    model PB with its own shift eps_k per basis spectrum and a baseline curve on the FFT's bins.
    With one eps for all and the poly-2 baseline it is "SignalModel" 'PB'. With a 'kernel' (Osprey's
    rows), the metabolites' spectra are convolved with it: their FIDs times
    h(t) = sum_j kernel_j exp(-2 pi i (c - j) bw / 2T t), c the centre tap.

    Args:
        rows: {con, gamma, eps: (N, n); sigma, phi0, phi1: (N,); baseline: (N, T) complex;
               optional kernel: (N, KERNEL_TAPS)}.
    """
    t = basis.t
    f = np.fft.fftfreq(len(t), d=float(t[1] - t[0]))
    step = basis.bw / (2 * len(t))
    metab = ~np.isin(basis.names, MM_NAMES)
    out = []
    for s in range(0, len(rows['con']), chunk):
        r = {k: np.asarray(rows[k][s:s + chunk]) for k in ROW_KEYS}
        E = np.exp(-(1j * r['eps'][:, None, :] + r['gamma'][:, None, :]
                     + (r['sigma'] ** 2)[:, None, None] * t[None, :, None]) * t[None, :, None])
        if 'kernel' in rows:
            k = np.asarray(rows['kernel'][s:s + chunk])
            j = (k.shape[1] - 1) // 2 - np.arange(k.shape[1])
            h = k @ np.exp(-2j * np.pi * step * np.outer(j, t))
            E[:, :, metab] = E[:, :, metab] * h[:, :, None]
        fid = np.einsum('btn,tn,bn->bt', E, basis.fids, r['con'])
        out.append(np.exp(-1j * (r['phi0'][:, None] + r['phi1'][:, None] * f[None, :]))
                   * np.fft.fft(fid, axis=-1) + r['baseline'])
    return np.concatenate(out)


def testset_rows(path):
    """
    The rows a test set copies (rows of "simulate_rows", in the data's units): cows_benchmark.py
    osprey-rows' (<fits>/rows_osprey.npz: each scan as Osprey fitted it, in Osprey's own model) or
    rows' (<fits>/rows.npz: each tool's concentrations, the rest fitted to the scan).

    Returns:
        (rows {key: (M, ...)}, tool per row, stem per row).
    """
    z = np.load(path)
    rows = {k: z[k] for k in ROW_KEYS + (('kernel',) if 'kernel' in z.files else ())}
    ok = np.all([np.isfinite(v).reshape(len(v), -1).all(1) for v in rows.values()], axis=0)
    return {k: v[ok] for k, v in rows.items()}, z['tool'][ok], [str(s) for s in z['stem'][ok]]


def invivo_ranges(fits_dir, processed, basis_dir=BASIS_DIR, smooth_pts=4.0):
    """
    The in-vivo spread behind every module's range, from FSL-MRS PB's fits of the 90 processed
    scans and the scans themselves, and the ranges it proposes next to the ones in "MODULES".

    Rule: the p5-p95 spread across the scans; symmetric parameters (shift, phases, Lorentzian
    width, which a negative lb_hz narrows) get +- half of it, one-sided ones (Gaussian width,
    macromolecules, baseline, residual water, echoes) 0 to it. Every size is against the module's
    reference, the real maximum of the whole spectrum ("ref"). Narrowing stops at the narrowest
    in-vivo Lorentzian (p5), so no line narrows past zero. Noise, eddy currents and the samplers
    are shown, not re-derived (see the entries).

    Returns:
        {module: {parameter: dict(invivo=percentiles, current=range, proposed=range, note)}}.
    """
    from scipy.ndimage import gaussian_filter1d
    basis = load_basis(basis_dir)
    model = SignalModel(basis, 'PB', dtype=torch.float64)
    theta, stems = pb_theta(fits_dir, model)
    z = np.load(processed)
    zstems = [str(s) for s in z['stems']]
    fids = z['fids'][[zstems.index(s) for s in stems]]
    spec = np.fft.fft(fids, axis=-1)
    ppm, names, w = basis.ppm, model.names, slice(model.first, model.last)
    band = lambda lo, hi: (ppm >= lo) & (ppm <= hi)
    pct = lambda x: {f'p{q}': float(np.percentile(x, q)) for q in (0, 5, 50, 95, 100)}
    spread = lambda x: float(np.percentile(x, 95) - np.percentile(x, 5))

    th = torch.from_numpy(theta)
    p = {k: v.numpy() for k, v in model.split(th).items()}
    full, base = (a.numpy() for a in model(th, baseline_out=True))
    # the fit is in the data's units; its phase removed, the real part is the absorption
    unphase = np.exp(1j * (p['phi0'] + p['phi1'] * model.f.numpy()))
    ref = np.abs((spec * unphase).real).max(1)                   # the modules' reference
    mm_only = model.join({k: torch.from_numpy(v) for k, v in dict(
        p, con=np.where(np.isin(names, MM_NAMES), p['con'], 0.0),
        baseline=np.zeros_like(p['baseline'])).items()}).numpy()
    mm_spec = model(torch.from_numpy(mm_only)).numpy()
    resid = ((spec - full) * unphase).real[:, w]
    smooth = gaussian_filter1d(resid, smooth_pts, axis=1)
    k = np.abs(smooth).argmax(1)
    naa = (spec * unphase).real[:, band(*PPM_NAA)].max(1)
    water = band(4.4, 4.9)
    widx = np.flatnonzero(water)[np.abs(spec[:, water]).argmax(1)]

    lor = p['gamma'][:, names.index('NAA')] / np.pi                    # FWHM, Hz
    gau = 2 * np.sqrt(np.log(2)) * p['sigma'][:, 0] / np.pi
    eps = p['eps'][:, 0] / (2 * np.pi)
    phi0 = np.rad2deg(np.angle(np.exp(1j * p['phi0'][:, 0])))
    phi1 = p['phi1'][:, 0] * BW / np.deg2rad(1.0)                    # Augmentrum's degrees
    mm = np.abs((mm_spec * unphase).real[:, w]).max(1) / ref
    bl = np.abs((base * unphase).real[:, w]).max(1) / ref
    wat = np.abs(spec[np.arange(len(spec)), widx]) / ref
    art = np.abs(smooth[np.arange(len(smooth)), k]) / ref
    art_ppm = ppm[w][k]
    T = fids.shape[1]
    t = np.arange(T) / BW
    kk = int(round(0.01 * BW))
    env = np.apply_along_axis(lambda a: np.convolve(a, np.ones(kk) / kk, mode='same'), 1,
                              np.abs(fids))
    late = (t >= 0.25) & (t <= 0.5)
    floor = np.array([spec_noise_sd(s_, ppm) for s_ in spec]) / np.sqrt(T) * np.sqrt(np.pi / 2)
    echo = (env[:, late].max(1) - floor) / np.abs(fids).max(1)
    snr = scan_snr(fids, ppm)

    sym = lambda x, cap=None: (lambda h: (-round(min(h, cap), 2) if cap else -h, h))(
        round(spread(x) / 2, 2))
    one = lambda x: (0.0, round(spread(x), 3 if spread(x) < 0.1 else 2))
    low = art_ppm[art_ppm < 2.5]
    m = MODULES
    out = {
        'line_broadening': {
            'lb_hz': dict(invivo=pct(lor), current=m['line_broadening']['line_broadening']['lb_hz'],
                          proposed=sym(lor, cap=float(np.percentile(lor, 5))),
                          note='NAA Lorentzian FWHM; negative narrows, down to the p5 width'),
            'gb_hz': dict(invivo=pct(gau), current=m['line_broadening']['line_broadening']['gb_hz'],
                          proposed=one(gau), note='Gaussian FWHM (one-sided: cannot narrow)')},
        'frequency_shift': {'shift_hz': dict(
            invivo=pct(eps), current=m['frequency_shift']['frequency_shift']['shift_hz'],
            proposed=sym(eps), note='fitted shift, Hz')},
        'phase_shift': {
            'zero_order_deg': dict(invivo=pct(phi0),
                                   current=m['phase_shift']['phase_shift']['zero_order_deg'],
                                   proposed=sym(phi0), note='phi0, deg'),
            'first_order_deg': dict(invivo=pct(phi1),
                                    current=m['phase_shift']['phase_shift']['first_order_deg'],
                                    proposed=sym(phi1), note='phi1 in Augmentrum degrees')},
        'macromolecules': {'mm_scale': dict(
            invivo=pct(mm), current=m['macromolecules']['macromolecules']['mm_scale'],
            proposed=one(mm), note='fitted MM + MM09 peak / ref')},
        'baseline': {'baseline_frac': dict(
            invivo=pct(bl), current=m['baseline']['baseline']['baseline_frac'], proposed=one(bl),
            note='fitted poly-2 baseline peak / ref')},
        'residual_water': {'amplitude_scale': dict(
            invivo=pct(wat), current=m['residual_water']['residual_water']['amplitude_scale'],
            proposed=one(wat), note='|spectrum| on 4.4-4.9 ppm / ref')},
        'artificial_peaks': {
            'amp': dict(invivo=pct(art),
                        current=m['artificial_peaks']['artificial_peaks']['peaks'][0]['amp'],
                        proposed=(0.0, round(float(np.percentile(art, 95)), 3)),
                        note=f'largest smoothed PB residual / ref (Gaussian SD {smooth_pts} points)'),
            'ppm': dict(invivo=pct(art_ppm),
                        current=m['artificial_peaks']['artificial_peaks']['peaks'][0]['ppm'],
                        proposed=(round(float(np.percentile(low, 5)), 2),
                                  round(float(np.percentile(low, 95)), 2)),
                        note=f'where it sits, below 2.5 ppm ({low.size} scans)')},
        'spurious_echoes': {'amp': dict(
            invivo=pct(echo),
            current=m['spurious_echoes']['spurious_echoes']['echoes'][0]['amp'],
            proposed=(0.0, round(float(np.ceil(100 * echo.max()) / 100), 2)),
            note='late-FID (0.25-0.5 s) excess over the noise / max |FID|: an upper bound')},
        'noise': {'snr': dict(
            invivo=pct(snr), current=m['noise']['noise']['snr'], proposed=m['noise']['noise']['snr'],
            note='NAA SNR of the scans; the module draws max|spectrum| / added SD (kept)')},
        'eddy_current': {'strength': dict(
            invivo={}, current=m['eddy_current']['eddy_current']['strength'],
            proposed=m['eddy_current']['eddy_current']['strength'],
            note='from the uncorrected waters (2026-09-18), not re-derived here')},
    }
    return out


def generate_testset(rows_path, processed, out_dir, seed=0, n=1000, jitter=0.1, snr_jitter=0.05,
                     tcr_ref=8.0, basis_dir=BASIS_DIR, subjects=None):
    """
    Simulate a test set: *n* rows drawn with replacement from "testset_rows", jittered (log-normal
    *jitter* on concentrations, widths and the baseline curve's size; one shift for all basis
    spectra and the phases + N(0, jitter x their robust spread across rows), phi0 wrapped),
    simulated by "simulate_rows", concentrations and baseline scaled so the median tCr of the rows
    is *tcr_ref*, and noised to the SNR of a random processed scan (x log-normal *snr_jitter*):
    complex white noise of SD max |clean| on 1.9-2.1 ppm / SNR per component. Seed 0 is the test
    set, seed 1 the selection set. *subjects*: only the rows (and the SNR pool) of these subjects'
    scans, so that the test and the selection set share no scan ("TESTSET_SUBJECTS").

    Returns:
        The .npz path (spectra (N, 2, T) float32, the rows' parameters ("ROW_KEYS"), snr,
        noise_sd, names, tool, stem, config, hash).
    """
    basis = load_basis(basis_dir)
    names = basis.names
    rows, tools, stems = testset_rows(rows_path)
    if subjects is not None:
        keep = np.array([st.split('_')[0] in subjects for st in stems])
        rows, tools = {k: v[keep] for k, v in rows.items()}, tools[keep]
        stems = [st for st, k in zip(stems, keep) if k]
    rng = np.random.default_rng([seed, 1729])
    idx = rng.integers(0, len(tools), n)
    p = {k: np.array(v[idx]) for k, v in rows.items()}
    spread = dict(eps=float(_spread(rows['eps'].mean(1))),
                  phi0=float(_spread(np.angle(np.exp(1j * rows['phi0'])))),
                  phi1=float(_spread(rows['phi1'])))
    for k in ('con', 'gamma', 'sigma'):
        p[k] = p[k] * np.exp(jitter * rng.standard_normal(p[k].shape))
    p['baseline'] = p['baseline'] * np.exp(jitter * rng.standard_normal(n))[:, None]
    p['eps'] = p['eps'] + jitter * spread['eps'] * rng.standard_normal(n)[:, None]
    p['phi0'] = np.angle(np.exp(1j * p['phi0']))
    for k in ('phi0', 'phi1'):
        p[k] = p[k] + jitter * spread[k] * rng.standard_normal(n)
    ix = [names.index(m) for m in TOTALS['tCr']]
    scale = tcr_ref / float(np.median(rows['con'][:, ix].sum(1)))
    p['con'] = p['con'] * scale
    p['baseline'] = p['baseline'] * scale

    z = np.load(processed)
    pool = (np.ones(len(z['fids']), bool) if subjects is None
            else np.isin([str(x) for x in z['subjects']], subjects))
    snr_pool = scan_snr(z['fids'][pool], basis.ppm)
    snr = rng.choice(snr_pool, n) * np.exp(snr_jitter * rng.standard_normal(n))
    clean = simulate_rows(p, basis)
    naa = (basis.ppm >= PPM_NAA[0]) & (basis.ppm <= PPM_NAA[1])
    sd = np.abs(clean[:, naa]).max(axis=1) / snr
    noisy = clean + sd[:, None] * (rng.standard_normal(clean.shape)
                                   + 1j * rng.standard_normal(clean.shape))
    spectra = np.stack((noisy.real, noisy.imag), axis=1).astype(np.float32)

    first, last = basis.window()
    config = dict(seed=seed, n=n, subjects=subjects, jitter=jitter, snr_jitter=snr_jitter,
                  tcr_ref=tcr_ref,
                  unit_scale=scale, rows=rows_path, processed=processed,
                  rows_per_tool={t: int((tools == t).sum()) for t in ('fsl_pb', 'lcmodel', 'osprey')},
                  drawn={t: int((tools[idx] == t).sum()) for t in ('fsl_pb', 'lcmodel', 'osprey')},
                  snr_pool=dict(n=int(snr_pool.size), median=float(np.median(snr_pool))),
                  window=[first, last], names=names)
    text = json.dumps(config, sort_keys=True, default=float)
    digest = hashlib.sha256(spectra.tobytes() + b''.join(p[k].tobytes() for k in sorted(p))
                            + text.encode()).hexdigest()[:16]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'test_n{n}_s{seed}.npz')
    np.savez(path, spectra=spectra, **p, snr=snr, noise_sd=sd, names=np.array(names),
             config=text, hash=digest, tool=tools[idx], stem=np.array(stems)[idx])
    return path


class TestSet:
    """A saved test set: spectra (N, 2, T), ground truth, and the network's scores on it."""

    def __init__(self, path):
        z = np.load(path)
        self.path, self.name = path, os.path.basename(path)[:-4]
        self.spectra = torch.from_numpy(z['spectra'])
        self.snr = z['snr']
        self.names = [str(s) for s in z['names']]
        self.config = json.loads(str(z['config']))
        self.hash = str(z['hash'])
        if 'con' in z.files:        # the rows of "simulate_rows"
            self.params = {k: z[k] for k in ROW_KEYS}
        else:                       # the first test sets: model PB's theta
            layout = json.loads(str(z['layout']))
            self.params = {k: z['theta'][:, slice(*v)] for k, v in layout.items()}
        self.concentrations = self.params['con']
        self.tool = z['tool'] if 'tool' in z.files else None
        self._on = {}

    @torch.no_grad()
    def predict(self, net, device, batch=500):
        """The network's parameters in the set's units (N, P) float64."""
        was_training = net.training
        net.eval()
        if str(device) not in self._on:
            self._on = {str(device): self.spectra.to(device)}
        x = self._on[str(device)]
        out = torch.cat([net.predict(x[i:i + batch]) for i in range(0, len(x), batch)])
        net.train(was_training)
        return out.double().cpu().numpy()

    def evaluate(self, net, device):
        pred = self.predict(net, device)[:, :len(self.names)]
        return concentration_metrics(pred, self.concentrations, self.names)


#**************************************************************************************************#
#                                              training                                            #
#**************************************************************************************************#
# test metrics logged at every evaluation
CURVE_METRICS = (['mosae', 'ccc_mean'] + [f'nmae_{k}' for k in TOTALS]
                 + [f'ccc_{k}' for k in TOTALS])


def run_id(cfg):
    reg = ((f"-do{cfg['dropout']:g}" if cfg['dropout'] else '')
           + (f"-wd{cfg['weight_decay']:g}" if cfg['weight_decay'] else ''))
    return (f"{cfg['condition']}{reg}__n{cfg['n_subjects']}__f{cfg['fold']}__{cfg['variant']}"
            f"__s{cfg['seed']}")


def atomic_save(obj, path):
    torch.save(obj, path + '.tmp')
    os.replace(path + '.tmp', path)


def write_json(obj, path):
    with open(path + '.tmp', 'w') as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(path + '.tmp', path)


def write_curve(history, path):
    keys = []
    for row in history:
        keys += [k for k in row if k not in keys]
    with open(path + '.tmp', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    os.replace(path + '.tmp', path)


def append_csv(row, path):
    """Append a row under an exclusive lock (parallel runs share the table)."""
    with open(path, 'a+', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        header = next(csv.reader(f), None)
        writer = csv.DictWriter(f, fieldnames=header or list(row), extrasaction='ignore')
        if header is None:
            writer.writeheader()
        writer.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)


def eval_steps(cfg):
    """Every eval_every steps, the last one, and 1, 2, 5, 10, 20, ... below eval_every."""
    extra = {m * 10 ** e for e in range(9) for m in (1, 2, 5) if m * 10 ** e < cfg['eval_every']}
    return lambda s: s % cfg['eval_every'] == 0 or s == cfg['max_steps'] or s in extra


class StepClock:
    """
    Data and model time per step from CUDA events, read at the evaluations only, so the host never
    waits for the device inside the loop (on the CPU: the wall clock). A step's two halves are
    timed end to end, so they add up to the step.
    """

    def __init__(self, device):
        self.cuda = device.type == 'cuda'
        self.marks, self.used = [], 0

    def _record(self, index):
        if not self.cuda:
            self.marks.append(time.perf_counter())
            return
        while len(self.marks) <= index:
            self.marks.append(torch.cuda.Event(enable_timing=True))
        self.marks[index].record()

    def start(self):
        self._record(3 * self.used)

    def data(self):
        self._record(3 * self.used + 1)

    def model(self):
        self._record(3 * self.used + 2)
        self.used += 1

    def drain(self):
        """(data seconds, model seconds, steps) since the last drain."""
        steps, self.used = self.used, 0
        if not steps:
            return 0.0, 0.0, 0
        m = self.marks
        if not self.cuda:
            self.marks = []
            return (sum(m[3 * i + 1] - m[3 * i] for i in range(steps)),
                    sum(m[3 * i + 2] - m[3 * i + 1] for i in range(steps)), steps)
        m[3 * steps - 1].synchronize()
        return (sum(m[3 * i].elapsed_time(m[3 * i + 1]) for i in range(steps)) / 1e3,
                sum(m[3 * i + 1].elapsed_time(m[3 * i + 2]) for i in range(steps)) / 1e3, steps)


class GraphedStep:
    """
    The training step (network, signal model, loss, backward, Adam) captured once as a CUDA graph
    and replayed: one launch instead of a few hundred small kernels, each of which waits for its
    turn while other jobs share the GPU. The first *warmup* steps run eagerly on a side stream
    (real training steps); the training loss is summed on the device and read at evaluations.
    """

    def __init__(self, net, model, opt, warmup=3):
        self.net, self.model, self.opt, self.warmup = net, model, opt, warmup
        self.graph, self.eager, self.count = None, 0, 0
        self.stream = torch.cuda.Stream()
        self.loss_sum = None

    def _step(self, x, y):
        theta_n, norm = self.net(x)
        loss = spectral_loss(self.model, theta_n, y, norm).mean()
        loss.backward()
        self.opt.step()
        self.loss_sum += loss.detach()

    def __call__(self, x, y):
        self.count += 1
        if self.graph is not None:
            self.x.copy_(x)
            self.y.copy_(y)
            self.graph.replay()
            return
        if self.loss_sum is None:
            self.loss_sum = torch.zeros((), device=x.device)
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.opt.zero_grad(set_to_none=True)
            self._step(x, y)
        torch.cuda.current_stream().wait_stream(self.stream)
        self.eager += 1
        if self.eager == self.warmup:
            self.x, self.y = x.clone(), y.clone()
            self.opt.zero_grad(set_to_none=True)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._step(self.x, self.y)

    def mean_loss(self):
        value = float(self.loss_sum) / max(self.count, 1)
        self.loss_sum.zero_()
        self.count = 0
        return value


@torch.no_grad()
def validate(net, model, x):
    """Mean per-spectrum loss on in-vivo spectra (target = input), eval mode."""
    net.eval()
    theta_n, norm = net(x)
    loss = float(spectral_loss(model, theta_n, x, norm).mean())
    net.train()
    return loss


@torch.no_grad()
def save_predictions(net, model, sets, path):
    """Parameters (units of the input) and per-spectrum loss for named sets of spectra."""
    out = {}
    net.eval()
    for key, (x, names) in sets.items():
        theta_n, norm = net(x)
        out[f'{key}_theta'] = net.rescale(theta_n, norm).double().cpu().numpy()
        out[f'{key}_loss'] = spectral_loss(model, theta_n, x, norm).double().cpu().numpy()
        out[f'{key}_names'] = np.array(names)
    net.train()
    np.savez(path, **out)


def build(cfg, device):
    """The signal model and the network of a run configuration."""
    model = SignalModel(load_basis(cfg['basis_dir']), cfg['variant']).to(device)
    net = QuantNet(model, width=cfg['width'], depth=cfg['depth'], activation=cfg['activation'],
                   dropout=cfg['dropout']).to(device)
    return model, net


def train(cfg):
    """
    One run (condition, n_subjects, fold, variant, seed): trains on the fly for max_steps with a
    CUDA-graphed Adam step, evaluates every eval_every steps (training and validation loss, the
    test set's scores, the selection set's MOSAE), keeps the checkpoint with the lowest selection
    MOSAE ('selected') and the last one ('final'), and tests both at the end. Resumable from
    last.pt; a finished run is skipped unless --force (or --extend with a larger budget). SIGTERM
    or SIGINT ends the run at its next evaluation, finished as if its budget ended there.
    """
    if cfg.get('augment'):
        cfg['condition'] = register_augment(cfg['augment'])
    rid = run_id(cfg)
    run_dir = os.path.join(cfg['out'], 'runs', rid)
    result_path = os.path.join(run_dir, 'result.json')
    paths = {k: os.path.join(run_dir, f'{k}.pt') for k in ('last', 'final', 'selected')}
    if os.path.isfile(result_path) and not cfg['force']:
        with open(result_path) as f:
            done = json.load(f)
        if not (cfg['extend'] and done['steps'] < cfg['max_steps']
                and os.path.isfile(paths['last'])):
            print(f'[skip] {rid} is finished', flush=True)
            return done
    os.makedirs(run_dir, exist_ok=True)
    t_start = time.time()
    device = torch.device(cfg['device'])
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    model, net = build(cfg, device)
    graphed = device.type == 'cuda' and cfg['graph']
    params = [p for p in net.parameters() if p.requires_grad]
    opt = (torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'],
                             capturable=graphed) if cfg['weight_decay'] > 0
           else torch.optim.Adam(params, lr=cfg['lr'], capturable=graphed))
    test, selection = TestSet(cfg['testset']), TestSet(cfg['selection_set'])
    if test.names != model.names or selection.names != model.names:
        raise ValueError('the test sets and the basis name their spectra differently')

    t0 = time.time()
    stream = COWSStream(cfg['condition'], cfg['fold'], cfg['n_subjects'], seed=cfg['seed'],
                        batch_size=cfg['batch'], device=device, precision=cfg['precision'],
                        data_dir=cfg['data_dir'], cache_dir=cfg['cache_dir'])
    x_val, val_names = stream.validation()
    x_fit, fit_names = stream.training_scans()
    t_setup = time.time() - t0

    state = dict(step=0, best_sel=math.inf, sel_step=-1, history=[], t_data=0.0, t_model=0.0,
                 t_eval=0.0, wall=0.0, resumes=0)
    if os.path.isfile(paths['last']) and not cfg['force']:
        ckpt = torch.load(paths['last'], map_location=device, weights_only=False)
        net.load_state_dict(ckpt['net'])
        opt.load_state_dict(ckpt['opt'])
        state = ckpt['state']
        state['resumes'] += 1
        torch.set_rng_state(ckpt['torch_rng'].cpu())
        stream.reseed(cfg['seed'] * 1_000_003 + state['step'])  # the stream cannot be rewound
        print(f'[resume] {rid} at step {state["step"]}', flush=True)

    wandb_run = None
    if cfg['wandb']:
        import wandb
        group = os.path.basename(os.path.normpath(cfg['out']))
        wandb_run = wandb.init(entity=cfg['wandb_entity'], project=cfg['wandb_project'],
                               group=group, name=rid, config=cfg, dir=run_dir, resume='allow',
                               id=re.sub(r'[^A-Za-z0-9_-]', '-', f'{group}-{rid}'),
                               tags=[cfg['condition'], f"n{cfg['n_subjects']}", f"f{cfg['fold']}"])

    # SIGTERM / SIGINT: stop at the next evaluation and finish the run as if its budget ended there
    stop = {'now': False}

    def request_stop(signum, frame):
        stop['now'] = True
        print(f'[stop] {rid}: signal {signum}, finishing at the next evaluation', flush=True)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    if cfg['checkpoint_every']:
        os.makedirs(ckpt_dir, exist_ok=True)

    clock = StepClock(device)
    is_eval = eval_steps(cfg)
    net.train()
    step_graph = GraphedStep(net, model, opt) if graphed else None
    running, mark = [], time.time()
    while state['step'] < cfg['max_steps']:
        clock.start()
        x, y = stream.next_batch()
        clock.data()
        if step_graph is not None:
            step_graph(x, y)
        else:
            theta_n, norm = net(x)
            loss = spectral_loss(model, theta_n, y, norm).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running.append(loss.detach())
        clock.model()
        state['step'] += 1
        step = state['step']
        if not is_eval(step):
            continue

        t0 = time.time()
        t_data, t_model, n_int = clock.drain()
        train_loss = (step_graph.mean_loss() if step_graph is not None
                      else float(torch.stack(running).mean()))
        if not math.isfinite(train_loss):
            raise FloatingPointError(f'training loss is {train_loss} at step {step}')
        val_loss = validate(net, model, x_val)
        fit_loss = validate(net, model, x_fit)
        metrics = test.evaluate(net, device)
        sel = selection.evaluate(net, device)['mosae']
        if sel < state['best_sel']:
            state['best_sel'], state['sel_step'] = sel, step
            atomic_save(net.state_dict(), paths['selected'])
        state['t_data'] += t_data
        state['t_model'] += t_model
        state['wall'] += t0 - mark
        state['history'].append(dict(
            step=step, train_wall_s=round(state['wall'], 2), train_loss=train_loss,
            val_loss=val_loss, train_scans_loss=fit_loss, sel_mosae=sel,
            data_ms=1e3 * t_data / max(n_int, 1), model_ms=1e3 * t_model / max(n_int, 1),
            **{f'test_{k}': metrics[k] for k in CURVE_METRICS if k in metrics}))
        running = []
        state['t_eval'] += time.time() - t0
        if step % cfg['eval_every'] == 0 or step == cfg['max_steps']:
            atomic_save(dict(net=net.state_dict(), opt=opt.state_dict(), state=state,
                             torch_rng=torch.get_rng_state()), paths['last'])
            write_curve(state['history'], os.path.join(run_dir, 'curve.csv'))
            print(f'{rid} step {step:7d}  train {train_loss:.3e}  val {val_loss:.3e}  sel {sel:.3f} '
                  f'best {state["best_sel"]:.3f}@{state["sel_step"]}  test MOSAE '
                  f'{metrics["mosae"]:.3f}  data {state["history"][-1]["data_ms"]:.1f} model '
                  f'{state["history"][-1]["model_ms"]:.1f} ms/step', flush=True)
        if wandb_run is not None:
            wandb_run.log({k: v for k, v in state['history'][-1].items() if k != 'step'},
                          step=step)
        if cfg['checkpoint_every'] and step % cfg['checkpoint_every'] == 0:
            atomic_save(net.state_dict(), os.path.join(ckpt_dir, f'step_{step:09d}.pt'))
        if stop['now']:
            atomic_save(dict(net=net.state_dict(), opt=opt.state_dict(), state=state,
                             torch_rng=torch.get_rng_state()), paths['last'])
            write_curve(state['history'], os.path.join(run_dir, 'curve.csv'))
            break
        mark = time.time()

    atomic_save(net.state_dict(), paths['final'])
    t0 = time.time()
    scores = dict(path=test.path, hash=test.hash, n=len(test.spectra))
    for which in ('final', 'selected'):
        if not os.path.isfile(paths[which]):
            continue
        net.load_state_dict(torch.load(paths[which], map_location=device, weights_only=True))
        scores[which] = test.evaluate(net, device)
        save_predictions(net, model, dict(val=(x_val, val_names), train=(x_fit, fit_names)),
                         os.path.join(run_dir, f'predictions_{which}.npz'))
    history = state['history']
    write_curve(history, os.path.join(run_dir, 'curve.csv'))
    steps = max(state['step'], 1)
    result = dict(
        run_id=rid, config={k: v for k, v in cfg.items() if k != 'force'},
        data=stream.describe(), model=dict(variant=cfg['variant'], n_params=model.n_params,
                                           layout=model.describe(), names=model.names,
                                           window=[model.first, model.last]),
        n_weights=sum(p.numel() for p in net.parameters()), steps=state['step'],
        selected_step=state['sel_step'], selected_mosae=state['best_sel'],
        selection_set=dict(path=selection.path, hash=selection.hash),
        last_val_loss=history[-1]['val_loss'] if history else None,
        last_train_loss=history[-1]['train_loss'] if history else None,
        history=history, resumes=state['resumes'],
        timing=dict(total_s=time.time() - t_start, train_wall_s=state['wall'], setup_s=t_setup,
                    data_ms_per_step=1e3 * state['t_data'] / steps,
                    model_ms_per_step=1e3 * state['t_model'] / steps, eval_s=state['t_eval'],
                    test_s=time.time() - t0),
        test=scores, device=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    write_json(result, result_path)
    row = dict(run_id=rid, **{k: v for k, v in cfg.items() if not isinstance(v, (dict, list))},
               steps=state['step'], selected_step=state['sel_step'],
               **{f'{w}_{k}': scores[w][k] for w in ('final', 'selected') if w in scores
                  for k in CURVE_METRICS if k in scores[w]})
    append_csv(row, os.path.join(cfg['out'], 'results.csv'))
    if os.path.isfile(paths['last']) and not cfg['keep_last']:
        os.remove(paths['last'])
    if wandb_run is not None:
        wandb_run.summary.update({f'test_{w}/{k}': v for w in ('final', 'selected')
                                  if w in scores for k, v in scores[w].items()})
        wandb_run.finish()
    return result


def load_run(run_dir, weights='selected', device='cpu'):
    """A finished run's network with its *weights* ('selected' or 'final'), and its result."""
    with open(os.path.join(run_dir, 'result.json')) as f:
        result = json.load(f)
    model, net = build(result['config'], torch.device(device))
    net.load_state_dict(torch.load(os.path.join(run_dir, f'{weights}.pt'), map_location=device,
                                   weights_only=True))
    net.eval()
    return result, model, net


def run_config(run_dir):
    """A run's configuration: its result.json's, or for a run still going, the defaults with the
    variant of its id (the open-budget runs set nothing else that the network depends on)."""
    path = os.path.join(run_dir, 'result.json')
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)['config']
    variant = os.path.basename(os.path.normpath(run_dir)).split('__')[3]
    return dict(TRAIN, variant=variant, basis_dir=BASIS_DIR)


@torch.no_grad()
def rescore(run_dir, testset, selection_set, device):
    """
    Every checkpoint of a run (checkpoints/step_*.pt) on a test set and a selection set:
    rescore_<set folder>_<test set>.csv ("rescore_name") in the run folder (step, test MOSAE /
    mean CCC / totals, selection MOSAE). Returns the rows.
    """
    ts, sel = TestSet(testset), TestSet(selection_set)
    _, net = build(run_config(run_dir), torch.device(device))
    rows = []
    ckpt = os.path.join(run_dir, 'checkpoints')
    for fn in sorted(os.listdir(ckpt)):
        if not re.fullmatch(r'step_\d+\.pt', fn):
            continue
        net.load_state_dict(torch.load(os.path.join(ckpt, fn), map_location=device,
                                       weights_only=True))
        m = ts.evaluate(net, device)
        rows.append(dict(step=int(fn[5:-3]), **{f'test_{k}': m[k] for k in CURVE_METRICS},
                         sel_mosae=sel.evaluate(net, device)['mosae']))
    write_curve(rows, os.path.join(run_dir, rescore_name(testset)))
    return rows


def rescore_name(testset):
    """rescore's file in a run folder for a test set: its folder in the name, as every set's file
    is test_n<n>_s<seed>.npz."""
    folder = os.path.basename(os.path.dirname(os.path.abspath(testset)))
    return f'rescore_{folder}_{os.path.basename(testset)[:-4]}.csv'


#**********#
#   main   #
#**********#
def add_paths(p):
    p.add_argument('--data-dir', default=DATA_DIR, help='OpenNeuro ds006812 root')
    p.add_argument('--basis-dir', default=BASIS_DIR, help='FSL-MRS JSON basis (summed Cr / PCr)')
    p.add_argument('--cache-dir', default=CACHE_DIR, help="Augmentrum's NIfTI cache of the scans")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    p = sub.add_parser('process', help='the 90 processed scans, the benchmark\'s input')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default=device)
    add_paths(p)

    p = sub.add_parser('ranges', help="every module's in-vivo spread and proposed range")
    p.add_argument('--fits', required=True, help="cows_benchmark.py invivo's output folder")
    p.add_argument('--processed', required=True, help="'process' output")
    p.add_argument('--out', required=True, help='JSON of the table')
    add_paths(p)

    p = sub.add_parser('testset', help='simulate a test set from the benchmark\'s in-vivo fits')
    p.add_argument('--rows', required=True, help="cows_benchmark.py osprey-rows' (or rows') .npz")
    p.add_argument('--processed', required=True, help="'process' output (the SNR pool)")
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=0, help='0: the test set, 1: the selection set')
    p.add_argument('--n', type=int, default=1000)
    add_paths(p)

    T = TRAIN
    p = sub.add_parser('train', help='one run')
    p.add_argument('--condition', choices=list(CONDITIONS),
                   help='a named condition, or --augment')
    p.add_argument('--augment', help="a condition as JSON ('register_augment'): its name, samplers "
                                     'and module entries')
    p.add_argument('--n-subjects', type=int, required=True, choices=range(1, 9))
    p.add_argument('--fold', type=int, required=True, choices=list(FOLDS))
    p.add_argument('--variant', default=T['variant'], choices=SignalModel.VARIANTS)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--testset', required=True)
    p.add_argument('--selection-set', required=True)
    p.add_argument('--out', required=True, help='experiment folder (runs/, results.csv)')
    for key in ('activation',):
        p.add_argument(f'--{key}', default=T[key])
    for key in ('dropout', 'lr', 'weight_decay'):
        p.add_argument(f'--{key.replace("_", "-")}', type=float, default=T[key])
    for key in ('width', 'depth', 'batch', 'max_steps', 'eval_every'):
        p.add_argument(f'--{key.replace("_", "-")}', type=int, default=T[key])
    p.add_argument('--precision', default=None, choices=('single', 'double'),
                   help="Augmentrum's working precision (default: the data's)")
    p.add_argument('--device', default=device)
    p.add_argument('--no-graph', dest='graph', action='store_false',
                   help='run the training step eagerly instead of as a CUDA graph')
    p.add_argument('--keep-last', action='store_true', help='keep last.pt (for --extend)')
    p.add_argument('--checkpoint-every', type=int, default=0,
                   help='also keep the weights every that many steps (checkpoints/), so any test or '
                        'selection set can score them later; 0: off')
    p.add_argument('--extend', action='store_true', help='continue to a larger --max-steps')
    p.add_argument('--force', action='store_true', help='rerun a finished run from scratch')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb-project', default='COWS-Ablation')
    p.add_argument('--wandb-entity', default='augmentrum')
    add_paths(p)

    p = sub.add_parser('score', help="a run's checkpoints on a test set")
    p.add_argument('run_dir')
    p.add_argument('--testset', required=True)
    p.add_argument('--weights', nargs='+', default=['selected', 'final'])
    p.add_argument('--device', default=device)

    p = sub.add_parser('rescore', help="every checkpoint of runs on a test and a selection set")
    p.add_argument('run_dirs', nargs='+')
    p.add_argument('--testset', required=True)
    p.add_argument('--selection-set', required=True)
    p.add_argument('--device', default=device)

    args = ap.parse_args(argv)
    if args.cmd == 'rescore':
        for run_dir in args.run_dirs:
            rows = rescore(run_dir, args.testset, args.selection_set, args.device)
            best = min(rows, key=lambda r: r['sel_mosae'])
            print(f"{os.path.basename(os.path.normpath(run_dir))}: {len(rows)} checkpoints; "
                  f"selected @ {best['step']}: test MOSAE {best['test_mosae']:.3f}; last @ "
                  f"{rows[-1]['step']}: {rows[-1]['test_mosae']:.3f}", flush=True)
    elif args.cmd == 'process':
        print(process_invivo(args.out, args.device, args.data_dir, args.cache_dir))
    elif args.cmd == 'ranges':
        table = invivo_ranges(args.fits, args.processed, args.basis_dir)
        write_json(table, args.out)
        print(f"{'module':17s} {'parameter':16s} {'in vivo p5 / p50 / p95':>28s}   {'current':>16s}"
              f"   {'proposed':>16s}")
        for mod, params in table.items():
            for par, e in params.items():
                iv = e['invivo']
                ivs = (f"{iv['p5']:8.3g} {iv['p50']:8.3g} {iv['p95']:8.3g}" if iv else '')
                fmt = lambda r: f'({r[0]:g}, {r[1]:g})'
                print(f'{mod:17s} {par:16s} {ivs:>28s}   {fmt(e["current"]):>16s}   '
                      f'{fmt(e["proposed"]):>16s}   {e["note"]}')
    elif args.cmd == 'testset':
        path = generate_testset(args.rows, args.processed, args.out, args.seed, args.n,
                                basis_dir=args.basis_dir,
                                subjects=list(TESTSET_SUBJECTS[args.seed]))
        ts = TestSet(path)
        print(f"{path}  hash {ts.hash}  rows {ts.config['rows_per_tool']}  SNR median "
              f"{np.median(ts.snr):.0f}")
    elif args.cmd == 'train':
        warnings.filterwarnings('ignore', message='.*global parameter reaches more than one step.*')
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 4)))
        cfg = vars(args)
        if cfg['augment']:
            with open(cfg['augment']) as f:
                cfg['augment'] = json.load(f)             # kept in the run's config
        elif not cfg['condition']:
            ap.error('train needs --condition or --augment')
        result = train(cfg)
        sel = result['test'].get('selected', {})
        print(f"done {result['run_id']}: test MOSAE final {result['test']['final']['mosae']:.3f}"
              + (f", selected {sel['mosae']:.3f} @ {result['selected_step']}" if sel else '')
              + f" ({result['timing']['total_s']:.0f} s)", flush=True)
    else:
        ts = TestSet(args.testset)
        out = {}
        for w in args.weights:
            _, _, net = load_run(args.run_dir, w, args.device)
            pred = ts.predict(net, torch.device(args.device))[:, :len(ts.names)]
            np.savez(os.path.join(args.run_dir, f'testpred_{ts.name}_{w}.npz'), con=pred,
                     names=np.array(ts.names), testset_hash=ts.hash)
            out[w] = concentration_metrics(pred, ts.concentrations, ts.names)
            print(f"{w}: MOSAE {out[w]['mosae']:.4f}  mean CCC {out[w]['ccc_mean']:.3f}")
        write_json(dict(testset=ts.path, hash=ts.hash, scores=out),
                   os.path.join(args.run_dir, f'scores_{ts.name}.json'))


if __name__ == '__main__':
    sys.exit(main())
