####################################################################################################
#                                          svs_ablation.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-23                                                                              #
#                                                                                                  #
# Purpose: The single-voxel augmentation ablation on the COWS data: does Augmentrum help a         #
#          self-supervised quantification network when in-vivo data are scarce? Trains the network #
#          on the fly on the raw transients of 1-8 subjects (OpenNeuro ds006812), with and without #
#          Augmentrum's modules, scores it on simulated spectra with known concentrations against  #
#          FSL-MRS, LCModel and Osprey, screens every augmentation and runs the cross-validated    #
#          grid on every visible GPU.                                                              #
#                                                                                                  #
# The grid on another machine (e.g. Kay's), from the Augmentrum root:                              #
#   1. env: Python 3.11, pip install -e .[torch] fsl-mrs==2.5.0 pymapvbvd==0.6.1 spec2nii==0.8.15  #
#      wandb; A100: torch 2.6 (cu124); Blackwell: torch >= 2.7 built for CUDA >= 12.8              #
#   2. tar xzf cows_grid_bundle.tar.gz (the basis, the test and selection sets, and the header-    #
#      repaired sub-01 acq-06 scan: OpenNeuro's copy has a broken multi-RAID header)               #
#   3. python scripts/svs_ablation.py grid (log: results/cows/grid/grid.log)                       #
#   4. send back results/cows/grid without the checkpoints: tar czf cows_grid_runs.tar.gz          #
#      --exclude=last.pt --exclude=checkpoints results/cows/grid                                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import sys

#: the fitting tools' commands: they fit in parallel processes, one BLAS thread each
FIT_COMMANDS = ('invivo', 'rows', 'osprey-rows', 'bench', 'challenge', 'check')
FITTING = (os.path.basename(sys.argv[0]) == 'svs_ablation.py' and len(sys.argv) > 1
           and sys.argv[1] in FIT_COMMANDS)
# uncapped BLAS threads stall numpy on a shared machine
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1' if FITTING else '4')
os.environ.setdefault('MRSUITE_OCTAVE_RUNTIME', 'docker')
# MRSuite's Octave container mounts the oct2py of the Python that started it: one per Python
os.environ.setdefault('MRSUITE_OCTAVE_CONTAINER',
                      f'mrsuite-octave-py{sys.version_info.major}{sys.version_info.minor}')

import argparse
import copy
import csv
import fcntl
import glob
import hashlib
import json
import math
import multiprocessing as mp
import re
import signal
import subprocess
import time
import urllib.request
import warnings
import xml.etree.ElementTree as ET
import zipfile
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize


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
    A condition given as data (the screen): {'name', 'samplers': [...], 'modules': [...]},
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


#**************************************************************************************************#
#                                          Class BasisSet                                          #
#**************************************************************************************************#
#                                                                                                  #
# The basis as the network uses it: fids (T, n) complex, conjugated so that fft of basis and data  #
# put NAA at the same bin; t (T,) starting at one dwell time; ppm (T,) of each unshifted FFT bin.  #
#                                                                                                  #
#**************************************************************************************************#
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
#                                         Class SignalModel                                        #
#**************************************************************************************************#
#                                                                                                  #
# The model spectrum fft( sum_k c_k b_k(t) exp(-(i eps + gamma_k + sigma^2 t) t) ) exp(-i (phi0 +  #
# phi1 f)) + B beta, f the true frequency of each unshifted FFT bin.                               #
#                                                                                                  #
#**************************************************************************************************#
class SignalModel(nn.Module):
    """
    The model spectrum fft( sum_k c_k b_k(t) exp(-(i eps + gamma_k + sigma^2 t) t) )
    exp(-i (phi0 + phi1 f)) + B beta, f the true frequency of each unshifted FFT bin. 'A':
    gamma_k = gamma for all k. 'PB': a gamma_k per basis spectrum (the network bounds how far they
    spread, see QuantNet).
    """

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
#                                             Class MLP                                            #
#**************************************************************************************************#
#                                                                                                  #
# BatchNorm over the two channels, flatten, fc layers halving the width, a linear output.          #
#                                                                                                  #
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


#**************************************************************************************************#
#                                          Class QuantNet                                          #
#**************************************************************************************************#
#                                                                                                  #
# The network: the window of the unit-norm input, the MLP, and a head that bounds every parameter  #
# kind; "rescale" gives the parameters of the input itself.                                        #
#                                                                                                  #
#**************************************************************************************************#
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


#**************#
#   the data   #
#**************#
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
#                                                                                                  #
# One run's data: an infinite training stream and the fixed validation set.                        #
#                                                                                                  #
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
    scans are: the input of every in-vivo fit ("invivo"). Saves fids, stems, subjects,
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


#*************#
#   scoring   #
#*************#
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


#***************#
#   test sets   #
#***************#
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
    """FSL-MRS PB's in-vivo fits ("invivo") in *model*'s layout: (theta, stems)."""
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
    The rows a test set copies (rows of "simulate_rows", in the data's units): osprey-rows'
    (<fits>/rows_osprey.npz: each scan as Osprey fitted it, in Osprey's own model) or rows'
    (<fits>/rows.npz: each tool's concentrations, the rest fitted to the scan).

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


#**************************************************************************************************#
#                                           Class TestSet                                          #
#**************************************************************************************************#
#                                                                                                  #
# A saved test set: spectra (N, 2, T), ground truth, and the network's scores on it.               #
#                                                                                                  #
#**************************************************************************************************#
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


#**************#
#   training   #
#**************#
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


#**************************************************************************************************#
#                                          Class StepClock                                         #
#**************************************************************************************************#
#                                                                                                  #
# Data and model time per step from CUDA events, read at the evaluations only, so the host never   #
# waits for the device inside the loop (on the CPU: the wall clock).                               #
#                                                                                                  #
#**************************************************************************************************#
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


#**************************************************************************************************#
#                                         Class GraphedStep                                        #
#**************************************************************************************************#
#                                                                                                  #
# The training step (network, signal model, loss, backward, Adam) captured once as a CUDA graph    #
# and replayed: one launch instead of a few hundred small kernels, each of which waits for its     #
# turn while other jobs share the GPU.                                                             #
#                                                                                                  #
#**************************************************************************************************#
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


#*******************#
#   tool settings   #
#*******************#
ECHO_TIME = 26.0                            # ms, sLASER TE26 (the JSON basis does not carry it)
MAX_WORKERS = 8                             # fitting processes (shared machine)
MAXFUN = 20000                              # TNC budget for PB (scipy's 10 per parameter is short)
LCM_NAME_LEN = 6                            # LCModel keeps the first 6 characters of a name
LCMODEL_CONTROL = {'nsimul': 0}             # the basis has its own macromolecules
OSPREY_CENTRE_PPM = 4.65                    # FSL-MRS' water reference, and the COWS basis'
METHODS = ('fsl_default', 'fsl_pb', 'lcmodel', 'osprey')
#: what "invivo" keeps of each fit besides the concentrations: the tool's own parameters ...
TOOL_PARAMS = {
    'fsl_pb': ('gamma', 'sigma', 'eps', 'phi0', 'phi1', 'baseline'),
    'lcmodel': ('fwhm', 'shift', 'ph0', 'ph1'),
    'osprey': ('ampl', 'scale', 'ph0', 'ph1', 'gaussLB', 'refShift', 'lorentzLB', 'freqShift',
               'basis_shift', 'basis_factor')}
#: ... and its curves (LCModel and Osprey: real parts on their own ppm labels)
CURVES = {'fsl_default': ('data', 'fit'), 'fsl_pb': ('data', 'fit'),
          'lcmodel': ('ppm', 'data', 'fit', 'background'),
          'osprey': ('ppm', 'data', 'fit', 'baseline', 'fit_nokernel', 'lineShape')}


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


def lcmodel_exec():
    """
    The LCModel binary as PyLCModel resolves it: LCMODEL_EXEC if set, else the native Linux
    binary PyLCModel downloads from the LCModel repository (schorschinho/LCModel) into
    data/lcmodel, once. A cache of its own, and no container or source-build fallback, so every
    fit runs the same native build (PyLCModel's shared cache may hold a Docker shim).
    """
    from lcmodel_wrapper.binaries import resolve_executable
    return resolve_executable(allow_download=True, allow_docker=False, allow_build=False,
                              cache_dir=os.path.abspath('data/lcmodel'))


#*************#
#   fsl-mrs   #
#*************#
_FSL = {}


def fsl_basis(basis_dir):
    """The JSON basis as FSL-MRS reads it (one per process and folder)."""
    from fsl_mrs.core.basis import Basis
    from fsl_mrs.utils.mrs_io.fsl_io import readFSLBasisFiles
    if basis_dir not in _FSL:
        _FSL[basis_dir] = Basis(*readFSLBasisFiles(basis_dir))
    return _FSL[basis_dir]


def prepare_mrs(fid, cf, bw, basis_dir, ppmlim=PPM_WINDOW):
    """An MRS object with FSL-MRS' own preparation: conjugation checks and rescaling."""
    from fsl_mrs.core import MRS
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mrs = MRS(FID=np.asarray(fid, complex), cf=cf, bw=bw, nucleus='1H',
                  basis=fsl_basis(basis_dir))
        mrs.processForFitting(ppmlim=ppmlim)
    return mrs


def fsl_orientation(fid, cf, bw, basis_dir, ppmlim=PPM_WINDOW):
    """FSL-MRS' conjugation flags (FID, basis) for data like *fid*."""
    from fsl_mrs.core import MRS
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mrs = MRS(FID=np.asarray(fid, complex), cf=cf, bw=bw, nucleus='1H',
                  basis=fsl_basis(basis_dir))
        mrs.check_FID(ppmlim=ppmlim, repair=True)
        mrs.check_Basis(ppmlim=ppmlim, repair=True)
    return bool(mrs.conj_FID), bool(mrs.conj_Basis)


def data_units(mrs, basis_dir):
    """The factor from FSL-MRS' rescaled concentrations to the data's (the formatted basis')."""
    basis = fsl_basis(basis_dir)
    s_basis = (np.linalg.norm(mrs.basis)
               / np.linalg.norm(basis.get_formatted_basis(mrs.bandwidth, mrs.numPoints)))
    return s_basis / np.abs(mrs._fid_scaling)


def fsl_voigt(mrs):
    """FSL-MRS' default fit of a prepared spectrum (one metabolite group, poly-2, Newton)."""
    from fsl_mrs.utils.fitting import fit_FSLModel
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fit_FSLModel(mrs, metab_groups=[0] * len(mrs.names), model='voigt', ppmlim=PPM_WINDOW,
                            method='Newton', baseline_order=2)


def fit_fsl_default(fid, cfg):
    """FSL-MRS at its defaults: concentrations (data units) and its fit in its own convention."""
    from fsl_mrs.utils.misc import FIDToSpec
    mrs = prepare_mrs(fid, cfg['cf'], cfg['bw'], cfg['basis_dir'])
    res = fsl_voigt(mrs)
    n = len(mrs.names)
    params = np.asarray(res.params, float)
    fsl_fid = np.asarray(mrs.FID).ravel()
    r = np.vdot(fsl_fid, fid) / np.vdot(fsl_fid, fsl_fid)        # FSL's scaling -> the data's
    spec = lambda x: np.fft.ifftshift(FIDToSpec(np.asarray(x).ravel() * r))  # halves point 0
    return dict(names=[m.split('.')[0] for m in mrs.names],
                con=params[:n] * data_units(mrs, cfg['basis_dir']),
                gamma=float(params[n]), sigma=float(params[n + 1]),
                data=spec(fsl_fid), fit=spec(res.pred))


#*********************#
#   model pb in fsl   #
#*********************#
# The parameter vector is the network's PB layout: con (n), gamma_k (n), sigma, eps, phi0, phi1,
# the poly-2 baseline (6). The optimiser works on (con, gamma0, delta_k, rest) with
# gamma_k = gamma0 + delta_k, gamma0 >= 0 and 0 <= delta_k <= Osprey's bound.
def _split(x, n):
    return (x[:n], x[n:2 * n], x[2 * n:2 * n + 1], x[2 * n + 1:2 * n + 2], x[2 * n + 2],
            x[2 * n + 3], x[2 * n + 4:])


def pb_forward(x, f, t, m, B):
    """The model spectrum (T,) in the network's convention (fft, no fftshift, no halving)."""
    n = m.shape[1]
    con, gamma, sigma, eps, phi0, phi1, b = _split(x, n)
    E = np.exp(-(1j * eps[0] + gamma[None, :] + sigma[0] ** 2 * t[:, None]) * t[:, None])
    return np.exp(-1j * (phi0 + phi1 * f)) * (np.fft.fft(m * E, axis=0) @ con) + B @ b


def pb_err(x, f, t, m, B, data, first, last):
    res = data[first:last] - pb_forward(x, f, t, m, B)[first:last]
    return float(np.real(np.sum(res * np.conj(res))))


def pb_grad(x, f, t, m, B, data, first, last):
    """Analytic gradient of "pb_err" (d sse / dp = -2 Re sum conj(dS/dp) (data - S))."""
    n = m.shape[1]
    con, gamma, sigma, eps, phi0, phi1, b = _split(x, n)
    tt = t[:, None]
    D = m * np.exp(-(1j * eps[0] + gamma[None, :] + sigma[0] ** 2 * tt) * tt)
    phase = np.exp(-1j * (phi0 + phi1 * f))
    M = np.fft.fft(D, axis=0)
    metab = phase * (M @ con)
    w = slice(first, last)
    res = (data - metab - B @ b)[w]
    dot = lambda col: -2.0 * np.real(np.sum(np.conj(col[w]) * res))
    g = np.zeros_like(x)
    Mg = np.fft.fft(-tt * D, axis=0) * con[None, :]
    for k in range(n):
        g[k] = dot(phase * M[:, k])
        g[n + k] = dot(phase * Mg[:, k])
    g[2 * n] = dot(phase * (np.fft.fft(-2.0 * sigma[0] * tt ** 2 * D, axis=0) @ con))
    g[2 * n + 1] = dot(phase * (np.fft.fft(-1j * tt * D, axis=0) @ con))
    g[2 * n + 2] = dot(-1j * metab)
    g[2 * n + 3] = dot(-1j * f * metab)
    for i in range(B.shape[1]):
        g[2 * n + 4 + i] = dot(B[:, i])
    return g


def _to_x(y, n):
    """(con, gamma0, delta, rest) -> PB's (con, gamma = gamma0 + delta, rest)."""
    return np.r_[y[:n], y[n] + y[n + 1:2 * n + 1], y[2 * n + 1:]]


def _to_y(x, n, delta_max):
    gamma = x[n:2 * n]
    g0 = max(float(gamma.min()), 0.0)
    return np.r_[x[:n], g0, np.clip(gamma - g0, 0.0, delta_max), x[2 * n:]]


def fit_fsl_pb(fid, cfg):
    """
    FSL-MRS on the network's model PB. FSL-MRS prepares the data (conjugation, rescaling) and fits
    its default model, whose optimum PB contains and starts from; then TNC with PB's bounds.

    Returns:
        Dict: con, gamma (n), sigma, eps, phi0, phi1, baseline (6) in the data's units and the
        network's conventions, the spectrum and fit on the scan's grid, residual / noise.
    """
    mrs = prepare_mrs(fid, cfg['cf'], cfg['bw'], cfg['basis_dir'])
    names = [m.split('.')[0] for m in mrs.names]
    n, T = len(names), mrs.numPoints
    ppm = np.fft.ifftshift(np.squeeze(mrs.ppmAxisShift))
    first = int(np.abs(ppm - PPM_WINDOW[0]).argmin())
    last = int(np.abs(ppm - PPM_WINDOW[1]).argmin())
    B = poly_baseline(T, first, last)
    f = np.fft.fftfreq(T, d=mrs.dwellTime)
    t = np.asarray(mrs.timeAxis).flatten()
    m = np.asarray(mrs.basis)
    data = np.fft.fft(np.asarray(mrs.FID))
    parent = np.asarray(fsl_voigt(mrs).params, float)             # con, gamma, sigma, eps, ...
    x0 = np.r_[parent[:n], np.full(n, parent[n]), parent[n + 1], parent[n + 2], parent[n + 3],
               parent[n + 4], parent[n + 5:]]
    delta_max = np.array([PB_DELTA_MAX_MM if nm in cfg['mm_names'] else PB_DELTA_MAX
                          for nm in names])
    args = (f, t, m, B, data, first, last)
    bounds = ([(0, None)] * n + [(0, None)] + [(0, float(d)) for d in delta_max]
              + [(0, None)] + [(None, None)] * (3 + B.shape[1]))
    res = minimize(lambda y: pb_err(_to_x(y, n), *args), _to_y(x0, n, delta_max), method='TNC',
                   jac=lambda y: (lambda g: np.r_[g[:n], g[n:2 * n].sum(), g[n:2 * n], g[2 * n:]])(
                       pb_grad(_to_x(y, n), *args)),
                   bounds=bounds, options={'maxfun': MAXFUN})
    x = _to_x(res.x, n)
    con, gamma, sigma, eps, phi0, phi1, b = _split(x, n)

    # the parameters in the data's units, and the model rebuilt on the network's basis
    basis = load_basis(cfg['basis_dir'])
    scale = float(np.abs(mrs._fid_scaling))
    con = con * data_units(mrs, cfg['basis_dir'])
    b = np.asarray(b) / scale
    order = [names.index(nm) for nm in basis.names]
    con, gamma = con[order], gamma[order]
    Bn = poly_baseline(len(fid), *basis.window())
    E = np.exp(-(1j * eps[0] + gamma[None, :] + sigma[0] ** 2 * basis.t[:, None]) * basis.t[:, None])
    fnet = np.fft.fftfreq(len(fid), d=float(basis.t[1] - basis.t[0]))
    model = (np.exp(-1j * (phi0 + phi1 * fnet)) * (np.fft.fft(basis.fids * E, axis=0) @ con)
             + Bn @ b)
    spec = np.fft.fft(np.asarray(fid, complex))
    return dict(names=list(basis.names), con=con, gamma=gamma, sigma=float(sigma[0]),
                eps=float(eps[0]), phi0=float(phi0), phi1=float(phi1), baseline=b,
                data=spec, fit=model, nfev=int(res.nfev), success=bool(res.status in (0, 1, 2)))


def check_pb_gradient(basis_dir, seed=0):
    """Largest relative error of "pb_grad" against central differences, on random parameters."""
    basis = load_basis(basis_dir)
    rng = np.random.default_rng(seed)
    n, T = len(basis.names), basis.fids.shape[0]
    first, last = basis.window()
    B = poly_baseline(T, first, last)
    f = np.fft.fftfreq(T, d=float(basis.t[1] - basis.t[0]))
    x = np.r_[rng.uniform(0.1, 9, n), rng.uniform(0.5, 30, n), rng.uniform(1, 8, 1),
              rng.uniform(-8, 8, 1), rng.uniform(-0.3, 0.3, 1), rng.uniform(-2e-3, 2e-3, 1),
              rng.uniform(-50, 50, B.shape[1])]
    clean = pb_forward(x, f, basis.t, basis.fids, B)
    data = clean + 0.05 * np.abs(clean).max() * (rng.standard_normal(T) + 1j * rng.standard_normal(T))
    args = (f, basis.t, basis.fids, B, data, first, last)
    g, num = pb_grad(x, *args), np.zeros_like(x)
    for i in range(len(x)):
        h = 1e-6 * max(abs(x[i]), 1e-3)
        up, dn = x.copy(), x.copy()
        up[i] += h
        dn[i] -= h
        num[i] = (pb_err(up, *args) - pb_err(dn, *args)) / (2 * h)
    return float(np.abs(g - num).max() / np.abs(num).max())


#*************#
#   lcmodel   #
#*************#
_CONC_ROW = re.compile(r'^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+(\d+)%\s+'
                       r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(\S+)\s*$')
_LCM = {}


def write_lcm_basis(path, basis_dir, conj_basis):
    """The JSON basis as an LCModel .BASIS (PyLCModel's writer: the zero-filled FFT of the FIDs),
    in LCModel's orientation, the conjugate of FSL-MRS'."""
    from lcmodel_wrapper.convert import write_basis
    basis = fsl_basis(basis_dir)
    fids = basis.original_basis_array
    fids = np.conj(np.conj(fids) if conj_basis else fids)
    names = [n.split('.')[0] for n in basis.names]
    if len({n[:LCM_NAME_LEN] for n in names}) != len(names):
        raise ValueError(f'basis names are not unique in their first {LCM_NAME_LEN} characters')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    write_basis(path, names, [fids[:, i] for i in range(len(names))],
                float(basis.original_dwell), float(basis.cf), ECHO_TIME)
    return path


def read_coord(path, names):
    """
    A .coord file: concentrations (basis order, matched on 6 characters), the misc. table (fwhm and
    shift in ppm, ph0 in whole degrees, ph1 in deg/ppm, as LCModel prints them) and the curves
    (ppm, phased data, fit, background), all real parts on LCModel's own ppm labels.
    """
    with open(path) as fh:
        text = fh.read()
    lines = text.splitlines()
    table = {}
    for i, line in enumerate(lines):
        if 'lines in following concentration table' in line:
            for row in lines[i + 1:i + int(line.split()[0]) + 1]:
                mt = _CONC_ROW.match(row)
                if mt:
                    table[mt.group(4)] = float(mt.group(1))
    num = r'[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?'
    npts = int(re.search(r'(\d+)\s+points on ppm-axis', text).group(1))
    curves = {}
    for key, head in (('ppm', r'points on ppm-axis[^\n]*'), ('data', r'phased data points follow'),
                      ('fit', r'points of the fit to the data follow'),
                      ('background', r'background values follow')):
        mt = re.search(head, text)
        start = text.index('\n', mt.end())
        curves[key] = np.array([float(v.replace('D', 'E'))
                                for v in re.findall(num, text[start:])[:npts]])
    ph = re.search(rf'Ph:\s*({num}) deg\s+({num}) deg/ppm', text)
    misc = dict(fwhm=float(re.search(rf'FWHM =\s*({num}) ppm', text).group(1)),
                shift=float(re.search(rf'Data shift =\s*({num}) ppm', text).group(1)),
                ph0=float(ph.group(1)), ph1=float(ph.group(2)))
    return dict(con=np.array([table.get(nm[:LCM_NAME_LEN], 0.0) for nm in names]), **misc,
                **curves)


def fit_lcmodel(fid, cfg):
    """One LCModel fit through PyLCModel; the .RAW / .coord files are removed afterwards."""
    from lcmodel_wrapper import PyLCModel
    from lcmodel_wrapper import control as lcm_control
    from lcmodel_wrapper import io as lcm_io
    key = (cfg['lcm_basis'], cfg['workdir'])
    if key not in _LCM:
        os.makedirs(cfg['workdir'], exist_ok=True)
        lcm = PyLCModel(cfg['lcm_basis'], ppmlim=PPM_WINDOW, conj=cfg['lcm_conj'], ignore='none',
                        save_path=cfg['workdir'], path2exec=cfg['lcmodel_exec'],
                        sample_points=len(fid), bandwidth=cfg['bw'], central_freq=cfg['cf'],
                        allow_download=False, allow_docker=False, allow_build=False)
        for k, v in LCMODEL_CONTROL.items():
            lcm_control.set_key(lcm.control, k, v)
        _LCM[key] = lcm
    lcm = _LCM[key]
    stem = os.path.join(cfg['workdir'], f"{cfg['tag']}_{os.getpid()}_{time.monotonic_ns()}")
    lcm_io.to_raw(np.conj(fid) if lcm.conj else fid, stem + '.raw')
    output = lcm.initiate(stem + '.raw')
    try:
        if not os.path.isfile(stem + '.coord'):
            raise RuntimeError(f'LCModel wrote no .coord: {output[-500:]}')
        return read_coord(stem + '.coord', cfg['names'])
    finally:
        for ext in ('.raw', '.ps', '.coord', '.control'):
            if os.path.isfile(stem + ext):
                os.remove(stem + ext)


#************#
#   osprey   #
#************#
_W = {}


def _osprey_init(lock, basis_path, fit_range, basis_dir):
    from mrsuite.bridges import osprey
    with lock:                                      # one Octave session start at a time
        osprey.bridge.centre_ppm = OSPREY_CENTRE_PPM
        _W['basis'] = osprey.bridge.basis_from_lcmodel(basis_path, add_mm=False)
    _W.update(osprey=osprey, range=list(fit_range), net=load_basis(basis_dir))


def _osprey_basis_relation(res_basis, names):
    """
    Osprey's resampled basis (the one its fit uses) against the network's: Osprey resamples the
    basis onto its own ppm axis, which shifts every function by the same fraction of a bin and
    scales each a little differently. Returns (shift, Hz: Osprey's = the network's shifted by
    it, and one complex factor per function in *names*' order, with the shift applied).
    """
    from scipy.optimize import minimize_scalar
    net = _W['net']
    osp = [str(n) for n in np.asarray(res_basis['name']).ravel()]
    T = net.fids.shape[0]
    a = np.asarray(res_basis['fids'])[1:T, [osp.index(nm) for nm in names]]
    c0 = net.fids[1:, [net.names.index(nm) for nm in names]]
    ramp = lambda d: np.exp(2j * np.pi * d * np.arange(1, T) / net.bw)[:, None]
    fac = lambda c: (np.conj(c) * a).sum(0) / (np.abs(c) ** 2).sum(0)

    def err(d):
        c = c0 * ramp(d)
        return float(np.linalg.norm(a - fac(c) * c))

    shift = float(minimize_scalar(err, bounds=(-2.0, 2.0), method='bounded',
                                  options=dict(xatol=1e-7)).x)
    return shift, fac(c0 * ramp(shift))


def _osprey_unit(names, fsl_fids):
    """
    The factor from Osprey's amplitudes to the data's units, after checking that Osprey reads the
    basis as FSL-MRS does (per function, up to one common complex factor; the first point left
    out, since Osprey removes a constant from each spectrum).
    """
    b = _W['basis']
    osp = [str(n) for n in np.asarray(b['name']).ravel()]
    fids = np.asarray(b['fids'])
    n = fsl_fids.shape[0]
    scales, worst = [], 0.0
    for k, nm in enumerate(names):
        a, c = fids[1:n, osp.index(nm)], fsl_fids[1:, k]
        s = np.vdot(c, a) / np.vdot(c, c)
        scales.append(s)
        worst = max(worst, np.linalg.norm(a - s * c) / np.linalg.norm(a))
    if worst > 1e-4:
        raise RuntimeError(f'Osprey reads the basis differently from FSL-MRS ({worst:.1e})')
    return float(np.median(np.real(scales)))


def _osprey_fit(job):
    i, fid, cf, bw = job
    from nifti_mrs.create_nmrs import gen_nifti_mrs
    osprey = _W['osprey']
    try:
        img = gen_nifti_mrs(np.asarray(fid, np.complex128).reshape(1, 1, 1, -1), 1.0 / bw, cf,
                            nucleus='1H')
        img.add_hdr_field('EchoTime', ECHO_TIME / 1e3)
        res = osprey.fit(img, _W['basis'], fit_opts=dict(range=_W['range']))
        params, res_basis, scale = (res.raw[k] for k in ('fitParams', 'resBasisSet', 'scale'))
        names = [str(n) for n in np.asarray(res_basis['name']).ravel()]
        bridge = osprey.bridge
        struct = bridge.codec.nifti2fida(img)
        struct['fids'] = np.asarray(struct['fids']) / scale
        struct['specs'] = np.asarray(struct['specs']) / scale
        full = bridge.fit_options(range=_W['range'])
        settings = dict(fitRangePPM=full['range'], minKnotSpacingPPM=full['bLineKnotSpace'],
                        scale=1.0, GAP=np.array([]), fitStyle='Separate')
        inputs = dict(dataToFit=struct, basisSet=res_basis)
        model = bridge.fit_OspreyParamsToModel(inputs, settings, params)
        # the same model without the lineshape kernel, the part the test set's rows leave out
        plain = bridge.fit_OspreyParamsToModel(inputs, settings,
                                               dict(params, lineShape=np.array([[1.0]])))
        curve = lambda m, k: np.asarray(m[k], dtype=float).ravel()
        vec = lambda k: np.asarray(params[k], dtype=float).ravel()
        shift, factors = _osprey_basis_relation(res_basis, names)
        return i, dict(names=names, ampl=np.array([res.conc('raw')[n] for n in names]),
                       ppm=curve(model, 'ppm'), data=curve(model, 'data'),
                       fit=curve(model, 'completeFit'), baseline=curve(model, 'baseline'),
                       fit_nokernel=curve(plain, 'completeFit'), scale=float(scale),
                       ph0=float(vec('ph0')[0]), ph1=float(vec('ph1')[0]),
                       gaussLB=float(vec('gaussLB')[0]), refShift=float(vec('refShift')[0]),
                       lorentzLB=vec('lorentzLB'), freqShift=vec('freqShift'),
                       lineShape=vec('lineShape'), basis_shift=shift,
                       basis_factor=factors), None
    except Exception as exc:                        # noqa: BLE001 - report and keep going
        return i, None, f'{type(exc).__name__}: {exc}'


def fit_osprey(fids, cf, bw, cfg, workers, label='osprey'):
    """Osprey on every FID in a pool of Octave sessions: {index: result}, {index: error}."""
    ctx = mp.get_context('spawn')
    lock = ctx.Lock()
    basis = fsl_basis(cfg['basis_dir'])
    fsl_fids = basis.original_basis_array
    fsl_fids = np.conj(fsl_fids) if cfg['conj_basis'] else fsl_fids
    names = [n.split('.')[0] for n in basis.names]
    results, errors, t0 = {}, {}, time.time()
    with ctx.Pool(min(workers, MAX_WORKERS), initializer=_osprey_init,
                  initargs=(lock, cfg['lcm_basis'], PPM_WINDOW, cfg['basis_dir'])) as pool:
        unit = pool.apply(_osprey_unit, (names, fsl_fids))
        for k, (i, r, err) in enumerate(pool.imap_unordered(
                _osprey_fit, [(i, f, cf, bw) for i, f in enumerate(fids)])):
            if err:
                errors[i] = err
            else:
                order = [r['names'].index(nm) for nm in cfg['names']]
                per_basis = {key: r[key][order] for key in ('ampl', 'lorentzLB', 'freqShift',
                                                            'basis_factor')}
                results[i] = dict(r, **per_basis, con=per_basis['ampl'] * unit,
                                  names=list(cfg['names']))
            if (k + 1) % 25 == 0 or k + 1 == len(fids):
                log(f'{label}: {k + 1}/{len(fids)} ({(time.time() - t0) / (k + 1):.1f} s per '
                    f'spectrum, {len(errors)} failed)')
    return results, errors


#****************#
#   batch fits   #
#****************#
_FITS = {'fsl_default': fit_fsl_default, 'fsl_pb': fit_fsl_pb, 'lcmodel': fit_lcmodel}


def _task(job):
    method, i, fid, cfg = job
    try:
        return i, _FITS[method](fid, cfg)
    except Exception:                               # noqa: BLE001 - report and keep going
        import traceback
        return i, traceback.format_exc()


def fit_all(method, fids, cfg, workers, label=None):
    """Every FID with *method*: a list of result dicts (None where a fit failed)."""
    label = label or method
    if method == 'osprey':
        res, errors = fit_osprey(fids, cfg['cf'], cfg['bw'], cfg, workers, label)
        results = [res.get(i) for i in range(len(fids))]
    else:
        results, errors, t0 = [None] * len(fids), {}, time.time()
        jobs = [(method, i, f, cfg) for i, f in enumerate(fids)]
        with ProcessPoolExecutor(min(workers, MAX_WORKERS),
                                 mp_context=mp.get_context('spawn')) as pool:
            for k, (i, r) in enumerate(pool.map(_task, jobs, chunksize=1)):
                if isinstance(r, str):
                    errors[i] = r
                else:
                    results[i] = r
                if (k + 1) % 25 == 0 or k + 1 == len(fids):
                    log(f'{label}: {k + 1}/{len(fids)} ({(time.time() - t0) / (k + 1):.2f} s per '
                        f'spectrum, {len(errors)} failed)')
    if errors:
        log(f'{label}: {len(errors)} failed, first:\n{next(iter(errors.values()))[-800:]}')
    return results


def setup(fid, cf, bw, basis_dir, work, mm_names=MM_NAMES):
    """What every fit needs: the basis names, FSL-MRS' orientation, the LCModel basis."""
    conj_fid, conj_basis = fsl_orientation(fid, cf, bw, basis_dir)
    lcm_basis = write_lcm_basis(os.path.join(work, 'basis.BASIS'), basis_dir, conj_basis)
    return dict(cf=cf, bw=bw, basis_dir=basis_dir, names=list(load_basis(basis_dir).names),
                mm_names=tuple(mm_names), conj_fid=conj_fid, conj_basis=conj_basis,
                lcm_basis=lcm_basis, lcm_conj=not conj_fid, lcmodel_exec=lcmodel_exec(),
                workdir=os.path.join(work, 'lcmodel'), tag='fit')


def residual_over_noise(results, fids, ppm_grid):
    """
    Residual / noise of every fit, all methods alike: the method's data and fit scaled onto the
    scan by least squares; FSL-MRS on the scan's points, LCModel and Osprey on their own
    (zero-filled) points, since read between points their noise averages down; RMS of the real
    residual over the scan's noise SD per real component.
    """
    w = (ppm_grid >= PPM_WINDOW[0]) & (ppm_grid <= PPM_WINDOW[1])
    out = np.full(len(fids), np.nan)
    for i, (r, fid) in enumerate(zip(results, fids)):
        if r is None:
            continue
        spec = np.fft.fft(fid)
        noise = spec_noise_sd(spec, ppm_grid)
        if 'ppm' in r:                              # LCModel / Osprey: their own points
            k = (r['ppm'] >= PPM_WINDOW[0]) & (r['ppm'] <= PPM_WINDOW[1])
            o = np.argsort(r['ppm'])
            d_at_scan = np.interp(ppm_grid[w], r['ppm'][o], r['data'][o])
            a = spec[w].real @ d_at_scan / (d_at_scan @ d_at_scan)
            res = a * (r['data'][k] - r['fit'][k])
        else:
            d, f_ = np.asarray(r['data'])[w].real, np.asarray(r['fit'])[w].real
            a = spec[w].real @ d / (d @ d)
            res = a * (d - f_)
        out[i] = np.sqrt(np.mean(res ** 2)) / noise
    return out


#***************************************#
#   the test sets' rows from the fits   #
#***************************************#
# A test-set row copies one scan as one tool quantified it: the tool's concentrations are the
# truth, and everything else (lineshape, shift, phases, baseline) is fitted to the scan itself with
# those concentrations held (times one common factor), the same way for every tool, on the real and
# imaginary parts. So every row is as close to its scan as that tool's concentrations allow, and no
# tool's own lineshape or baseline model is built into the set. The model: model PB (a shared
# Voigt plus a Lorentzian per basis spectrum within Osprey's bounds, one shift, zero- and first-
# order phase) and a smooth baseline, a causal FID of ROW_BASELINE_POINTS points (20 ms: the
# resolution of Osprey's default 0.4 ppm spline knots) fitted on the window, zero outside it.
ROW_BASELINE_POINTS = 80


def _row_design(x, con, basis, first, last, f, M):
    """The linear model of a row with the nonlinear parameters *x*: the concentrations' spectrum
    (one factor) and the baseline FID's M complex points, as real columns on the window."""
    n = len(basis.names)
    g0, d, sig, eps, p0, p1 = x[0], x[1:n + 1], x[n + 1], x[n + 2], x[n + 3], x[n + 4]
    t = basis.t
    E = np.exp(-(1j * eps + g0 + d[None, :] + sig ** 2 * t[:, None]) * t[:, None])
    w = slice(first, last)
    m = np.exp(-1j * (p0 + p1 * f[w])) * np.fft.fft((basis.fids * E) @ con)[w]
    Bf = np.exp(-2j * np.pi * np.outer(f[w], np.arange(M)) / basis.bw)
    C = np.c_[m, Bf, 1j * Bf]
    return np.r_[C.real, C.imag]


def fitted_row(fid, con, start, basis, mm_names=MM_NAMES, M=ROW_BASELINE_POINTS):
    """
    One row: *con* held (up to one factor), the rest fitted to *fid* from *start* (the scan's
    FSL-MRS PB fit), the linear parts (factor, baseline) solved at each step. Returns the row and
    its residual / noise over the window (real part, both parts).
    """
    from scipy.optimize import least_squares
    n, T = len(basis.names), len(basis.t)
    first, last = basis.window()
    f = np.fft.fftfreq(T, d=1.0 / basis.bw)
    spec = np.fft.fft(fid)
    y = np.r_[spec[first:last].real, spec[first:last].imag]
    dmax = np.array([PB_DELTA_MAX_MM if nm in mm_names else PB_DELTA_MAX
                     for nm in basis.names])
    g = np.asarray(start['gamma'], float)
    x0 = np.r_[g.min(), np.clip(g - g.min(), 0, dmax), float(start['sigma']),
               float(start['eps']), float(start['phi0']), float(start['phi1'])]
    solve = lambda x: (lambda A: (A, np.linalg.lstsq(A, y, rcond=None)[0]))(
        _row_design(x, con, basis, first, last, f, M))
    res = least_squares(lambda x: (lambda A, c: A @ c - y)(*solve(x)), x0, x_scale='jac',
                        bounds=(np.r_[0.0, np.zeros(n), 0.0, -np.inf, -np.inf, -np.inf],
                                np.r_[np.inf, dmax, np.inf, np.inf, np.inf, np.inf]))
    x = res.x
    A, c = solve(x)
    baseline = np.zeros(T, complex)
    baseline[:M] = c[1:M + 1] + 1j * c[M + 1:]
    # fitted on the window only, the curve is unbounded outside it (~1e10 x the signal): zero
    # there, as FSL-MRS' polynomial baseline ("poly_baseline")
    curve = np.zeros(T, complex)
    curve[first:last] = np.fft.fft(baseline)[first:last]
    row = dict(con=con * c[0], gamma=x[0] + x[1:n + 1], sigma=float(x[n + 1]),
               eps=np.full(n, float(x[n + 2])), phi0=float(x[n + 3]), phi1=float(x[n + 4]),
               baseline=curve)
    sd = spec_noise_sd(spec, basis.ppm)
    r = A @ c - y
    return row, dict(resid_real=float(np.sqrt(np.mean(r[:len(r) // 2] ** 2)) / sd),
                     resid=float(np.sqrt(np.mean(r ** 2)) / sd), factor=float(c[0]),
                     success=bool(res.success))


def _row_task(job):
    fid, con, start, basis_dir = job
    return fitted_row(fid, con, start, load_basis(basis_dir))


def tool_rows(fits_dir, processed, basis_dir, workers):
    """
    Every scan as each tool quantified it, as rows ("fitted_row"), with each row's residual /
    noise against its scan: <fits_dir>/rows.npz and rows_check.csv.
    """
    z = np.load(processed)
    zstems = [str(s) for s in z['stems']]
    basis = load_basis(basis_dir)
    load = lambda m: np.load(os.path.join(fits_dir, f'invivo_{m}.npz'), allow_pickle=True)
    pb = load('fsl_pb')
    pb_stems = [str(s) for s in pb['stems']]
    jobs, meta = [], []
    for tool in ('fsl_pb', 'lcmodel', 'osprey'):
        f = load(tool)
        if [str(s) for s in f['names']] != basis.names:
            raise ValueError(f'{tool} names its basis spectra differently from the basis')
        for i, stem in enumerate(str(s) for s in f['stems']):
            j = pb_stems.index(stem)
            con = np.clip(np.asarray(f['con'][i], float), 0, None)
            if not (f['ok'][i] and pb['ok'][j] and np.isfinite(con).all()):
                continue
            start = {k: pb[k][j] for k in ('gamma', 'sigma', 'eps', 'phi0', 'phi1')}
            jobs.append((z['fids'][zstems.index(stem)], con, start, basis_dir))
            meta.append((tool, stem))
    rows, checks, t0 = [], [], time.time()
    with ProcessPoolExecutor(min(workers, MAX_WORKERS),
                             mp_context=mp.get_context('spawn')) as pool:
        for k, (row, check) in enumerate(pool.map(_row_task, jobs, chunksize=2)):
            rows.append(row)
            checks.append(check)
            if (k + 1) % 50 == 0 or k + 1 == len(jobs):
                log(f'rows: {k + 1}/{len(jobs)} ({time.time() - t0:.0f} s)')
    out = {k: np.array([r[k] for r in rows]) for k in ROW_KEYS}
    np.savez(os.path.join(fits_dir, 'rows.npz'), **out, tool=np.array([m[0] for m in meta]),
             stem=np.array([m[1] for m in meta]), names=np.array(basis.names),
             **{f'check_{k}': np.array([c[k] for c in checks]) for k in checks[0]})
    with open(os.path.join(fits_dir, 'rows_check.csv'), 'w') as fh:
        fh.write('tool,stem,' + ','.join(checks[0]) + '\n')
        for (tool, stem), c in zip(meta, checks):
            fh.write(f'{tool},{stem},' + ','.join(f'{c[k]:.6g}' for k in checks[0]) + '\n')
    for tool in ('fsl_pb', 'lcmodel', 'osprey'):
        sel = [c for m, c in zip(meta, checks) if m[0] == tool]
        q = lambda k: ' / '.join(f'{v:.2f}' for v in np.percentile([c[k] for c in sel],
                                                                    [50, 90, 100]))
        log(f'{tool} ({len(sel)} rows): row vs scan, residual / noise (median / p90 / max) real '
            f'{q("resid_real")}, both parts {q("resid")}; factor {q("factor")}; '
            f'{sum(not c["success"] for c in sel)} not converged')


#*************************************#
#   the test sets' rows from osprey   #
#*************************************#
# Osprey as the truth: every scan as Osprey fitted it, in Osprey's own model (fit_OspreyParams-
# ToModel.m): per basis spectrum a Lorentzian and a shift, one Gaussian, the lineshape kernel
# convolved with the metabolites, a cubic-spline baseline; the phases and the reference shift it
# applies to the data are applied to the model instead, so the row lies in the data's frame.
# Osprey models the real part only: the rows' metabolites are complex by construction (the kernel
# as a time-domain factor, "simulate_rows"); the baseline is Osprey's in its frame's real part,
# and its imaginary part, which Osprey leaves undefined, is fitted to the scan with a cubic spline
# at Osprey's default knot spacing (OSPREY_KNOT_PPM); the truth is Osprey's alone either way. Found against Osprey's own curves (invivo_osprey.npz): its data curve is the scan with
# its labels OSPREY_PPM_OFFSET below ours (fitted per scan), its per-basis shifts run opposite to
# our frequency axis; the rebuilt model is 0.26 x the noise from Osprey's own fit (median).
OSPREY_PIVOT_PPM = 4.68                    # fit_OspreyParamsToModel's first-order phase pivot
OSPREY_KNOT_PPM = 0.4                      # Osprey's default baseline knot spacing (bLineKnotSpace)


def _spline_design(x, lo, hi, spacing=OSPREY_KNOT_PPM, k=3):
    """Cubic B-splines on [lo, hi], knots about *spacing* apart: (len(x), m)."""
    from scipy.interpolate import BSpline
    inner = np.linspace(lo, hi, max(2, int(round((hi - lo) / spacing)) + 1))
    knots = np.r_[[lo] * k, inner, [hi] * k]
    return BSpline.design_matrix(np.clip(x, lo, hi), knots, k).toarray()


def _osprey_axis(basis):
    """Our FFT bins' ppm = a f + c (f in Hz)."""
    f = np.fft.fftfreq(len(basis.t), d=1.0 / basis.bw)
    a, c = np.polyfit(f, basis.ppm, 1)
    return f, float(a), float(c)


def _osprey_offset(fid, fit, i, basis):
    """Where Osprey's ppm labels sit on our axis: the offset that turns the scan into Osprey's own
    data curve (its reference shift, phases and scale applied), and that curve's mismatch."""
    from scipy.optimize import minimize_scalar
    _, a, c = _osprey_axis(basis)
    t = np.arange(len(fid)) / basis.bw
    ppm = np.asarray(fit['curve_ppm'][i], float)
    data = np.asarray(fit['curve_data'][i], float)
    x = fid * np.exp(2j * np.pi * fit['refShift'][i] * t) / fit['scale'][i]
    ph = np.exp(-1j * np.deg2rad(fit['ph0'][i] + fit['ph1'][i] * (ppm - OSPREY_PIVOT_PPM)))

    def err(d):
        nu = (ppm + d - c) / a
        sp = (np.exp(-2j * np.pi * np.outer(nu, t)) @ x) * ph
        return float(np.linalg.norm(sp.real - data) / np.linalg.norm(data))

    r = minimize_scalar(err, bounds=(-0.02, 0.02), method='bounded', options=dict(xatol=1e-7))
    return float(r.x), float(r.fun)


def osprey_row(fid, fit, i, basis):
    """
    Scan *i* as Osprey fitted it (*fit*: invivo_osprey.npz), a row of "simulate_rows" with its
    kernel, and the row against the scan: residual / noise over the window (real part, both
    parts), with Osprey's own fit's (real, on its grid) beside it.
    """
    f, a, c = _osprey_axis(basis)
    T, (first, last) = len(f), basis.window()
    off, data_err = _osprey_offset(fid, fit, i, basis)
    rs, scale = float(fit['refShift'][i]), float(fit['scale'][i])
    ph0, ph1 = np.deg2rad(fit['ph0'][i]), np.deg2rad(fit['ph1'][i])      # rad, rad / ppm
    fac = np.asarray(fit['basis_factor'][i])
    # one common phase (Osprey's resampling adds ~1 deg per function; the check measures it)
    theta = float(np.angle(np.mean(fac)))
    spread = float(np.ptp(np.angle(fac * np.exp(-1j * theta))))
    # Osprey's frame: data(p) = S_x(p + off) exp(-i (ph0 + ph1 (p - pivot))), x = fid e^{2 pi i rs t};
    # so fid's spectrum Y(f) = S_M(f + rs) exp(i (ph0 + ph1 (a (f + rs) + c - off - pivot)))
    phi1 = -ph1 * a
    phi0 = -(ph0 + ph1 * (a * rs + c - off - OSPREY_PIVOT_PPM)) - theta
    eps = -np.asarray(fit['freqShift'][i], float) - 2 * np.pi * (fit['basis_shift'][i] - rs)
    k = np.asarray(fit['curve_lineShape'][i], float)
    # the kernel's step on our axis: Osprey's grid step in Hz of ours
    ppm_o = np.asarray(fit['curve_ppm'][i], float)
    step_ours = basis.bw / (2 * T)
    if abs(np.diff(ppm_o).mean() / a - step_ours) > 1e-3 * step_ours:
        raise ValueError("Osprey's grid is not the zero-filled one")
    # the baseline in Osprey's frame, B + i g, into the data's (its phases), zero outside the window
    # (as every row's): B Osprey's curve at the bins' Osprey ppm, g fitted to the scan
    from scipy.interpolate import CubicSpline
    w = slice(first, last)
    p_bins = (a * (f + rs) + c - off)[w]
    real = CubicSpline(ppm_o, np.asarray(fit['curve_baseline'][i], float))(
        np.clip(p_bins, ppm_o[0], ppm_o[-1])) * scale
    phase = np.exp(1j * (ph0 + ph1 * (p_bins - OSPREY_PIVOT_PPM)))
    row = dict(con=np.asarray(fit['ampl'][i], float) * np.abs(fac),
               gamma=np.asarray(fit['lorentzLB'][i], float),
               sigma=float(np.sqrt(fit['gaussLB'][i])), eps=eps, phi0=float(phi0),
               phi1=float(phi1), baseline=np.zeros(T, complex), kernel=k / k.sum())
    spec = np.fft.fft(fid)
    metab = simulate_rows({key: np.asarray(v)[None] for key, v in row.items()}, basis)[0]
    spl = _spline_design(p_bins, ppm_o[0], ppm_o[-1])
    D = 1j * phase[:, None] * spl
    y = spec[w] - metab[w] - real * phase
    g = np.linalg.lstsq(np.r_[D.real, D.imag], np.r_[y.real, y.imag], rcond=None)[0]
    row['baseline'][w] = (real + 1j * (spl @ g)) * phase
    sd = spec_noise_sd(spec, basis.ppm)
    sim = metab + row['baseline']
    r = (sim - spec)[first:last]
    own = np.asarray(fit['curve_data'][i], float) - np.asarray(fit['curve_fit'][i], float)
    return row, dict(resid_real=float(np.sqrt(np.mean(r.real ** 2)) / sd),
                     resid=float(np.sqrt(np.mean(r.real ** 2 + r.imag ** 2) / 2) / sd),
                     osprey_own=float(np.sqrt(np.mean(own ** 2)) * scale / sd),
                     offset_ppm=off, data_curve_err=data_err, basis_phase_spread=spread)


def osprey_rows(fits_dir, processed, basis_dir):
    """Every scan Osprey fitted, as a row ("osprey_row"): <fits_dir>/rows_osprey.npz and
    rows_osprey_check.csv."""
    z = np.load(processed)
    zstems = [str(s) for s in z['stems']]
    basis = load_basis(basis_dir)
    fit = np.load(os.path.join(fits_dir, 'invivo_osprey.npz'), allow_pickle=True)
    if [str(s) for s in fit['names']] != basis.names:
        raise ValueError('Osprey names its basis spectra differently from the basis')
    rows, checks, stems = [], [], []
    for i, stem in enumerate(str(s) for s in fit['stems']):
        if not fit['ok'][i]:
            continue
        row, check = osprey_row(z['fids'][zstems.index(stem)], fit, i, basis)
        rows.append(row)
        checks.append(check)
        stems.append(stem)
    for r in rows:                         # Osprey sizes the kernel to the lines: pad, centred
        pad = (KERNEL_TAPS - len(r['kernel'])) // 2
        r['kernel'] = np.pad(r['kernel'], pad)
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    np.savez(os.path.join(fits_dir, 'rows_osprey.npz'), **out, tool=np.array(['osprey'] * len(rows)),
             stem=np.array(stems), names=np.array(basis.names),
             **{f'check_{k}': np.array([c[k] for c in checks]) for k in checks[0]})
    with open(os.path.join(fits_dir, 'rows_osprey_check.csv'), 'w') as fh:
        fh.write('stem,' + ','.join(checks[0]) + '\n')
        for stem, c in zip(stems, checks):
            fh.write(f'{stem},' + ','.join(f'{c[k]:.6g}' for k in checks[0]) + '\n')
    q = lambda k: ' / '.join(f'{v:.3g}' for v in np.percentile([c[k] for c in checks],
                                                                [50, 90, 100]))
    log(f'osprey rows ({len(rows)}): row vs scan, residual / noise (median / p90 / max) real '
        f'{q("resid_real")}, both parts {q("resid")}; Osprey\'s own fit (real, its grid) '
        f'{q("osprey_own")}; label offset {q("offset_ppm")} ppm, data-curve mismatch '
        f'{q("data_curve_err")}')


#*******************************#
#   in-vivo and test-set fits   #
#*******************************#
def invivo(processed, out, basis_dir, workers, methods=METHODS):
    """
    The 90 processed scans fitted by every method: invivo_<method>.npz (names, stems, con, and for
    fsl_pb its parameters; data / fit curves for the figures) and invivo_quality.csv.
    """
    z = np.load(processed)
    fids, stems, cf, bw = z['fids'], [str(s) for s in z['stems']], float(z['cf']), float(z['bw'])
    os.makedirs(out, exist_ok=True)
    cfg = setup(fids[0], cf, bw, basis_dir, os.path.join(out, 'work'))
    log(f"{len(fids)} scans; FSL-MRS flags conj_FID={cfg['conj_fid']}, conj_Basis="
        f"{cfg['conj_basis']}; window {PPM_WINDOW}")
    ppm_grid = load_basis(basis_dir).ppm
    quality = {}
    for method in methods:
        t0 = time.time()
        results = fit_all(method, fids, cfg, workers)
        ok = np.array([r is not None for r in results])
        con = np.array([r['con'] if r is not None else np.full(len(cfg['names']), np.nan)
                        for r in results])
        save = dict(names=np.array(cfg['names']), stems=np.array(stems), con=con, ok=ok,
                    method=method, window=np.array(PPM_WINDOW), seconds=time.time() - t0)
        # each tool's own parameters, which "rows" translates into the network's signal model
        for k in TOOL_PARAMS.get(method, ()):
            save[k] = np.array([r[k] if r is not None else np.full_like(
                np.asarray(next(q[k] for q in results if q is not None)), np.nan)
                for r in results])
        for k in CURVES[method]:
            save[f'curve_{k}'] = np.array([r[k] if r is not None else np.array([])
                                           for r in results], dtype=object)
        np.savez(os.path.join(out, f'invivo_{method}.npz'), **save)
        quality[method] = residual_over_noise(results, fids, ppm_grid)
        log(f'{method}: {int(ok.sum())}/{len(fids)} fitted in {time.time() - t0:.0f} s; residual '
            f'/ noise median {np.nanmedian(quality[method]):.2f}')
    with open(os.path.join(out, 'invivo_quality.csv'), 'w') as fh:
        fh.write('stem,' + ','.join(quality) + '\n')
        for i, s in enumerate(stems):
            fh.write(s + ',' + ','.join(f'{quality[m][i]:.4f}' for m in quality) + '\n')


def bench_testset(path, out, basis_dir, workers, methods=METHODS):
    """A test set fitted by every method, scored like the network: <set>_<method>.npz, scores."""
    ts = TestSet(path)
    x = ts.spectra.numpy().astype(np.float64)
    fids = np.fft.ifft(x[:, 0] + 1j * x[:, 1], axis=-1)   # the network's representation, inverted
    basis = load_basis(basis_dir)
    os.makedirs(out, exist_ok=True)
    cfg = setup(fids[0], basis.cf, basis.bw, basis_dir, os.path.join(out, 'work'))
    scores = {'no information': concentration_metrics(
        np.repeat(ts.concentrations.mean(0, keepdims=True), len(fids), 0), ts.concentrations,
        ts.names)}
    for method in methods:
        results = fit_all(method, fids, cfg, workers)
        con = np.array([r['con'] if r is not None else np.zeros(len(ts.names)) for r in results])
        np.savez(os.path.join(out, f'{ts.name}_{method}.npz'), con=con, names=np.array(ts.names),
                 ok=np.array([r is not None for r in results]), testset_hash=ts.hash)
        scores[method] = concentration_metrics(con, ts.concentrations, ts.names)
        log(f"{method}: MOSAE {scores[method]['mosae']:.4f}, mean CCC "
            f"{scores[method]['ccc_mean']:.3f}")
    with open(os.path.join(out, f'{ts.name}_scores.json'), 'w') as fh:
        json.dump(dict(testset=path, hash=ts.hash, scores=scores), fh, indent=1)


#*************************#
#   the ismrm challenge   #
#*************************#
CHALLENGE_CF, CHALLENGE_BW = 123.261703, 4000.0     # the organisers' .BASIS (HZPPPM, 1 / BADELT)


def challenge_truth(truth_dir, k):
    """{metabolite: mM} and the lipid amount of dataset *k* from the organisers' sheet."""
    zf = zipfile.ZipFile(os.path.join(truth_dir, f'dataset{k}.xlsx'))
    strings = re.findall(r'<t[^>]*>(.*?)</t>', zf.read('xl/sharedStrings.xml').decode())
    sheet = zf.read('xl/worksheets/sheet1.xml').decode()
    rows = {}
    for col, row, attrs, v in re.findall(r'<c r="([A-Z]+)(\d+)"([^>]*)>(?:<f>.*?</f>)?<v>(.*?)</v>',
                                         sheet):
        rows.setdefault(int(row), {})[col] = strings[int(v)] if 't="s"' in attrs else v
    gt, lipids = {}, 0.0
    for r in sorted(rows):
        a, b = rows[r].get('A', '').strip(), rows[r].get('B')
        if r <= 18 or b is None or '+' in a:
            continue
        try:
            val = float(b)
        except ValueError:
            continue
        if a == 'lipids':
            lipids = val
        else:
            gt['Mac' if a == 'MMBL' else a] = val
    return gt, lipids


def challenge(data_dir, truth_dir, out, workers, methods=METHODS):
    """
    The 28 datasets of the ISMRM 2016 MRS fitting challenge (Marjanska, Deelchand, Kreis; PRESS
    TE 30 ms, 3 T, 2048 points at 4000 Hz, one T2 for every metabolite): every method on
    0.5-4.2 ppm with the organisers' basis (Mac, the measured macromolecules, as MM), scored
    with the study's MOSAE (Mac left out, metabolites absent from the truth count as 0).
    """
    os.makedirs(out, exist_ok=True)
    basis_dir = os.path.join(out, 'basis_json')          # the organisers' text basis as JSON
    os.makedirs(basis_dir, exist_ok=True)
    src = os.path.join(data_dir, 'basisset_text_noTMS')
    for fn in sorted(os.listdir(src)):
        if fn.endswith('.txt'):
            a = np.loadtxt(os.path.join(src, fn))
            with open(os.path.join(basis_dir, fn[:-4] + '.json'), 'w') as fh:
                json.dump(dict(basis=dict(basis_re=a[:, 0].tolist(), basis_im=a[:, 1].tolist(),
                                          basis_dwell=1.0 / CHALLENGE_BW, basis_centre=CHALLENGE_CF,
                                          basis_width=None, basis_name=fn[:-4]),
                               meta=dict(source='ISMRM 2016 fitting challenge')), fh)
    ks = list(range(1, 29))
    fids = np.array([(lambda a: a[:, 0] + 1j * a[:, 1])(
        np.loadtxt(os.path.join(data_dir, 'datasets_text', f'dataset{k}.txt'))) for k in ks])
    if fsl_orientation(fids[0], CHALLENGE_CF, CHALLENGE_BW, basis_dir)[0]:
        fids = np.conj(fids)                               # into FSL-MRS' orientation
    cfg = setup(fids[0], CHALLENGE_CF, CHALLENGE_BW, basis_dir, os.path.join(out, 'work'),
                mm_names=('Mac',))
    names = cfg['names']
    keep = [j for j, nm in enumerate(names) if nm != 'Mac']
    truth = [challenge_truth(truth_dir, k) for k in ks]
    t = np.array([[gt.get(names[j], 0.0) for j in keep] for gt, _ in truth])
    lipids = np.array([lip for _, lip in truth])
    scores = {}
    for method in methods:
        results = fit_all(method, fids, cfg, workers, f'challenge {method}')
        con = np.array([r['con'][keep] if r is not None else np.zeros(len(keep))
                        for r in results])
        y_hat = optimal_scale(t, con) * con
        per = np.abs(t - y_hat).mean(1) / t.mean(1)            # MOSAE per dataset, relative
        scores[method] = dict(per_dataset=per.tolist(), median=float(np.median(per)),
                              mean=float(per.mean()),
                              median_no_lipids=float(np.median(per[lipids == 0])),
                              median_lipids=float(np.median(per[lipids > 0])))
        np.savez(os.path.join(out, f'challenge_{method}.npz'), con=con,
                 names=np.array([names[j] for j in keep]), truth=t, lipids=lipids)
        log(f"challenge {method}: MOSAE / mean truth median {scores[method]['median']:.3f} "
            f"(no lipids {scores[method]['median_no_lipids']:.3f}, lipids "
            f"{scores[method]['median_lipids']:.3f})")
    with open(os.path.join(out, 'challenge_scores.json'), 'w') as fh:
        json.dump(scores, fh, indent=1)


#**************#
#   pipeline   #
#**************#
def pipeline():
    """
    The data, end to end, each step its own command: the 90 scans processed and fitted by every
    tool, Osprey's fits as test-set rows, the test (seed 0) and the selection set (seed 1), every
    tool on the test set, and the in-vivo fits figure.
    """
    r, me, nice = 'results/cows', [sys.executable, 'scripts/svs_ablation.py'], ['nice', '-n', '5']
    scans = f'{r}/processed_scans.npz'
    steps = ([me + ['process', '--out', scans],
              nice + me + ['invivo', '--processed', scans, '--out', f'{r}/invivo'],
              nice + me + ['osprey-rows', '--fits', f'{r}/invivo', '--processed', scans]]
             + [me + ['testset', '--rows', f'{r}/invivo/rows_osprey.npz', '--processed', scans,
                      '--out', f'{r}/testsets_osprey', '--seed', str(seed)] for seed in (0, 1)]
             + [nice + me + ['bench', f'{r}/testsets_osprey/test_n1000_s0.npz', '--out',
                             f'{r}/benchmark_osprey'],
                [sys.executable, 'scripts/svs_figures.py', 'invivo', '--invivo', f'{r}/invivo',
                 '--out', f'{r}/figures']])
    for cmd in steps:
        log(' '.join(cmd[cmd.index(sys.executable) + 1:]))
        subprocess.run(cmd, check=True)
    log('done')


#*********************#
#   screen settings   #
#*********************#
SCREEN = 'results/cows/screen'
TESTSET = 'results/cows/testsets_osprey/test_n1000_s0.npz'
SELECTION = 'results/cows/testsets_osprey/test_n1000_s1.npz'
N_SUBJECTS = (1, 8)                 # stage A and the picks
B_SUBJECTS = tuple(range(1, 9))     # stage B: every training-set size (user, 2026-09-27)
SMOKE_STEPS, STEPS_A, EVAL_A = 200, 300_000, 5000
STEPS_B, EVAL_B, CKPT_B = 1_000_000, 1000, 250_000
MAX_PARALLEL = 16                   # runs per GPU at most (here the GPU and CPUs are shared)
MIN_FREE_MB = 6000                  # launch only while this much GPU memory is free ...
MIN_FREE_RAM_GB = 8                 # ... and this much RAM (sampling runs hold raw scans)
PY = sys.executable

#: the grid: the main paper's conditions at their stage-B strengths, longest first (stage-B hours)
GRID = ('all-best', 'average_sampling-min1', 'coil_sampling-min1', 'eddy_current-x2',
        'spurious_echoes-echo-amp0p1', 'phase_shift-x4', 'baseline-bspline-x2',
        'broadening-voigt-x4', 'noise-snr15', 'none', 'frequency_shift-x4')
GRID_FOLDS, GRID_SEEDS, GRID_STEPS = (0, 1, 2, 3, 4), (0,), 2_000_000
OUT_GRID = 'results/cows/grid'
#: all-best = the samplers and per family the best module of the screen (stage_b's picks)
ALL_BEST = ('coil_sampling-min1', 'average_sampling-min1', 'apodization-truncate-keep0p25',
            'artificial_peaks-voigt-phase-x4', 'baseline-bspline-x2', 'eddy_current-x2',
            'frequency_shift-x4', 'broadening-voigt-x4', 'macromolecules-measured-x2',
            'noise-snr15', 'phase_shift-x4', 'residual_water-turco-x2',
            'spurious_echoes-echo-amp0p1')
OPENNEURO = 'https://s3.amazonaws.com/openneuro.org'
#: OpenNeuro's copy has a broken multi-RAID header (the second measurement 516 bytes off); ours is
#: header-repaired (45 bytes, data untouched; 2026-09-17) and comes with the bundle: md5
REPAIRED = {'sub-01/mrs/sourcedata/sub-01_acq-06_svs_slaser_vapor7_metab_Occipital.dat':
            'd9160d10ef5a8c1c1d416e5eb80e7016'}
LOG = os.path.join(SCREEN, 'screen.log')


#******************#
#   the variants   #
#******************#
def _fmt(x):
    return f'{x:g}'.replace('.', 'p').replace('-', 'm')


def _ladder(name, family, kind, levels, build):
    """A variant: its strength ladder as [(label, entry)]."""
    return dict(name=name, family=family, kind=kind,
                levels=[(label, build(v)) for label, v in levels])


times = lambda *fs: [(f'x{_fmt(f)}', f) for f in fs]             # multiples of the in-vivo range


def variants():
    """Every screened augmentation, each with its strength ladder."""
    V = []
    # samplers: at least n of the 32 coils / transients
    for s, key in (('coil_sampling', 'n_coils'), ('average_sampling', 'n_averages')):
        V.append(_ladder(s, s, 'sampler', [(f'min{n}', n) for n in (1, 4, 8, 16, 24)],
                         lambda n, s=s, key=key: {s: {'per_sample': True, key: [n, 32]}}))
    V.append(_ladder('average_sampling-consecutive', 'average_sampling', 'sampler',
                     [(f'min{n}', n) for n in (4, 8, 16)],
                     lambda n: {'average_sampling': {'per_sample': True, 'n_averages': [n, 32],
                                                     'scheme': 'consecutive'}}))
    # noise: max |spectrum| / added SD from the lower bound up to 330 (little added noise)
    V.append(_ladder('noise', 'noise', 'module', [(f'snr{n}', n) for n in (165, 100, 60, 30, 15)],
                     lambda n: {'noise': {'snr': [float(n), 330.0]}}))
    # line broadening (in vivo: NAA Lorentzian FWHM 0.8-4.0 Hz, Gaussian 1.8-4.1 Hz)
    V.append(_ladder('broadening-voigt', 'line_broadening', 'module', times(0.5, 1, 2, 4),
                     lambda f: {'line_broadening': {'mode': 'voigt', 'lb_hz': [0.0, 3.3 * f],
                                                    'gb_hz': [0.0, 4.1 * f]}}))
    V.append(_ladder('broadening-lorentzian', 'line_broadening', 'module', times(0.5, 1, 2, 4),
                     lambda f: {'line_broadening': {'mode': 'lorentzian',
                                                    'lb_hz': [0.0, 3.3 * f]}}))
    V.append(_ladder('broadening-gaussian', 'line_broadening', 'module', times(0.5, 1, 2, 4),
                     lambda f: {'line_broadening': {'mode': 'gaussian', 'gb_hz': [0.0, 4.1 * f]}}))
    V.append(_ladder('broadening-voigt-narrowing', 'line_broadening', 'module', times(0.5, 1, 2, 4),
                     lambda f: {'line_broadening': {'mode': 'voigt', 'lb_hz': [-0.78, 3.3 * f],
                                                    'gb_hz': [0.0, 4.1 * f],
                                                    'narrow_cap_s': 0.2}}))
    V.append(_ladder('broadening-kernel', 'line_broadening', 'module',
                     [(f'spread{_fmt(s)}', s) for s in (0.5, 1, 2, 4)],
                     lambda s: {'line_broadening': {'mode': 'voigt', 'lb_hz': 0.0, 'gb_hz': 0.0,
                                                    'kernel': 'random',
                                                    'kernel_spread_hz': [0.0, float(s)]}}))
    # frequency shift (fitted shift -0.70 to 0.61 Hz) up to Augmentrum's typical +-40 Hz scale
    V.append(_ladder('frequency_shift', 'frequency_shift', 'module', times(0.5, 1, 2, 4, 8, 16),
                     lambda f: {'frequency_shift': {'shift_hz': [-0.69 * f, 0.69 * f]}}))
    # phases (phi0 spans 25.5 deg, phi1 about 180 deg)
    V.append(_ladder('phase_shift', 'phase_shift', 'module', times(0.5, 1, 2, 4),
                     lambda f: {'phase_shift': {'zero_order_deg': [-min(14.0 * f, 180.0),
                                                                   min(14.0 * f, 180.0)],
                                                'first_order_deg': [-90.0 * f, 90.0 * f]}}))
    # macromolecules (fitted MM + MM09 peak 0.13-0.29 of ref)
    for src in ('semi_parametrized', 'parametrized', 'measured'):
        V.append(_ladder(f'macromolecules-{src.replace("_", "")}', 'macromolecules', 'module',
                         times(0.5, 1, 2, 4) if src != 'measured' else times(1, 2),
                         lambda f, src=src: {'macromolecules': {'mm_source': src,
                                                                'mm_scale': [0.0, 0.15 * f]}}))
    # residual water (0.23-1.06 of ref, any phase)
    for model in ('lobes', 'turco'):
        V.append(_ladder(f'residual_water-{model}', 'residual_water', 'module',
                         times(0.25, 0.5, 1, 2),
                         lambda f, model=model: {'residual_water': {
                             'model': model, 'amplitude_scale': [0.0, 0.83 * f],
                             'phase_deg': [-180.0, 180.0]}}))
    # baselines (fitted poly-2 baseline 0.016-0.12 of ref, any phase)
    for mode in ('bspline', 'polynomial', 'random_walk'):
        V.append(_ladder(f'baseline-{mode.replace("_", "")}', 'baseline', 'module',
                         times(0.25, 0.5, 1, 2),
                         lambda f, mode=mode: {'baseline': {'mode': mode,
                                                            'baseline_frac': [0.0, 0.10 * f],
                                                            'phase_deg': [-180.0, 180.0]}}))
    # artificial (lipid / MM-region) peaks: largest unexplained residual <= 0.05 of ref
    V.append(_ladder('artificial_peaks', 'artificial_peaks', 'module', times(0.5, 1, 2, 4),
                     lambda f: {'artificial_peaks': {'peaks': [
                         {'ppm': [0.5, 1.75], 'amp': [0.0, 0.05 * f], 'lb_hz': [5.0, 20.0],
                          'gb_hz': 0.0, 'phase_deg': 0.0}]}}))
    V.append(_ladder('artificial_peaks-voigt-phase', 'artificial_peaks', 'module', times(0.5, 1, 2, 4),
                     lambda f: {'artificial_peaks': {'peaks': [
                         {'ppm': [0.5, 1.75], 'amp': [0.0, 0.05 * f], 'lb_hz': [5.0, 20.0],
                          'gb_hz': [0.0, 10.0], 'phase_deg': [-180.0, 180.0]}]}}))
    # eddy currents (uncorrected waters: strength 0-2)
    V.append(_ladder('eddy_current', 'eddy_current', 'module', times(0.25, 0.5, 1, 2),
                     lambda f: {'eddy_current': {'mode': 'synthetic', 'strength': [0.0, 2.0 * f]}}))
    # spurious echoes (none visible in vivo: <= 1 % of max |FID|)
    V.append(_ladder('spurious_echoes-echo', 'spurious_echoes', 'module',
                     [(f'amp{_fmt(a)}', a) for a in (0.01, 0.03, 0.1, 0.2)],
                     lambda a: {'spurious_echoes': {'mode': 'echo', 'echoes': [
                         {'t_echo_frac': [0.1, 0.9], 'T2': [0.01, 0.05], 'ppm': [0.0, 8.0],
                          'phase_deg': [0.0, 360.0], 'amp': [0.0, a]}]}}))
    V.append(_ladder('spurious_echoes-replica', 'spurious_echoes', 'module',
                     [(f'amp{_fmt(a)}', a) for a in (0.01, 0.03, 0.1, 0.2)],
                     lambda a: {'spurious_echoes': {'mode': 'replica', 'echoes': [
                         {'delay_s': [0.02, 0.3], 'amp': [0.0, a], 'decay_hz': [2.0, 10.0],
                          'phase_deg': [0.0, 360.0]}]}}))
    # a shorter acquisition: the FID truncated (kept fraction drawn from [lo, 1]) and zero-filled
    # back to the network's length (truncate alone shortens the FID)
    V.append(_ladder('apodization-truncate', 'apodization', 'module',
                     [(f'keep{_fmt(k)}', k) for k in (0.9, 0.75, 0.5, 0.25)],
                     lambda k: [{'apodization': {'mode': 'truncate', 'frac_pts': [k, 1.0]}},
                                {'zero_fill': {'target_pts': 2048}}]))
    return V


def spec(name, samplers=(), modules=()):
    return dict(name=name, samplers=list(samplers), modules=list(modules))


def stage_a():
    """Every variant x strength: [spec]."""
    out = []
    for v in variants():
        for label, entry in v['levels']:
            name = f"{v['name']}-{label}"
            entries = entry if isinstance(entry, list) else [entry]
            out.append(dict(spec(name, *((entries, []) if v['kind'] == 'sampler' else
                                         ([], entries))), variant=v['name'], family=v['family'],
                            kind=v['kind'], label=label))
    return out


#*************#
#   running   #
#*************#
def driver_log(*a):
    msg = time.strftime('%Y-%m-%d %H:%M:%S ') + ' '.join(str(x) for x in a)
    print(msg, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, 'a') as f:
        f.write(msg + '\n')


def free_gpu_mb(gpu):
    try:
        out = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.split()[0])
    except Exception:                                   # noqa: BLE001 - no reading: be careful
        return 0


def visible_gpus():
    """The GPUs to use: CUDA_VISIBLE_DEVICES if set, else every GPU nvidia-smi lists."""
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        return [int(g) for g in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    out = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                         capture_output=True, text=True, timeout=30).stdout
    return [int(g) for g in out.split()]


def free_ram_gb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 2 ** 20
    return 0.0


def live_cmds():
    """Command lines (tuples) of the train processes alive now, e.g. left by an
    earlier driver: they are not launched again and count against MAX_PARALLEL."""
    cmds = set()
    for p in glob.glob('/proc/[0-9]*/cmdline'):
        try:
            with open(p, 'rb') as f:
                args = f.read().decode(errors='replace').split('\0')[:-1]
        except OSError:
            continue
        if len(args) > 2 and args[1] == 'scripts/svs_ablation.py' and args[2] == 'train':
            cmds.add(tuple(args))
    return cmds


def screen_run_id(name, n, fold=0, seed=0):
    return f'{name}__n{n}__f{fold}__A__s{seed}'


def result_path(out, name, n, fold=0, seed=0):
    return os.path.join(out, 'runs', screen_run_id(name, n, fold, seed), 'result.json')


def finished(path, steps=None):
    """A run has result.json at *path* (and, with *steps*, has trained that many steps)."""
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        return steps is None or json.load(f)['steps'] >= steps


def job_cmd(sp, n, out, steps, eval_every, extra=(), fold=0, seed=0):
    os.makedirs(os.path.join(out, 'specs'), exist_ok=True)
    common = ['--n-subjects', str(n), '--fold', str(fold), '--variant', 'A', '--seed', str(seed),
              '--max-steps', str(steps), '--eval-every', str(eval_every), '--precision', 'single',
              '--testset', TESTSET, '--selection-set', SELECTION, '--out', out, '--wandb',
              *extra]
    if sp.get('builtin'):
        return [PY, 'scripts/svs_ablation.py', 'train', '--condition', sp['name'], *common]
    path = os.path.join(out, 'specs', f"{sp['name']}.json")
    with open(path, 'w') as f:
        json.dump({k: sp[k] for k in ('name', 'samplers', 'modules')}, f, indent=1)
    return [PY, 'scripts/svs_ablation.py', 'train', '--augment', path, *common]


def run_queue(jobs, label, stagger=30, steps=None, gpus=None, per_gpu=None):
    """
    jobs: [(run id, cmd, out)]. At most *per_gpu* at once on each of *gpus* (default: every
    visible GPU, one CPU core per run up to MAX_PARALLEL per GPU), each launched on the least
    busy GPU with MIN_FREE_MB free, while MIN_FREE_RAM_GB are free, *stagger* s after the previous
    one (runs allocate memory while they set up); finished runs (result.json; with *steps*,
    trained that many steps) are skipped. Jobs already running (an earlier driver's) are adopted:
    waited for, counted against the slots, their result.json read as the exit (0 if present,
    else 1). Returns {run id: exit code}.
    """
    gpus = gpus or visible_gpus()
    per_gpu = per_gpu or min(MAX_PARALLEL, max(1, os.cpu_count() // len(gpus)))
    done = lambda j: finished(os.path.join(j[2], 'runs', j[0], 'result.json'), steps)
    todo = [j for j in jobs if not done(j)]
    live = live_cmds()
    adopted = {j[0]: j for j in todo if tuple(j[1]) in live}
    todo = [j for j in todo if j[0] not in adopted]
    driver_log(f'{label}: {len(jobs)} jobs, {len(jobs) - len(todo) - len(adopted)} already finished, '
        f'{len(adopted)} already running; GPUs {gpus}, {per_gpu} runs each')
    running, codes, last = {}, {}, 0.0
    env = dict(os.environ, WANDB_MODE='offline', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               CUDA_DEVICE_ORDER='PCI_BUS_ID')                  # nvidia-smi's numbering
    status = lambda: (f'({len(codes)}/{len(jobs)} done, {len(running) + len(adopted)} running, '
                      f'{len(todo)} queued)')
    while todo or running or adopted:
        for rid, (proc, fh, gpu) in list(running.items()):
            if proc.poll() is not None:
                fh.close()
                codes[rid] = proc.returncode
                del running[rid]
                driver_log(f'{label}: {rid} exit {proc.returncode} {status()}')
        if adopted:
            live = live_cmds()
            for rid, j in list(adopted.items()):
                if tuple(j[1]) not in live:
                    del adopted[rid]
                    codes[rid] = 0 if done(j) else 1
                    driver_log(f'{label}: {rid} exit {codes[rid]} (adopted) {status()}')
        if (todo and len(running) + len(adopted) < len(gpus) * per_gpu
                and time.time() - last > stagger and free_ram_gb() > MIN_FREE_RAM_GB):
            load = {g: sum(r[2] == g for r in running.values()) for g in gpus}
            free = [g for g in sorted(gpus, key=load.get)
                    if load[g] < per_gpu and free_gpu_mb(g) > MIN_FREE_MB]
            if free:
                rid, cmd, out = todo.pop(0)
                os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
                fh = open(os.path.join(out, 'logs', f'{rid}.log'), 'a')
                running[rid] = (subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                                 env=dict(env, CUDA_VISIBLE_DEVICES=str(free[0])),
                                                 start_new_session=True), fh, free[0])
                last = time.time()
        time.sleep(10)
    return codes


def best_sel(out, name, n):
    p = result_path(out, name, n)
    if not os.path.isfile(p):
        return np.nan
    with open(p) as f:
        return float(json.load(f)['selected_mosae'])


def pick(jobs_a):
    """Per variant the strength with the lowest selection MOSAE, mean over N_SUBJECTS."""
    out_a = os.path.join(SCREEN, 'A')
    picks = {}
    for v in sorted({j['variant'] for j in jobs_a}):
        rows = []
        for j in (j for j in jobs_a if j['variant'] == v):
            s = [best_sel(out_a, j['name'], n) for n in N_SUBJECTS]
            rows.append((float(np.mean(s)) if np.all(np.isfinite(s)) else np.inf, j, s))
        score, j, s = min(rows, key=lambda r: r[0])
        if np.isfinite(score):
            picks[v] = dict(label=j['label'], name=j['name'], family=j['family'], kind=j['kind'],
                            sel_mean=score, sel=dict(zip(map(str, N_SUBJECTS), s)),
                            ladder={r[1]['label']: r[0] for r in rows},
                            samplers=j['samplers'], modules=j['modules'])
    with open(os.path.join(SCREEN, 'picks.json'), 'w') as f:
        json.dump(picks, f, indent=1)
    return picks


def stage_b(picks):
    """The long runs: none, each variant at its best, sampling / all at the best, the old 'all'."""
    out = [dict(name='none', builtin=True), dict(name='all', builtin=True)]
    for v, p in picks.items():
        out.append(spec(p['name'], p['samplers'], p['modules']))
    # per family the best variant (lowest selection MOSAE), all families together
    fam = {}
    for v, p in picks.items():
        if p['family'] not in fam or p['sel_mean'] < fam[p['family']]['sel_mean']:
            fam[p['family']] = p
    samp = [e for f in ('coil_sampling', 'average_sampling') if f in fam for e in fam[f]['samplers']]
    mods = [e for f, p in sorted(fam.items()) if p['kind'] == 'module' for e in p['modules']]
    out.append(spec('sampling-best', samp, []))
    out.append(spec('all-best', samp, mods))
    return out


def run(args):
    os.makedirs(SCREEN, exist_ok=True)
    jobs_a = stage_a()
    with open(os.path.join(SCREEN, 'plan.json'), 'w') as f:
        json.dump(jobs_a, f, indent=1)
    # smoke: every condition on 1 subject for a few steps; drop what fails
    out_s = os.path.join(SCREEN, 'smoke')
    codes = run_queue([(screen_run_id(j['name'], 1), job_cmd(j, 1, out_s, SMOKE_STEPS, 100), out_s)
                       for j in jobs_a], 'smoke', stagger=15)
    bad = {j['name'] for j in jobs_a if codes.get(screen_run_id(j['name'], 1), 0) != 0}
    bad |= {j['name'] for j in jobs_a if not os.path.isfile(result_path(out_s, j['name'], 1))}
    if bad:
        driver_log(f'smoke: {len(bad)} failed and are dropped: {sorted(bad)}')
    jobs_a = [j for j in jobs_a if j['name'] not in bad]
    out_a = os.path.join(SCREEN, 'A')
    run_queue([(screen_run_id(j['name'], n), job_cmd(j, n, out_a, STEPS_A, EVAL_A), out_a)
               for n in N_SUBJECTS for j in jobs_a], 'A')
    picks = pick(jobs_a)
    driver_log('picks: ' + ', '.join(f"{v}={p['label']} ({p['sel_mean']:.3f})" for v, p in picks.items()))
    out_b = os.path.join(SCREEN, 'B')
    jobs_b = stage_b(picks)
    with open(os.path.join(SCREEN, 'plan_B.json'), 'w') as f:
        json.dump(jobs_b, f, indent=1)
    # longest first over every n (user, 2026-09-28): the slow sampler runs start early and the
    # fast ones fill the free slots; the hours are the finished runs' longest, unknown first
    hours = {j['name']: max((json.load(open(p))['timing']['total_s'] / 3600 for p in
                             glob.glob(os.path.join(out_b, 'runs', f"{j['name']}__n*", 'result.json'))),
                            default=np.inf) for j in jobs_b}
    keep = ('--keep-last', '--checkpoint-every', str(CKPT_B))
    run_queue([(screen_run_id(j['name'], n), job_cmd(j, n, out_b, STEPS_B, EVAL_B, keep), out_b)
               for j in sorted(jobs_b, key=lambda j: -hours[j['name']]) for n in B_SUBJECTS], 'B')
    driver_log('all stages done')


#************#
#   extend   #
#************#
def extend(args):
    """Continue stage B runs (last.pt) to --steps, longest first; the same queue as run."""
    out_b = os.path.join(SCREEN, 'B')
    with open(os.path.join(SCREEN, 'plan_B.json')) as f:
        jobs_b = [j for j in json.load(f) if not args.names or j['name'] in args.names]
    if args.names and len(jobs_b) != len(set(args.names)):
        raise SystemExit(f'not in stage B: {sorted(set(args.names) - {j["name"] for j in jobs_b})}')
    hours = {j['name']: max(json.load(open(result_path(out_b, j['name'], n)))['timing']['total_s']
                            for n in args.n) for j in jobs_b}
    keep = ('--keep-last', '--checkpoint-every', str(CKPT_B), '--extend')
    run_queue([(screen_run_id(j['name'], n), job_cmd(j, n, out_b, args.steps, EVAL_B, keep), out_b)
               for j in sorted(jobs_b, key=lambda j: -hours[j['name']]) for n in args.n],
              f'extend {args.steps:,}', steps=args.steps)
    driver_log('extend done')


#**********#
#   grid   #
#**********#
def grid_specs():
    """GRID's specs: stage A's by name, none, and all-best as the union of ALL_BEST."""
    a = {j['name']: j for j in stage_a()}
    special = {'none': dict(name='none', builtin=True),
               'all-best': spec('all-best', [e for c in ALL_BEST for e in a[c]['samplers']],
                                [e for c in ALL_BEST for e in a[c]['modules']])}
    return [special[c] if c in special else a[c] for c in GRID]


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2 ** 24), b''):
            h.update(block)
    return h.hexdigest()


def fetch():
    """The raw scans (sub-XX/mrs/sourcedata/*.dat) from OpenNeuro, once, each checked by its md5
    (S3 ETag); REPAIRED comes with the bundle, never from OpenNeuro."""
    ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    for i in range(1, 11):
        prefix = f'ds006812/sub-{i:02d}/mrs/sourcedata/'
        with urllib.request.urlopen(f'{OPENNEURO}?list-type=2&prefix={prefix}', timeout=60) as r:
            listing = ET.fromstring(r.read())
        for c in listing.findall('s3:Contents', ns):
            rel, size, etag = (c.findtext(f's3:{k}', namespaces=ns)
                               for k in ('Key', 'Size', 'ETag'))
            rel, size, etag = rel[len('ds006812/'):], int(size), etag.strip('"')
            path = os.path.join(DATA_DIR, rel)
            if (not rel.endswith('.dat') or rel in REPAIRED
                    or (os.path.isfile(path) and os.path.getsize(path) == size)):
                continue
            driver_log(f'fetch: {rel} ({size / 2 ** 20:.0f} MB)')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(f'{OPENNEURO}/ds006812/{rel}', path + '.part')
            # the ETag is the md5 unless the upload was multipart ('-' in it)
            if (os.path.getsize(path + '.part') != size
                    or ('-' not in etag and md5(path + '.part') != etag)):
                raise SystemExit(f'fetch: {rel} arrived damaged; run again')
            os.replace(path + '.part', path)
    for rel, digest in REPAIRED.items():
        if not os.path.isfile(os.path.join(DATA_DIR, rel)) or md5(os.path.join(DATA_DIR, rel)) != digest:
            raise SystemExit(f'{rel} is not the header-repaired copy: '
                             'untar cows_grid_bundle.tar.gz in the Augmentrum root')


def grid(args):
    """GRID x 1-8 subjects x GRID_FOLDS x GRID_SEEDS for GRID_STEPS on every visible GPU."""
    global LOG
    LOG = os.path.join(OUT_GRID, 'grid.log')
    specs, gpus = grid_specs(), args.gpus or visible_gpus()
    runs = [(sp, n, fold, seed) for sp in specs for fold in GRID_FOLDS for seed in GRID_SEEDS
            for n in B_SUBJECTS]
    print(f'{len(specs)} conditions x {len(B_SUBJECTS)} subject counts x {len(GRID_FOLDS)} folds x '
          f'{len(GRID_SEEDS)} seeds = {len(runs)} runs of {GRID_STEPS:,} steps, on GPUs {gpus}')
    if args.dry_run:
        for sp in specs:
            print(f"  {sp['name']}: {json.dumps({k: sp.get(k) for k in ('samplers', 'modules')})}")
        return
    missing = [p for p in (BASIS_DIR, TESTSET, SELECTION) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f'missing {missing}: untar cows_grid_bundle.tar.gz in the Augmentrum root')
    fetch()
    # the NIfTI cache once, before parallel runs (they would all write it at the same time)
    subprocess.run([PY, '-c', 'import sys; sys.path.insert(0, "scripts"); '
                              'import svs_ablation; svs_ablation.load_scans()'], check=True)
    out_s = os.path.join(OUT_GRID, 'smoke')
    run_queue([(screen_run_id(sp['name'], 1), job_cmd(sp, 1, out_s, SMOKE_STEPS, 100), out_s)
               for sp in specs], 'smoke', stagger=15, gpus=gpus, per_gpu=args.per_gpu)
    bad = [sp['name'] for sp in specs if not os.path.isfile(result_path(out_s, sp['name'], 1))]
    if bad:
        raise SystemExit(f'smoke failed: {bad}; see {out_s}/logs')
    # --extend: a larger GRID_STEPS later continues the finished runs from last.pt
    keep = ('--keep-last', '--checkpoint-every', str(CKPT_B), '--extend')
    run_queue([(screen_run_id(sp['name'], n, fold, seed),
                job_cmd(sp, n, OUT_GRID, GRID_STEPS, EVAL_B, keep, fold, seed), OUT_GRID)
               for sp, n, fold, seed in runs],
              f'grid {GRID_STEPS:,}', steps=GRID_STEPS, gpus=gpus, per_gpu=args.per_gpu)
    left = [r for r in runs
            if not finished(result_path(OUT_GRID, r[0]['name'], *r[1:]), GRID_STEPS)]
    driver_log(f'grid done: {len(runs) - len(left)}/{len(runs)} runs at {GRID_STEPS:,} steps'
        + (f'; rerun for {len(left)} unfinished' if left else ''))


#************#
#   report   #
#************#
def report(args):
    """Stage A ladders and stage B results (selection / test MOSAE, test mean CCC per n), from what exists."""
    def test_sel(out, name, n):
        p = result_path(out, name, n)
        if not os.path.isfile(p):
            return None
        with open(p) as f:
            r = json.load(f)
        t = r['test']['selected']
        return r['selected_mosae'], t['mosae'], t['ccc_mean'], r['selected_step']
    out_a = os.path.join(SCREEN, 'A')
    print(f"{'stage A condition':44s}" + ''.join(f'   n{n}: sel / test / CCC @ step  ' for n in N_SUBJECTS))
    for j in stage_a():
        cells = []
        for n in N_SUBJECTS:
            r = test_sel(out_a, j['name'], n)
            cells.append(f'{r[0]:6.3f} / {r[1]:5.3f} / {r[2]:5.3f} @ {r[3] / 1e3:4.0f}k' if r else ' ' * 32)
        if any(c.strip() for c in cells):
            print(f"{j['name']:44s}" + ''.join(f'   {c}' for c in cells))
    pb = os.path.join(SCREEN, 'plan_B.json')
    if os.path.isfile(pb):
        print(f"\n{'stage B: test MOSAE / CCC':36s}" + ''.join(f'{f"n{n}":>13s}' for n in B_SUBJECTS))
        with open(pb) as f:
            for j in json.load(f):
                cells = []
                for n in B_SUBJECTS:
                    r = test_sel(os.path.join(SCREEN, 'B'), j['name'], n)
                    cells.append(f'{r[1]:.3f}/{r[2]:.3f}' if r else '')
                print(f"{j['name']:36s}" + ''.join(f'{c:>13s}' for c in cells))


def plan(args):
    os.makedirs(SCREEN, exist_ok=True)
    jobs = stage_a()
    with open(os.path.join(SCREEN, 'plan.json'), 'w') as f:
        json.dump(jobs, f, indent=1)
    fams = {}
    for j in jobs:
        fams.setdefault(j['family'], set()).add(j['variant'])
    print(f'{len(jobs)} stage-A conditions x {len(N_SUBJECTS)} subject counts = '
          f'{len(jobs) * len(N_SUBJECTS)} runs of {STEPS_A:,} steps')
    for f, vs in fams.items():
        print(f'  {f}: {", ".join(sorted(vs))}')


#**********#
#   main   #
#**********#
def add_paths(p):
    p.add_argument('--data-dir', default=DATA_DIR, help='OpenNeuro ds006812 root')
    p.add_argument('--basis-dir', default=BASIS_DIR, help='FSL-MRS JSON basis (summed Cr / PCr)')
    p.add_argument('--cache-dir', default=CACHE_DIR, help="Augmentrum's NIfTI cache of the scans")


def main(argv=None):
    ap = argparse.ArgumentParser(description='The single-voxel augmentation ablation on COWS.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    p = sub.add_parser('process', help='the 90 processed scans, the benchmark\'s input')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default=device)
    add_paths(p)

    p = sub.add_parser('ranges', help="every module's in-vivo spread and proposed range")
    p.add_argument('--fits', required=True, help="invivo's output folder")
    p.add_argument('--processed', required=True, help="'process' output")
    p.add_argument('--out', required=True, help='JSON of the table')
    add_paths(p)

    p = sub.add_parser('testset', help='simulate a test set from the benchmark\'s in-vivo fits')
    p.add_argument('--rows', required=True, help="osprey-rows' (or rows') .npz")
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

    p = sub.add_parser('invivo', help='the 90 processed scans')
    p.add_argument('--processed', required=True, help="process's output")
    p.add_argument('--out', required=True)
    p = sub.add_parser('rows', help="the in-vivo fits as test-set rows, each checked")
    p.add_argument('--fits', required=True, help="invivo's output folder (rows.npz goes there)")
    p.add_argument('--processed', required=True, help="process's output")
    p = sub.add_parser('osprey-rows', help="Osprey's in-vivo fits as test-set rows, each checked")
    p.add_argument('--fits', required=True, help="invivo's output folder (rows_osprey.npz goes there)")
    p.add_argument('--processed', required=True, help="process's output")
    p = sub.add_parser('bench', help='a simulated test set')
    p.add_argument('testset')
    p.add_argument('--out', required=True)
    p = sub.add_parser('challenge', help='the ISMRM 2016 fitting challenge')
    p.add_argument('--data', required=True, help='the challenge folder (datasets_text, ...)')
    p.add_argument('--truth', required=True, help='the ground-truth .xlsx folder')
    p.add_argument('--out', required=True)
    p = sub.add_parser('check', help="model PB's analytic gradient against finite differences")
    for name in FIT_COMMANDS:
        p = sub.choices[name]
        p.add_argument('--basis-dir', default=BASIS_DIR)
        p.add_argument('--workers', type=int, default=MAX_WORKERS)
        p.add_argument('--methods', nargs='+', default=list(METHODS), choices=METHODS)
    sub.add_parser('pipeline', help='the data, end to end: process, invivo, osprey-rows, '
                                    'testset (seeds 0, 1), bench, the in-vivo figure')

    p = sub.add_parser('screen', help='the augmentation screen: plan, run, report, extend')
    p.add_argument('stage', choices=('plan', 'run', 'report', 'extend'))
    p.add_argument('names', nargs='*', help='extend: stage B conditions (default: all)')
    p.add_argument('--n', type=int, nargs='+', default=list(B_SUBJECTS), help='extend: subjects')
    p.add_argument('--steps', type=int, default=10_000_000, help='extend: the new budget')
    p = sub.add_parser('grid', help='the cross-validated grid on every visible GPU')
    p.add_argument('--gpus', type=int, nargs='+', help='GPU indices (default: all visible)')
    p.add_argument('--per-gpu', type=int, help='runs per GPU (default: one per CPU core, '
                                               f'at most {MAX_PARALLEL})')
    p.add_argument('--dry-run', action='store_true', help='print the plan only')

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
    elif args.cmd == 'score':
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
    elif args.cmd == 'invivo':
        invivo(args.processed, args.out, args.basis_dir, args.workers, args.methods)
    elif args.cmd == 'rows':
        tool_rows(args.fits, args.processed, args.basis_dir, args.workers)
    elif args.cmd == 'osprey-rows':
        osprey_rows(args.fits, args.processed, args.basis_dir)
    elif args.cmd == 'bench':
        bench_testset(args.testset, args.out, args.basis_dir, args.workers, args.methods)
    elif args.cmd == 'challenge':
        challenge(args.data, args.truth, args.out, args.workers, args.methods)
    elif args.cmd == 'check':
        print(f'PB gradient, largest relative error: {check_pb_gradient(args.basis_dir):.2e}')
    elif args.cmd == 'pipeline':
        pipeline()
    elif args.cmd == 'screen':
        {'plan': plan, 'run': run, 'report': report, 'extend': extend}[args.stage](args)
    elif args.cmd == 'grid':
        grid(args)


if __name__ == '__main__':
    sys.exit(main())
