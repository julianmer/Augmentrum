####################################################################################################
#                                        cows_benchmark.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-23                                                                              #
#                                                                                                  #
# Purpose: The fitting tools the COWS study compares the network with: FSL-MRS at its defaults,    #
#          FSL-MRS on the network's model PB, LCModel and Osprey at their defaults, on the 90      #
#          processed in-vivo scans, on the simulated test sets, and on the ISMRM 2016 fitting      #
#          challenge.                                                                              #
#                                                                                                  #
####################################################################################################

"""
FSL-MRS, LCModel and Osprey on the COWS scans, the test sets and the ISMRM 2016 challenge.

Methods (every fit on 0.5-4.2 ppm)
----------------------------------
- fsl_default  FSL-MRS' default model: one Voigt for all basis spectra, a complex
               poly-2 baseline, zero- and first-order phase (fit_FSLModel, Newton).
- fsl_pb       The network's model PB fitted by FSL-MRS' optimiser (scipy TNC, as
               fit_FSLModel's Newton path calls it): the global Voigt plus an extra
               Lorentzian per basis spectrum within Osprey's bounds. It starts from
               the fsl_default fit, which it contains. Written in the network's
               conventions, so its parameters are the network's.
- lcmodel      LCModel at its defaults (PyLCModel), no simulated macromolecules.
- osprey       Osprey at its defaults (MRSuite, Octave in Docker).

Commands
--------
    # the 90 scans ("cows_study.py process" first): concentrations, PB's parameters,
    # the fitted spectra and residual / noise of every method
    python scripts/cows_benchmark.py invivo --processed results/cows/processed_scans.npz \\
        --out results/cows/invivo
    # the fits as test-set rows: each tool's concentrations, the rest fitted to the scan
    # (rows.npz, rows_check.csv; cows_study.py testset reads them)
    python scripts/cows_benchmark.py rows --fits results/cows/invivo \\
        --processed results/cows/processed_scans.npz
    # a simulated test set, scored like the network (MOSAE, CCC)
    python scripts/cows_benchmark.py testset results/cows/testsets/test_n1000_s0.npz \\
        --out results/cows/benchmark
    # the ISMRM 2016 fitting challenge (PyLCModel's example data)
    python scripts/cows_benchmark.py challenge --data <2016_fitting_challenge> \\
        --truth <2016_fitting_challenge_gts> --out results/cows/challenge

Run from the Augmentrum root. Needs FSL-MRS, PyLCModel (which fetches the LCModel binary
itself) and MRSuite for Osprey (Octave in Docker, one container per Python version). Fits
are never cached: every command fits afresh.
"""

#*************#
#   imports   #
#*************#
import os
import sys

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')                # one thread per fitting process
os.environ.setdefault('MRSUITE_OCTAVE_RUNTIME', 'docker')
# MRSuite's Octave container mounts the oct2py of the Python that started it: one per Python
os.environ.setdefault('MRSUITE_OCTAVE_CONTAINER',
                      f'mrsuite-octave-py{sys.version_info.major}{sys.version_info.minor}')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import multiprocessing as mp
import re
import time
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize

import cows_study as S


#**************#
#   settings   #
#**************#
WINDOW = S.PPM_WINDOW
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


#**************************************************************************************************#
#                                              FSL-MRS                                             #
#**************************************************************************************************#
_FSL = {}


def fsl_basis(basis_dir):
    """The JSON basis as FSL-MRS reads it (one per process and folder)."""
    from fsl_mrs.core.basis import Basis
    from fsl_mrs.utils.mrs_io.fsl_io import readFSLBasisFiles
    if basis_dir not in _FSL:
        _FSL[basis_dir] = Basis(*readFSLBasisFiles(basis_dir))
    return _FSL[basis_dir]


def prepare_mrs(fid, cf, bw, basis_dir, ppmlim=WINDOW):
    """An MRS object with FSL-MRS' own preparation: conjugation checks and rescaling."""
    from fsl_mrs.core import MRS
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mrs = MRS(FID=np.asarray(fid, complex), cf=cf, bw=bw, nucleus='1H',
                  basis=fsl_basis(basis_dir))
        mrs.processForFitting(ppmlim=ppmlim)
    return mrs


def fsl_orientation(fid, cf, bw, basis_dir, ppmlim=WINDOW):
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
        return fit_FSLModel(mrs, metab_groups=[0] * len(mrs.names), model='voigt', ppmlim=WINDOW,
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
#   model PB in FSL   #
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
    first = int(np.abs(ppm - WINDOW[0]).argmin())
    last = int(np.abs(ppm - WINDOW[1]).argmin())
    B = S.poly_baseline(T, first, last)
    f = np.fft.fftfreq(T, d=mrs.dwellTime)
    t = np.asarray(mrs.timeAxis).flatten()
    m = np.asarray(mrs.basis)
    data = np.fft.fft(np.asarray(mrs.FID))
    parent = np.asarray(fsl_voigt(mrs).params, float)             # con, gamma, sigma, eps, ...
    x0 = np.r_[parent[:n], np.full(n, parent[n]), parent[n + 1], parent[n + 2], parent[n + 3],
               parent[n + 4], parent[n + 5:]]
    delta_max = np.array([S.PB_DELTA_MAX_MM if nm in cfg['mm_names'] else S.PB_DELTA_MAX
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
    basis = S.load_basis(cfg['basis_dir'])
    scale = float(np.abs(mrs._fid_scaling))
    con = con * data_units(mrs, cfg['basis_dir'])
    b = np.asarray(b) / scale
    order = [names.index(nm) for nm in basis.names]
    con, gamma = con[order], gamma[order]
    Bn = S.poly_baseline(len(fid), *basis.window())
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
    basis = S.load_basis(basis_dir)
    rng = np.random.default_rng(seed)
    n, T = len(basis.names), basis.fids.shape[0]
    first, last = basis.window()
    B = S.poly_baseline(T, first, last)
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


#**************************************************************************************************#
#                                              LCModel                                             #
#**************************************************************************************************#
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
        lcm = PyLCModel(cfg['lcm_basis'], ppmlim=WINDOW, conj=cfg['lcm_conj'], ignore='none',
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


#**************************************************************************************************#
#                                              Osprey                                              #
#**************************************************************************************************#
_W = {}


def _osprey_init(lock, basis_path, fit_range, basis_dir):
    from mrsuite.bridges import osprey
    with lock:                                      # one Octave session start at a time
        osprey.bridge.centre_ppm = OSPREY_CENTRE_PPM
        _W['basis'] = osprey.bridge.basis_from_lcmodel(basis_path, add_mm=False)
    _W.update(osprey=osprey, range=list(fit_range), net=S.load_basis(basis_dir))


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
                  initargs=(lock, cfg['lcm_basis'], WINDOW, cfg['basis_dir'])) as pool:
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


#**************************************************************************************************#
#                                            batch fits                                            #
#**************************************************************************************************#
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


def setup(fid, cf, bw, basis_dir, work, mm_names=S.MM_NAMES):
    """What every fit needs: the basis names, FSL-MRS' orientation, the LCModel basis."""
    conj_fid, conj_basis = fsl_orientation(fid, cf, bw, basis_dir)
    lcm_basis = write_lcm_basis(os.path.join(work, 'basis.BASIS'), basis_dir, conj_basis)
    return dict(cf=cf, bw=bw, basis_dir=basis_dir, names=list(S.load_basis(basis_dir).names),
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
    w = (ppm_grid >= WINDOW[0]) & (ppm_grid <= WINDOW[1])
    out = np.full(len(fids), np.nan)
    for i, (r, fid) in enumerate(zip(results, fids)):
        if r is None:
            continue
        spec = np.fft.fft(fid)
        noise = S.spec_noise_sd(spec, ppm_grid)
        if 'ppm' in r:                              # LCModel / Osprey: their own points
            k = (r['ppm'] >= WINDOW[0]) & (r['ppm'] <= WINDOW[1])
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


#**************************************************************************************************#
#                                   the test sets' rows from the fits                              #
#**************************************************************************************************#
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


def fitted_row(fid, con, start, basis, mm_names=S.MM_NAMES, M=ROW_BASELINE_POINTS):
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
    dmax = np.array([S.PB_DELTA_MAX_MM if nm in mm_names else S.PB_DELTA_MAX
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
    # there, as FSL-MRS' polynomial baseline ("S.poly_baseline")
    curve = np.zeros(T, complex)
    curve[first:last] = np.fft.fft(baseline)[first:last]
    row = dict(con=con * c[0], gamma=x[0] + x[1:n + 1], sigma=float(x[n + 1]),
               eps=np.full(n, float(x[n + 2])), phi0=float(x[n + 3]), phi1=float(x[n + 4]),
               baseline=curve)
    sd = S.spec_noise_sd(spec, basis.ppm)
    r = A @ c - y
    return row, dict(resid_real=float(np.sqrt(np.mean(r[:len(r) // 2] ** 2)) / sd),
                     resid=float(np.sqrt(np.mean(r ** 2)) / sd), factor=float(c[0]),
                     success=bool(res.success))


def _row_task(job):
    fid, con, start, basis_dir = job
    return fitted_row(fid, con, start, S.load_basis(basis_dir))


def tool_rows(fits_dir, processed, basis_dir, workers):
    """
    Every scan as each tool quantified it, as rows ("fitted_row"), with each row's residual /
    noise against its scan: <fits_dir>/rows.npz and rows_check.csv.
    """
    z = np.load(processed)
    zstems = [str(s) for s in z['stems']]
    basis = S.load_basis(basis_dir)
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
    out = {k: np.array([r[k] for r in rows]) for k in S.ROW_KEYS}
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


#**************************************************************************************************#
#                                   the test sets' rows from Osprey                                #
#**************************************************************************************************#
# Osprey as the truth: every scan as Osprey fitted it, in Osprey's own model (fit_OspreyParams-
# ToModel.m): per basis spectrum a Lorentzian and a shift, one Gaussian, the lineshape kernel
# convolved with the metabolites, a cubic-spline baseline; the phases and the reference shift it
# applies to the data are applied to the model instead, so the row lies in the data's frame.
# Osprey models the real part only: the rows' metabolites are complex by construction (the kernel
# as a time-domain factor, "S.simulate_rows"); the baseline is Osprey's in its frame's real part,
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
    Scan *i* as Osprey fitted it (*fit*: invivo_osprey.npz), a row of "S.simulate_rows" with its
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
    metab = S.simulate_rows({key: np.asarray(v)[None] for key, v in row.items()}, basis)[0]
    spl = _spline_design(p_bins, ppm_o[0], ppm_o[-1])
    D = 1j * phase[:, None] * spl
    y = spec[w] - metab[w] - real * phase
    g = np.linalg.lstsq(np.r_[D.real, D.imag], np.r_[y.real, y.imag], rcond=None)[0]
    row['baseline'][w] = (real + 1j * (spl @ g)) * phase
    sd = S.spec_noise_sd(spec, basis.ppm)
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
    basis = S.load_basis(basis_dir)
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
        pad = (S.KERNEL_TAPS - len(r['kernel'])) // 2
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


#**************************************************************************************************#
#                                             commands                                             #
#**************************************************************************************************#
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
        f"{cfg['conj_basis']}; window {WINDOW}")
    ppm_grid = S.load_basis(basis_dir).ppm
    quality = {}
    for method in methods:
        t0 = time.time()
        results = fit_all(method, fids, cfg, workers)
        ok = np.array([r is not None for r in results])
        con = np.array([r['con'] if r is not None else np.full(len(cfg['names']), np.nan)
                        for r in results])
        save = dict(names=np.array(cfg['names']), stems=np.array(stems), con=con, ok=ok,
                    method=method, window=np.array(WINDOW), seconds=time.time() - t0)
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


def testset(path, out, basis_dir, workers, methods=METHODS):
    """A test set fitted by every method, scored like the network: <set>_<method>.npz, scores."""
    ts = S.TestSet(path)
    x = ts.spectra.numpy().astype(np.float64)
    fids = np.fft.ifft(x[:, 0] + 1j * x[:, 1], axis=-1)   # the network's representation, inverted
    basis = S.load_basis(basis_dir)
    os.makedirs(out, exist_ok=True)
    cfg = setup(fids[0], basis.cf, basis.bw, basis_dir, os.path.join(out, 'work'))
    scores = {'no information': S.concentration_metrics(
        np.repeat(ts.concentrations.mean(0, keepdims=True), len(fids), 0), ts.concentrations,
        ts.names)}
    for method in methods:
        results = fit_all(method, fids, cfg, workers)
        con = np.array([r['con'] if r is not None else np.zeros(len(ts.names)) for r in results])
        np.savez(os.path.join(out, f'{ts.name}_{method}.npz'), con=con, names=np.array(ts.names),
                 ok=np.array([r is not None for r in results]), testset_hash=ts.hash)
        scores[method] = S.concentration_metrics(con, ts.concentrations, ts.names)
        log(f"{method}: MOSAE {scores[method]['mosae']:.4f}, mean CCC "
            f"{scores[method]['ccc_mean']:.3f}")
    with open(os.path.join(out, f'{ts.name}_scores.json'), 'w') as fh:
        json.dump(dict(testset=path, hash=ts.hash, scores=scores), fh, indent=1)


#*************************#
#   the ISMRM challenge   #
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
        y_hat = S.optimal_scale(t, con) * con
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


#**********#
#   main   #
#**********#
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('invivo', help='the 90 processed scans')
    p.add_argument('--processed', required=True, help="cows_study.py process's output")
    p.add_argument('--out', required=True)
    p = sub.add_parser('rows', help="the in-vivo fits as test-set rows, each checked")
    p.add_argument('--fits', required=True, help="invivo's output folder (rows.npz goes there)")
    p.add_argument('--processed', required=True, help="cows_study.py process's output")
    p = sub.add_parser('osprey-rows', help="Osprey's in-vivo fits as test-set rows, each checked")
    p.add_argument('--fits', required=True, help="invivo's output folder (rows_osprey.npz goes there)")
    p.add_argument('--processed', required=True, help="cows_study.py process's output")
    p = sub.add_parser('testset', help='a simulated test set')
    p.add_argument('testset')
    p.add_argument('--out', required=True)
    p = sub.add_parser('challenge', help='the ISMRM 2016 fitting challenge')
    p.add_argument('--data', required=True, help='the challenge folder (datasets_text, ...)')
    p.add_argument('--truth', required=True, help='the ground-truth .xlsx folder')
    p.add_argument('--out', required=True)
    p = sub.add_parser('check', help="model PB's analytic gradient against finite differences")
    for p in sub.choices.values():
        p.add_argument('--basis-dir', default=S.BASIS_DIR)
        p.add_argument('--workers', type=int, default=MAX_WORKERS)
        p.add_argument('--methods', nargs='+', default=list(METHODS), choices=METHODS)
    args = ap.parse_args(argv)
    if args.cmd == 'invivo':
        invivo(args.processed, args.out, args.basis_dir, args.workers, args.methods)
    elif args.cmd == 'rows':
        tool_rows(args.fits, args.processed, args.basis_dir, args.workers)
    elif args.cmd == 'osprey-rows':
        osprey_rows(args.fits, args.processed, args.basis_dir)
    elif args.cmd == 'testset':
        testset(args.testset, args.out, args.basis_dir, args.workers, args.methods)
    elif args.cmd == 'challenge':
        challenge(args.data, args.truth, args.out, args.workers, args.methods)
    else:
        print(f'PB gradient, largest relative error: {check_pb_gradient(args.basis_dir):.2e}')


if __name__ == '__main__':
    sys.exit(main())
