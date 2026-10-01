####################################################################################################
#                                          cows_screen.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-25                                                                              #
#                                                                                                  #
# Purpose: The COWS augmentation screen: every Augmentrum augmentation for single-                 #
#          voxel 1H MRS over a ladder of strengths (stage A), each at its best                     #
#          strength for the long runs at 1-8 training subjects (stage B), the                      #
#          extension of the stage-B runs to a larger budget, and the cross-                        #
#          validated grid of the main conditions on every visible GPU.                             #
#                                                                                                  #
####################################################################################################

"""
The COWS augmentation screen: every augmentation Augmentrum offers for single-voxel 1H MRS, each
over a ladder of strengths, then each at its best strength for the long runs.

Stages (all model A, fold 0, seed 0, 1 and 8 training subjects, the Osprey test / selection sets):
    smoke  every stage-A condition for SMOKE_STEPS on 1 subject: does it run at all
    A      every variant x strength for STEPS_A (300k: the early ranking of conditions predicts the
           2M one from ~200-300k on, rank correlation 0.86-0.97; at 10k it is 0.26-0.58)
    pick   per variant the strength with the lowest selection-set MOSAE (mean over 1 and 8
           subjects); the test set is never used to choose
    B      none, each variant at its best strength, sampling / all at the best strengths, and the
           old 'all', at 1-8 subjects, for STEPS_B with --keep-last (extendable)

Not screened, with the reason: AmplitudeScaling (the network divides each input by its norm, so a
global scale changes nothing), ZeroFill (changes the input length), TransientSynthesizer (the COWS
scans have real transients), exponential / Gaussian Apodization (the same as line broadening).

Usage (from the Augmentrum root, env augmentrum311):
    python scripts/cows_screen.py plan      # writes results/cows/screen/plan.json, prints the counts
    python scripts/cows_screen.py run       # smoke -> A -> pick -> B; resumable, skips finished runs
    python scripts/cows_screen.py report    # tables of what exists
    python scripts/cows_screen.py extend [NAME ...] [--n ...] [--steps 10000000]
                                            # continue stage B runs from last.pt (default: all)
    python scripts/cows_screen.py grid      # the cross-validated grid (below), every visible GPU
W&B logs offline (WANDB_MODE=offline) into each run folder; upload later with `wandb sync`.

The grid: GRID x 1-8 subjects x GRID_FOLDS x GRID_SEEDS for GRID_STEPS, every condition at its
stage-B strength, into results/cows/grid. It downloads the raw COWS scans from OpenNeuro once,
builds the NIfTI cache once, runs every condition for SMOKE_STEPS (stops if one fails), then
trains the grid longest first, at most --per-gpu runs per GPU (default: one CPU core per run, up
to MAX_PARALLEL), each launched while MIN_FREE_MB is free on its GPU. Rerun the same command after
an interruption: finished runs are skipped, interrupted ones resume from last.pt.
On another machine (e.g. Kay's), from the Augmentrum root:
    1. env: Python 3.11, `pip install -e .[torch] fsl-mrs==2.5.0 pymapvbvd==0.6.1 spec2nii==0.8.15
       wandb`; A100: torch 2.6 (cu124) as here; Blackwell: torch >= 2.7 built for CUDA >= 12.8
    2. `tar xzf cows_grid_bundle.tar.gz` (the basis, the test and selection sets, and the
       header-repaired sub-01 acq-06 scan: OpenNeuro's copy has a broken multi-RAID header)
    3. `python scripts/cows_screen.py grid` (log: results/cows/grid/grid.log)
    4. send back results/cows/grid without the checkpoints:
       `tar czf cows_grid_runs.tar.gz --exclude=last.pt --exclude=checkpoints results/cows/grid`
The bundle, made here: `tar czf cows_grid_bundle.tar.gz data/BasisSets/TE26_basis_summed
results/cows/testsets_osprey/test_n1000_s[01].npz data/openneuro_ds006812/<REPAIRED file>`.
"""
import os
import sys
import json
import glob
import time
import hashlib
import argparse
import subprocess
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

OUT = 'results/cows/screen'
TESTSET = 'results/cows/testsets_osprey/test_n1000_s0.npz'
SELECTION = 'results/cows/testsets_osprey/test_n1000_s1.npz'
DATA, BASIS = 'data/openneuro_ds006812', 'data/BasisSets/TE26_basis_summed'   # cows_study's
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
LOG = os.path.join(OUT, 'screen.log')


#**************************************************************************************************#
#                                            the variants                                          #
#**************************************************************************************************#
def _fmt(x):
    return f'{x:g}'.replace('.', 'p').replace('-', 'm')


def _ladder(name, family, kind, levels, build):
    """A variant: its strength ladder as [(label, entry)]."""
    return dict(name=name, family=family, kind=kind,
                levels=[(label, build(v)) for label, v in levels])


F = lambda *fs: [(f'x{_fmt(f)}', f) for f in fs]             # multiples of the in-vivo range


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
    V.append(_ladder('broadening-voigt', 'line_broadening', 'module', F(0.5, 1, 2, 4),
                     lambda f: {'line_broadening': {'mode': 'voigt', 'lb_hz': [0.0, 3.3 * f],
                                                    'gb_hz': [0.0, 4.1 * f]}}))
    V.append(_ladder('broadening-lorentzian', 'line_broadening', 'module', F(0.5, 1, 2, 4),
                     lambda f: {'line_broadening': {'mode': 'lorentzian',
                                                    'lb_hz': [0.0, 3.3 * f]}}))
    V.append(_ladder('broadening-gaussian', 'line_broadening', 'module', F(0.5, 1, 2, 4),
                     lambda f: {'line_broadening': {'mode': 'gaussian', 'gb_hz': [0.0, 4.1 * f]}}))
    V.append(_ladder('broadening-voigt-narrowing', 'line_broadening', 'module', F(0.5, 1, 2, 4),
                     lambda f: {'line_broadening': {'mode': 'voigt', 'lb_hz': [-0.78, 3.3 * f],
                                                    'gb_hz': [0.0, 4.1 * f],
                                                    'narrow_cap_s': 0.2}}))
    V.append(_ladder('broadening-kernel', 'line_broadening', 'module',
                     [(f'spread{_fmt(s)}', s) for s in (0.5, 1, 2, 4)],
                     lambda s: {'line_broadening': {'mode': 'voigt', 'lb_hz': 0.0, 'gb_hz': 0.0,
                                                    'kernel': 'random',
                                                    'kernel_spread_hz': [0.0, float(s)]}}))
    # frequency shift (fitted shift -0.70 to 0.61 Hz) up to Augmentrum's typical +-40 Hz scale
    V.append(_ladder('frequency_shift', 'frequency_shift', 'module', F(0.5, 1, 2, 4, 8, 16),
                     lambda f: {'frequency_shift': {'shift_hz': [-0.69 * f, 0.69 * f]}}))
    # phases (phi0 spans 25.5 deg, phi1 about 180 deg)
    V.append(_ladder('phase_shift', 'phase_shift', 'module', F(0.5, 1, 2, 4),
                     lambda f: {'phase_shift': {'zero_order_deg': [-min(14.0 * f, 180.0),
                                                                   min(14.0 * f, 180.0)],
                                                'first_order_deg': [-90.0 * f, 90.0 * f]}}))
    # macromolecules (fitted MM + MM09 peak 0.13-0.29 of ref)
    for src in ('semi_parametrized', 'parametrized', 'measured'):
        V.append(_ladder(f'macromolecules-{src.replace("_", "")}', 'macromolecules', 'module',
                         F(0.5, 1, 2, 4) if src != 'measured' else F(1, 2),
                         lambda f, src=src: {'macromolecules': {'mm_source': src,
                                                                'mm_scale': [0.0, 0.15 * f]}}))
    # residual water (0.23-1.06 of ref, any phase)
    for model in ('lobes', 'turco'):
        V.append(_ladder(f'residual_water-{model}', 'residual_water', 'module',
                         F(0.25, 0.5, 1, 2),
                         lambda f, model=model: {'residual_water': {
                             'model': model, 'amplitude_scale': [0.0, 0.83 * f],
                             'phase_deg': [-180.0, 180.0]}}))
    # baselines (fitted poly-2 baseline 0.016-0.12 of ref, any phase)
    for mode in ('bspline', 'polynomial', 'random_walk'):
        V.append(_ladder(f'baseline-{mode.replace("_", "")}', 'baseline', 'module',
                         F(0.25, 0.5, 1, 2),
                         lambda f, mode=mode: {'baseline': {'mode': mode,
                                                            'baseline_frac': [0.0, 0.10 * f],
                                                            'phase_deg': [-180.0, 180.0]}}))
    # artificial (lipid / MM-region) peaks: largest unexplained residual <= 0.05 of ref
    V.append(_ladder('artificial_peaks', 'artificial_peaks', 'module', F(0.5, 1, 2, 4),
                     lambda f: {'artificial_peaks': {'peaks': [
                         {'ppm': [0.5, 1.75], 'amp': [0.0, 0.05 * f], 'lb_hz': [5.0, 20.0],
                          'gb_hz': 0.0, 'phase_deg': 0.0}]}}))
    V.append(_ladder('artificial_peaks-voigt-phase', 'artificial_peaks', 'module', F(0.5, 1, 2, 4),
                     lambda f: {'artificial_peaks': {'peaks': [
                         {'ppm': [0.5, 1.75], 'amp': [0.0, 0.05 * f], 'lb_hz': [5.0, 20.0],
                          'gb_hz': [0.0, 10.0], 'phase_deg': [-180.0, 180.0]}]}}))
    # eddy currents (uncorrected waters: strength 0-2)
    V.append(_ladder('eddy_current', 'eddy_current', 'module', F(0.25, 0.5, 1, 2),
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


#**************************************************************************************************#
#                                               running                                            #
#**************************************************************************************************#
def log(*a):
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
    """Command lines (tuples) of the cows_study.py train processes alive now, e.g. left by an
    earlier driver: they are not launched again and count against MAX_PARALLEL."""
    cmds = set()
    for p in glob.glob('/proc/[0-9]*/cmdline'):
        try:
            with open(p, 'rb') as f:
                args = f.read().decode(errors='replace').split('\0')[:-1]
        except OSError:
            continue
        if len(args) > 2 and args[1] == 'scripts/cows_study.py' and args[2] == 'train':
            cmds.add(tuple(args))
    return cmds


def run_id(name, n, fold=0, seed=0):
    return f'{name}__n{n}__f{fold}__A__s{seed}'


def result_path(out, name, n, fold=0, seed=0):
    return os.path.join(out, 'runs', run_id(name, n, fold, seed), 'result.json')


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
        return [PY, 'scripts/cows_study.py', 'train', '--condition', sp['name'], *common]
    path = os.path.join(out, 'specs', f"{sp['name']}.json")
    with open(path, 'w') as f:
        json.dump({k: sp[k] for k in ('name', 'samplers', 'modules')}, f, indent=1)
    return [PY, 'scripts/cows_study.py', 'train', '--augment', path, *common]


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
    log(f'{label}: {len(jobs)} jobs, {len(jobs) - len(todo) - len(adopted)} already finished, '
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
                log(f'{label}: {rid} exit {proc.returncode} {status()}')
        if adopted:
            live = live_cmds()
            for rid, j in list(adopted.items()):
                if tuple(j[1]) not in live:
                    del adopted[rid]
                    codes[rid] = 0 if done(j) else 1
                    log(f'{label}: {rid} exit {codes[rid]} (adopted) {status()}')
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
    out_a = os.path.join(OUT, 'A')
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
    with open(os.path.join(OUT, 'picks.json'), 'w') as f:
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
    os.makedirs(OUT, exist_ok=True)
    jobs_a = stage_a()
    with open(os.path.join(OUT, 'plan.json'), 'w') as f:
        json.dump(jobs_a, f, indent=1)
    # smoke: every condition on 1 subject for a few steps; drop what fails
    out_s = os.path.join(OUT, 'smoke')
    codes = run_queue([(run_id(j['name'], 1), job_cmd(j, 1, out_s, SMOKE_STEPS, 100), out_s)
                       for j in jobs_a], 'smoke', stagger=15)
    bad = {j['name'] for j in jobs_a if codes.get(run_id(j['name'], 1), 0) != 0}
    bad |= {j['name'] for j in jobs_a if not os.path.isfile(result_path(out_s, j['name'], 1))}
    if bad:
        log(f'smoke: {len(bad)} failed and are dropped: {sorted(bad)}')
    jobs_a = [j for j in jobs_a if j['name'] not in bad]
    out_a = os.path.join(OUT, 'A')
    run_queue([(run_id(j['name'], n), job_cmd(j, n, out_a, STEPS_A, EVAL_A), out_a)
               for n in N_SUBJECTS for j in jobs_a], 'A')
    picks = pick(jobs_a)
    log('picks: ' + ', '.join(f"{v}={p['label']} ({p['sel_mean']:.3f})" for v, p in picks.items()))
    out_b = os.path.join(OUT, 'B')
    jobs_b = stage_b(picks)
    with open(os.path.join(OUT, 'plan_B.json'), 'w') as f:
        json.dump(jobs_b, f, indent=1)
    # longest first over every n (user, 2026-09-28): the slow sampler runs start early and the
    # fast ones fill the free slots; the hours are the finished runs' longest, unknown first
    hours = {j['name']: max((json.load(open(p))['timing']['total_s'] / 3600 for p in
                             glob.glob(os.path.join(out_b, 'runs', f"{j['name']}__n*", 'result.json'))),
                            default=np.inf) for j in jobs_b}
    keep = ('--keep-last', '--checkpoint-every', str(CKPT_B))
    run_queue([(run_id(j['name'], n), job_cmd(j, n, out_b, STEPS_B, EVAL_B, keep), out_b)
               for j in sorted(jobs_b, key=lambda j: -hours[j['name']]) for n in B_SUBJECTS], 'B')
    log('all stages done')


#*************#
#   extend    #
#*************#
def extend(args):
    """Continue stage B runs (last.pt) to --steps, longest first; the same queue as run."""
    out_b = os.path.join(OUT, 'B')
    with open(os.path.join(OUT, 'plan_B.json')) as f:
        jobs_b = [j for j in json.load(f) if not args.names or j['name'] in args.names]
    if args.names and len(jobs_b) != len(set(args.names)):
        raise SystemExit(f'not in stage B: {sorted(set(args.names) - {j["name"] for j in jobs_b})}')
    hours = {j['name']: max(json.load(open(result_path(out_b, j['name'], n)))['timing']['total_s']
                            for n in args.n) for j in jobs_b}
    keep = ('--keep-last', '--checkpoint-every', str(CKPT_B), '--extend')
    run_queue([(run_id(j['name'], n), job_cmd(j, n, out_b, args.steps, EVAL_B, keep), out_b)
               for j in sorted(jobs_b, key=lambda j: -hours[j['name']]) for n in args.n],
              f'extend {args.steps:,}', steps=args.steps)
    log('extend done')


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
            path = os.path.join(DATA, rel)
            if (not rel.endswith('.dat') or rel in REPAIRED
                    or (os.path.isfile(path) and os.path.getsize(path) == size)):
                continue
            log(f'fetch: {rel} ({size / 2 ** 20:.0f} MB)')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(f'{OPENNEURO}/ds006812/{rel}', path + '.part')
            # the ETag is the md5 unless the upload was multipart ('-' in it)
            if (os.path.getsize(path + '.part') != size
                    or ('-' not in etag and md5(path + '.part') != etag)):
                raise SystemExit(f'fetch: {rel} arrived damaged; run again')
            os.replace(path + '.part', path)
    for rel, digest in REPAIRED.items():
        if not os.path.isfile(os.path.join(DATA, rel)) or md5(os.path.join(DATA, rel)) != digest:
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
    missing = [p for p in (BASIS, TESTSET, SELECTION) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f'missing {missing}: untar cows_grid_bundle.tar.gz in the Augmentrum root')
    fetch()
    # the NIfTI cache once, before parallel runs (they would all write it at the same time)
    subprocess.run([PY, '-c', 'import sys; sys.path.insert(0, "scripts"); '
                              'import cows_study; cows_study.load_scans()'], check=True)
    out_s = os.path.join(OUT_GRID, 'smoke')
    run_queue([(run_id(sp['name'], 1), job_cmd(sp, 1, out_s, SMOKE_STEPS, 100), out_s)
               for sp in specs], 'smoke', stagger=15, gpus=gpus, per_gpu=args.per_gpu)
    bad = [sp['name'] for sp in specs if not os.path.isfile(result_path(out_s, sp['name'], 1))]
    if bad:
        raise SystemExit(f'smoke failed: {bad}; see {out_s}/logs')
    # --extend: a larger GRID_STEPS later continues the finished runs from last.pt
    keep = ('--keep-last', '--checkpoint-every', str(CKPT_B), '--extend')
    run_queue([(run_id(sp['name'], n, fold, seed),
                job_cmd(sp, n, OUT_GRID, GRID_STEPS, EVAL_B, keep, fold, seed), OUT_GRID)
               for sp, n, fold, seed in runs],
              f'grid {GRID_STEPS:,}', steps=GRID_STEPS, gpus=gpus, per_gpu=args.per_gpu)
    left = [r for r in runs
            if not finished(result_path(OUT_GRID, r[0]['name'], *r[1:]), GRID_STEPS)]
    log(f'grid done: {len(runs) - len(left)}/{len(runs)} runs at {GRID_STEPS:,} steps'
        + (f'; rerun for {len(left)} unfinished' if left else ''))


#**************************************************************************************************#
#                                               report                                             #
#**************************************************************************************************#
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
    out_a = os.path.join(OUT, 'A')
    print(f"{'stage A condition':44s}" + ''.join(f'   n{n}: sel / test / CCC @ step  ' for n in N_SUBJECTS))
    for j in stage_a():
        cells = []
        for n in N_SUBJECTS:
            r = test_sel(out_a, j['name'], n)
            cells.append(f'{r[0]:6.3f} / {r[1]:5.3f} / {r[2]:5.3f} @ {r[3] / 1e3:4.0f}k' if r else ' ' * 32)
        if any(c.strip() for c in cells):
            print(f"{j['name']:44s}" + ''.join(f'   {c}' for c in cells))
    pb = os.path.join(OUT, 'plan_B.json')
    if os.path.isfile(pb):
        print(f"\n{'stage B: test MOSAE / CCC':36s}" + ''.join(f'{f"n{n}":>13s}' for n in B_SUBJECTS))
        with open(pb) as f:
            for j in json.load(f):
                cells = []
                for n in B_SUBJECTS:
                    r = test_sel(os.path.join(OUT, 'B'), j['name'], n)
                    cells.append(f'{r[1]:.3f}/{r[2]:.3f}' if r else '')
                print(f"{j['name']:36s}" + ''.join(f'{c:>13s}' for c in cells))


def plan(args):
    os.makedirs(OUT, exist_ok=True)
    jobs = stage_a()
    with open(os.path.join(OUT, 'plan.json'), 'w') as f:
        json.dump(jobs, f, indent=1)
    fams = {}
    for j in jobs:
        fams.setdefault(j['family'], set()).add(j['variant'])
    print(f'{len(jobs)} stage-A conditions x {len(N_SUBJECTS)} subject counts = '
          f'{len(jobs) * len(N_SUBJECTS)} runs of {STEPS_A:,} steps')
    for f, vs in fams.items():
        print(f'  {f}: {", ".join(sorted(vs))}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('cmd', choices=('plan', 'run', 'report', 'extend', 'grid'))
    ap.add_argument('names', nargs='*', help='extend: stage B conditions (default: all)')
    ap.add_argument('--n', type=int, nargs='+', default=list(B_SUBJECTS), help='extend: subjects')
    ap.add_argument('--steps', type=int, default=10_000_000, help='extend: the new budget')
    ap.add_argument('--gpus', type=int, nargs='+', help='grid: GPU indices (default: all visible)')
    ap.add_argument('--per-gpu', type=int, help='grid: runs per GPU (default: one per CPU core, '
                                                f'at most {MAX_PARALLEL})')
    ap.add_argument('--dry-run', action='store_true', help='grid: print the plan only')
    args = ap.parse_args(argv)
    {'plan': plan, 'run': run, 'report': report, 'extend': extend, 'grid': grid}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
