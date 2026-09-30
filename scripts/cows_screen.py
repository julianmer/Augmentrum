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
#          strength for the long runs at 1-8 training subjects (stage B), and                      #
#          the extension of the stage-B runs to a larger budget.                                   #
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
W&B logs offline (WANDB_MODE=offline) into each run folder; upload later with `wandb sync`.
"""
import os
import sys
import json
import glob
import time
import argparse
import subprocess

import numpy as np

OUT = 'results/cows/screen'
TESTSET = 'results/cows/testsets_osprey/test_n1000_s0.npz'
SELECTION = 'results/cows/testsets_osprey/test_n1000_s1.npz'
N_SUBJECTS = (1, 8)                 # stage A and the picks
B_SUBJECTS = tuple(range(1, 9))     # stage B: every training-set size (user, 2026-09-27)
SMOKE_STEPS, STEPS_A, EVAL_A = 200, 300_000, 5000
STEPS_B, EVAL_B, CKPT_B = 1_000_000, 1000, 250_000
MAX_PARALLEL = 16                   # the GPU and CPUs are shared (CNM-MUSIC, zea jobs)
MIN_FREE_MB = 6000                  # launch only while this much GPU memory is free ...
MIN_FREE_RAM_GB = 8                 # ... and this much RAM (sampling runs hold raw scans)
PY = sys.executable


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
    with open(os.path.join(OUT, 'screen.log'), 'a') as f:
        f.write(msg + '\n')


def free_gpu_mb():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.split()[0])
    except Exception:                                   # noqa: BLE001 - no reading: be careful
        return 0


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


def result_path(out, name, n):
    return os.path.join(out, 'runs', f'{name}__n{n}__f0__A__s0', 'result.json')


def finished(out, name, n, steps=None):
    """A run has result.json (and, with *steps*, has trained that many steps)."""
    p = result_path(out, name, n)
    if not os.path.isfile(p):
        return False
    with open(p) as f:
        return steps is None or json.load(f)['steps'] >= steps


def job_cmd(sp, n, out, steps, eval_every, extra=()):
    os.makedirs(os.path.join(out, 'specs'), exist_ok=True)
    common = ['--n-subjects', str(n), '--fold', '0', '--variant', 'A', '--seed', '0',
              '--max-steps', str(steps), '--eval-every', str(eval_every), '--precision', 'single',
              '--testset', TESTSET, '--selection-set', SELECTION, '--out', out, '--wandb',
              *extra]
    if sp.get('builtin'):
        return [PY, 'scripts/cows_study.py', 'train', '--condition', sp['name'], *common]
    path = os.path.join(out, 'specs', f"{sp['name']}.json")
    with open(path, 'w') as f:
        json.dump({k: sp[k] for k in ('name', 'samplers', 'modules')}, f, indent=1)
    return [PY, 'scripts/cows_study.py', 'train', '--augment', path, *common]


def run_queue(jobs, label, stagger=30, steps=None):
    """
    jobs: [(name, n, cmd, out)]. At most MAX_PARALLEL at once, each launched only while
    MIN_FREE_MB of GPU memory and MIN_FREE_RAM_GB are free, *stagger* s after the previous
    one (runs allocate memory while they set up); finished runs
    (result.json; with *steps*, trained that many steps) are skipped. Jobs already running (an earlier driver's) are adopted: waited for,
    counted against MAX_PARALLEL, their result.json read as the exit (0 if present, else 1).
    Returns {(name, n): exit code}.
    """
    todo = [j for j in jobs if not finished(j[3], j[0], j[1], steps)]
    live = live_cmds()
    adopted = {(j[0], j[1]): j for j in todo if tuple(j[2]) in live}
    todo = [j for j in todo if (j[0], j[1]) not in adopted]
    log(f'{label}: {len(jobs)} jobs, {len(jobs) - len(todo) - len(adopted)} already finished, '
        f'{len(adopted)} already running')
    running, codes, last = {}, {}, 0.0
    env = dict(os.environ, WANDB_MODE='offline', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    while todo or running or adopted:
        for key, (proc, fh) in list(running.items()):
            if proc.poll() is not None:
                fh.close()
                codes[key] = proc.returncode
                del running[key]
                log(f'{label}: {key[0]} n{key[1]} exit {proc.returncode} '
                    f'({len(codes)}/{len(jobs)} done, {len(running) + len(adopted)} running, '
                    f'{len(todo)} queued)')
        if adopted:
            live = live_cmds()
            for key, j in list(adopted.items()):
                if tuple(j[2]) not in live:
                    del adopted[key]
                    codes[key] = 0 if finished(j[3], j[0], j[1], steps) else 1
                    log(f'{label}: {key[0]} n{key[1]} exit {codes[key]} (adopted) '
                        f'({len(codes)}/{len(jobs)} done, {len(running) + len(adopted)} running, '
                        f'{len(todo)} queued)')
        if (todo and len(running) + len(adopted) < MAX_PARALLEL and time.time() - last > stagger
                and free_gpu_mb() > MIN_FREE_MB and free_ram_gb() > MIN_FREE_RAM_GB):
            name, n, cmd, out = todo.pop(0)
            os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
            fh = open(os.path.join(out, 'logs', f'{name}_n{n}.log'), 'a')
            running[(name, n)] = (subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                                   env=env, start_new_session=True), fh)
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
    codes = run_queue([(j['name'], 1, job_cmd(j, 1, out_s, SMOKE_STEPS, 100), out_s)
                       for j in jobs_a], 'smoke', stagger=15)
    bad = {name for (name, n), c in codes.items() if c != 0}
    bad |= {j['name'] for j in jobs_a if not os.path.isfile(result_path(out_s, j['name'], 1))}
    if bad:
        log(f'smoke: {len(bad)} failed and are dropped: {sorted(bad)}')
    jobs_a = [j for j in jobs_a if j['name'] not in bad]
    out_a = os.path.join(OUT, 'A')
    run_queue([(j['name'], n, job_cmd(j, n, out_a, STEPS_A, EVAL_A), out_a)
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
    run_queue([(j['name'], n, job_cmd(j, n, out_b, STEPS_B, EVAL_B,
                                      ('--keep-last', '--checkpoint-every', str(CKPT_B))), out_b)
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
    run_queue([(j['name'], n, job_cmd(j, n, out_b, args.steps, EVAL_B,
                                      ('--keep-last', '--checkpoint-every', str(CKPT_B), '--extend')),
                out_b)
               for j in sorted(jobs_b, key=lambda j: -hours[j['name']]) for n in args.n],
              f'extend {args.steps:,}', steps=args.steps)
    log('extend done')


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
    ap.add_argument('cmd', choices=('plan', 'run', 'report', 'extend'))
    ap.add_argument('names', nargs='*', help='extend: stage B conditions (default: all)')
    ap.add_argument('--n', type=int, nargs='+', default=list(B_SUBJECTS), help='extend: subjects')
    ap.add_argument('--steps', type=int, default=10_000_000, help='extend: the new budget')
    args = ap.parse_args(argv)
    {'plan': plan, 'run': run, 'report': report, 'extend': extend}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
