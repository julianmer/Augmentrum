####################################################################################################
#                                          svs_figures.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-23                                                                              #
#                                                                                                  #
# Purpose: The figures of the single-voxel augmentation ablation (svs_ablation.py): the paper's,   #
#          from the augmentation screen, the fitting tools and the in-vivo fits, and the earlier   #
#          ones.                                                                                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import svs_ablation as S


#***********#
#   style   #
#***********#
INK = '#333333'                  # phantom_ablation.py's line colour
ACCENT = '#1f8a93'               # Augmentrum, the one thing in colour
ACCENT_LIGHT = '#8fc4c9'         # the same teal, lighter: fewer training subjects
INK_LIGHT = '#999999'
MID = '#7a7a7a'
REF = '#a6a6a6'                  # every reference level
REF_TEXT = '#666666'
TOTALS = ('tNAA', 'tCr', 'tCho', 'Glx', 'mI')
LABELS = {'mI': 'Ins', 'sI': 'sIns'}
TOOLS = {'fsl_default': 'FSL-MRS', 'fsl_pb': 'FSL-MRS PB', 'lcmodel': 'LCModel',
         'osprey': 'Osprey'}
NAMES = {'none': 'No augmentation', 'coil_sampling': 'Coil sampling',
         'average_sampling': 'Transient sampling', 'sampling': 'Coil + transient sampling',
         'line_broadening': 'Line broadening', 'frequency_shift': 'Frequency shift',
         'phase_shift': 'Phase shift', 'macromolecules': 'Macromolecules',
         'residual_water': 'Residual water', 'baseline': 'Baseline',
         'artificial_peaks': 'Artificial peaks', 'eddy_current': 'Eddy currents',
         'spurious_echoes': 'Spurious echoes', 'noise': 'Noise', 'all': 'All'}


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 8, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.titlesize': 9, 'axes.linewidth': 0.8, 'axes.edgecolor': INK,
        'xtick.color': INK, 'ytick.color': INK, 'axes.labelcolor': INK, 'text.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.6,
        'lines.markersize': 4.5, 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    return plt


def grid(ax, axis='both'):
    ax.grid(True, axis=axis, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, f'{name}.png'), dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(out, f'{name}.pdf'), dpi=600, bbox_inches='tight', facecolor='white')
    print(os.path.join(out, f'{name}.png'))


def label(name):
    return LABELS.get(name, name)


def subjects(n):
    return f'{n} subject' + ('' if n == 1 else 's')


def num(x, digits=2):
    """A number for figure text: no negative zero, a typographic minus."""
    text = f'{x + 0.0:.{digits}f}'
    return '0.' + '0' * digits if float(text) == 0 else text.replace('-', '−')


def end_labels(ax, items, x, gap):
    """Direct labels (y, text, colour) in one column at *x*, nudged apart where they meet."""
    items = sorted(items)
    placed = []
    for y, _, _ in items:
        placed.append(y if not placed or y - placed[-1] >= gap else placed[-1] + gap)
    for (_, text, color), yl in zip(items, placed):
        ax.text(x, yl, text, va='center', ha='left', fontsize=7.5, color=color, clip_on=False)


def mean_se(values):
    v = np.asarray(values, float)
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)


#**************************************************************************************************#
#                                           Class Results                                          #
#**************************************************************************************************#
#                                                                                                  #
# An experiment's runs, a test set, and the tools' fits of it.                                     #
#                                                                                                  #
#**************************************************************************************************#
class Results:
    """An experiment's runs, a test set, and the tools' fits of it."""

    def __init__(self, args):
        self.args = args
        self.ts = S.TestSet(args.testset) if args.testset else None
        self._pred = {}

    def run_dir(self, cond, n, fold=0, seed=0):
        return os.path.join(self.args.exp, 'runs',
                            f'{cond}__n{n}__f{fold}__{self.args.variant}__s{seed}')

    def runs(self):
        """(condition, n, fold, seed) of every finished run."""
        out = []
        for path in glob.glob(os.path.join(self.args.exp, 'runs', '*', 'result.json')):
            with open(path) as f:
                c = json.load(f)['config']
            if c['variant'] == self.args.variant and not c['dropout'] and not c['weight_decay']:
                out.append((c['condition'], c['n_subjects'], c['fold'], c['seed']))
        return out

    def net(self, cond, n, fold=0, seed=0):
        """A network's test-set concentrations (computed from its checkpoint)."""
        key = (cond, n, fold, seed)
        if key not in self._pred:
            import torch
            _, _, net = S.load_run(self.run_dir(*key), self.args.weights, 'cpu')
            self._pred[key] = self.ts.predict(net, torch.device('cpu'))[:, :len(self.ts.names)]
        return self._pred[key]

    def tool(self, method):
        z = np.load(os.path.join(self.args.bench, f'{self.ts.name}_{method}.npz'))
        if str(z['testset_hash']) != self.ts.hash:
            raise ValueError(f'{method} was fitted on another version of {self.ts.name}')
        order = [list(map(str, z['names'])).index(n) for n in self.ts.names]
        return np.asarray(z['con'], float)[:, order]

    def on_scale(self, pred):
        """(truth, estimate): macromolecules zeroed, the estimate at each spectrum's scale."""
        mm = [i for i, nm in enumerate(self.ts.names) if nm in S.MM_NAMES]
        y, y_hat = self.ts.concentrations.copy(), np.asarray(pred, float).copy()
        y[:, mm] = 0
        y_hat[:, mm] = 0
        return y, S.optimal_scale(y, y_hat) * y_hat

    def spectrum_mosae(self, pred):
        y, y_hat = self.on_scale(pred)
        return np.abs(y - y_hat).mean(1)

    def metric(self, pred, metric):
        m = S.concentration_metrics(pred, self.ts.concentrations, self.ts.names)
        return m['mosae'] if metric == 'mosae' else m['ccc_mean']

    def complete(self, conditions, ns):
        """The (fold, seed) pairs that every (condition, n) cell has finished."""
        done = self.runs()
        keys = None
        for c in conditions:
            for n in ns:
                k = {(f, s) for cc, nn, f, s in done if cc == c and nn == n}
                keys = k if keys is None else keys & k
        return sorted(keys or [])


#*************#
#   figures   #
#*************#
def against_subjects(R, conditions, metric, ylabel, name, tools=True):
    """A metric against the number of in-vivo training subjects, one line per condition."""
    plt = style()
    ns = list(range(1, 9))
    keys = R.complete(list(conditions), ns)
    if not keys:
        print(f'{name}: no (fold, seed) has every cell yet')
        return
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    ends = []
    for cond, (color, marker, text) in conditions.items():
        stats = [mean_se([R.metric(R.net(cond, n, *k), metric) for k in keys]) for n in ns]
        mean, se = np.array(stats).T
        if len(keys) > 1:
            ax.fill_between(ns, mean - se, mean + se, color=color, alpha=0.15, lw=0)
        ax.plot(ns, mean, '-', marker=marker, color=color, zorder=3, mfc=color, mec=color)
        ends.append((mean[-1], text, INK))
    if tools:
        for method, text in (TOOLS if tools is True else tools).items():
            if os.path.isfile(os.path.join(R.args.bench, f'{R.ts.name}_{method}.npz')):
                y = R.metric(R.tool(method), metric)
                ax.hlines(y, 0.6, 8.15, color=REF, lw=0.9, zorder=1)
                ends.append((y, text, REF_TEXT))
    lo, hi = ax.get_ylim()
    end_labels(ax, ends, 8.3, 0.045 * (hi - lo))
    ax.set_xlim(0.6, 8.2)
    ax.set_xticks(ns)
    ax.set_xlabel('In-vivo training subjects')
    ax.set_ylabel(ylabel)
    top = ax.secondary_xaxis('top', functions=(lambda n: 12.5 * n, lambda p: p / 12.5))
    top.set_xticks([25, 50, 75, 100])
    top.set_xlabel('Share of the training data (%)', fontsize=8)
    top.tick_params(labelsize=7.5)
    grid(ax, 'y')
    save(fig, R.args.out, name)
    plt.close(fig)


def fig_scaling(R):
    """MOSAE and the mean CCC against the training subjects, without and with Augmentrum."""
    conds = {'none': (INK, 'o', 'No augmentation'), 'all': (ACCENT, 'o', 'Augmentrum')}
    against_subjects(R, conds, 'mosae', 'MOSAE', 'fig_scaling')
    against_subjects(R, conds, 'ccc', "Lin's CCC (mean over metabolites)", 'fig_scaling_ccc')


def fig_samplers(R):
    against_subjects(R, {'none': (INK, 'o', 'No augmentation'),
                         'coil_sampling': (MID, 's', 'Coil sampling'),
                         'average_sampling': (MID, '^', 'Transient sampling'),
                         'sampling': (ACCENT, 'o', 'Coil + transient')},
                     'mosae', 'MOSAE', 'fig_samplers', tools=False)


def fig_modules(R):
    """The change in MOSAE of every condition against no augmentation, paired by fold and seed."""
    from matplotlib.lines import Line2D
    plt = style()
    groups = [('Samplers', ['coil_sampling', 'average_sampling', 'sampling']),
              ('Modules', list(S.MODULES)), ('Combined', ['all'])]
    deltas = {}
    for _, members in groups:
        for m in members:
            deltas[m] = {}
            for n in (1, 8):
                keys = R.complete(['none', m], [n])
                if keys:
                    d = [100 * (R.metric(R.net(m, n, *k), 'mosae')
                                - R.metric(R.net('none', n, *k), 'mosae'))
                         / R.metric(R.net('none', n, *k), 'mosae') for k in keys]
                    deltas[m][n] = (*mean_se(d), len(d))
    fig, ax = plt.subplots(figsize=(4.6, 0.9 + 0.23 * sum(len(g) + 1 for _, g in groups)))
    ypos, y = {}, 0.0
    for head, members in groups:
        ax.text(0.0, y, head, transform=ax.get_yaxis_transform(), ha='right', va='center',
                fontsize=7.5, color=REF_TEXT, style='italic')
        y += 1.0
        for m in members:
            ypos[m], y = y, y + 1.0
        y += 0.3
    ax.axvline(0, color=REF, lw=0.9, zorder=1)
    for m, by_n in deltas.items():
        color = ACCENT if m == 'all' else INK
        for n, (mean, se, k) in by_n.items():
            yy = ypos[m] + (-0.16 if n == 1 else 0.16)
            if k > 1:
                ax.errorbar(mean, yy, xerr=se, color=color, lw=0.8, capsize=0, zorder=2)
            ax.plot(mean, yy, 'o', color=color, mfc='white' if n == 1 else color, mew=1.0, ms=4.5,
                    zorder=3)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([NAMES[m] for m in ypos])
    ax.set_ylim(y - 0.5, -0.8)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Change in MOSAE against no augmentation (%)')
    handles = [Line2D([], [], ls='none', marker='o', ms=4.5, color=INK, mfc='white', mew=1.0,
                      label='1 subject (12.5 %)'),
               Line2D([], [], ls='none', marker='o', ms=4.5, color=INK, label='8 subjects (100 %)')]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2,
              handletextpad=0.2, columnspacing=1.6, borderaxespad=0.3)
    grid(ax, 'x')
    save(fig, R.args.out, 'fig_modules')
    plt.close(fig)


def methods(R, n):
    """(name, concentrations, colour) of the networks at *n* subjects and the main tools."""
    out = [('No augmentation', R.net('none', n), INK), ('Augmentrum', R.net('all', n), ACCENT)]
    return out + [(TOOLS[m], R.tool(m), INK) for m in ('fsl_default', 'lcmodel')]


def regression(true, est):
    """The OoD paper's est = alpha true + beta, R^2, sigma (RMSE of est - true), and Lin's CCC."""
    alpha, beta = np.polyfit(true, est, 1)
    return dict(alpha=alpha, beta=beta, r2=np.corrcoef(true, est)[0, 1] ** 2,
                sigma=np.sqrt(np.mean((est - true) ** 2)), ccc=S.lin_ccc(est, true))


def fig_bias(R):
    """Estimated against true totals on the MOSAE scale, with the regression numbers."""
    plt = style()
    meths = methods(R, R.args.n)
    fig, axes = plt.subplots(len(TOTALS), len(meths), figsize=(7.0, 1.62 * len(TOTALS) + 0.4),
                             sharex='row', sharey='row')
    for c, (name, con, color) in enumerate(meths):
        y, y_hat = R.on_scale(con)
        tt, tp = S.totals(y, R.ts.names), S.totals(y_hat, R.ts.names)
        for r, met in enumerate(TOTALS):
            ax = axes[r, c]
            lo, hi = np.percentile(tt[met], [0.5, 99.5])
            lim = (lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo))
            s = regression(tt[met], tp[met])
            ax.plot(lim, lim, color=REF, lw=0.9, zorder=1)
            ax.scatter(tt[met], tp[met], s=2, color=color, alpha=0.25, lw=0, rasterized=True,
                       zorder=2)
            ax.plot(lim, [s['alpha'] * lim[0] + s['beta'], s['alpha'] * lim[1] + s['beta']],
                    color=color, lw=1.3, zorder=3)
            ax.text(0.04, 0.97, f"α {num(s['alpha'])}   β {num(s['beta'])}\n"
                                f"R² {num(s['r2'])}   σ {num(s['sigma'])}\nCCC {num(s['ccc'])}",
                    transform=ax.transAxes, ha='left', va='top', fontsize=6.3, linespacing=1.25)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect('equal')
            ax.locator_params(nbins=4)
            ax.tick_params(labelsize=7)
            grid(ax)
            if r == 0:
                ax.set_title(name if name in TOOLS.values() else f'{name}\n{subjects(R.args.n)}',
                             fontsize=8.5, pad=5)
            if c == 0:
                ax.set_ylabel(f'Estimated {label(met)}', fontsize=8)
            if r == len(TOTALS) - 1:
                ax.set_xlabel('True', fontsize=8)
    fig.tight_layout(h_pad=0.9, w_pad=0.5)
    save(fig, R.args.out, f'fig_bias_n{R.args.n}')
    plt.close(fig)


def series(R):
    return [('FSL-MRS', R.tool('fsl_default'), REF_TEXT, 's', 'white'),
            ('LCModel', R.tool('lcmodel'), REF_TEXT, 'D', 'white'),
            ('No augmentation', R.net('none', R.args.n), INK, 'o', INK),
            ('Augmentrum', R.net('all', R.args.n), ACCENT, 'o', ACCENT)]


def fig_metabolites(R):
    """Lin's CCC per metabolite and method; the networks at --n subjects."""
    from matplotlib.lines import Line2D
    plt = style()
    ts = R.ts
    order = [i for i in np.argsort(-ts.concentrations.mean(0)) if ts.names[i] not in S.MM_NAMES]
    ser = series(R)
    fig, ax = plt.subplots(figsize=(4.4, 0.2 * len(order) + 0.9))
    for (name, con, color, marker, face), dy in zip(ser, np.linspace(-0.24, 0.24, len(ser))):
        y, y_hat = R.on_scale(con)
        ax.plot([S.lin_ccc(y_hat[:, i], y[:, i]) for i in order], np.arange(len(order)) + dy,
                ls='none', marker=marker, ms=3.8, color=color, mfc=face, mew=0.9, zorder=3)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([label(ts.names[i]) for i in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(-0.1, 1.02)
    ax.axvline(0, color=REF, lw=0.8, zorder=1)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel("Lin's CCC, estimated against true")
    ax.text(1.0, 1.0, f'Networks: {subjects(R.args.n)}; metabolites by mean concentration',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7, color=REF_TEXT)
    grid(ax, 'x')
    handles = [Line2D([], [], ls='none', marker=m, ms=3.8, color=c, mfc=f, mew=0.9, label=n)
               for n, _, c, m, f in ser]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.45, 1.04), ncol=4,
              handletextpad=0.1, columnspacing=1.2, borderaxespad=0.0)
    save(fig, R.args.out, f'fig_metabolites_n{R.args.n}')
    plt.close(fig)


def binned(R, x, xlabel, name, bins=6):
    """Mean MOSAE per spectrum (± SE) in equal-count bins of *x*, one line per method."""
    plt = style()
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, bins - 1)
    centres = np.array([np.median(x[idx == b]) for b in range(bins)])
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ends = []
    for text, con, color, marker, face in series(R):
        err = R.spectrum_mosae(con)
        stats = np.array([mean_se(err[idx == b]) for b in range(bins)])
        ax.errorbar(centres, stats[:, 0], yerr=stats[:, 1], color=color, lw=1.2, capsize=0,
                    marker=marker, ms=4, mfc=face, mew=0.9, zorder=3)
        ends.append((stats[-1, 0], text, REF_TEXT if color == REF_TEXT else INK))
    lo, hi = ax.get_ylim()
    end_labels(ax, ends, centres[-1] + 0.04 * (centres[-1] - centres[0]), 0.05 * (hi - lo))
    ax.set_xlabel(xlabel)
    ax.set_ylabel('MOSAE per spectrum')
    ax.text(0.0, 1.02, f'Networks: {subjects(R.args.n)}; {bins} equal-count bins, mean ± SE',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=7, color=REF_TEXT)
    grid(ax, 'y')
    save(fig, R.args.out, f'{name}_n{R.args.n}')
    plt.close(fig)


def fig_snr(R):
    binned(R, np.asarray(R.ts.snr, float), 'SNR (NAA)', 'fig_snr')


def fig_linewidth(R):
    """MOSAE against the NAA linewidth (FWHM of the simulated NAA singlet, Hz)."""
    ts = R.ts
    i = ts.names.index('NAA')
    gamma = ts.params['gamma'][:, i]
    sigma = np.asarray(ts.params['sigma']).reshape(len(gamma), -1)[:, 0]
    t = np.arange(8192) / 4000.0
    f = np.fft.fftshift(np.fft.fftfreq(8192, 1 / 4000.0))
    fwhm = []
    for g, s in zip(gamma, sigma):
        line = np.fft.fftshift(np.fft.fft(np.exp(-(g + s ** 2 * t) * t))).real
        above = f[line >= line.max() / 2]
        fwhm.append(above.max() - above.min())
    binned(R, np.array(fwhm), 'NAA linewidth (Hz)', 'fig_linewidth')


def fig_curves(R):
    """
    The selection-set MOSAE over training of every condition against no augmentation, at 1
    (light) and 8 (dark) subjects, one figure per condition (fold 0, seed 0).
    """
    import pandas as pd
    plt = style()
    curves = {}
    for cond, n, fold, seed in R.runs():
        if fold == 0 and seed == 0:
            c = pd.read_csv(os.path.join(R.run_dir(cond, n), 'curve.csv'))
            c = c[c.step % int(c.step.diff().mode().iloc[-1] or 1) == 0]
            curves[(cond, n)] = c
    for cond in [c for c in NAMES if c != 'none' and any(k[0] == c for k in curves)]:
        fig, ax = plt.subplots(figsize=(4.4, 2.8))
        for c, color_1, color_8 in ((cond, ACCENT_LIGHT, ACCENT), ('none', INK_LIGHT, INK)):
            for n, color in ((1, color_1), (8, color_8)):
                if (c, n) in curves:
                    d = curves[(c, n)]
                    ax.plot(d.step / 1e6, d.sel_mosae, color=color, lw=1.1)
        ax.set_xlabel('Training steps (millions)')
        ax.set_ylabel('MOSAE, selection set')
        ax.text(0.0, 1.02, f'{NAMES[cond]} (teal) against no augmentation (grey); light 1 '
                           'subject, dark 8', transform=ax.transAxes, ha='left', va='bottom',
                fontsize=7, color=REF_TEXT)
        grid(ax, 'y')
        save(fig, R.args.out, f'fig_curves_{cond}')
        plt.close(fig)


#***********#
#   paper   #
#***********#
# The paper's figures, from the augmentation screen (svs_ablation.py screen): stage B for the
# results, stage A for the strength ladders. Each figure stands on its own at the full A4 text width (MRM
# 6.92 in), text >= 7 pt, data lines > 1 pt. One colour per method, the same in every figure,
# COWS and Deep-ER (train_deep_er.py --figures) alike: Paul Tol's colour-blind-safe colours
# (https://sronpersonalpages.nl/~pault/), Augmentrum cool, the fitting tools warm, no
# augmentation grey. Every pair stays apart for normal vision (OKLab dE >= 15.7) and under the
# three colour-vision deficiencies (>= 6.0; there markers and dashes separate them too).
# Tools dashed, networks solid, labels at the lines rather than in legends.
AUGMENTRUM = '#009988'                        # all augmentations, at the picked strengths
BEST_SINGLE_COLOR = '#332288'                 # the best single augmentation (selection set)
NONE_COLOR = '#4D4D4D'
#: (condition, label, colour, marker): the networks of the figures that hold only a few (fits,
#: estimated against true, in vivo, SNR); every stage-B condition is at its picked strength
#: (the old 'all', never picked, is left out)
PAPER_NETS = (('none', 'No augmentation', NONE_COLOR, 'o'),
              ('average_sampling-min1', 'Transient sampling', BEST_SINGLE_COLOR, 's'))
ALL_BEST = ('all-best', 'All augmentations', AUGMENTRUM, 'o')
FEATURED = 'average_sampling-min1'            # the network whose errors pick the example fits
#: the main paper's conditions (the user's selection, 2026-09-30), grouped by what they model,
#: each with its label, colour and marker: one cool hue per group (the tools are warm), darker
#: for the better on the selection set
MAIN_GROUPS = (
    ('Sampling and noise', (('average_sampling-min1', 'Transients', '#00665E', 'o'),
                            ('coil_sampling-min1', 'Coils', '#009988', 's'),
                            ('noise-snr15', 'Noise', '#5DBFB3', '^'))),
    ('Frequency, phase and lineshape', (('phase_shift-x4', 'Phase shift', '#221A6E', 'o'),
                                        ('eddy_current-x2', 'Eddy currents', '#4B3FA0', 's'),
                                        ('broadening-voigt-x4', 'Voigt broadening', '#7F75C4',
                                         '^'),
                                        ('frequency_shift-x4', 'Frequency shift', '#AFA8DE',
                                         'D'))),
    ('Artefacts', (('spurious_echoes-echo-amp0p1', 'Spurious echo', '#0077BB', 'o'),
                   ('baseline-bspline-x2', 'B-spline baseline', '#6CB0E0', 's'))))
MAIN = [c for _, members in MAIN_GROUPS for c, _, _, _ in members]
#: the variants of one module family, in order (one panel each): colours and markers
VARIANT_STYLE = (('#009988', 'o'), ('#332288', 's'), ('#33BBEE', '^'), ('#999933', 'D'),
                 ('#117733', 'v'))
TOOL_BAND = '#E6E6E6'
BURD = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7', '#FDDBC7', '#F4A582', '#D6604D',
        '#B2182B']                            # Tol's diverging BuRd
PAPER_TOOLS = (('fsl_default', 'FSL-MRS', '#EE7733', 's'),
               ('lcmodel', 'LCModel', '#997700', 'D'),
               ('osprey', 'Osprey', '#AA3377', '^'))
COL1, COL2 = 3.42, 6.92                      # MRM single and double column (inches)
DASH = (0, (3.5, 2))
PAPER_TOTALS = ('tNAA', 'tCho', 'Glx', 'mI')
#: the stage B conditions, grouped by module family, in words (n×: n times the in-vivo range)
FAMILIES = {'coil_sampling': 'Sampling', 'average_sampling': 'Sampling', 'noise': 'Noise',
            'line_broadening': 'Line broadening', 'frequency_shift': 'Frequency shift',
            'phase_shift': 'Phase shift', 'macromolecules': 'Macromolecules',
            'residual_water': 'Residual water', 'baseline': 'Baseline',
            'artificial_peaks': 'Artificial peaks', 'eddy_current': 'Eddy currents',
            'spurious_echoes': 'Spurious echoes', 'apodization': 'Truncation'}
SHORT = {
    'none': 'No augmentation', 'all': 'All, in-vivo ranges', 'all-best': 'All augmentations',
    'sampling-best': 'Coils + transients',
    'average_sampling-min1': 'Transients, random (1–32 of 32)',
    'average_sampling-consecutive-min4': 'Transients, consecutive (4–32)',
    'coil_sampling-min1': 'Coils (1–32 of 32)', 'noise-snr15': 'SNR 15–330',
    'spurious_echoes-echo-amp0p1': 'Echo (amplitude ≤ 0.1)',
    'spurious_echoes-replica-amp0p2': 'Delayed replica (≤ 0.2)',
    'baseline-bspline-x2': 'B-spline (2×)', 'baseline-randomwalk-x2': 'Random walk (2×)',
    'baseline-polynomial-x2': 'Polynomial (2×)', 'phase_shift-x4': 'Zero and first order (4×)',
    'eddy_current-x2': 'Synthetic (2×)', 'broadening-voigt-x4': 'Voigt (4×)',
    'broadening-voigt-narrowing-x4': 'Voigt with narrowing (4×)',
    'broadening-gaussian-x4': 'Gaussian (4×)', 'broadening-lorentzian-x4': 'Lorentzian (4×)',
    'broadening-kernel-spread4': 'Lineshape kernel (4 Hz)',
    'artificial_peaks-voigt-phase-x4': 'Voigt, random phase (4×)',
    'artificial_peaks-x4': 'Lorentzian (4×)', 'frequency_shift-x4': 'Global (4×)',
    'apodization-truncate-keep0p25': 'Keep 25–100 % of the FID',
    'residual_water-turco-x2': 'Turco (2×)', 'residual_water-lobes-x2': 'Three lobes (2×)',
    'macromolecules-measured-x2': 'Measured (2×)',
    'macromolecules-semiparametrized-x1': 'Semi-parametric (1×)',
    'macromolecules-parametrized-x1': 'Parametric (1×)'}
#: the stage B conditions in a few words, for labels inside a family's panel
TERSE = {
    'none': 'None', 'all-best': 'All', 'sampling-best': 'Coils + transients',
    'average_sampling-min1': 'Transients 1–32', 'average_sampling-consecutive-min4':
    'Consecutive 4–32', 'coil_sampling-min1': 'Coils 1–32', 'noise-snr15': 'SNR 15–330',
    'spurious_echoes-echo-amp0p1': 'Echo', 'spurious_echoes-replica-amp0p2': 'Replica',
    'baseline-bspline-x2': 'B-spline', 'baseline-randomwalk-x2': 'Random walk',
    'baseline-polynomial-x2': 'Polynomial', 'phase_shift-x4': 'Zero + first order',
    'eddy_current-x2': 'Synthetic', 'broadening-voigt-x4': 'Voigt',
    'broadening-voigt-narrowing-x4': 'Voigt, narrowing', 'broadening-gaussian-x4': 'Gaussian',
    'broadening-lorentzian-x4': 'Lorentzian', 'broadening-kernel-spread4': 'Kernel',
    'artificial_peaks-voigt-phase-x4': 'Voigt + phase', 'artificial_peaks-x4': 'Lorentzian',
    'frequency_shift-x4': 'Global', 'apodization-truncate-keep0p25': 'Keep 25–100 %',
    'residual_water-turco-x2': 'Turco', 'residual_water-lobes-x2': 'Three lobes',
    'macromolecules-measured-x2': 'Measured', 'macromolecules-semiparametrized-x1':
    'Semi-parametric', 'macromolecules-parametrized-x1': 'Parametric'}
#: stage A: each variant's panel title and what its ladder steps through
LADDERS = {
    'coil_sampling': ('Coil sampling', 'Fewest coils (of 32)'),
    'average_sampling': ('Transient sampling', 'Fewest transients (of 32)'),
    'average_sampling-consecutive': ('Transients, consecutive', 'Fewest transients (of 32)'),
    'noise': ('Noise', 'Lowest SNR'),
    'broadening-voigt': ('Voigt broadening', '× in-vivo range'),
    'broadening-lorentzian': ('Lorentzian broadening', '× in-vivo range'),
    'broadening-gaussian': ('Gaussian broadening', '× in-vivo range'),
    'broadening-voigt-narrowing': ('Voigt with narrowing', '× in-vivo range'),
    'broadening-kernel': ('Lineshape kernel', 'Spread [Hz]'),
    'frequency_shift': ('Frequency shift', '× in-vivo range'),
    'phase_shift': ('Phase shift', '× in-vivo range'),
    'macromolecules-semiparametrized': ('MM, semi-parametric', '× in-vivo range'),
    'macromolecules-parametrized': ('MM, parametric', '× in-vivo range'),
    'macromolecules-measured': ('MM, measured', '× in-vivo range'),
    'residual_water-lobes': ('Water, three lobes', '× in-vivo range'),
    'residual_water-turco': ('Water, Turco', '× in-vivo range'),
    'baseline-bspline': ('Baseline, B-spline', '× in-vivo range'),
    'baseline-polynomial': ('Baseline, polynomial', '× in-vivo range'),
    'baseline-randomwalk': ('Baseline, random walk', '× in-vivo range'),
    'artificial_peaks': ('Peaks, Lorentzian', '× in-vivo range'),
    'artificial_peaks-voigt-phase': ('Peaks, Voigt + phase', '× in-vivo range'),
    'eddy_current': ('Eddy currents', '× in-vivo range'),
    'spurious_echoes-echo': ('Spurious echo', 'Largest amplitude'),
    'spurious_echoes-replica': ('Delayed replica', 'Largest amplitude'),
    'apodization-truncate': ('Truncation', 'Least of the FID kept')}


def paper_style():
    plt = style()
    plt.rcParams.update({
        'font.size': 7, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'axes.linewidth': 0.8, 'xtick.major.width': 0.8,
        'ytick.major.width': 0.8, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'lines.linewidth': 1.5, 'lines.markersize': 4, 'savefig.pad_inches': 0.02})
    return plt


def stored(R, cond, n):
    """A run's test-set scores as svs_ablation.py train stored them (at --weights)."""
    with open(os.path.join(R.run_dir(cond, n), 'result.json')) as f:
        return json.load(f)['test'][R.args.weights]


def tool_score(R, method, metric):
    """A tool's score on the test set; *metric* a key of svs_ablation.concentration_metrics."""
    return S.concentration_metrics(R.tool(method), R.ts.concentrations, R.ts.names)[metric]


def side_labels(ax, items, x, gap, x_from=None):
    """Direct labels (y, text, colour) in one column at data x, spread apart where they meet;
    with *x_from*, a thin connector from each line's end to its label."""
    items = sorted(items)
    for (y0, text, color), y in zip(items, spread_labels([y for y, _, _ in items], gap)):
        ax.text(x, y, text, color=color, va='center', ha='left', clip_on=False)
        if x_from is not None:
            ax.plot([x_from, x_from + 0.6 * (x - x_from), x - 0.04 * (x - x_from)], [y0, y, y],
                    color=color, lw=0.6, clip_on=False, zorder=2)


def top_labels(ax, items, gap):
    """Labels (x, text, colour) above the axes at data x; neighbours closer than *gap* stacked."""
    level, prev = 0, None
    for x, text, color in sorted(items):
        level = level + 1 if prev is not None and x - prev < gap else 0
        ax.annotate(text, (x, 1.0), xycoords=('data', 'axes fraction'), xytext=(0, 2 + 9 * level),
                    textcoords='offset points', ha='center', va='bottom', color=color)
        prev = x


def paper_accuracy(R, metric, ylabel, name):
    """
    A score against the in-vivo training subjects for every main-paper condition in one plot
    (colour by group, marker by condition), no augmentation grey, the tools dashed.
    """
    plt = paper_style()
    ns = list(range(1, 9))
    fig, ax = plt.subplots(figsize=(COL2 + 0.75, 3.4))            # with the labels: 6.9 in
    fig.subplots_adjust(right=0.78)
    items = []
    for m, text, color, _ in PAPER_TOOLS:
        y = tool_score(R, m, metric)
        ax.axhline(y, xmax=0.96, color=color, lw=1.2, ls=DASH, zorder=1)
        items.append((y, text, color))
    lines = [('none', 'No augmentation', NONE_COLOR, 'o')] + [
        x for _, members in MAIN_GROUPS for x in members]
    for cond, text, color, marker in lines:
        y = [stored(R, cond, n)[metric] for n in ns]
        ax.plot(ns, y, '-', marker=marker, color=color, ms=3.6, lw=1.3, mec='white', mew=0.4,
                zorder=3)
        items.append((y[-1], text, color))
    lo, hi = ax.get_ylim()
    side_labels(ax, items, 8.75, 0.05 * (hi - lo), x_from=8.15)
    ax.set_xlim(0.7, 8.3)
    ax.set_xticks(ns)
    ax.set_xlabel('In-vivo training subjects')
    ax.set_ylabel(ylabel)
    grid(ax, 'y')
    save(fig, R.args.out, name)
    plt.close(fig)


def screen_groups():
    """[(family, [conditions])] of stage B at the picked strengths: combinations, then families."""
    with open(os.path.join(S.SCREEN, 'picks.json')) as f:
        picks = json.load(f)
    groups = {'Combinations': ['all-best', 'sampling-best']}
    for p in picks.values():
        groups.setdefault(FAMILIES[p['family']], []).append(p['name'])
    return list(groups.items())


def screen_order(R, metric='mosae'):
    """screen_groups(), families and their members ordered by the mean over 1-8 subjects."""
    sign = 1 if metric == 'mosae' else -1
    mean = lambda c: sign * np.mean([stored(R, c, n)[metric] for n in range(1, 9)])
    groups = [(g, sorted(m, key=mean)) for g, m in screen_groups()]
    return sorted(groups, key=lambda g: mean(g[1][0]))


def full_name(cond):
    """A stage-B condition in words, with its module family."""
    for family, members in screen_groups():
        if cond in members and family != 'Combinations':
            first, rest = SHORT[cond].split(' ', 1)
            proper = first.rstrip(',') in ('Voigt', 'Lorentzian', 'Gaussian', 'Turco', 'B-spline',
                                           'SNR')
            return f"{family}: {first if proper else first.lower()} {rest}"
    return SHORT[cond]


def paper_screen(R, metric, xlabel, name, groups=None, width=COL2 - 0.6):
    """
    Every stage-B condition: the mean over 1-8 training subjects (dot) and the range (bar),
    grouped by module family, families and rows ordered by their best; the tools dashed.
    """
    plt = paper_style()
    colors, names = {}, dict(SHORT)
    if groups is None:
        groups = screen_order(R, metric)
    else:                                         # [(title, [(cond, label, colour, marker)])]
        colors = {c: col for _, m in groups for c, _, col, _ in m}
        names |= {c: f'{text} ({SHORT[c].split("(")[-1]}' if '(' in SHORT[c]
                  else f'{text} ({SHORT[c]})' for _, m in groups for c, text, _, _ in m}
        groups = [(g, [c for c, _, _, _ in m]) for g, m in groups]
    groups = [('', ['none'])] + groups
    val = {c: np.array([stored(R, c, n)[metric] for n in range(1, 9)])
           for _, m in groups for c in m}
    rows = sum(len(m) + 1 for _, m in groups)
    fig, ax = plt.subplots(figsize=(width, 0.15 * rows + 0.95))    # the labels hang ~1.5 in out
    ypos, labels, y = [], [], 0.0
    for head, members in groups:
        if head:
            ax.text(0.01, y, head, transform=ax.get_yaxis_transform(), ha='left', va='center',
                    fontweight='bold', zorder=5,
                    bbox=dict(facecolor='white', edgecolor='none', pad=0.8))
            y += 1
        for c in members:
            v, color = val[c], colors.get(c, INK)
            ax.plot([v.min(), v.max()], [y, y], color=color, lw=1.2, alpha=0.45,
                    solid_capstyle='round', zorder=2)
            ax.plot(v.mean(), y, 'o', color=color, ms=4, mec='white', mew=0.4, zorder=3)
            ypos.append(y)
            labels.append(names[c])
            y += 1
        y += 0.4
    items = []
    for m, text, color, _ in PAPER_TOOLS:
        x = tool_score(R, m, metric)
        ax.axvline(x, color=color, lw=1.2, ls=DASH, zorder=1)
        items.append((x, text, color))
    ax.axvline(val['none'].mean(), color=NONE_COLOR, lw=0.8, alpha=0.5, zorder=1)
    lo, hi = ax.get_xlim()
    top_labels(ax, items, 0.16 * (hi - lo))
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_ylim(y - 0.4, -0.7)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(xlabel)
    grid(ax, 'x')
    save(fig, R.args.out, name)
    plt.close(fig)


def family_panels(R, draw, xlabel, ylabel, name, label_x, band=None, xlog=False, ylim=None,
                  xlim=None, xticks=None, note=None, groups=None):
    """
    One panel per module family, every stage-B condition in it, labelled at the line ends:
    draw(ax, cond, colour, marker) -> the line's last y, for no augmentation (grey) and each
    variant; *band* (lo, hi) shaded: the tools' range.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    plt = paper_style()
    styled = groups is not None               # [(title, [(cond, label, colour, marker)])]
    if not styled:
        groups = [(g, [(c, TERSE[c], *st) for c, st in zip(m, VARIANT_STYLE)])
                  for g, m in screen_order(R)]
    cols = 3
    rows = -(-(len(groups) + (0 if styled else 1)) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(COL2, 1.4 * rows + (0.45 if styled else 0.1)),
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()
    if xlog:
        flat[0].set_xscale('log')
    if ylim is not None:
        flat[0].set_ylim(ylim)
    if xlim is not None:
        flat[0].set_xlim(xlim)
    if xticks is not None:
        flat[0].set_xticks(xticks)
    labels = []
    for ax, (family, members) in zip(flat, groups):
        if band is not None:
            ax.axhspan(*band, color=TOOL_BAND, lw=0, zorder=0)
        items = [(draw(ax, 'none', NONE_COLOR, 'o'), TERSE['none'], NONE_COLOR)]
        for c, text, color, marker in members:
            items.append((draw(ax, c, color, marker), text, color))
        labels.append((ax, items))
        ax.set_title(family, fontsize=7.5, fontweight='bold', pad=3)
        grid(ax, 'y')
    lo, hi = flat[0].get_ylim()
    for ax, items in labels:                      # after every panel set the shared limits
        side_labels(ax, items, label_x, 0.1 * (hi - lo))
    handles = [Line2D([], [], color=NONE_COLOR, lw=1.1, label='No augmentation (every panel)')]
    if band is not None:
        handles.append(Patch(color=TOOL_BAND, label='FSL-MRS, LCModel, Osprey (range)'))
    if styled:
        fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0),
                   title=note, title_fontsize=7)
    else:
        key = flat[-1]                            # the last slot, clear of the labels
        key.axis('off')
        key.legend(handles=handles, loc='center', fontsize=7, title=note, title_fontsize=7)
    for i, ax in enumerate(flat):
        if i >= len(groups):
            ax.axis('off')
        if i < len(groups) and i + cols >= len(groups):
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if styled else 1), h_pad=0.7, w_pad=0.3)
    save(fig, R.args.out, name)
    plt.close(fig)


def paper_families(R, metric, ylabel, name, groups=None):
    """The score against the training subjects for every stage-B condition, one panel per family."""
    ns = list(range(1, 9))

    def draw(ax, c, color, marker):
        y = [stored(R, c, n)[metric] for n in ns]
        ax.plot(ns, y, '-', marker=marker, color=color, ms=3, lw=1.1, mec='white', mew=0.3,
                zorder=3)
        return y[-1]
    tools = [tool_score(R, m, metric) for m, *_ in PAPER_TOOLS]
    family_panels(R, draw, 'In-vivo training subjects', ylabel, name, 8.5,
                  band=(min(tools), max(tools)), xticks=ns, groups=groups)


def paper_series(R, n):
    """(label, concentrations, colour, marker, filled) of the tools and the networks at *n*."""
    return ([(text, R.tool(m), color, mk, False) for m, text, color, mk in PAPER_TOOLS]
            + [(text, R.net(c, n), color, mk, True) for c, text, color, mk in PAPER_NETS])


def paper_metabolite_grid(R, n, metric, name, conds=None):
    """
    Every stage-B condition (rows, as in the screen figure) and tool per metabolite (columns, by
    mean concentration) at *n* subjects: the value in the cell, the colour its difference from
    the best of the three tools on that metabolite (blue better, red worse; Tol's BuRd).
    """
    from matplotlib.colors import LinearSegmentedColormap
    plt = paper_style()
    ts = R.ts
    mets = [ts.names[i] for i in np.argsort(-ts.concentrations.mean(0))
            if ts.names[i] not in S.MM_NAMES]
    tools = {text: S.concentration_metrics(R.tool(m), ts.concentrations, ts.names)
             for m, text, _, _ in PAPER_TOOLS}
    conds = ['none'] + (conds or [c for _, m in screen_order(R) for c in m])
    rows = [(t, [tools[t][f'{metric}_{m}'] for m in mets]) for t in tools]
    rows += [(full_name(c), [stored(R, c, n)[f'{metric}_{m}'] for m in mets]) for c in conds]
    v = np.array([r[1] for r in rows])
    best = v[:len(tools)].min(0) if metric == 'mosae' else v[:len(tools)].max(0)
    diff = (np.log2(v / best) if metric == 'mosae' else best - v)
    lim = 1.0 if metric == 'mosae' else 0.5
    cmap = LinearSegmentedColormap.from_list('BuRd', BURD)
    fig, ax = plt.subplots(figsize=(COL2 - 0.98, 0.15 * len(rows) + 0.9))
    ax.imshow(diff, cmap=cmap, vmin=-lim, vmax=lim, aspect='auto')
    for i in range(len(rows)):
        for j in range(len(mets)):
            text = f'{v[i, j]:.2f}'.replace('0.', '.', 1) if v[i, j] < 1 else f'{v[i, j]:.1f}'
            ax.text(j, i, text.replace('-', '−'), ha='center', va='center', fontsize=5.8,
                    color='white' if abs(diff[i, j]) > 0.75 * lim else INK)
    ax.axhline(len(tools) - 0.5, color='white', lw=2.5)
    ax.set_xticks(range(len(mets)))
    ax.set_xticklabels([label(m) for m in mets], rotation=90)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    for tick, (m, text, color, _) in zip(ax.get_yticklabels(), PAPER_TOOLS):
        tick.set_color(color)
        tick.set_fontweight('bold')
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    what = 'MOSAE' if metric == 'mosae' else "Lin's CCC"
    ax.set_xlabel(f'{what} per metabolite, networks at {subjects(n)}\n'
                  'colour: against the best tool (blue better, red worse)')
    save(fig, R.args.out, name)
    plt.close(fig)


def paper_bias(R, n, name):
    """Estimated against true totals on the MOSAE scale, with the regression numbers."""
    plt = paper_style()
    ser = paper_series(R, n)
    fig, axes = plt.subplots(len(PAPER_TOTALS), len(ser),
                             figsize=(COL2, 1.36 * len(PAPER_TOTALS) + 0.3),
                             sharex='row', sharey='row')
    for c, (text, con, color, _, _) in enumerate(ser):
        y, y_hat = R.on_scale(con)
        tt, tp = S.totals(y, R.ts.names), S.totals(y_hat, R.ts.names)
        for r, met in enumerate(PAPER_TOTALS):
            ax = axes[r, c]
            lo, hi = np.percentile(tt[met], [0.5, 99.5])
            lim = (lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo))
            s = regression(tt[met], tp[met])
            ax.plot(lim, lim, color=REF, lw=0.8, zorder=1)
            ax.scatter(tt[met], tp[met], s=1.5, color=color, alpha=0.3, lw=0, rasterized=True,
                       zorder=2)
            ax.plot(lim, [s['alpha'] * lim[0] + s['beta'], s['alpha'] * lim[1] + s['beta']],
                    color=color, lw=1.3, zorder=3)
            ax.text(0.04, 0.97, f"CCC {num(s['ccc'])}\nα {num(s['alpha'])}  β {num(s['beta'])}"
                                f"\nR² {num(s['r2'])}  σ {num(s['sigma'])}",
                    transform=ax.transAxes, ha='left', va='top', fontsize=6.5, linespacing=1.2)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect('equal')
            ax.locator_params(nbins=3)
            if r == 0:
                ax.set_title(text if c < len(PAPER_TOOLS) else f'{text}\n{subjects(n)}',
                             color=color, fontweight='bold', fontsize=7, pad=4)
            if c == 0:
                ax.set_ylabel(f'Estimated {label(met)}')
            if r == len(PAPER_TOTALS) - 1:
                ax.set_xlabel('True')
    fig.tight_layout(h_pad=0.5, w_pad=0.4)
    save(fig, R.args.out, name)
    plt.close(fig)


def paper_fits(R, n, name, percentiles=(10, 50, 90)):
    """
    Test spectra at percentiles of the featured network's per-spectrum MOSAE: both networks'
    fits and baselines (left; residuals above), their concentrations against the truth on the
    MOSAE scale (right).
    """
    import torch
    from matplotlib.lines import Line2D
    plt = paper_style()
    ts = R.ts
    nets = [(c, text, color) for c, text, color, _ in PAPER_NETS]
    err = {c: R.spectrum_mosae(R.net(c, n)) for c, _, _ in nets}
    order = np.argsort(err[FEATURED])
    picks = [int(order[int(round(p / 100 * (len(order) - 1)))]) for p in percentiles]
    fits = {}
    for c, _, _ in nets:
        _, model, net = S.load_run(R.run_dir(c, n), R.args.weights, 'cpu')
        with torch.no_grad():
            spec, base = model(net.predict(ts.spectra[picks]).to(model.basis_ri.dtype),
                               baseline_out=True)
        fits[c] = (spec.real.numpy(), base.real.numpy())
    w = slice(model.first, model.last)
    ppm = S.load_basis(S.BASIS_DIR).ppm[w]
    show = [ts.names.index(m) for m in ('NAA', 'Cr', 'PCr', 'GPC', 'Glu', 'Gln', 'mI', 'GSH',
                                        'Tau', 'NAAG')]
    fig, axes = plt.subplots(len(picks), 2, figsize=(COL2, 1.75 * len(picks) + 0.45),
                             gridspec_kw=dict(width_ratios=(2.5, 1.0)))
    for r, (i, p) in enumerate(zip(picks, percentiles)):
        ax = axes[r, 0]
        d = ts.spectra[i, 0, w].numpy()
        s = np.abs(d).max()
        ax.plot(ppm, d / s, color='black', lw=0.8, zorder=2)
        for k, (c, _, color) in enumerate(nets):
            f, b = fits[c][0][r, w] / s, fits[c][1][r, w] / s
            ax.plot(ppm, f, color=color, lw=1.1, alpha=0.9, zorder=3 + k)
            ax.plot(ppm, b, color=color, lw=0.9, ls=DASH, zorder=3 + k)
            ax.plot(ppm, d / s - f + 1.2 + 0.28 * k, color=color, lw=0.8, zorder=2)
        ax.set_xlim(S.PPM_WINDOW[1], S.PPM_WINDOW[0])
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.text(0.0, 1.0, f"{p}th percentile of the {nets[1][1].lower()} network's errors",
                transform=ax.transAxes, ha='left', va='bottom')
        ax2 = axes[r, 1]
        ax2.text(0.0, 1.0, 'MOSAE', transform=ax2.transAxes, ha='left', va='bottom')
        for x, (c, _, color) in zip((0.36, 0.68), nets):
            ax2.text(x, 1.0, num(err[c][i], 3), color=color, fontweight='bold',
                     transform=ax2.transAxes, ha='left', va='bottom')
        for dy, (c, _, color) in zip((-0.15, 0.15), nets):
            y, y_hat = R.on_scale(R.net(c, n)[[i]])
            ax2.plot(y_hat[0, show], np.arange(len(show)) + dy, 'o', color=color, ms=3.6,
                     mec='white', mew=0.3, zorder=3)
        ax2.plot(y[0, show], np.arange(len(show)), '|', color='black', ms=8, mew=1.5, zorder=4)
        ax2.set_yticks(np.arange(len(show)))
        ax2.set_yticklabels([label(ts.names[k]) for k in show])
        ax2.set_ylim(len(show) - 0.5, -0.5)
        ax2.tick_params(axis='y', length=0)
        ax2.spines['left'].set_visible(False)
        grid(ax2, 'x')
        if r == len(picks) - 1:
            ax.set_xlabel('Chemical shift [ppm]')
            ax2.set_xlabel('Concentration [MOSAE scale]')
    handles = ([Line2D([], [], color='black', lw=1.0, label='Data; truth |')]
               + [Line2D([], [], color=color, lw=1.4, label=f'{text}, {subjects(n)}')
                  for _, text, color in nets]
               + [Line2D([], [], color=INK, lw=1.0, ls=DASH, label='Fitted baseline'),
                  Line2D([], [], color=INK, lw=0.8, label='Residuals, offset (top)')])
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0),
               frameon=False, handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.4, w_pad=1.2)
    save(fig, R.args.out, name)
    plt.close(fig)


def paper_grid(R, metric, label_, name, conds=None):
    """Every condition (rows, by mean over n) x training subjects; the tools on the colour bar."""
    from matplotlib.colors import LinearSegmentedColormap
    plt = paper_style()
    ns = list(range(1, 9))
    conds = ['none'] + (conds or [c for _, m in screen_groups() for c in m])
    val = {c: [stored(R, c, n)[metric] for n in ns] for c in conds}
    low = metric == 'mosae'
    conds.sort(key=lambda c: np.mean(val[c]) * (1 if low else -1))
    tools = {text: tool_score(R, m, metric) for m, text, _, _ in PAPER_TOOLS}
    v = np.array([val[c] for c in conds])
    lo, hi = min(v.min(), *tools.values()), max(v.max(), *tools.values())
    ramp = ['#FFFFFF', '#CDEBE7', '#7FC8BE', AUGMENTRUM]
    cmap = LinearSegmentedColormap.from_list('', ramp[::-1] if low else ramp)
    fig, ax = plt.subplots(figsize=(COL2 - 2.12, 0.17 * len(conds) + 0.75))
    im = ax.imshow(v, cmap=cmap, vmin=lo, vmax=hi, aspect='auto')
    for i in range(len(conds)):
        for j in range(len(ns)):
            dark = (hi - v[i, j] if low else v[i, j] - lo) > 0.8 * (hi - lo)
            ax.text(j, i, f'{v[i, j]:.3f}', ha='center', va='center', fontsize=6.5,
                    color='white' if dark else INK)
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels(ns)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('In-vivo training subjects')
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([full_name(c) for c in conds])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_ticks(list(tools.values()), labels=[])
    cb.ax.tick_params(length=3, width=0.8)
    items = sorted((y, f'{t} {y:.3f}') for t, y in tools.items())
    for (_, text), y in zip(items, spread_labels([y for y, _ in items], 0.075 * (hi - lo))):
        cb.ax.text(1.6, y, text, transform=cb.ax.get_yaxis_transform(), va='center')
    cb.set_label(label_, labelpad=62)
    save(fig, R.args.out, name)
    plt.close(fig)


def paper_ladders(R, name):
    """Stage A: every variant's strength ladder, selection-set MOSAE at 1 and 8 subjects."""
    import re
    from matplotlib.lines import Line2D
    plt = paper_style()
    with open(os.path.join(S.SCREEN, 'plan.json')) as f:
        plan = json.load(f)
    with open(os.path.join(S.SCREEN, 'picks.json')) as f:
        picks = json.load(f)
    ladders = {}
    for j in plan:
        ladders.setdefault(j['variant'], []).append(j)
    cols = 5
    rows = -(-len(ladders) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(COL2, 1.42 * rows + 0.3), sharey=True)
    for ax, (variant, jobs) in zip(axes.ravel(), ladders.items()):
        x = np.arange(len(jobs))
        k = [j['label'] for j in jobs].index(picks[variant]['label'])
        ax.axvspan(k - 0.35, k + 0.35, color=AUGMENTRUM, alpha=0.2, lw=0)
        for n, face in ((1, 'white'), (8, INK)):
            y = []
            for j in jobs:
                p = S.result_path(os.path.join(S.SCREEN, 'A'), j['name'], n)
                if os.path.isfile(p):
                    with open(p) as f:
                        y.append(json.load(f)['selected_mosae'])
                else:
                    y.append(np.nan)
            ax.plot(x, y, '-o', color=INK, mfc=face, ms=3.2, mew=0.9, lw=1.0)
        title, xlabel = LADDERS[variant]
        ax.set_xticks(x)
        ax.set_xticklabels([re.sub(r'^[a-z]+', '', j['label']).replace('p', '.') for j in jobs])
        ax.set_title(title, fontsize=7, pad=3)
        ax.set_xlabel(xlabel, fontsize=6.5, labelpad=1)
        grid(ax, 'y')
    for ax in axes.ravel()[len(ladders):]:
        ax.axis('off')
    for ax in axes[:, 0]:
        ax.set_ylabel('Selection MOSAE')
    handles = [Line2D([], [], color=INK, marker='o', mfc='white', ms=3.2, mew=0.9, lw=1.0,
                      label=subjects(1)),
               Line2D([], [], color=INK, marker='o', mfc=INK, ms=3.2, lw=1.0, label=subjects(8)),
               plt.Rectangle((0, 0), 1, 1, color=AUGMENTRUM, alpha=0.2, lw=0,
                             label='Picked (lowest mean of the two)')]
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0),
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.9, w_pad=0.5)
    save(fig, R.args.out, name)
    plt.close(fig)


def paper_convergence(R, n, name, bins=60):
    """Selection-set MOSAE over training at *n* subjects (median per log-spaced bin), every
    stage-B condition, one panel per family."""
    import pandas as pd
    curves = {}

    def curve(c):
        if c not in curves:
            d = pd.read_csv(os.path.join(R.run_dir(c, n), 'curve.csv')).dropna(subset=['sel_mosae'])
            b = np.digitize(d.step, np.geomspace(d.step.min(), d.step.max() + 1, bins))
            curves[c] = d.groupby(b).agg(step=('step', 'median'), y=('sel_mosae', 'median'))
        return curves[c]

    def draw(ax, c, color, marker):
        g = curve(c)
        ax.plot(g.step, g.y, color=color, lw=1.1, zorder=3)
        return g.y.iloc[-1]
    tails = [curve(c).y[curve(c).step > 3e5] for c in
             ['none'] + [c for _, m in screen_groups() for c in m]]
    lo = min(curve(c).y.min() for c in ['none'] + [c for _, m in screen_groups() for c in m])
    hi = max(t.max() for t in tails)
    family_panels(R, draw, 'Training steps', 'Selection MOSAE', name, 1.4e6, xlog=True,
                  xlim=(1e3, 1e6), note=f'Networks trained on {subjects(n)}',
                  ylim=(lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo)))


def paper_binned(R, x, xlabel, name, ns=(1, 8), bins=6):
    """Mean per-spectrum MOSAE (± SE) in equal-count bins of *x*, the networks at 1 and at 8
    subjects side by side: tools dashed, networks solid."""
    plt = paper_style()
    idx = np.empty(len(x), int)
    idx[np.argsort(x, kind='stable')] = np.arange(len(x)) * bins // len(x)
    centres = np.array([np.median(x[idx == b]) for b in range(bins)])
    fig, axes = plt.subplots(1, len(ns), figsize=(COL2 + 0.55, 2.8), sharey=True)
    fig.subplots_adjust(right=0.84, wspace=0.55)
    for ax, n in zip(axes, ns):
        items = []
        for text, con, color, marker, filled in paper_series(R, n):
            err = R.spectrum_mosae(con)
            m = np.array([mean_se(err[idx == b]) for b in range(bins)])
            ax.errorbar(centres, m[:, 0], yerr=m[:, 1], color=color, lw=1.2, capsize=0,
                        marker=marker, ms=3.4, mfc=color if filled else 'white', mew=0.9,
                        ls='-' if filled else DASH, zorder=3)
            items.append((m[-1, 0], text, color))
        lo, hi = ax.get_ylim()
        side_labels(ax, items, centres[-1] + 0.06 * (centres[-1] - centres[0]), 0.065 * (hi - lo))
        ax.set_title(f'Networks trained on {subjects(n)}', fontsize=7.5, loc='left')
        ax.set_xlabel(xlabel)
        grid(ax, 'y')
    axes[0].set_ylabel('MOSAE per spectrum')
    save(fig, R.args.out, name)
    plt.close(fig)


def naa_linewidth(ts, points=2 ** 16, chunk=50):
    """
    The FWHM (Hz) of each test spectrum's NAA singlet as simulated: its Lorentzian, the shared
    Gaussian and Osprey's lineshape kernel (svs_ablation.simulate_rows), the FID continued to
    *points* for a fine frequency grid.
    """
    basis = S.load_basis(S.BASIS_DIR)
    dt = float(basis.t[1] - basis.t[0])
    t = np.arange(points) * dt
    f = np.fft.fftshift(np.fft.fftfreq(points, d=dt))
    i = ts.names.index('NAA')
    gamma, sigma = ts.params['gamma'][:, i], np.asarray(ts.params['sigma']).ravel()
    kernel = np.load(ts.path)['kernel']
    j = (kernel.shape[1] - 1) // 2 - np.arange(kernel.shape[1])
    shift = np.exp(-2j * np.pi * basis.bw / (2 * len(basis.t)) * np.outer(j, t))
    fwhm = []
    for s in range(0, len(gamma), chunk):
        fid = (np.exp(-(gamma[s:s + chunk, None] + sigma[s:s + chunk, None] ** 2 * t) * t)
               * (kernel[s:s + chunk] @ shift))
        lines = np.fft.fftshift(np.fft.fft(fid, axis=-1), axes=-1).real
        for line in lines:
            above = f[line >= line.max() / 2]
            fwhm.append(above.max() - above.min())
    return np.array(fwhm)


def paper_invivo(R, n, name, invivo='results/cows/invivo',
                 processed='results/cows/processed_scans.npz'):
    """
    One held-out in-vivo scan (fold 0's validation subjects, the median of FSL-MRS's residual /
    noise) fitted by each tool and by the headline networks at *n* subjects: data black, fit in
    the method's colour, residual above; residual / noise as svs_ablation.py invivo computes it.
    """
    import pandas as pd
    import torch
    plt = paper_style()
    with open(os.path.join(R.run_dir(FEATURED, n), 'result.json')) as f:
        held_out = json.load(f)['data']['val_subjects']
    q = pd.read_csv(os.path.join(invivo, 'invivo_quality.csv')).set_index('stem')
    q = q[[s.split('_')[0] in held_out for s in q.index]]
    stem = q.fsl_default.sort_values().index[len(q) // 2]
    z = np.load(processed)
    fid = z['fids'][[str(s) for s in z['stems']].index(stem)]
    basis = S.load_basis(S.BASIS_DIR)
    w = (basis.ppm >= S.PPM_WINDOW[0]) & (basis.ppm <= S.PPM_WINDOW[1])
    spec = np.fft.fft(fid)
    noise = S.spec_noise_sd(spec, basis.ppm)
    panels = []
    for m, text, color, _ in PAPER_TOOLS:
        t = np.load(os.path.join(invivo, f'invivo_{m}.npz'), allow_pickle=True)
        i = [str(x) for x in t['stems']].index(stem)
        d = np.asarray(t['curve_data'][i], complex).real
        f = np.asarray(t['curve_fit'][i], complex).real
        x = np.asarray(t['curve_ppm'][i], float) if 'curve_ppm' in t else basis.ppm
        k = (x >= S.PPM_WINDOW[0]) & (x <= S.PPM_WINDOW[1])
        panels.append((text, color, x[k], d[k], f[k], q.loc[stem, m]))
    x_in = torch.as_tensor(np.stack((spec.real, spec.imag))[None], dtype=torch.float32)
    for c, text, color, _ in PAPER_NETS + (ALL_BEST,):
        _, model, net = S.load_run(R.run_dir(c, n), R.args.weights, 'cpu')
        with torch.no_grad():
            fit = model(net.predict(x_in).to(model.basis_ri.dtype))[0].real.numpy()
        ratio = np.sqrt(np.mean((spec.real[w] - fit[w]) ** 2)) / noise
        panels.append((f'{text}, {subjects(n)}', color, basis.ppm[w], spec.real[w], fit[w], ratio))
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 3.7), sharex=True)
    for ax, (text, color, x, d, f, ratio) in zip(axes.ravel(), panels):
        o = np.argsort(x)
        x, d, f = x[o], d[o], f[o]
        s_ = np.abs(d).max()
        ax.plot(x, d / s_, color='black', lw=0.8)
        ax.plot(x, f / s_, color=color, lw=1.2, alpha=0.9)
        ax.plot(x, (d - f) / s_ + 1.2, color=color, lw=0.8)
        ax.set_title(f'{text}\nresidual / noise {ratio:.2f}', loc='left', color=color,
                     fontsize=7.5, pad=2)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_ylim(-0.15, 1.4)
    axes[0, 0].set_xlim(S.PPM_WINDOW[1], S.PPM_WINDOW[0])
    for ax in axes[1]:
        ax.set_xlabel('Chemical shift [ppm]')
    fig.tight_layout(h_pad=1.2, w_pad=1.5)
    save(fig, R.args.out, name)
    plt.close(fig)
    return stem


def paper_augmentations(R, name, stem='sub-01_acq-01_vapor7_metab_PFL', draws=6):
    """
    What each main-paper augmentation does to one held-out in-vivo scan: the scan as processed
    without augmentation (grey, thick) and *draws* augmented versions (the condition's colour),
    each drawn through the training pipeline itself (sampling on the raw coils and transients,
    processing, modules). Real part over the network's window, each spectrum over its own
    maximum magnitude, as the network normalises its input.
    """
    import torch
    plt = paper_style()
    data, water, names, _ = S.load_scans([stem.split('_')[0]])
    i = list(names).index(stem)
    basis = S.load_basis(S.BASIS_DIR)
    first, last = basis.window()
    ppm = basis.ppm[first:last]

    def spectrum(cond, seed):
        fid = S.process_scans(data[i:i + 1], water[i:i + 1], S.pipeline_spec(cond), 'cpu',
                              seed=seed).numpy()[0]
        spec = np.fft.fft(fid)[first:last]
        return spec.real / np.abs(spec).max()
    torch.set_num_threads(8)
    ref = spectrum('none', 0)
    panels = [x for _, m in MAIN_GROUPS for x in m]
    cols = 3
    rows = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(COL2, 1.55 * rows + 0.35), sharex=True)
    for ax, (cond, text, color, _) in zip(axes.ravel(), panels):
        S.register_augment(os.path.join(S.SCREEN, 'B', 'specs', f'{cond}.json'))
        ax.plot(ppm, ref, color=REF, lw=2.2, zorder=1)
        for k in range(draws):
            ax.plot(ppm, spectrum(cond, 100 + k), color=color, lw=0.6, alpha=0.8, zorder=2)
        ax.set_title(text, color=color, fontweight='bold', fontsize=7.5, loc='left', pad=2)
        strength = SHORT[cond].rsplit('(', 1)[-1].rstrip(')') if '(' in SHORT[cond] else SHORT[cond]
        ax.set_title(strength, color=color, fontsize=7, loc='right', pad=2)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
    for ax in axes.ravel()[len(panels):]:
        ax.axis('off')
    axes[0, 0].set_xlim(S.PPM_WINDOW[1], S.PPM_WINDOW[0])
    for ax in axes[-1]:
        ax.set_xlabel('Chemical shift [ppm]')
    fig.legend(handles=[plt.Line2D([], [], color=REF, lw=2.2, label='No augmentation'),
                        plt.Line2D([], [], color=INK, lw=0.6,
                                   label=f'{draws} augmented draws (colour of the condition)')],
               loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.9, w_pad=0.8)
    save(fig, R.args.out, name)
    plt.close(fig)


def paper_distributions(R, n, name):
    """Per-spectrum MOSAE on the test set: the tools, no augmentation and every main-paper
    condition at *n* subjects; box: quartiles, whiskers: 5th-95th percentile, dot: mean."""
    plt = paper_style()
    rows = [(text, R.tool(m), color, True) for m, text, color, _ in PAPER_TOOLS]
    rows.append(('No augmentation', R.net('none', n), NONE_COLOR, False))
    rows += [(text, R.net(c, n), color, False) for _, m in MAIN_GROUPS for c, text, color, _ in m]
    fig, ax = plt.subplots(figsize=(COL2 - 0.3, 0.24 * len(rows) + 0.6))
    for y, (text, con, color, tool) in enumerate(rows):
        e = R.spectrum_mosae(con)
        q5, q25, q50, q75, q95 = np.percentile(e, [5, 25, 50, 75, 95])
        ax.plot([q5, q95], [y, y], color=color, lw=1.0, zorder=2)
        ax.add_patch(plt.Rectangle((q25, y - 0.3), q75 - q25, 0.6, facecolor='white' if tool
                                   else color, edgecolor=color, lw=1.0, alpha=1 if tool else 0.35,
                                   zorder=3))
        ax.plot([q50, q50], [y - 0.3, y + 0.3], color=color, lw=1.4, zorder=4)
        ax.plot(e.mean(), y, 'o', color=color, ms=3.2, mec='white', mew=0.4, zorder=5)
    ax.axhline(len(PAPER_TOOLS) - 0.5, color=REF, lw=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    for tick, (_, _, color, _) in zip(ax.get_yticklabels(), rows):
        tick.set_color(color)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(f'MOSAE per test spectrum, networks at {subjects(n)}')
    grid(ax, 'x')
    save(fig, R.args.out, name)
    plt.close(fig)


def paper_gain(R, name):
    """
    What each main-paper condition gains over no augmentation at the same number of training
    subjects: MOSAE in percent (left, lower is better) and mean CCC (right, higher is better);
    open circle 1 subject, filled 8, the bar the range over 1-8.
    """
    from matplotlib.lines import Line2D
    plt = paper_style()
    ns = range(1, 9)
    rows, y, ypos = [], 0.0, {}
    for group, members in MAIN_GROUPS:
        rows.append((y, group))
        y += 1
        for c, text, color, marker in members:
            ypos[c] = (y, text, color)
            y += 1
        y += 0.35
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 0.2 * y + 0.75), sharey=True)
    for ax, metric, xlabel, change in (
            (axes[0], 'mosae', 'MOSAE against no augmentation [%]',
             lambda v, b: 100 * (v - b) / b),
            (axes[1], 'ccc_mean', "Mean CCC against no augmentation",
             lambda v, b: v - b)):
        base = np.array([stored(R, 'none', n)[metric] for n in ns])
        for c, (yy, _, color) in ypos.items():
            d = change(np.array([stored(R, c, n)[metric] for n in ns]), base)
            ax.plot([d.min(), d.max()], [yy, yy], color=color, lw=3.2, alpha=0.25,
                    solid_capstyle='round', zorder=2)
            ax.plot(d[0], yy, 'o', color=color, mfc='white', mew=1.1, ms=4.2, zorder=3)
            ax.plot(d[-1], yy, 'o', color=color, ms=4.2, mec='white', mew=0.4, zorder=4)
        ax.axvline(0, color=NONE_COLOR, lw=0.9, zorder=1)
        ax.set_xlabel(xlabel)
        grid(ax, 'x')
    for yy, group in rows:
        axes[0].text(0.0, yy, group, transform=axes[0].get_yaxis_transform(), ha='left',
                     va='center', fontweight='bold')
    axes[0].set_yticks([v[0] for v in ypos.values()])
    axes[0].set_yticklabels([v[1] for v in ypos.values()])
    for tick, (_, _, color) in zip(axes[0].get_yticklabels(), ypos.values()):
        tick.set_color(color)
    axes[0].set_ylim(y - 0.2, -0.7)
    for ax in axes:
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
    axes[0].legend(handles=[
        Line2D([], [], ls='none', marker='o', mfc='white', color=INK, mew=1.1, ms=4.2,
               label=subjects(1)),
        Line2D([], [], ls='none', marker='o', color=INK, ms=4.2, label=subjects(8)),
        Line2D([], [], color=INK, lw=3.2, alpha=0.25, label='Range over 1–8')],
        loc='lower left', ncol=3, bbox_to_anchor=(0.0, 1.0), handletextpad=0.2,
        columnspacing=1.0)
    fig.tight_layout(w_pad=1.5)
    save(fig, R.args.out, name)
    plt.close(fig)


def paper_selection_check(R, name):
    """Model selection: every stage-B run's selection-set MOSAE (what picks its checkpoint and
    the screen's strengths) against its test MOSAE; Pearson r and Spearman rho."""
    from scipy.stats import spearmanr
    plt = paper_style()
    conds = ['none'] + [c for _, m in screen_groups() for c in m]
    x, y = [], []
    for c in conds:
        for n in range(1, 9):
            with open(os.path.join(R.run_dir(c, n), 'result.json')) as f:
                r = json.load(f)
            x.append(r['selected_mosae'])
            y.append(r['test'][R.args.weights]['mosae'])
    x, y = np.array(x), np.array(y)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.scatter(x, y, s=9, color=AUGMENTRUM, alpha=0.6, lw=0, zorder=3)
    k, b = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, k * xs + b, color=INK, lw=1.0, zorder=2)
    ax.text(0.04, 0.96, f'r {num(np.corrcoef(x, y)[0, 1])}\n'
                        f'ρ {num(spearmanr(x, y).statistic)}\n{len(x)} runs',
            transform=ax.transAxes, ha='left', va='top')
    ax.set_xlabel('Selection-set MOSAE')
    ax.set_ylabel('Test-set MOSAE')
    grid(ax)
    save(fig, R.args.out, name)
    plt.close(fig)


def fig_paper(R):
    """Every figure of the paper: <out>/main (the selected conditions) and <out>/appendix (all)."""
    out = R.args.out
    R.args.out = os.path.join(out, 'main')
    paper_accuracy(R, 'mosae', 'MOSAE', 'fig_accuracy_mosae')
    paper_accuracy(R, 'ccc_mean', "Lin's CCC, mean over metabolites", 'fig_accuracy_ccc')
    paper_gain(R, 'fig_gain')
    paper_screen(R, 'mosae', 'MOSAE (dot: mean over 1–8 subjects; bar: range)', 'fig_screen_mosae',
                 groups=MAIN_GROUPS)
    paper_grid(R, 'mosae', 'MOSAE', 'fig_grid_mosae', conds=MAIN)
    paper_metabolite_grid(R, 1, 'ccc', 'fig_metabolites_ccc_n1', conds=MAIN)
    paper_bias(R, 1, 'fig_bias_n1')
    paper_fits(R, 1, 'fig_fits_n1')
    paper_distributions(R, 1, 'fig_distributions_n1')
    paper_augmentations(R, 'fig_augmentations')
    paper_invivo(R, 8, 'fig_invivo_fits')
    R.args.out = os.path.join(out, 'appendix')
    paper_families(R, 'mosae', 'MOSAE', 'fig_groups_mosae', groups=MAIN_GROUPS)
    paper_distributions(R, 8, 'fig_distributions_n8')
    paper_families(R, 'ccc_mean', "Lin's CCC, mean", 'fig_groups_ccc', groups=MAIN_GROUPS)
    paper_screen(R, 'ccc_mean', "Lin's CCC (dot: mean over 1–8 subjects; bar: range)",
                 'fig_screen_ccc', groups=MAIN_GROUPS)
    paper_grid(R, 'ccc_mean', "Lin's CCC, mean over metabolites", 'fig_grid_ccc', conds=MAIN)
    paper_families(R, 'mosae', 'MOSAE', 'fig_families_mosae')
    paper_families(R, 'ccc_mean', "Lin's CCC, mean", 'fig_families_ccc')
    paper_screen(R, 'mosae', 'MOSAE (dot: mean over 1–8 subjects; bar: range)',
                 'fig_screen_all_mosae')
    paper_screen(R, 'ccc_mean', "Lin's CCC (dot: mean over 1–8 subjects; bar: range)",
                 'fig_screen_all_ccc')
    paper_grid(R, 'mosae', 'MOSAE', 'fig_grid_all_mosae')
    paper_grid(R, 'ccc_mean', "Lin's CCC, mean over metabolites", 'fig_grid_all_ccc')
    paper_ladders(R, 'fig_ladders')
    paper_selection_check(R, 'fig_selection_check')
    paper_metabolite_grid(R, 1, 'mosae', 'fig_metabolites_mosae_n1', conds=MAIN)
    for n in (1, 8):
        for metric in ('ccc', 'mosae'):
            paper_metabolite_grid(R, n, metric, f'fig_metabolites_all_{metric}_n{n}')
    paper_metabolite_grid(R, 8, 'ccc', 'fig_metabolites_ccc_n8', conds=MAIN)
    paper_bias(R, 8, 'fig_bias_n8')
    paper_fits(R, 8, 'fig_fits_n8')
    for n in (1, 8):
        paper_convergence(R, n, f'fig_convergence_n{n}')
    paper_binned(R, np.asarray(R.ts.snr, float), 'SNR (NAA)', 'fig_snr')
    paper_binned(R, naa_linewidth(R.ts), 'NAA linewidth [Hz]', 'fig_linewidth')
    R.args.out = out


#******************#
#   in-vivo fits   #
#******************#
PANELS = (('fsl_default', 'fsl_pb', 'lcmodel'), ('osprey',))
PANEL_NAMES = {'fsl_default': 'FSL-MRS: Voigt, shared widths',
               'fsl_pb': 'FSL-MRS PB: Voigt + a bounded Lorentzian per metabolite',
               'lcmodel': 'LCModel, defaults', 'osprey': 'Osprey, defaults'}


def fig_invivo(args):
    """
    The processed scans fitted by every method, side by side (svs_ablation.py invivo): one
    figure per percentile of FSL-MRS default's residual / noise, data black, fit red, residual
    grey above; FSL-MRS on the scan's points, LCModel and Osprey on their own.
    """
    import pandas as pd
    plt = style()
    q = pd.read_csv(os.path.join(args.invivo, 'invivo_quality.csv')).set_index('stem')
    z = {m: np.load(os.path.join(args.invivo, f'invivo_{m}.npz'), allow_pickle=True)
         for m in TOOLS if os.path.isfile(os.path.join(args.invivo, f'invivo_{m}.npz'))}
    stems = [str(s) for s in z['fsl_default']['stems']]
    ppm = S.load_basis(args.basis_dir).ppm
    w = (ppm >= S.PPM_WINDOW[0]) & (ppm <= S.PPM_WINDOW[1])
    order = q['fsl_default'].sort_values()
    keys = [m for row in PANELS for m in row if m in z]
    for p in args.percentiles:
        stem = order.index[int(round(p / 100 * (len(order) - 1)))]
        i = stems.index(stem)
        cols = 2
        rows = -(-len(keys) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(8.0, 2.6 * rows), sharex=True, sharey=True,
                                 squeeze=False)
        scale = None
        for ax, m in zip(axes.ravel(), keys):
            d = np.asarray(z[m]['curve_data'][i], complex)       # object arrays of curves
            f = np.asarray(z[m]['curve_fit'][i], complex)
            if 'curve_ppm' in z[m]:
                x = np.asarray(z[m]['curve_ppm'][i], float)
                k = (x >= S.PPM_WINDOW[0]) & (x <= S.PPM_WINDOW[1])
                x, d, f = x[k], np.real(d[k]), np.real(f[k])
            else:
                x, d, f = ppm[w], np.real(d[w]), np.real(f[w])
            s = np.abs(d).max()
            d, f = d / s, f / s
            ax.plot(x, d, color='k', lw=0.7, label='Data')
            ax.plot(x, f, color='r', lw=1.0, alpha=0.6, label='Fit')
            ax.plot(x, d - f + 1.25, color='dimgray', lw=0.6, label='Residual (offset)')
            ax.set_title(f'{PANEL_NAMES[m]}\nresidual / noise {q.loc[stem, m]:.2f}', fontsize=8,
                         loc='left')
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
        for ax in axes.ravel()[len(keys):]:
            ax.axis('off')
        axes[0, 0].set_xlim(S.PPM_WINDOW[1], S.PPM_WINDOW[0])
        axes[0, 0].set_ylim(-0.15, 1.45)
        for ax in axes[-1]:
            ax.set_xlabel('Chemical shift (ppm)')
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper right', ncol=3,
                   fontsize=8, frameon=False)
        fig.suptitle(f'{stem}: {p}th percentile of FSL-MRS default\'s residual / noise',
                     fontsize=9, x=0.01, ha='left')
        fig.tight_layout()
        save(fig, args.out, f'fig_invivo_fits_p{p}')
        plt.close(fig)


#******************#
#   module check   #
#******************#
def proposed_modules(ranges):
    """"MODULES" with the ranges of "svs_ablation.py ranges" (the 'proposed' column) put in."""
    import copy
    mods = copy.deepcopy(S.MODULES)
    for mod, params in ranges.items():
        step = mods[mod][mod]
        for par, e in params.items():
            value = tuple(e['proposed'])
            if mod == 'artificial_peaks':
                step['peaks'][0][par] = value
            elif mod == 'spurious_echoes':
                step['echoes'][0][par] = value
            else:
                step[par] = value
    return mods


def module_draws(fid, cf, bw, spec, draws, seed):
    """*draws* draws of one pipeline step on one processed scan: (before, after) spectra (T,)."""
    import warnings
    import torch
    from augmentrum import Augmentrum
    from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
    nii = gen_nifti_mrs(np.asarray(fid, np.complex128).reshape(1, 1, 1, -1), dwelltime=1.0 / bw,
                        spec_freq=cf, nucleus='1H')
    nii.add_hdr_field('EchoTime', 0.026)
    nii.add_hdr_field('SubjectID', 'sub-00', doc='Subject identifier')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        aug = Augmentrum(data=[nii] * draws, water=None, split_indices={'t': list(range(draws))},
                         pipelines={'t': ['tap:clean', spec]}, modes={'t': 'on-the-fly'},
                         outputs={'t': ('data', 'clean')}, batch_size=draws, device='cpu',
                         seed=seed, volatile=True)
        d, c = next(iter(aug.dataloader(split='t')))
    to = lambda x: torch.fft.fft(x.reshape(x.shape[0], -1), dim=-1).numpy()
    return to(c)[0], to(d)


def fig_module_check(args):
    """
    Every module on one processed scan (the median SNR), current ranges (left) against the ranges
    "svs_ablation.py ranges" proposes (right): the scan in grey, six draws in black, real part
    over the network's window, divided by the scan's maximum.
    """
    plt = style()
    with open(args.ranges) as f:
        ranges = json.load(f)
    z = np.load(args.processed)
    basis = S.load_basis(args.basis_dir)
    snr = S.scan_snr(z['fids'], basis.ppm)
    i = int(np.argsort(snr)[len(snr) // 2])
    fid, cf, bw = z['fids'][i], float(z['cf']), float(z['bw'])
    first, last = basis.window()
    ppm = basis.ppm[first:last]
    o = np.argsort(ppm)
    new = proposed_modules(ranges)
    mods = list(S.MODULES)
    fig, axes = plt.subplots(len(mods), 2, figsize=(7.0, 1.15 * len(mods)), sharex=True)
    for r, m in enumerate(mods):
        for c, (spec, head) in enumerate(((S.MODULES[m], 'Current'), (new[m], 'Proposed'))):
            ax = axes[r, c]
            before, after = module_draws(fid, cf, bw, spec, 6, seed=11 + r)
            scale = np.abs(before[first:last].real).max()
            ax.plot(ppm[o], before[first:last].real[o] / scale, color=REF, lw=1.6)
            for a in after:
                ax.plot(ppm[o], a[first:last].real[o] / scale, color=INK, lw=0.45, alpha=0.7)
            ax.set_xlim(S.PPM_WINDOW[1], S.PPM_WINDOW[0])
            ax.yaxis.set_visible(False)
            ax.spines['left'].set_visible(False)
            step = spec[m]
            text = ', '.join(f'{k} {v}' for k, v in step.items() if k not in ('mode', 'mm_source'))
            ax.set_title(f'{NAMES.get(m, m)}: {head.lower()}\n{text[:70]}', fontsize=7,
                         loc='left', pad=2)
    for ax in axes[-1]:
        ax.set_xlabel('Chemical shift (ppm)')
    fig.suptitle(f'{str(z["stems"][i])} (median SNR); grey: the scan, black: six draws',
                 fontsize=8.5, x=0.01, ha='left')
    fig.tight_layout(h_pad=0.8, w_pad=1.2)
    save(fig, args.out, 'fig_module_check')
    plt.close(fig)


#**********#
#   main   #
#**********#
FIGURES = {'scaling': fig_scaling, 'samplers': fig_samplers, 'modules': fig_modules,
           'bias': fig_bias, 'metabolites': fig_metabolites, 'snr': fig_snr,
           'linewidth': fig_linewidth, 'curves': fig_curves, 'paper': fig_paper}


#: the convergence figures: (source, column, y label, note, file name, log y); source 'curve' is
#: the run's own curve.csv (every evaluation), 'rescore' its checkpoints on the given sets
CONVERGENCE = (
    ('curve', 'train_loss', 'Train MSE',
     'Training batches (augmented), mean since the last evaluation', 'train_mse', True),
    ('curve', 'val_loss', 'Val MSE (in vivo)',
     "The fold's two held-out subjects, all coils and transients", 'val_mse', True),
    ('rescore', 'sel_mosae', 'Val MOSAE (simulated)',
     'Selection set, 1000 simulated spectra, every 50k steps', 'val_mosae', False),
    ('rescore', 'test_mosae', 'Test MOSAE (simulated)',
     'Test set, 1000 simulated spectra, every 50k steps', 'test_mosae', False))


def spread_labels(ys, gap):
    """Label heights for sorted line ends *ys*, at least *gap* apart: each crowded group centred on
    its lines' mean, so no label moves further than it must."""
    groups = [[y] for y in ys]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups) - 1):
            a, b = groups[i], groups[i + 1]
            top_a = np.mean(a) + (len(a) - 1) * gap / 2
            low_b = np.mean(b) - (len(b) - 1) * gap / 2
            if low_b - top_a < gap:
                groups[i:i + 2] = [a + b]
                merged = True
                break
    out = []
    for g in groups:
        out += list(np.mean(g) + (np.arange(len(g)) - (len(g) - 1) / 2) * gap)
    return out


def fig_convergence(args):
    """
    The runs of an experiment folder over training, one figure per quantity ("CONVERGENCE") and
    signal model: no augmentation in grey, the other condition in teal, light at 1 subject, dark
    at 8; open markers at the checkpoint the selection set picks. The test-set figure carries the
    tools' MOSAE on the same set (--bench) as grey levels.
    """
    import pandas as pd
    plt = style()
    name = os.path.basename(args.testset)[:-4]
    runs = {}
    for d in sorted(glob.glob(os.path.join(args.exp, 'runs', '*__n*__f0__*__s0'))):
        cond, n, _, variant, _ = os.path.basename(d).split('__')
        rs = os.path.join(d, S.rescore_name(args.testset))
        if not os.path.isfile(rs):
            continue
        runs[(variant, cond, int(n[1:]))] = (pd.read_csv(os.path.join(d, 'curve.csv')),
                                             pd.read_csv(rs))
    tools = {}
    for m, text in TOOLS.items():
        path = os.path.join(args.bench or '', f'{name}_{m}.npz')
        if args.bench and os.path.isfile(path):
            ts = S.TestSet(args.testset)
            tools[text] = S.concentration_metrics(np.load(path)['con'], ts.concentrations,
                                                  ts.names)
    for variant in sorted({k[0] for k in runs}):
        conds = sorted({k[1] for k in runs if k[0] == variant}, key=lambda c: c != 'none')
        last = max(int(r[0].step.iloc[-1]) for k, r in runs.items() if k[0] == variant)
        xmax = np.ceil(last / 1e6)
        for source, column, ylabel, note, fname, log in CONVERGENCE:
            levels = ({text: m['mosae'] for text, m in tools.items()}
                      if column == 'test_mosae' else {})
            fig, ax = plt.subplots(figsize=(4.6, 2.9))
            ends = []
            for cond in conds:
                dark, light = (INK, INK_LIGHT) if cond == 'none' else (ACCENT, ACCENT_LIGHT)
                for n, color in ((1, light), (8, dark)):
                    if (variant, cond, n) not in runs:
                        continue
                    curve, rescore = runs[(variant, cond, n)]
                    c = (curve[curve.step >= 5000] if source == 'curve' else rescore)
                    x, y = c.step.values / 1e6, c[column].values
                    ax.plot(x, y, color=color, lw=0.8 if source == 'curve' else 1.0, zorder=3)
                    sel = rescore.loc[rescore.sel_mosae.idxmin(), 'step']
                    k = int(np.abs(c.step.values - sel).argmin())
                    ax.plot(x[k], y[k], 'o', ms=3.5, color=color, mfc='white', mew=1.0, zorder=4)
                    tail = np.median(y[-20:]) if source == 'curve' else y[-1]
                    ends.append((np.log10(tail) if log else tail,
                                 f'{NAMES.get(cond, cond)}, {subjects(n)}', INK))
            if levels:
                for text, v in levels.items():
                    ax.axhline(v, color=REF, lw=0.8, ls='--', zorder=2)
                    ends.append((v, text, REF_TEXT))
            if log:
                ax.set_yscale('log')
            else:
                top = max(np.percentile(r[1][column], 95) for k, r in runs.items()
                          if k[0] == variant)
                ax.set_ylim(None, max(top, *levels.values()) * 1.05 if levels else top * 1.05)
            lo, hi = ax.get_ylim()
            span = np.log10(hi) - np.log10(lo) if log else hi - lo
            items = sorted(ends)
            placed = spread_labels([yv for yv, _, _ in items], 0.052 * span)
            for (_, text, color), yl in zip(items, placed):
                ax.text(xmax * 1.02, 10 ** yl if log else yl, text, va='center', ha='left',
                        fontsize=7.5, color=color, clip_on=False)
            ax.set_xlim(0, xmax)
            ax.set_xlabel('Training step (millions)')
            ax.set_ylabel(ylabel)
            ax.text(0.0, 1.02, f'{note}; model {variant}; open markers: selected checkpoints '
                               f'(at {last / 1e6:.1f}M)',
                    transform=ax.transAxes, ha='left', va='bottom', fontsize=7, color=REF_TEXT)
            grid(ax, 'y')
            save(fig, args.out, f'fig_convergence_{variant}_{fname}')
            plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description='The figures of the single-voxel ablation.',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('figure', choices=list(FIGURES) + ['invivo', 'module-check',
                                                         'convergence'])
    ap.add_argument('--exp', help="an experiment folder of svs_ablation.py train (runs/)")
    ap.add_argument('--testset', help='the test set the networks and tools are scored on')
    ap.add_argument('--bench', help="svs_ablation.py bench's output folder")
    ap.add_argument('--weights', default='selected', choices=('selected', 'final'))
    ap.add_argument('--variant', default='A', choices=('A', 'PB'))
    ap.add_argument('--n', type=int, default=1, help='training subjects of the networks shown')
    ap.add_argument('--invivo', help="svs_ablation.py invivo's output folder")
    ap.add_argument('--percentiles', type=int, nargs='+', default=[10, 50, 90])
    ap.add_argument('--basis-dir', default=S.BASIS_DIR)
    ap.add_argument('--ranges', help="svs_ablation.py ranges' JSON (module-check)")
    ap.add_argument('--processed', help="svs_ablation.py process' output (module-check)")
    ap.add_argument('--out', default='results/cows/figures')
    args = ap.parse_args(argv)
    if args.figure == 'invivo':
        fig_invivo(args)
    elif args.figure == 'module-check':
        fig_module_check(args)
    elif args.figure == 'convergence':
        fig_convergence(args)
    else:
        FIGURES[args.figure](Results(args))


if __name__ == '__main__':
    sys.exit(main())