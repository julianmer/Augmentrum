####################################################################################################
#                                       spatial_ablation.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-17                                                                              #
#                                                                                                  #
# Purpose: Ablate Augmentrum's spatial augmentation on MRSI reconstruction. Networks learn to      #
#          turn an undersampled acquisition into the same scan fully sampled, on MRSI              #
#          Challenge data with the trajectory, coils, noise and undersampling simulated on the     #
#          fly: a 3-D U-Net, Deep-ER and a conditional diffusion model.                            #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from augmentrum.sampling.kspace_reconstructor import GriddingNUFFT

import train_deep_er as deep_er


#*******************#
#   configuration   #
#*******************#
#: The networks, each with the learning rate it was published with: Deep-ER (Weiser et al. 2025),
#: an unrolled network in the variational network's design (Hammernik et al. 2018, Sriram et al.
#: 2020) with NC-PDNet's density-compensated operator (Ramzi et al. 2022), a conditional diffusion
#: model in SR3's design (Saharia et al. 2022) and the fastMRI U-Net baseline (Zbontar et al.
#: 2018). In the grid's order, the slowest first.
MODELS = {'deep-er': 1e-5, 'varnet': 1e-3, 'diffusion': 1e-4, 'unet': 1e-3}
#: The spatial conditions: none, each of Augmentrum's transforms alone (sizes in the Deep-ER
#: script's SPATIAL), all of them, and Deep-ER's own augmentation in image space (its PAUL).
SPATIAL = ('none',) + deep_er.SPATIAL_KINDS + ('all', 'paul')
#: What this study fixes of the Deep-ER script's acquisition: no spectral augmentation, and the
#: target is the input's own scan fully sampled, the challenge's noise in its coil images, so in
#: the input and the target alike.
DEEP_ER_SETTINGS = dict(arm='none', target='noisy', noise=False, noise_sigma_k=0.0,
                        ksp_mode='nufft', crop_t=None, pool_on_gpu=None)
#: U-Net channels at the first level, doubled at each of its poolings (fastMRI's 32; three
#: poolings take the 64 x 64 x 32 grid to 8 x 8 x 4).
WIDTH, DEPTH = 32, 3
#: Diffusion: noise levels in training, on a cosine schedule (Nichol & Dhariwal 2021), and the
#: deterministic DDIM steps (Song et al. 2021) that sample an image.
DIFFUSION_STEPS, SAMPLING_STEPS = 1000, 50
#: The unrolled network's cascades, each with a U-Net of this width as its regulariser. The
#: data-consistency step is taken on the normal operator divided by its largest eigenvalue (3 at
#: acceleration 2 to 8 at 4 and above, POWER_ITERATIONS per trajectory), so the learned step lives
#: in (0, 2), the range in which a gradient step converges, starting at 1.
CASCADES, VARNET_WIDTH, POWER_ITERATIONS = 5, 16, 10


#************#
#   models   #
#************#
def sinusoid(t, dim: int):
    """The noise levels *t* (n,) as sinusoids of *dim* frequencies, (n, dim)."""
    freqs = torch.exp(-math.log(10000.0) * torch.arange(dim // 2, device=t.device) / (dim // 2))
    angles = t.float()[:, None] * freqs[None]
    return torch.cat((angles.sin(), angles.cos()), dim=1)


#**************************************************************************************************#
#                                           Class Block                                            #
#**************************************************************************************************#
#                                                                                                  #
# Two convolutions, each normed and activated, with the noise level added after the first.         #
#                                                                                                  #
#**************************************************************************************************#
class Block(nn.Module):
    """Two convolutions, each normed and activated, with the noise level added after the first."""

    def __init__(self, c_in: int, c_out: int, t_dim: int = 0):
        super().__init__()
        self.conv1, self.norm1 = nn.Conv3d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out)
        self.conv2, self.norm2 = nn.Conv3d(c_out, c_out, 3, padding=1), nn.GroupNorm(8, c_out)
        self.time = nn.Linear(t_dim, c_out) if t_dim else None

    def forward(self, x, emb=None):
        x = F.silu(self.norm1(self.conv1(x)))
        if self.time is not None:
            x = x + self.time(emb)[:, :, None, None, None]
        return F.silu(self.norm2(self.conv2(x)))


#**************************************************************************************************#
#                                           Class UNet3D                                           #
#**************************************************************************************************#
#                                                                                                  #
# A 3-D U-Net: average-pooled encoder, transposed-convolution decoder, skips by concatenation.     #
#                                                                                                  #
#**************************************************************************************************#
class UNet3D(nn.Module):
    """
    A 3-D U-Net: average-pooled encoder, transposed-convolution decoder, skips by concatenation.

    Laid out as fastMRI's baseline, WIDTH channels doubling over DEPTH poolings; *timed* adds
    the diffusion model's noise level to every block.
    """

    def __init__(self, c_in: int, c_out: int, timed: bool = False, width: int = WIDTH):
        super().__init__()
        self.width = width
        widths = [width * 2 ** i for i in range(DEPTH + 1)]
        t_dim = 4 * width if timed else 0
        self.time = nn.Sequential(nn.Linear(width, t_dim), nn.SiLU(),
                                  nn.Linear(t_dim, t_dim)) if timed else None
        self.down = nn.ModuleList(Block(a, b, t_dim) for a, b in zip([c_in] + widths, widths))
        self.up = nn.ModuleList(nn.ConvTranspose3d(c, c // 2, 2, stride=2) for c in widths[:0:-1])
        self.merge = nn.ModuleList(Block(c, c // 2, t_dim) for c in widths[:0:-1])
        self.out = nn.Conv3d(width, c_out, 1)

    def forward(self, x, t=None):
        emb = None if self.time is None else self.time(sinusoid(t, self.width))
        skips = []
        for block in self.down[:-1]:
            skips.append(block(x, emb))
            x = F.avg_pool3d(skips[-1], 2)
        x = self.down[-1](x, emb)
        for up, merge in zip(self.up, self.merge):
            x = merge(torch.cat((up(x), skips.pop()), dim=1), emb)
        return self.out(x)


#**************************************************************************************************#
#                                            Class UNet                                            #
#**************************************************************************************************#
#                                                                                                  #
# The fastMRI baseline in 3-D: a U-Net adding its correction to the input image.                   #
#                                                                                                  #
#**************************************************************************************************#
class UNet(UNet3D):
    """The fastMRI baseline in 3-D: a U-Net adding its correction to the input image."""

    def __init__(self):
        super().__init__(2, 2)

    def forward(self, img0, k0, sense, batch=None):
        return img0 + super().forward(img0).float(), None


#**************************************************************************************************#
#                                           Class VarNet                                           #
#**************************************************************************************************#
#                                                                                                  #
# An unrolled reconstruction: cascades of a U-Net and a data-consistency step through the          #
# acquisition's own operator.                                                                      #
#                                                                                                  #
#**************************************************************************************************#
class VarNet(nn.Module):
    """
    An unrolled reconstruction: cascades of a U-Net and a data-consistency step through the
    acquisition's own operator.

    From the gridded input x0, each cascade takes x <- x - w (N(x) - x0) / L + R(x): R its U-Net,
    w a learned step in (0, 2) and N the acquisition's normal operator, gain * adjoint(dcf *
    forward(.)) along the trajectory the input was measured on, so that N(x) - x0 is the gradient
    of the distance between x's samples and the measured ones; L is N's largest eigenvalue, by
    power iteration the first time a trajectory is seen. N acts on the coil-combined image in the
    acquisition's B0 frame: the coils' sum of squares is one, so it is within 1 % of the per-coil
    operator at a thirty-second of the cost.
    """

    def __init__(self):
        super().__init__()
        self.cascades = nn.ModuleList(UNet3D(2, 2, width=VARNET_WIDTH) for _ in range(CASCADES))
        for cascade in self.cascades:         # starts as data-consistency steps alone
            nn.init.zeros_(cascade.out.weight)
            nn.init.zeros_(cascade.out.bias)
        self.steps = nn.Parameter(torch.zeros(CASCADES))     # 2 sigmoid(0) = 1
        self.nufft, self.eigen = None, {}

    def gradient(self, x, x0, batch):
        """(N(x) - x0) / L of real-imaginary images (n, 2, X, Y, Z), in float32."""
        with torch.autocast('cuda', enabled=False):
            if self.nufft is None:
                self.nufft = GriddingNUFFT(tuple(x.shape[2:]))
            coords = batch['acq_coords'].cpu().numpy()
            apply = lambda v: (self.nufft.adjoint(self.nufft.forward(v * batch['b0'].conj(), coords)
                                                  * batch['acq_dcf'][None], coords)
                               * batch['acq_gain'] * batch['b0'])
            key = hash(coords.tobytes())
            if key not in self.eigen:
                with torch.no_grad():
                    v = torch.randn_like(batch['b0'][:1])
                    for _ in range(POWER_ITERATIONS):
                        v = apply(v / v.norm())
                    self.eigen[key] = v.norm()
            cplx = lambda v: torch.complex(v[:, 0].float(), v[:, 1].float())
            g = (apply(cplx(x)) - cplx(x0)) / self.eigen[key]
            return torch.stack((g.real, g.imag), dim=1)

    def forward(self, img0, k0, sense, batch=None):
        x = img0
        for cascade, w in zip(self.cascades, 2 * torch.sigmoid(self.steps)):
            x = x - w * self.gradient(x, img0, batch) + cascade(x).float()
        return x, None


#**************************************************************************************************#
#                                         Class Diffusion                                          #
#**************************************************************************************************#
#                                                                                                  #
# A conditional diffusion model in SR3's design: the input image conditions the denoiser.          #
#                                                                                                  #
#**************************************************************************************************#
class Diffusion(nn.Module):
    """
    A conditional diffusion model in SR3's design: the input image conditions the denoiser.

    The denoiser sees the noisy target and the input image as channels and predicts v (Salimans
    & Ho 2022), which keeps the few DDIM steps stable at every noise level. Sampling starts from
    fixed noise, so an input always gives the same image.
    """

    def __init__(self):
        super().__init__()
        self.denoiser = UNet3D(4, 2, timed=True)
        s = 0.008
        t = torch.arange(1, DIFFUSION_STEPS + 1, dtype=torch.float64) / DIFFUSION_STEPS
        alpha_bar = (torch.cos((t + s) / (1 + s) * math.pi / 2)
                     / math.cos(s / (1 + s) * math.pi / 2)) ** 2
        self.register_buffer('alpha_bar', alpha_bar.float(), persistent=False)

    def predict(self, x, img0, level):
        """v of the noisy images *x* at their noise *level*s, given the input images."""
        return self.denoiser(torch.cat((x, img0), dim=1), level).float()

    def loss(self, target, img0):
        """The v-prediction MSE at a random noise level per item."""
        level = torch.randint(0, DIFFUSION_STEPS, (len(target),), device=target.device)
        a = self.alpha_bar[level].view(-1, 1, 1, 1, 1)
        noise = torch.randn_like(target)
        x = a.sqrt() * target + (1 - a).sqrt() * noise
        return F.mse_loss(self.predict(x, img0, level), a.sqrt() * noise - (1 - a).sqrt() * target)

    def forward(self, img0, k0, sense, batch=None):
        """The image, sampled in SAMPLING_STEPS deterministic DDIM steps."""
        x = torch.randn(img0.shape, device=img0.device,
                        generator=torch.Generator(img0.device).manual_seed(0))
        levels = torch.linspace(DIFFUSION_STEPS - 1, 0, SAMPLING_STEPS,
                                device=img0.device).round().long()
        for k, level in enumerate(levels):
            a = self.alpha_bar[level]
            v = self.predict(x, img0, level.expand(len(x)))
            clean, noise = a.sqrt() * x - (1 - a).sqrt() * v, (1 - a).sqrt() * x + a.sqrt() * v
            a = self.alpha_bar[levels[k + 1]] if k + 1 < SAMPLING_STEPS else torch.ones_like(a)
            x = a.sqrt() * clean + (1 - a).sqrt() * noise
        return x, None


def build_model(args) -> nn.Module:
    """The network of --model; Deep-ER as published, its image from the k-space branch."""
    if args.model == 'unet':
        return UNet()
    if args.model == 'varnet':
        return VarNet()
    if args.model == 'diffusion':
        return Diffusion()
    return deep_er.DeepER(args.n_coils, channels_last=True, from_kspace=True).to(
        memory_format=torch.channels_last_3d)


def train_loss(model, loss_func, batch, amp: bool):
    """The diffusion model's loss on its noise; the others' Deep-ER image loss on their output."""
    if isinstance(model, Diffusion):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            return model.loss(batch['img_gt'], batch['inputs_img'])
    return deep_er.run_batch(model, loss_func, batch, amp)[0]


#**************#
#   training   #
#**************#
def deep_er_args(args) -> argparse.Namespace:
    """*args* as the Deep-ER script's pipeline and evaluation read them."""
    paul = args.spatial == 'paul'
    return argparse.Namespace(**{**vars(args), **DEEP_ER_SETTINGS, 'crop_t': args.crop_t,
                                 'target': args.target,
                                 # the noise-free target gets the noise in the input (INPUT_NOISE)
                                 'noise': args.target == 'truth',
                                 'spatial': 'none' if paul else args.spatial, 'paul': paul})


def run_name(args) -> str:
    crop = f"_t{args.crop_t}" if args.crop_t else ''
    target = '' if args.target == 'noisy' else f"_{args.target}"
    scale = '' if args.scale == 'timepoint' else f"_{args.scale}scale"
    loss = '' if args.loss == 'mse+ssim' else f"_{args.loss}"
    return (f"{args.model}_{args.spatial}_{args.trajectory}_n{args.train}_s{args.seed}"
            f"{crop}{target}{scale}{loss}")


def train(args):
    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    amp = args.amp and device.type == 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    settings = deep_er_args(args)
    deep_er.SCALE = args.scale
    aug = deep_er.build_train_data(settings, device)
    model = build_model(args).to(device)
    loss_func = deep_er.make_loss(use_ssim=args.loss == 'mse+ssim')
    if args.compile:                      # the networks; the NUFFT's plans stay outside
        nets = ((model.denoiser,) if isinstance(model, Diffusion) else
                model.cascades if isinstance(model, VarNet) else (model,))
        for net in nets:
            net.compile()

    if args.dry_run:
        batch, _ = next(deep_er.pulls(aug, 'train', device))
        for key, value in batch.items():
            print(f"  {key:14s} {tuple(value.shape)!s:24s} {value.dtype}")
        loss = train_loss(model, loss_func, batch, amp)
        loss.backward()
        grads = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        val, _ = deep_er.evaluate(model, loss_func, [batch], amp)
        print(f"  {args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M "
              f"parameters, train loss {loss.item():.4f} (grad magnitude {grads:.3g}), "
              f"evaluated loss {val['loss']:.4f}  nrmse {val['nrmse']:.4f}  "
              f"ssim {val['ssim']:.4f}")
        print("Dry run OK: data, model, loss, gradients and evaluation are wired.")
        return

    out_dir = Path(args.out_dir) / run_name(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.lr = args.lr or MODELS[args.model]
    (out_dir / 'config.json').write_text(json.dumps(
        {'args': vars(args),
         'settings': {k: getattr(settings, k) for k in (*DEEP_ER_SETTINGS, 'crop_t', 'target')}
         | {'scale': args.scale, 'loss': args.loss},
         'spatial': deep_er.SPATIAL,
         'paul': deep_er.PAUL, 'evaluation': deep_er.EVALUATION}, indent=2, default=str))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, fused=device.type == 'cuda')
    step, best_maps = 0, float('inf')
    last_path, log_path = out_dir / 'last.pt', out_dir / 'log.csv'
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        step, best_maps = state['step'], state['best_maps']
        # a fresh stream, so the resumed run does not replay its first batches
        aug.reseed(args.seed + step)
        print(f"resumed at step {step}")
    else:
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'train_loss', 'val_loss', 'val_nrmse',
                                    'val_ssim', 'val_maps', 'val_maps_target', 'val_lcmodel', 'ms_per_step'])

    val_names = deep_er.subjects(settings)['val']
    val_set = deep_er.load_fixed(settings, device, 'val', val_names,
                                 deep_er.EVALUATION['val_acc'],
                                 deep_er.EVALUATION['val_timepoints'], keep_on=device)
    val_maps = deep_er.load_val_maps(settings, device, val_names)
    print(f"validation: {len(val_set)} subjects at acceleration "
          f"{deep_er.EVALUATION['val_acc']:g}, and their maps (the input's "
          f"{deep_er.scores_line(deep_er.map_score(None, val_maps, device, amp))})")

    run = None
    if args.wandb:
        import wandb
        group = Path(args.out_dir).name
        run = wandb.init(project='spatial-ablation', entity='augmentrum', group=group,
                         name=out_dir.name, config=vars(args), resume='allow',
                         id=f"{group}-{out_dir.name}")

    batches = deep_er.Prefetcher(lambda: deep_er.pulls(aug, 'train', device), device)
    model.train()
    running, t0 = [], time.time()
    while step < args.steps:
        batch, _ = next(batches)
        optimizer.zero_grad(set_to_none=True)
        loss = train_loss(model, loss_func, batch, amp)
        loss.backward()
        optimizer.step()
        running.append(loss.detach())
        step += 1

        if step % args.eval_every and step != args.steps:
            continue
        ms = (time.time() - t0) / len(running) * 1e3
        mean_loss = torch.stack(running).mean().item()
        val, _ = deep_er.evaluate(model, loss_func, val_set, amp)
        scores = deep_er.map_score(model, val_maps, device, amp)
        print(f"step {step:7d}/{args.steps}  train {mean_loss:.5f}  val {val['loss']:.5f}  "
              f"nrmse {val['nrmse']:.4f}  ssim {val['ssim']:.4f}  "
              f"{deep_er.scores_line(scores)}  ({ms:.0f} ms/step)", flush=True)
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([step, mean_loss, val['loss'], val['nrmse'], val['ssim'],
                                    *scores.values(), round(ms, 1)])
        if run:
            run.log({'loss/train': mean_loss, 'loss/val': val['loss'],
                     'val/nrmse': val['nrmse'], 'val/ssim': val['ssim'],
                     **{f'val/{k}': v for k, v in scores.items()}, 'ms_per_step': ms}, step=step)
        if scores['lcmodel'] < best_maps:     # the checkpoint whose concentrations are best
            best_maps = scores['lcmodel']
            torch.save(model.state_dict(), out_dir / 'best.pt')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'step': step, 'best_maps': best_maps}, last_path)
        running, t0 = [], time.time()

    batches.close()
    del batches, val_set, val_maps
    results = {}
    for weights in ('last', 'best'):
        if weights == 'best':
            model.load_state_dict(torch.load(out_dir / 'best.pt', map_location=device))
        print(f"test, {weights} weights:")
        results[weights] = deep_er.test(settings, model, loss_func, device, amp)
    (out_dir / 'test.json').write_text(json.dumps(results, indent=2))
    if run:
        run.summary.update({f'test/{w}/{k}/{m}': v['mean'][m]
                            for w, r in results.items() for k, v in r.items()
                            for m in v['mean']})
        run.finish()
    print(f"Done. Checkpoints, logs and test.json in {out_dir}")


#**************#
#   ablation   #
#**************#
def grid(parser, args):
    """
    Every network x spatial condition, one after another on the visible GPU: finished runs
    (test.json) are skipped, interrupted ones resume; each run logs to <run>/run.log.
    """
    deep_er.fetch_all(args)             # once, before any run
    common = deep_er.passed_on(parser, args)
    for model in MODELS:
        for spatial in SPATIAL:
            out = Path(args.out_dir) / run_name(argparse.Namespace(
                **{**vars(args), 'model': model, 'spatial': spatial}))
            if deep_er.finished(out, args.steps):
                continue
            out.mkdir(parents=True, exist_ok=True)
            print(f"{time.strftime('%H:%M')}  started {out.name}", flush=True)
            with open(out / 'run.log', 'a') as log:
                code = subprocess.call([sys.executable, __file__, *common, '--model', model,
                                        '--spatial', spatial, '--seed', str(args.seed),
                                        '--resume'], stdout=log, stderr=subprocess.STDOUT)
            print(f"{time.strftime('%H:%M')}  {'done' if code == 0 else 'FAILED'} {out.name}",
                  flush=True)
    summarize(args.out_dir)


def summarize(out_dir):
    """
    Every finished run's test results in one table, <out_dir>/summary.csv: one row per run,
    weights (last/best), acceleration and test track.
    """
    rows = []
    for path in sorted(Path(out_dir).glob('*/test.json')):
        config = json.loads((path.parent / 'config.json').read_text())['args']
        for weights, results in json.loads(path.read_text()).items():
            for key, result in results.items():
                acc, track = key.split('/')
                rows.append({'run': path.parent.name, 'model': config['model'],
                             'spatial': config['spatial'], 'seed': config['seed'],
                             'weights': weights, 'acceleration': acc[3:], 'track': track,
                             **result['mean']})
    if not rows:
        print("no finished runs to summarize")
        return
    path = Path(out_dir) / 'summary.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} results of {len({r['run'] for r in rows})} runs in {path}")


#**********#
#   main   #
#**********#
def main():
    parser = argparse.ArgumentParser(
        description="Train networks to turn undersampled MRSI into the fully sampled scan, with "
                    "and without Augmentrum's spatial augmentation, on MRSI Challenge data.")
    parser.add_argument('--data-dir', default='data/mrsi_challenge',
                        help="Release root; missing subjects are fetched from Zenodo.")
    parser.add_argument('--out-dir', default='results/spatial_ablation')
    parser.add_argument('--model', choices=list(MODELS), default=None,
                        help="Train this network alone. Without it, the whole ablation runs: "
                             "every network x spatial condition.")
    parser.add_argument('--spatial', choices=SPATIAL, default='none',
                        help="One of Augmentrum's spatial transforms, all of them, or paul: "
                             "Deep-ER's own augmentation, in image space.")
    parser.add_argument('--trajectory', choices=sorted(deep_er.TRAJECTORIES),
                        default='eccentric-stack')
    parser.add_argument('--trajectories', type=int, default=19,
                        help="Training acquisitions come from a bank of this many trajectories, "
                             "like scans of as many subjects; 0 places them anew every pull.")
    parser.add_argument('--acc-low', type=int, default=2,
                        help="Training accelerations are integers drawn uniformly in "
                             "[low, high].")
    parser.add_argument('--acc-high', type=int, default=6)
    parser.add_argument('--train', type=int, default=19, help="Contest subjects for training.")
    parser.add_argument('--val', type=int, default=5)
    parser.add_argument('--test', nargs='*', choices=['track1', 'track2'],
                        default=['track1', 'track2'],
                        help="Test sets to evaluate at the end; pass --test alone for none.")
    parser.add_argument('--n-coils', type=int, default=32)
    parser.add_argument('--n-timepoints', type=int, default=4,
                        help="FID timepoints per batch (one volume each pull).")
    parser.add_argument('--target', choices=['noisy', 'truth'], default='noisy',
                        help="noisy: the input's own scan fully sampled, the challenge's noise "
                             "in both; truth: the noise-free truth band-limited to the "
                             "trajectory's k-space, the noise in the input alone.")
    parser.add_argument('--loss', choices=['mse+ssim', 'mse'], default='mse+ssim',
                        help="Deep-ER's image loss, MSE + (1 - SSIM), or the MSE alone (SSIM is "
                             "blind to a gain).")
    parser.add_argument('--scale', choices=['timepoint', 'pull'], default='timepoint',
                        help="What the volumes are scaled by: each timepoint's input maximum "
                             "(Deep-ER) or the pull's, so the noise-only tail of the FID stays "
                             "small in the loss.")
    parser.add_argument('--crop-t', type=int, default=None,
                        help="Train on the first this many FID timepoints only (the metabolite "
                             "signal lives there); validation and test stay on the whole FID.")
    parser.add_argument('--lr', type=float, default=None,
                        help="Default: the network's own (MODELS).")
    parser.add_argument('--steps', type=int, default=20_000)
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True,
                        help="bfloat16 convolutions; FFTs and the loss stay float32.")
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
                        help="One batch through data, model, loss, backward and evaluation; "
                             "no training.")

    args = parser.parse_args()
    if args.model is None and not args.dry_run:
        grid(parser, args)
    else:
        args.model = args.model or 'unet'
        train(args)


if __name__ == '__main__':
    main()
