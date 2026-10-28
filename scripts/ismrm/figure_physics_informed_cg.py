#!/usr/bin/env python3
"""
Physics-informed CG reconstruction experiment, per AUGMENTRUM_PLUS_CG_RECON_HANDOVER.md.

Reuses the SAME anatomy, maps, measured GIRF, and spiral trajectory as
scripts/spatial_physics_ablation.py (handover section 10 - "reuse the same setup... so the
figures form a pair") via RepoAdapter, and the SAME real physics phase terms
(_MultiShotStaticOffResonancePhase, _MultiShotConcomitantPhase, GradientNonlinearityPhase) rather
than reimplementing them - only the ENCODING/RECONSTRUCTION architecture (LinearMRIOperator + CG)
is new.

One reference dataset y_ref is generated ONCE with the full-physics operator (handover section
11); every reconstruction below (cumulative R0-R5, leave-one-out) solves against that SAME
y_ref, using the SAME CG budget (handover section 15), and is scored against x_true AND against
y_ref via the common/reference-model residual (handover section 14) - not each solver's own
native residual, which can converge perfectly to the wrong inverse problem.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spatial_physics_ablation as spa  # noqa: E402

from augmentrum.reconstruction.linear_operator import (  # noqa: E402
    LinearMRIOperator, PhysicsConfig, _GIRFTrajectoryErrorPhase,
)
from augmentrum.reconstruction.cg import cg_solve  # noqa: E402


def _prepare_context(adapter: "spa.RepoAdapter", inputs: "spa.FigureInputs", n_coils: int = 8) -> Dict:
    nx = ny = inputs.anatomy.shape[0]
    b0_tesla = float(getattr(adapter, "_field_strength_t", 3.0))
    device = adapter.device

    spiral = adapter.get_spiral(nx, inputs.fov_m)
    k_nominal = spiral["k_nominal"]
    ndim, T = k_nominal.shape
    dt_s = spiral["dt_s"]
    gradients_per_shot = spiral["gradients_per_shot"]
    n_shots = spiral["n_shots"]
    samples_per_shot = T // n_shots

    measured = adapter.load_bacon_measured_girf()
    dk_error = (measured.dk_error.permute(0, 2, 1).reshape(-1, ndim)
               .numpy().astype(np.float64).T)
    k_actual = k_nominal + dk_error

    positions_cpu = adapter.gs.make_grid_2d(nx, ny, inputs.fov_m, inputs.fov_m,
                                            z=adapter._slice_offset_m)
    positions = positions_cpu.to(device)

    x_true = torch.from_numpy(adapter._flatten(inputs.anatomy).astype(np.complex64)).to(device)
    b1plus_map = torch.from_numpy(adapter._flatten(inputs.b1_rel).astype(np.complex64)).to(device)
    b0_hz_flat = adapter._flatten(inputs.b0_hz).astype(np.float32)

    # Synthetic multi-coil receive array (augmentrum.sampling.coil_sampling.Birdcage - built from
    # pure grid geometry, no data/download needed; RSS-normalized to 1). Testing whether real
    # coil-sensitivity diversity resolves the single-coil CG instability found for B0-without-GIRF
    # configurations (see CG_FITTING_STATUS.md) - single-coil non-Cartesian reconstruction lacks
    # the implicit conditioning multi-coil parallel imaging normally provides, which may be why
    # that instability doesn't resemble everyday (always multi-coil) clinical reconstruction.
    from augmentrum.sampling.coil_sampling import Birdcage
    coil_grid = Birdcage(n_coils=n_coils, radius=1.5).maps(
        (nx, ny, 1), rng=np.random.default_rng(0))[:, :, 0, :]   # (nx, ny, n_coils)
    coil_map = torch.from_numpy(np.stack(
        [adapter._flatten(coil_grid[:, :, c]) for c in range(n_coils)], axis=0
    ).astype(np.complex64)).to(device)   # (n_coils, N)

    # Plain numpy, NOT a device tensor: GriddingNUFFT.forward/adjoint call np.asarray(coords)
    # internally (augmentrum/sampling/kspace_reconstructor.py's _for_interpolator), which fails
    # outright on a CUDA tensor - this is independent of whatever device the image/position
    # tensors passed through LinearMRIOperator itself are on.
    kmax_xy = nx / (2.0 * inputs.fov_m)
    coords_nominal = (k_nominal.T[:, :2] / (2.0 * kmax_xy)).astype(np.float32)
    coords_actual = (k_actual.T[:, :2] / (2.0 * kmax_xy)).astype(np.float32)
    t_axis = (torch.arange(T, dtype=torch.float32) * dt_s).to(device)

    b0_term = spa._MultiShotStaticOffResonancePhase(
        torch.from_numpy(b0_hz_flat).to(device), n_shots, samples_per_shot, dt_s)
    concomitant_term = spa._MultiShotConcomitantPhase(gradients_per_shot, dt_s, b0_tesla)

    gnl_coeffs = adapter._load_gnl_coefficients()
    if adapter._gnl_coefficient_scale != 1.0:
        gnl_coeffs = gnl_coeffs._replace(**{
            f: getattr(gnl_coeffs, f) * adapter._gnl_coefficient_scale
            for f in gnl_coeffs._fields if f != "R0_m"
        })
    gnl_term_nominal = adapter._GradientNonlinearityPhase(gnl_coeffs, k_nominal, max_order=9)
    gnl_term_actual = adapter._GradientNonlinearityPhase(gnl_coeffs, k_actual, max_order=9)

    # GIRF trajectory error as a phase term, NOT a second NUFFT trajectory - see
    # augmentrum.reconstruction.linear_operator's _GIRFTrajectoryErrorPhase/PhysicsConfig
    # docstrings for the exact algebraic identity this relies on. dk_error is in the SAME
    # cycles/m units as k_nominal/k_actual; positions are in meters, so phi = -2*pi*dk.r is in
    # radians, matching every other phase term here.
    girf_term = _GIRFTrajectoryErrorPhase(torch.from_numpy(dk_error).float().to(device))

    nufft = adapter._GriddingNUFFT((nx, ny), osf=2.0)

    return dict(
        nx=nx, ny=ny, im_size=(nx, ny), device=device, T=T, dt_s=dt_s, n_shots=n_shots,
        positions=positions, x_true=x_true, b1plus_map=b1plus_map, t_axis=t_axis,
        coords_nominal=coords_nominal, coords_actual=coords_actual, nufft=nufft,
        b0_term=b0_term, concomitant_term=concomitant_term, girf_term=girf_term,
        gnl_term_nominal=gnl_term_nominal, gnl_term_actual=gnl_term_actual, coil_map=coil_map,
    )


def _operator_from_context(ctx: Dict, config: PhysicsConfig) -> LinearMRIOperator:
    gnl_term = ctx["gnl_term_actual"] if config.girf else ctx["gnl_term_nominal"]
    op = LinearMRIOperator(
        positions=ctx["positions"], t=ctx["t_axis"],
        k_nominal=ctx["coords_nominal"],   # NUFFT always grids on k_nominal, regardless of config.girf
        gridding_nufft=ctx["nufft"], im_size=ctx["im_size"],
        b1plus_map=ctx["b1plus_map"] if config.b1plus else None,
        coil_map=ctx["coil_map"],
        b0_term=ctx["b0_term"], concomitant_term=ctx["concomitant_term"], gnl_term=gnl_term,
        girf_term=ctx["girf_term"],
        config=config,
    )
    return op.prepare()


CUMULATIVE_SERIES: List[Tuple[str, PhysicsConfig]] = [
    ("R0_nominal", PhysicsConfig()),
    ("R1_b0", PhysicsConfig(b0=True)),
    ("R2_b0_b1", PhysicsConfig(b0=True, b1plus=True)),
    ("R3_b0_b1_girf", PhysicsConfig(b0=True, b1plus=True, girf=True)),
    ("R4_plus_concomitant", PhysicsConfig(b0=True, b1plus=True, girf=True, concomitant=True)),
    ("R5_full", PhysicsConfig(b0=True, b1plus=True, girf=True, concomitant=True, gnl=True)),
]

LEAVE_ONE_OUT: List[Tuple[str, PhysicsConfig]] = [
    ("leaveout_b0", PhysicsConfig(b0=False, b1plus=True, girf=True, concomitant=True, gnl=True)),
    ("leaveout_b1", PhysicsConfig(b0=True, b1plus=False, girf=True, concomitant=True, gnl=True)),
    ("leaveout_girf", PhysicsConfig(b0=True, b1plus=True, girf=False, concomitant=True, gnl=True)),
    ("leaveout_concomitant", PhysicsConfig(b0=True, b1plus=True, girf=True, concomitant=False, gnl=True)),
    ("leaveout_gnl", PhysicsConfig(b0=True, b1plus=True, girf=True, concomitant=True, gnl=False)),
]


def nrmse(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:
    a, b = x_hat.abs(), x_true.abs()
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))


def _snapshot_callback(snapshot_dir: Path, name: str, im_size: Tuple[int, int], every: int):
    """
    Builds a cg_solve "callback(iteration, x, residual)" that periodically saves the current CG
    image estimate to disk - for inspecting/animating convergence later, not needed for the
    final metrics themselves (those come from the LAST iterate, saved separately as usual).
    Always saves iteration 0 (the starting point) so a convergence montage has a true baseline.
    """
    if every <= 0:
        return None
    out = snapshot_dir / name
    out.mkdir(parents=True, exist_ok=True)

    def _cb(iteration: int, x: torch.Tensor, residual: float):
        if iteration % every != 0:
            return
        img = x.detach().cpu().numpy().reshape(*im_size)
        np.savez_compressed(out / f"iter_{iteration:04d}.npz", image=img, residual=residual)

    return _cb


def _reconstruct_one(name: str, config: PhysicsConfig, ctx: Dict, op_full: LinearMRIOperator,
                     y_ref: torch.Tensor, x_true: torch.Tensor, max_iter: int, rtol: float,
                     lam: float, outdir: Path, snapshot_every: int) -> Dict:
    t_start = time.time()
    op = _operator_from_context(ctx, config)
    rhs = op.adjoint(y_ref)
    callback = _snapshot_callback(outdir / "intermediate", name, ctx["im_size"], snapshot_every)
    result = cg_solve(lambda v: op.normal(v, lam=lam), rhs, max_iter=max_iter, rtol=rtol,
                      callback=callback)
    elapsed = time.time() - t_start

    x_hat = result.image
    true_model_residual = float(
        torch.linalg.vector_norm(op_full.forward(x_hat) - y_ref) / torch.linalg.vector_norm(y_ref))
    row = dict(
        reconstruction=name, nrmse=nrmse(x_hat, x_true), true_model_residual=true_model_residual,
        native_final_residual=result.rel_residual_history[-1], iterations=result.n_iter,
        converged=result.converged, runtime_s=elapsed,
    )
    img = x_hat.detach().cpu().numpy().reshape(*ctx["im_size"])
    np.savez_compressed(outdir / f"recon_{name}.npz", image=img)
    print(f"[{name}] nrmse={row['nrmse']:.4f} true_model_residual={true_model_residual:.4f} "
         f"iters={result.n_iter} converged={result.converged} ({elapsed:.1f}s)", flush=True)
    return row


def _worker_run_subset(gpu_id: Optional[int], args_dict: dict, config_items: List[Tuple[str, dict]],
                       n_segments: int, max_iter: int, rtol: float, lam: float, outdir_str: str,
                       snapshot_every: int) -> List[Dict]:
    """
    Runs ONE GPU's assigned subset of reconstructions, in its own process (required for clean
    CUDA context isolation - each worker gets exactly one GPU via CUDA_VISIBLE_DEVICES, set
    BEFORE torch ever touches CUDA in this process). Rebuilds the shared context ONCE per worker
    (not once per config) since it's the same anatomy/trajectory/maps for every config - only
    y_ref/x_true are loaded from disk rather than recomputed, so every worker reconstructs
    against the EXACT SAME reference dataset regardless of which GPU/process generated it.
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Imports deferred until after CUDA_VISIBLE_DEVICES is set, so this process's CUDA context
    # only ever sees its own assigned GPU.
    import spatial_physics_ablation as spa_local
    from augmentrum.reconstruction.linear_operator import LinearMRIOperator as LMO, PhysicsConfig as PC

    outdir = Path(outdir_str)
    # Each worker rebuilds its OWN RepoAdapter/inputs from scratch (independent processes can't
    # share Python objects), and that rebuild writes real files: a .seq sequence file
    # (RepoAdapter.seq_cache_dir) and HCP-derived B0/B1+ NIfTI caches (load_or_prepare_inputs'
    # own use of args.outdir). Pointing every worker at the SAME shared outdir for those writes
    # caused exactly the failure this comment replaces: two workers writing/reading the same
    # .seq path concurrently corrupted one worker's read of it (a pypulseq parse error deep in
    # a totally unrelated-looking TypeError). Give each worker its own private scratch
    # subdirectory for this internal I/O; only the FINAL reconstruction outputs (recon_*.npz,
    # via _reconstruct_one below) go to the real, shared outdir.
    worker_scratch = outdir / "_worker_scratch" / f"gpu{gpu_id if gpu_id is not None else 'cpu'}"
    worker_scratch.mkdir(parents=True, exist_ok=True)
    args_dict_local = dict(args_dict, outdir=str(worker_scratch))
    args_ns = argparse.Namespace(**args_dict_local)
    inputs = spa_local.load_or_prepare_inputs(args_ns)
    device = "cuda" if (gpu_id is not None) else "cpu"
    adapter = spa_local.RepoAdapter(
        Path(args_dict["augmentrum_root"]).resolve() if args_dict.get("augmentrum_root") else None,
        Path(args_dict["girfsim_root"]).resolve() if args_dict.get("girfsim_root") else None,
        outdir=worker_scratch, field_strength_t=args_dict["field_strength_t"], device=device,
        slice_offset_m=args_dict.get("slice_offset_mm", 0.0) / 1000.0,
    )
    ctx = _prepare_context(adapter, inputs)

    gt = np.load(outdir / "ground_truth.npz")
    kr = np.load(outdir / "kspace_reference.npz")
    x_true = torch.from_numpy(gt["x_true"].reshape(-1).astype(np.complex64)).to(ctx["device"])
    y_ref = torch.from_numpy(kr["y_ref"].astype(np.complex64)).to(ctx["device"])

    full_config = PC(b0=True, b1plus=True, girf=True, concomitant=True, gnl=True,
                    n_segments=n_segments)
    op_full = _operator_from_context(ctx, full_config)

    rows = []
    for name, cfg_kwargs in config_items:
        cfg = PC(n_segments=n_segments, **cfg_kwargs)
        rows.append(_reconstruct_one(name, cfg, ctx, op_full, y_ref, x_true, max_iter, rtol, lam,
                                     outdir, snapshot_every))
    return rows


def run_experiment(adapter, inputs, n_segments: int, max_iter: int, rtol: float, lam: float,
                   outdir: Path, snapshot_every: int = 0, parallel: bool = False,
                   gpu_ids: Optional[List[int]] = None, args_dict: Optional[dict] = None,
                   only: Optional[List[str]] = None) -> Dict:
    outdir.mkdir(parents=True, exist_ok=True)

    ctx = _prepare_context(adapter, inputs)
    x_true = ctx["x_true"]

    full_config = PhysicsConfig(b0=True, b1plus=True, girf=True, concomitant=True, gnl=True,
                                n_segments=n_segments)
    t0 = time.time()
    op_full = _operator_from_context(ctx, full_config)
    y_ref = op_full.forward(x_true)
    print(f"[reference] y_ref generated with full-physics operator in {time.time()-t0:.1f}s",
         flush=True)

    # Saved to disk and (if parallel) re-loaded by every worker, rather than each worker
    # recomputing its own copy - guarantees every reconstruction is scored against the EXACT
    # SAME y_ref regardless of which GPU/process ran it (handover section 11), rather than
    # relying on bit-for-bit reproducibility across independent CUDA kernel runs.
    np.savez_compressed(outdir / "ground_truth.npz",
                        x_true=x_true.detach().cpu().numpy().reshape(ctx["im_size"]))
    np.savez_compressed(outdir / "kspace_reference.npz", y_ref=y_ref.detach().cpu().numpy())

    all_configs = CUMULATIVE_SERIES + LEAVE_ONE_OUT
    if only:
        all_configs = [(name, cfg) for name, cfg in all_configs if name in only]
    rows: List[Dict] = []

    if parallel and gpu_ids and len(gpu_ids) > 1:
        print(f"[parallel] distributing {len(all_configs)} reconstructions across GPUs {gpu_ids}",
             flush=True)
        # PhysicsConfig instances aren't needed in the child (rebuilt there from plain kwargs) -
        # pass simple dicts so this stays picklable with the 'spawn' start method, which is what
        # cleanly isolates each worker's CUDA context onto its own assigned GPU.
        config_kwargs = [(name, dict(b0=cfg.b0, b1plus=cfg.b1plus, girf=cfg.girf,
                                     concomitant=cfg.concomitant, gnl=cfg.gnl))
                         for name, cfg in all_configs]
        chunks: List[List[Tuple[str, dict]]] = [[] for _ in gpu_ids]
        for i, item in enumerate(config_kwargs):
            chunks[i % len(gpu_ids)].append(item)

        ctx_mp = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=ctx_mp) as pool:
            futures = [
                pool.submit(_worker_run_subset, gpu_id, args_dict, chunk, n_segments, max_iter,
                           rtol, lam, str(outdir), snapshot_every)
                for gpu_id, chunk in zip(gpu_ids, chunks) if chunk
            ]
            for fut in futures:
                rows.extend(fut.result())
    else:
        for name, cfg in all_configs:
            rows.append(_reconstruct_one(name, cfg, ctx, op_full, y_ref, x_true, max_iter, rtol,
                                         lam, outdir, snapshot_every))

    # Re-order to match the canonical CUMULATIVE_SERIES + LEAVE_ONE_OUT sequence regardless of
    # which order parallel workers finished in.
    order = {name: i for i, (name, _) in enumerate(all_configs)}
    rows.sort(key=lambda r: order[r["reconstruction"]])

    with open(outdir / "metrics.json", "w") as f:
        json.dump(rows, f, indent=2)

    full_row = next((r for r in rows if r["reconstruction"] == "R5_full"), None)
    nominal_row = next((r for r in rows if r["reconstruction"] == "R0_nominal"), None)
    if full_row and nominal_row:
        print(f"\nSummary: R0 (nominal) NRMSE={nominal_row['nrmse']:.4f} vs "
             f"R5 (full) NRMSE={full_row['nrmse']:.4f} "
             f"({'IMPROVEMENT' if full_row['nrmse'] < nominal_row['nrmse'] else 'NO IMPROVEMENT'})")
    return dict(rows=rows, ctx=ctx)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--augmentrum-root", type=str, default=None)
    p.add_argument("--girfsim-root", type=str, default=None)
    p.add_argument("--outdir", type=str, default="outputs/ismrm_cg_recon")
    p.add_argument("--matrix", type=int, default=32)
    p.add_argument("--fov-mm", type=float, default=220.0)
    p.add_argument("--field-strength-t", type=float, default=3.0)
    p.add_argument("--n-segments", type=int, default=8)
    p.add_argument("--cg-max-iter", type=int, default=30)
    p.add_argument("--cg-rtol", type=float, default=1e-5)
    p.add_argument("--cg-lambda", type=float, default=1e-6)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--synthetic-b0-hz", type=float, default=90.0)
    p.add_argument("--anatomy", type=str, default=None)
    p.add_argument("--anatomy-source", choices=["shepp_logan", "brainweb", "hcp"], default="shepp_logan")
    p.add_argument("--hcp-slice-offset-vox", type=int, default=0)
    p.add_argument("--brainweb-subject", type=int, default=4)
    p.add_argument("--b0-map", type=str, default="auto")
    p.add_argument("--b1-map", type=str, default="auto")
    p.add_argument("--rotate90-ccw", type=int, default=0)
    p.add_argument("--slice-offset-mm", type=float, default=30.0,
                   help="Off-isocenter slice position (mm) - matches the ablation figure's own "
                        "default. At 0mm (isocenter) the concomitant-field term is analytically "
                        "EXACTLY zero for this in-plane-only trajectory, not just small, which "
                        "misrepresents real 3T scanning. 30mm gives a genuinely small-but-nonzero "
                        "concomitant effect consistent with real off-center slices at 3T.")
    p.add_argument("--snapshot-every", type=int, default=0,
                   help="Save an intermediate CG image estimate every N iterations (to "
                        "<outdir>/intermediate/<reconstruction>/iter_NNNN.npz), for inspecting "
                        "or animating convergence. 0 (default) disables snapshotting.")
    p.add_argument("--parallel", action="store_true",
                   help="Distribute the 11 independent reconstructions (6 cumulative + 5 "
                        "leave-one-out) across multiple GPUs, one worker process per GPU - CG "
                        "itself is sequential and gains nothing from this, but the "
                        "reconstructions are fully independent of each other. y_ref/x_true are "
                        "generated once and loaded from disk by every worker so all "
                        "reconstructions are scored against an identical reference regardless of "
                        "which GPU ran them.")
    p.add_argument("--gpus", type=str, default=None,
                   help="Comma-separated GPU ids to use with --parallel (e.g. '0,1'). Defaults "
                        "to all visible CUDA devices (torch.cuda.device_count()).")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated subset of reconstruction names to run (e.g. "
                        "'R0_nominal,R5_full'), skipping the rest of the usual 11. For "
                        "figure-asset regeneration, not for the full ablation/leave-one-out study.")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    outdir = Path(args.outdir)

    inputs = spa.load_or_prepare_inputs(args)
    adapter = spa.RepoAdapter(
        Path(args.augmentrum_root).resolve() if args.augmentrum_root else None,
        Path(args.girfsim_root).resolve() if args.girfsim_root else None,
        outdir=outdir, field_strength_t=args.field_strength_t, device=args.device,
        slice_offset_m=args.slice_offset_mm / 1000.0,
    )

    gpu_ids = None
    if args.parallel:
        if args.gpus:
            gpu_ids = [int(g) for g in args.gpus.split(",")]
        elif torch.cuda.is_available():
            gpu_ids = list(range(torch.cuda.device_count()))
        if not gpu_ids or len(gpu_ids) < 2:
            print("--parallel requested but fewer than 2 GPUs are available/specified; "
                 "running sequentially instead.", flush=True)
            gpu_ids = None

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    run_experiment(adapter, inputs, n_segments=args.n_segments, max_iter=args.cg_max_iter,
                  rtol=args.cg_rtol, lam=args.cg_lambda, outdir=outdir,
                  snapshot_every=args.snapshot_every, parallel=bool(gpu_ids), gpu_ids=gpu_ids,
                  args_dict=vars(args), only=only)


if __name__ == "__main__":
    main()
