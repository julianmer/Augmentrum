#!/usr/bin/env python3
"""
Starter utilities for the Augmentrum+ overview diagram.

This script does two things:
1. recrops the original overview image into named assets;
2. provides helper functions for saving real data panels with the right sizes.

Claude Code can extend this to:
- render B0/B1+/concomitant/GNL maps from real arrays;
- render trajectories;
- export final replacement PNGs matching the SVG placeholder sizes.
"""

from pathlib import Path
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
MANIFEST = BASE / "crop_manifest.json"

def load_manifest():
    with open(MANIFEST, "r", encoding="utf-8") as f:
        return json.load(f)

def recrop_original(source_png: str, outdir: str):
    data = load_manifest()
    source = Image.open(source_png)
    out = Path(outdir)
    out.mkdir(exist_ok=True, parents=True)
    for item in data["crops"]:
        crop = source.crop((item["x0"], item["y0"], item["x1"], item["y1"]))
        crop.save(out / item["file"])

def save_map(array, out_png, cmap="gray", vmin=None, vmax=None, facecolor="white"):
    out_png = Path(out_png)
    h, w = array.shape[:2]
    fig = plt.figure(figsize=(w/100, h/100), dpi=100, facecolor=facecolor)
    ax = plt.axes([0,0,1,1])
    ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_axis_off()
    fig.savefig(out_png, dpi=100, bbox_inches="tight", pad_inches=0, facecolor=facecolor)
    plt.close(fig)

def save_trajectory(k_nominal, k_realized=None, out_png="trajectory.png"):
    out_png = Path(out_png)
    fig = plt.figure(figsize=(1.2, 1.6), dpi=100, facecolor="white")
    ax = plt.axes([0.12, 0.08, 0.82, 0.84])
    ax.plot(k_nominal[:,0], k_nominal[:,1], '--', lw=1.5, color='#4C78A8', label='Nominal')
    if k_realized is not None:
        ax.plot(k_realized[:,0], k_realized[:,1], '-', lw=1.3, color='#F28E2B', label='Realized')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=6, loc='upper right')
    fig.savefig(out_png, dpi=100, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    print("Use this as a starting point for generating replacement overview assets.")
