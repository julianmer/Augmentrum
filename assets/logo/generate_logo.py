####################################################################################################
#                                          generate_logo.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 13/08/26                                                                                #
#                                                                                                  #
# Purpose: Generate the Augmentrum logo: a sampling sphere woven from trajectory lines with the    #
#          original spectrum riding its equator. STYLE picks the variant:                          #
#            "solid"   - woven ring, centre spectrum as one solid stroke    -> logo.png            #
#            "strands" - woven ring, centre spectrum as five tight strands  -> logo_strands.png    #
#            "classic" - classic ring, curved solid centre, echo spectra    -> logo_classic.png    #
#            "minimal" - classic ring, straight solid centre, no echoes     -> logo_minimal.png    #
#            "original"- the old logo: ring and trace alone                 -> logo_original.png   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize


#**************#
#   settings   #
#**************#
STYLE = "solid"          # "solid" | "strands" | "classic" | "minimal" | "original"

OUT_NAMES = {"solid": "logo.png", "strands": "logo_strands.png",
             "classic": "logo_classic.png", "minimal": "logo_minimal.png",
             "original": "logo_original.png"}

HERE = Path(__file__).parent
CMAP = LinearSegmentedColormap.from_list("augmentrum", ["#6C4AA8", "#2B6CB7", "#3AA0A8"])
SPEC = np.load(HERE / "spectrum.npy")     # exact trace of the original logo, x in [-1, 1]

R = 1.0                  # sphere radius
Y0 = 0.10                # vertical centring of the curved centre spectrum
RC = np.sqrt(R**2 - Y0**2)
XE = 0.999 * RC          # reach of the curved centre spectrum's baseline
RING_C = 0.958           # squash of the boundary-weave circles
MESH_LW = 2.0            # uniform line width of the sampling lines
SPEC_LW = 14.0           # width of the solid centre spectrum and classic ring


#**********************#
#   drawing helpers    #
#**********************#
def gradient_line(ax, x, y, lw, zorder=5, alpha=1.0):
    """Polyline coloured purple -> teal along its x-position."""
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    t = np.clip(((x[:-1] + x[1:]) / 2 + R) / (2 * R), 0, 1)
    lc = LineCollection(segs, cmap=CMAP, norm=Normalize(0, 1), zorder=zorder,
                        alpha=alpha, capstyle="round", joinstyle="round")
    lc.set_array(t)
    lc.set_linewidth(lw)
    ax.add_collection(lc)


def spec_at(x):
    return np.interp(x, SPEC[0], SPEC[1], left=0.0, right=0.0)


def curved_base(x):
    """Baseline of the curved centre spectrum: centred, sagging around the ball."""
    return spec_at(x) + Y0 - 0.16 * (1 - (x / XE) ** 2)


def flat_base(x):
    """Baseline of the straight centre spectrum (the minimal look)."""
    return spec_at(x)


#*******************#
#   boundary ring   #
#*******************#
def ring_weave(ax):
    """The outer circle as six overlapping near-circular trajectories."""
    t = np.linspace(0, 2 * np.pi, 720)
    for ang in range(0, 180, 30):
        a = np.radians(ang)
        ca, sa = np.cos(a), np.sin(a)
        ex, ey = np.cos(t), RING_C * np.sin(t)
        gradient_line(ax, ex * ca - ey * sa, ex * sa + ey * ca, MESH_LW, zorder=6)


def ring_solid(ax):
    """The outer circle as one classic full-weight stroke."""
    t = np.linspace(0, 2 * np.pi, 720)
    gradient_line(ax, R * np.cos(t), R * np.sin(t), SPEC_LW, zorder=6)


#*******************#
#   sampling mesh   #
#*******************#
def sampling_mesh(ax, lw_front, lw_back, target):
    """Great-circle weave with depth fade; lines near the equator braid along
    the centre spectrum and release smoothly before the poles."""
    th = np.linspace(0, 2 * np.pi, 900)
    segs, deps, xs = [], [], []
    for ang in range(0, 180, 45):
        a = np.radians(ang)
        ca, sa = np.cos(a), np.sin(a)
        for c in [0.06, 0.18, 0.40, 0.70]:
            X = R * np.cos(th) * ca - R * c * np.sin(th) * sa
            Y = R * np.cos(th) * sa + R * c * np.sin(th) * ca
            w = 0.93 * np.exp(-((Y / 0.22) ** 2))
            u = np.clip((1 - np.abs(X)) / 0.15, 0, 1)
            w *= u * u * (3 - 2 * u)
            Y = Y + w * (target(X) - Y)
            sign = -1.0 if np.cos(a) > 1e-6 else 1.0
            D = sign * R * np.sqrt(1 - c ** 2) * np.sin(th)
            P = np.column_stack([X, Y]).reshape(-1, 1, 2)
            segs.append(np.concatenate([P[:-1], P[1:]], axis=1))
            deps.append((D[:-1] + D[1:]) / 2)
            xs.append((X[:-1] + X[1:]) / 2)
    segs = np.concatenate(segs); deps = np.concatenate(deps); xs = np.concatenate(xs)
    order = np.argsort(deps)
    segs, deps, xs = segs[order], deps[order], xs[order]
    d = (deps + R) / (2 * R)
    cols = CMAP(np.clip((xs + R) / (2 * R), 0, 1))
    cols[:, 3] = 0.12 + (0.95 - 0.12) * d
    lc = LineCollection(segs, colors=cols, capstyle="round")
    lc.set_linewidths(lw_back + (lw_front - lw_back) * d)
    lc.set_zorder(3)
    ax.add_collection(lc)


#*******************#
#   echo spectra    #
#*******************#
def echo_rings(ax):
    """Latitude rings that carry the identical trace, scaled into the disk."""
    for y0 in [-0.60, 0.48, 0.78]:
        r = np.sqrt(R ** 2 - y0 ** 2)
        amp = min(0.5 * r, max(0.10, (y0 + 0.95 - 0.18 * r) / 0.79))
        xb = np.linspace(r, -r, 300)
        yb = y0 + 0.18 * np.sqrt(np.clip(r ** 2 - xb ** 2, 0, None))
        gradient_line(ax, xb, yb, 1.8, zorder=2, alpha=0.35)
        xf = np.linspace(-r, r, 700)
        base = y0 - 0.18 * np.sqrt(np.clip(r ** 2 - xf ** 2, 0, None))
        gradient_line(ax, xf, base + spec_at(xf / r) / 0.70 * amp, 2.3, zorder=5)


#**********************#
#   centre spectrum    #
#**********************#
def centre_solid(ax, base, xe):
    """One full-weight stroke from ring to ring on the given baseline."""
    keep = np.abs(SPEC[0]) < xe - 1e-4
    x = np.concatenate([[-xe], SPEC[0][keep], [xe]])
    y = base(x) - spec_at(x) + np.concatenate([[0.0], SPEC[1][keep], [0.0]])
    gradient_line(ax, x, y, SPEC_LW, zorder=8)


def centre_strands(ax, n=5, span=0.044):
    """Five identical strands offset along the curve normals; each runs into
    the boundary band and is absorbed where it meets the weave."""
    x = np.linspace(-0.9985, 0.9985, 1400)
    y = curved_base(x)
    dx, dy = np.gradient(x), np.gradient(y)
    L = np.hypot(dx, dy) + 1e-12
    nx, ny = -dy / L, dx / L
    k5 = np.ones(5) / 5.0
    for k in range(n):
        off = (k / (n - 1) - 0.5) * span
        xs = np.convolve(np.pad(x + off * nx, 5, mode="edge"), k5, mode="same")[5:-5]
        ys = np.convolve(np.pad(y + off * ny, 5, mode="edge"), k5, mode="same")[5:-5]
        inside = np.hypot(xs, ys) <= 0.988
        gradient_line(ax, xs[inside], ys[inside], MESH_LW, zorder=8)


#**********#
#   main   #
#**********#
def main():
    fig = plt.figure(figsize=(8, 8), dpi=256)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")

    if STYLE in ("strands", "solid"):
        ring_weave(ax)
        sampling_mesh(ax, MESH_LW, MESH_LW, curved_base)
        echo_rings(ax)
        if STYLE == "solid":
            centre_solid(ax, curved_base, 0.999 * np.sqrt(RING_C ** 2 - Y0 ** 2))
        else:
            centre_strands(ax)
    elif STYLE == "classic":
        ring_solid(ax)
        sampling_mesh(ax, MESH_LW, MESH_LW, curved_base)
        echo_rings(ax)
        centre_solid(ax, curved_base, XE)
    elif STYLE == "minimal":
        ring_solid(ax)
        sampling_mesh(ax, 3.4, 1.6, flat_base)
        centre_solid(ax, flat_base, 0.995)
    elif STYLE == "original":
        ring_solid(ax)
        centre_solid(ax, flat_base, 0.995)
    else:
        raise ValueError(f"unknown STYLE {STYLE!r}")

    out = HERE / OUT_NAMES[STYLE]
    fig.savefig(out, transparent=True)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
