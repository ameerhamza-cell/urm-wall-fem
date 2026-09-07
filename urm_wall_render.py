#!/usr/bin/env python3
"""
urm_wall_render.py — side-by-side PNG comparison of the two URM mesh styles.

  LEFT  : Explicit mortar — nominal bricks + physical mortar hex elements
  RIGHT : Zero-thickness  — expanded bricks + zero-thickness interface quads
                            (Lourenco & Rots 1997 simplified micro-modelling)

Output : urm_wall_styles.png  in the same folder as this script.

Both styles use the same underlying geometry (same brick positions, same
interface detection).  The difference is purely in how the mortar is
represented:

  Explicit mortar:
    - Brick geometry is at nominal dimensions (as built)
    - Mortar occupies the physical gap between bricks
    - Requires two material laws: brick + mortar
    - More degrees of freedom; realistic thickness for plastic deformation
    - Better for thick mortar joints or when mortar crushing matters

  Zero-thickness interface elements:
    - Brick is expanded by mortar_t/2 on each face (absorbs mortar volume)
    - Interface law lives at a zero-thickness quad (constitutive law only,
      no geometric thickness → no volume change under compressive yield)
    - Fewer DOF, standard in masonry FEM literature
    - Requires penalty stiffness kn, ks instead of elastic constants
"""

from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')   # file output — no display / X server needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines   import Line2D
from matplotlib.patches import Rectangle

from urm_wall_mesh import WallConfig, place_bricks, find_interfaces_kdtree

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
BRICK_FACE   = '#c8945c'   # warm tan
BRICK_EDGE   = '#4a3520'   # dark brown
MORTAR_FACE  = '#b0a898'   # cool gray
MORTAR_EDGE  = '#888070'
BED_COLOR    = '#cc2222'   # red  — bed joint interface line
HEAD_COLOR   = '#1a52c0'   # blue — head joint interface line

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _rect(ax, x0, y0, w, h, fc, ec, lw=0.5, zorder=2, alpha=1.0):
    ax.add_patch(Rectangle(
        (x0, y0), w, h,
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder, alpha=alpha,
    ))


def _ax_setup(ax, cfg: WallConfig, title: str, subtitle: str):
    pad = 0.015
    ax.set_xlim(-pad, cfg.wall_width  + pad)
    ax.set_ylim(-pad, cfg.wall_height + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('Width [m]', fontsize=8)
    ax.set_ylabel('Height [m]', fontsize=8)
    ax.set_title(f'{title}\n{subtitle}', fontsize=9, fontweight='bold',
                 pad=5, linespacing=1.4)
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ---------------------------------------------------------------------------
# Style 1 : Explicit mortar
# ---------------------------------------------------------------------------

def render_explicit_mortar(ax, bricks, interfaces, cfg: WallConfig):
    ax.set_facecolor('#f0ede8')

    # 1. Mortar joints first (zorder=1 — below bricks)
    for ifc in interfaces:
        if ifc.joint_type == 'perp':
            continue
        q   = ifc.quad
        mid = ifc.center
        if ifc.joint_type == 'bed':
            t  = cfg.mortar_bed
            x0 = q[:, 0].min();  x1 = q[:, 0].max()
            y0 = mid[1] - t/2;   y1 = mid[1] + t/2
        else:   # head
            t  = cfg.mortar_head
            y0 = q[:, 1].min();  y1 = q[:, 1].max()
            x0 = mid[0] - t/2;   x1 = mid[0] + t/2
        if (x1 - x0) > 1e-6 and (y1 - y0) > 1e-6:
            _rect(ax, x0, y0, x1-x0, y1-y0, MORTAR_FACE, MORTAR_EDGE, 0.3, zorder=1)

    # 2. Nominal bricks on top
    for b in bricks:
        cx, cy, _ = b.center
        lx, ly, _ = b.size
        _rect(ax, cx - lx/2, cy - ly/2, lx, ly, BRICK_FACE, BRICK_EDGE, 0.6, zorder=2)

    _ax_setup(ax, cfg,
              'Style 1 — Explicit Mortar',
              'Nominal brick hexes  +  mortar hex elements')

    ax.legend(handles=[
        mpatches.Patch(fc=BRICK_FACE,  ec=BRICK_EDGE,  label='Brick (nominal size, hex element)'),
        mpatches.Patch(fc=MORTAR_FACE, ec=MORTAR_EDGE, label='Mortar joint  (physical hex element)'),
    ], loc='lower right', fontsize=7, framealpha=0.90)


# ---------------------------------------------------------------------------
# Style 2 : Zero-thickness interface elements
# ---------------------------------------------------------------------------

def render_zero_thickness(ax, bricks, interfaces, cfg: WallConfig):
    ax.set_facecolor('#f0ede8')

    # 1. Expanded bricks
    for b in bricks:
        cx, cy, _ = b.center
        lx, ly, _ = b.size
        x0 = max(0.0,             cx - (lx + cfg.mortar_head) / 2)
        x1 = min(cfg.wall_width,  cx + (lx + cfg.mortar_head) / 2)
        y0 = max(0.0,             cy - (ly + cfg.mortar_bed)  / 2)
        y1 = min(cfg.wall_height, cy + (ly + cfg.mortar_bed)  / 2)
        _rect(ax, x0, y0, x1-x0, y1-y0, BRICK_FACE, BRICK_EDGE, 0.6, zorder=2)

    # 2. Zero-thickness interface lines at the midplane
    for ifc in interfaces:
        if ifc.joint_type == 'perp':
            continue
        q = ifc.quad
        if ifc.joint_type == 'bed':
            y  = ifc.center[1]
            x0 = q[:, 0].min();  x1 = q[:, 0].max()
            ax.plot([x0, x1], [y, y],
                    color=BED_COLOR, lw=1.1, zorder=3, solid_capstyle='butt')
        else:   # head
            x  = ifc.center[0]
            y0 = q[:, 1].min();  y1 = q[:, 1].max()
            ax.plot([x, x], [y0, y1],
                    color=HEAD_COLOR, lw=1.1, zorder=3, solid_capstyle='butt')

    _ax_setup(ax, cfg,
              'Style 2 — Zero-thickness Interface Elements',
              'Expanded brick hexes  +  zero-thickness quad interfaces  '
              '(Lourenço & Rots 1997)')

    ax.legend(handles=[
        mpatches.Patch(fc=BRICK_FACE, ec=BRICK_EDGE,
                       label='Brick (expanded by mortar/2 each side, hex element)'),
        Line2D([0], [0], color=BED_COLOR,  lw=1.5,
               label='Bed interface  (zero-thickness quad)'),
        Line2D([0], [0], color=HEAD_COLOR, lw=1.5,
               label='Head interface (zero-thickness quad)'),
    ], loc='lower right', fontsize=7, framealpha=0.90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = WallConfig(
        wall_width=1.20, wall_height=1.00,
        brick_length=0.210, brick_height=0.052,
        brick_depth=0.100,
        mortar_bed=0.010, mortar_head=0.010, mortar_perp=0.010,
        bond='running', n_wythes=1,
    )

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, 'urm_wall_styles.png')

    print('Placing bricks ...')
    bricks     = place_bricks(cfg)
    print('Detecting interfaces ...')
    interfaces = find_interfaces_kdtree(bricks, cfg)
    print(f'  {len(bricks)} bricks,  {len(interfaces)} interfaces')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#faf9f7')

    n_bed  = sum(1 for i in interfaces if i.joint_type == 'bed')
    n_head = sum(1 for i in interfaces if i.joint_type == 'head')
    fig.suptitle(
        f'URM Simplified Micro-Modelling — Running Bond Wall  '
        f'({cfg.wall_width:.2f} m × {cfg.wall_height:.2f} m)\n'
        f'Brick {cfg.brick_length*1e3:.0f}×{cfg.brick_height*1e3:.0f} mm  |  '
        f'Mortar bed={cfg.mortar_bed*1e3:.0f} mm  head={cfg.mortar_head*1e3:.0f} mm  |  '
        f'{len(bricks)} bricks,  {n_bed} bed interfaces,  {n_head} head interfaces',
        fontsize=10, fontweight='bold', y=1.01,
    )

    render_explicit_mortar(axes[0], bricks, interfaces, cfg)
    render_zero_thickness( axes[1], bricks, interfaces, cfg)

    plt.tight_layout(pad=1.5)
    plt.savefig(out_png, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'\nPNG saved → {out_png}')
    print('Open in Windows Explorer or any image viewer.')


if __name__ == '__main__':
    main()
