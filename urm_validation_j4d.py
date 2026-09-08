#!/usr/bin/env python3
"""
urm_validation_j4d.py
=====================
Validation of the ZT micro-model against the Raijmakers & Vermeltfoort (1992)
shear-wall test, specimen J4D.

Geometry  : 990 mm × 1000 mm running-bond, single wythe (100 mm deep)
Loading   : σ_V = −0.30 MPa, horizontal ramp to u_top = 3.0 mm
Parameters: Lourenço (1994) calibrated values for J4D
              G_f^I  = 18 N/m  (vs default 12 N/m)
              G_f^II = 125 N/m (vs default 40 N/m)
              ν_brick = 0.15   (vs default 0.20)

Experimental reference (approximate digitisation from Lourenço 1994, Fig. 5.8):
  Raijmakers & Vermeltfoort (1992) J4D – monotonic push
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---- Import solver; monkey-patch ν before mesh build ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import urm_wall_fem2d as urm
import urm_wall_mesh  as uwm

# Override Poisson ratio BEFORE building mesh (used inside build_zt_mesh)
urm.NU_BRICK = 0.15

# ---- J4D specimen geometry ----
cfg = uwm.WallConfig(
    wall_width=0.990,
    wall_height=1.000,
    brick_length=0.210,
    brick_height=0.052,
    brick_depth=0.100,
    mortar_bed=0.010,
    mortar_head=0.010,
    bond='running',
)

# ---- Lourenco (1994) calibrated interface parameters ----
p_j4d = urm.IFaceParams(
    kn=82.0e9,
    ks=36.0e9,
    ft=0.25e6,
    GfI=18.0,      # N/m — longer Mode I tail
    c0=0.35e6,
    GfII=125.0,    # N/m — much longer cohesion softening (key calibration param)
    tanPhi=0.75,
)

SIGMA_V_J4D = -0.30e6   # Pa
U_TOTAL_J4D =  3.00e-3  # m
N_STEPS_J4D =  250

# ---- Build mesh ----
print("=" * 60)
print("Building ZT mesh  (J4D: 990 × 1000 mm)")
print("=" * 60)
mesh = urm.build_zt_mesh(cfg)
print(f"  {len(mesh.brick_elems)} bricks, "
      f"{len(mesh.iface_nodes)} interfaces "
      f"({sum(mesh.iface_horiz)} bed + {sum(not h for h in mesh.iface_horiz)} head)")

# ---- Solve ----
print("\nRunning Newton-Raphson (250 steps to 3.0 mm) ...")
res = urm.solve_wall(
    mesh, cfg,
    p_iface_override=p_j4d,
    sigma_v=SIGMA_V_J4D,
    u_total=U_TOTAL_J4D,
    n_steps=N_STEPS_J4D,
    rigid_top=True,
)

u_mm = res['u_history'] * 1e3
F_kN = -res['F_history'] / 1e3   # positive = rightward resistance

# ---- Key metrics ----
F_peak   = F_kN.max()
u_peak   = u_mm[F_kN.argmax()]
# Friction plateau: last 20% of steps
F_plateau = F_kN[int(0.80 * len(F_kN)):].mean()

# Analytical friction plateau (full contact)
A_top     = cfg.wall_width * urm.THICK           # m²
N_total   = abs(SIGMA_V_J4D) * A_top             # N
F_plateau_analytical = N_total * p_j4d.tanPhi / 1e3  # kN

print(f"\n--- Results ---")
print(f"  Peak shear      : {F_peak:.1f} kN  at u = {u_peak:.2f} mm")
print(f"  FEM plateau     : {F_plateau:.1f} kN  (last 20% of steps)")
print(f"  Analyt. plateau : {F_plateau_analytical:.1f} kN  "
      f"(N_total × tanφ = {N_total/1e3:.1f} × {p_j4d.tanPhi})")

# ---- Approximate experimental data (J4D, σ_V = 0.30 MPa) ----
# Sources: Raijmakers & Vermeltfoort (1992); reproduced in Lourenco (1994)
# and Petracca et al. (2017, Table 2: V_max = 57 kN).
# Digitised from published load-displacement plots; uncertainty ≈ ±8 kN.
exp_u    = np.array([0.00, 0.20, 0.40, 0.70, 1.00, 1.40, 1.80, 2.20, 2.60, 3.00])
exp_F    = np.array([0.00, 18.0, 34.0, 48.0, 57.0, 54.0, 48.0, 42.0, 38.0, 35.0])
exp_F_lo = exp_F - 8.0
exp_F_hi = exp_F + 8.0

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#F5F6F8')

# --- Left: load-displacement comparison ---
ax = axes[0]
ax.set_facecolor('#F5F6F8')

# Experimental (digitised with uncertainty band)
ax.fill_between(exp_u, exp_F_lo, exp_F_hi, color='#888', alpha=0.18, zorder=0)
ax.plot(exp_u, exp_F, 's--', color='#444', markersize=5, lw=1.4,
        label='Experiment — J4D\n(Raijmakers & Vermeltfoort 1992,\napprox. ±8 kN digitisation)')

# Our FEM
ax.plot(u_mm, F_kN, color='#A8282A', lw=2.2,
        label='ZT FEM — this study\n(Lourenço 1994 parameters)')

# Analytical plateau
ax.axhline(F_plateau_analytical, color='#1B3872', lw=1.0, ls=':',
           label=f'Friction plateau (analyt.)\n{F_plateau_analytical:.1f} kN = N·tanφ')

# Peak annotation
ax.annotate(f'FEM peak\n{F_peak:.1f} kN @ {u_peak:.2f} mm',
            xy=(u_peak, F_peak), xytext=(u_peak + 0.4, F_peak - 5),
            arrowprops=dict(arrowstyle='->', color='#A8282A', lw=1.2),
            fontsize=8.5, color='#A8282A')

ax.set_xlabel('Top displacement  u  (mm)', fontsize=10)
ax.set_ylabel('Shear resistance  V  (kN)', fontsize=10)
ax.set_title('ZT Model Validation — J4D Specimen\n'
             r'990 × 1000 mm, $\sigma_V$ = 0.30 MPa',
             fontsize=10, fontweight='bold')
ax.set_xlim(0, 3.0)
ax.set_ylim(bottom=0)
ax.legend(fontsize=8, framealpha=0.92, loc='upper right')
ax.grid(True, alpha=0.3)

# --- Right: interface damage map at final step ---
ax2 = axes[1]
ax2.set_facecolor('#EDE9DF')
ax2.set_aspect('equal')
ax2.set_title('Interface damage map (final step)\n'
              'Colour: bed joint kappa2 plastic slip (m)',
              fontsize=9, fontweight='bold')

# Draw bricks
for e, conn in enumerate(mesh.brick_elems):
    xy = mesh.nodes[conn]
    xs = np.append(xy[:, 0], xy[0, 0])
    ys = np.append(xy[:, 1], xy[0, 1])
    ax2.fill(xs, ys, color='#D4C8A8', zorder=1)
    ax2.plot(xs, ys, color='#9B8870', lw=0.4, zorder=2)

# Draw interfaces coloured by kappa2
iface_states = res['iface_states']
kappa2_vals  = np.array([s.kappa2 for s in iface_states])
k2_max = max(kappa2_vals.max(), 1e-10)
cmap_bed  = plt.cm.Reds
cmap_head = plt.cm.Blues

for i, (nodes4, L, horiz, st) in enumerate(zip(
        mesh.iface_nodes, mesh.iface_len, mesh.iface_horiz, iface_states)):
    nA0, nA1, nB0, nB1 = nodes4
    # Midline of the interface
    xA = 0.5 * (mesh.nodes[nA0, 0] + mesh.nodes[nA1, 0])
    yA = 0.5 * (mesh.nodes[nA0, 1] + mesh.nodes[nA1, 1])
    xB = 0.5 * (mesh.nodes[nB0, 0] + mesh.nodes[nB1, 0])
    yB = 0.5 * (mesh.nodes[nB0, 1] + mesh.nodes[nB1, 1])
    x0 = mesh.nodes[nA0, 0]; x1 = mesh.nodes[nA1, 0]
    y0 = mesh.nodes[nA0, 1]; y1 = mesh.nodes[nA1, 1]

    norm_k2 = st.kappa2 / k2_max
    lw = 0.6 + 2.4 * norm_k2   # thicker = more plastic slip

    if horiz:
        colour = cmap_bed(0.2 + 0.8 * norm_k2)
        ax2.plot([x0, x1], [y0, y1], color=colour, lw=lw, zorder=3)
    else:
        colour = cmap_head(0.2 + 0.8 * (st.d1))  # head joints: Mode I damage
        ax2.plot([x0, x1], [y0, y1], color=colour, lw=0.5 + 1.5 * st.d1, zorder=3)

ax2.set_xlim(-0.02, cfg.wall_width + 0.02)
ax2.set_ylim(-0.02, cfg.wall_height + 0.10)
ax2.set_xlabel('x  (m)', fontsize=9)
ax2.set_ylabel('y  (m)', fontsize=9)

# Legends for damage map
bed_patch  = mpatches.Patch(color=cmap_bed(0.9),  label=f'Bed joint slip (max {k2_max*1e3:.2f} mm)')
head_patch = mpatches.Patch(color=cmap_head(0.7), label='Head joint Mode I damage')
ax2.legend(handles=[bed_patch, head_patch], fontsize=7.5, loc='upper right')

plt.tight_layout(pad=1.5)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'urm_validation_j4d.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved plot → {out}")
