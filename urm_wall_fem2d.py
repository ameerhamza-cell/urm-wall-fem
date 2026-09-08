#!/usr/bin/env python3
"""
urm_wall_fem2d.py — 2D plane-stress FEM for URM walls, both mesh styles.

Two styles are solved on the same wall geometry and compared:
  'zt' : Zero-thickness interface elements (Lourenco & Rots 1997)
          Expanded bricks + penalty-based nonlinear interface law
  'em' : Explicit mortar
          Nominal bricks + linear elastic mortar quad elements

Loading: vertical pre-compression, then ramp horizontal shear at top.
Output : load-displacement curves + crack/damage map PNG.

Usage
-----
  python urm_wall_fem2d.py              # runs both and plots comparison
  python urm_wall_fem2d.py --style zt
  python urm_wall_fem2d.py --style em
  python urm_wall_fem2d.py --wall_w 0.60 --wall_h 0.60  # smaller wall
"""

from __future__ import annotations
import sys, os, argparse, copy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

from urm_wall_mesh import WallConfig, place_bricks, find_interfaces_kdtree, Brick, Interface

# ============================================================
# Material parameters  (Lourenco 1996 / van der Pluijm 1993)
# ============================================================
E_BRICK   = 16.7e9   # Pa
NU_BRICK  = 0.20
E_MORTAR  = 0.78e9   # Pa  (explicit mortar elastic stiffness)
NU_MORTAR = 0.20
THICK     = 0.100    # m   wall thickness (out-of-plane, for force/area)

# Interface parameters (zero-thickness style)
KN     = 82.0e9   # Pa/m
KS     = 36.0e9   # Pa/m
FT     = 0.25e6   # Pa
GFI    = 12.0     # N/m
C0     = 0.35e6   # Pa
GFII   = 40.0     # N/m
TANPHI = 0.75

# Loading
SIGMA_V   = -0.30e6   # Pa  uniform vertical pre-compression (negative = compressive)
U_TOTAL   =  1.50e-3  # m   total horizontal displacement at top
N_STEPS   =  150      # load steps
MAX_ITER  = 50        # Newton iterations / step
CONV_TOL  = 1e-3     # 0.1% relative tolerance


# ============================================================
# Plane-stress D matrix
# ============================================================

def plane_stress_D(E: float, nu: float) -> np.ndarray:
    c = E / (1.0 - nu**2)
    return c * np.array([[1.0, nu, 0.0],
                          [nu, 1.0, 0.0],
                          [0.0, 0.0, (1.0 - nu) / 2.0]])


# ============================================================
# Q4 element (bilinear quad, 8 DOF)
# ============================================================

_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)
_GW = np.array([ 1.0, 1.0])
_GAUSS = [(xi, eta, wi * wj)
          for xi, wi in zip(_GP, _GW)
          for eta, wj in zip(_GP, _GW)]


def _q4_shape(xi: float, eta: float):
    """Shape functions and d/dxi, d/deta at (xi,eta)."""
    xs = np.array([-1.0,  1.0,  1.0, -1.0])
    es = np.array([-1.0, -1.0,  1.0,  1.0])
    N       = 0.25 * (1.0 + xs * xi) * (1.0 + es * eta)
    dN_dxi  = 0.25 * xs * (1.0 + es * eta)
    dN_deta = 0.25 * es * (1.0 + xs * xi)
    return N, dN_dxi, dN_deta


def _q4_B(xi: float, eta: float, xy: np.ndarray) -> Tuple[np.ndarray, float]:
    """B matrix (3×8) and determinant of Jacobian."""
    N, dN_dxi, dN_deta = _q4_shape(xi, eta)
    J = np.array([dN_dxi, dN_deta]) @ xy  # (2,2)
    detJ = np.linalg.det(J)
    invJ = np.linalg.inv(J)
    dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
    dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
    B = np.zeros((3, 8))
    B[0, 0::2] = dN_dx
    B[1, 1::2] = dN_dy
    B[2, 0::2] = dN_dy
    B[2, 1::2] = dN_dx
    return B, detJ


def q4_stiffness(xy: np.ndarray, D: np.ndarray, t: float = 1.0) -> np.ndarray:
    """8×8 stiffness matrix.  xy: (4,2) node coords."""
    Ke = np.zeros((8, 8))
    for xi, eta, w in _GAUSS:
        B, detJ = _q4_B(xi, eta, xy)
        Ke += (B.T @ D @ B) * (detJ * w * t)
    return Ke


def q4_fint(xy: np.ndarray, D: np.ndarray, ue: np.ndarray, t: float = 1.0) -> np.ndarray:
    """Internal force vector (8,) for Q4 element."""
    fe = np.zeros(8)
    for xi, eta, w in _GAUSS:
        B, detJ = _q4_B(xi, eta, xy)
        fe += (B.T @ (D @ B @ ue)) * (detJ * w * t)
    return fe


# ============================================================
# Interface constitutive law (2D: [un, us] → [sigma_n, tau])
# Lourenco Mode I + Mode II; compression = elastic (no cap).
# ============================================================

@dataclass
class IFaceState:
    kappa1: float = 0.0   # Mode I: max historical opening
    d1:     float = 0.0   # Mode I damage ∈ [0,1)
    kappa2: float = 0.0   # Mode II: cumulative plastic slip (m)
    up:     float = 0.0   # Mode II: plastic slip (scalar in 2D)


@dataclass
class MortarGPState:
    """Plastic state at one Gauss point of an EM mortar element."""
    up:     float = 0.0   # plastic shear STRAIN (dimensionless)
    kappa2: float = 0.0   # cumulative plastic SLIP in metres (= up × h_mortar summed)


@dataclass
class IFaceParams:
    kn:     float = KN
    ks:     float = KS
    ft:     float = FT
    GfI:    float = GFI
    c0:     float = C0
    GfII:   float = GFII
    tanPhi: float = TANPHI


def iface_update(p: IFaceParams, s: IFaceState,
                 un: float, us: float,
                 skip_mode2: bool = False) -> Tuple[float, float, float, float]:
    """
    Update interface state and return (sigma_n, tau, K_nn, K_ss).
    skip_mode2=True: shear remains elastic (head joints — only Mode I cracking allowed).
    """
    from math import exp

    # ---- Mode I (normal) ----
    if un > 0.0:
        loading = un > s.kappa1
        if loading:
            s.kappa1 = un
        un0 = p.ft / p.kn
        if s.kappa1 <= un0:
            sigma_n = p.kn * un
            K_nn = p.kn
        else:
            sigma_env = p.ft * exp(-p.ft * (s.kappa1 - un0) / p.GfI)
            sigma_n   = sigma_env * (un / s.kappa1)
            d1_new    = 1.0 - sigma_env / (p.kn * s.kappa1)
            if d1_new > s.d1:
                s.d1 = d1_new
            K_nn = p.kn
    else:
        sigma_n = p.kn * un
        K_nn    = p.kn

    # ---- Mode II (shear) ----
    # Head joints (skip_mode2=True): elastic shear only.
    # Running-bond kinematics cause large vertical shear at head joints; allowing
    # Mode II yield there drives K_ff → singular and prevents bed-joint sliding.
    if skip_mode2:
        tau  = p.ks * (us - s.up)   # s.up stays 0 for head joints
        K_ss = p.ks
        return sigma_n, tau, K_nn, K_ss

    c0_eff = p.c0 * (1.0 - s.d1)

    def coh(k: float) -> float:
        return c0_eff * exp(-c0_eff * k / p.GfII)

    friction   = max(0.0, -sigma_n) * p.tanPhi
    tau_trial  = p.ks * (us - s.up)
    tt_abs     = abs(tau_trial)
    f_trial    = tt_abs - (coh(s.kappa2) + friction)

    if tt_abs < 1e-14 or f_trial <= 0.0:
        tau  = tau_trial
        K_ss = p.ks
    else:
        sign   = 1.0 if tau_trial >= 0.0 else -1.0
        dGamma = 0.0
        k0     = s.kappa2
        for _ in range(50):
            k     = k0 + dGamma
            c     = coh(k)
            R     = tt_abs - p.ks * dGamma - c - friction
            dcdk  = -c0_eff**2 / p.GfII * exp(-c0_eff * k / p.GfII)
            dR    = -p.ks - dcdk
            dGamma -= R / dR
            if abs(R / dR) < 1e-14:
                break
        if dGamma < 0.0:
            dGamma = 0.0
        s.kappa2 = k0 + dGamma
        s.up    += dGamma * sign
        tau      = (tt_abs - p.ks * dGamma) * sign
        # Secant stiffness: |tau|/|us_total|. Positive definite; avoids both
        # K_ss=0 cascade and K_ss=KS spurious-equilibrium. Capped at KS.
        K_ss = min(p.ks, abs(tau) / max(abs(us), 1e-15))

    return sigma_n, tau, K_nn, K_ss


def mortar_gp_update(p: IFaceParams, st: MortarGPState,
                     sigma_n: float, gamma_xy: float,
                     G: float, h_mortar: float) -> Tuple[float, float]:
    """
    Mohr-Coulomb cohesion-softening return mapping for one mortar Gauss point.

    st.up     — plastic shear strain (dimensionless, updated in-place)
    st.kappa2 — cumulative plastic SLIP in metres (= Σ|Δγ_p| × h_mortar)
                same scale as ZT kappa2 so GfII applies without rescaling.

    Returns (tau_actual, G_secant) where G_secant replaces D[2,2] in D_ep.
    """
    from math import exp
    c0_eff = p.c0   # Mode I coupling not tracked at mortar GP level

    def coh(k_slip: float) -> float:
        return c0_eff * exp(-c0_eff * k_slip / p.GfII)

    friction  = max(0.0, -sigma_n) * p.tanPhi
    tau_trial = G * (gamma_xy - st.up)
    tt_abs    = abs(tau_trial)
    f_trial   = tt_abs - (coh(st.kappa2) + friction)

    if tt_abs < 1e-14 or f_trial <= 0.0:
        return tau_trial, G

    sign = 1.0 if tau_trial >= 0.0 else -1.0
    dGs  = 0.0           # plastic shear STRAIN increment
    k0   = st.kappa2     # slip in metres
    for _ in range(50):
        k_slip = k0 + dGs * h_mortar
        c      = coh(k_slip)
        R      = tt_abs - G * dGs - c - friction
        dcdk   = -c0_eff**2 / p.GfII * exp(-c0_eff * k_slip / p.GfII)
        dR     = -G - dcdk * h_mortar
        dGs   -= R / dR
        if abs(R / dR) < 1e-14:
            break
    if dGs < 0.0:
        dGs = 0.0
    st.kappa2 += dGs * h_mortar
    st.up     += dGs * sign
    tau_actual = (tt_abs - G * dGs) * sign
    G_secant   = min(G, abs(tau_actual) / max(abs(gamma_xy), 1e-15))
    return tau_actual, G_secant


# ============================================================
# Interface element stiffness / internal force  (4-node, 8 DOF)
# ============================================================

# Gauss rule along 1D interface ξ ∈ [−1,1]
_GP1 = np.array([-1.0, 1.0]) / np.sqrt(3.0)
_GW1 = np.array([ 1.0, 1.0])


def _iface_B(xi: float, is_horizontal: bool) -> np.ndarray:
    """
    2×8 B matrix for zero-thickness interface element.
    DOF order: [uA0x, uA0y, uA1x, uA1y, uB0x, uB0y, uB1x, uB1y]
    A = side A nodes (e.g., top of lower brick), B = side B nodes.
    Δu = u_B − u_A.
    Rows: [Δu_n, Δu_s]
    """
    N0 = 0.5 * (1.0 - xi)
    N1 = 0.5 * (1.0 + xi)
    if is_horizontal:     # normal = +y
        # Δu_n = u_B_y − u_A_y
        # Δu_s = u_B_x − u_A_x
        return np.array([
            [ 0.0, -N0,  0.0, -N1,  0.0,  N0,  0.0,  N1],   # Δu_n
            [-N0,   0.0, -N1,  0.0,  N0,   0.0,  N1,  0.0],  # Δu_s
        ])
    else:                  # normal = +x
        # Δu_n = u_B_x − u_A_x
        # Δu_s = u_B_y − u_A_y
        return np.array([
            [-N0,   0.0, -N1,  0.0,  N0,   0.0,  N1,  0.0],  # Δu_n
            [ 0.0, -N0,  0.0, -N1,  0.0,  N0,   0.0,  N1],   # Δu_s
        ])


def iface_stiffness(length: float, is_horiz: bool,
                    K_nn: float, K_ss: float, t: float = 1.0) -> np.ndarray:
    """
    8×8 stiffness matrix using midpoint (1-point) rule to match iface_fint,
    making the tangent consistent with the internal force (quadratic convergence).
    """
    B_mid = _iface_B(0.0, is_horiz)
    Kmat  = np.diag([K_nn, K_ss])
    return (B_mid.T @ Kmat @ B_mid) * (length * t)


def iface_fint(length: float, is_horiz: bool,
               sigma_n: float, tau: float,
               ue: np.ndarray, t: float = 1.0) -> np.ndarray:
    """
    Internal force vector (8,) for interface element.
    sigma_n, tau: converged tractions at mid-point (constant along element).
    """
    fe = np.zeros(8)
    traction = np.array([sigma_n, tau])
    half_L = length / 2.0
    for xi, w in zip(_GP1, _GW1):
        B = _iface_B(xi, is_horiz)
        fe += (B.T @ traction) * (half_L * w * t)
    return fe


# ============================================================
# Node map (tolerance-based merging)  — used for EM mesh
# ============================================================

class NodeMap:
    def __init__(self, tol: float = 1e-9):
        self._nodes: List[List[float]] = []
        self._map:   Dict[Tuple[int, int], int] = {}
        self._tol = tol

    def get_or_add(self, x: float, y: float) -> int:
        key = (round(x / self._tol), round(y / self._tol))
        if key not in self._map:
            self._map[key] = len(self._nodes)
            self._nodes.append([x, y])
        return self._map[key]

    def coords(self) -> np.ndarray:
        return np.array(self._nodes, dtype=float)


# ============================================================
# Mesh data class
# ============================================================

@dataclass
class FEMesh:
    """Complete 2D FEM mesh for one wall style."""
    style:      str                   # 'zt' or 'em'
    nodes:      np.ndarray            # (n_nodes, 2)
    # Quad elements: [n0,n1,n2,n3] CCW, brick material
    brick_elems: np.ndarray           # (n_bricks, 4) int
    brick_D:     np.ndarray           # (n_bricks, 3, 3)
    # Mortar quad elements (EM only, empty for ZT)
    mortar_elems: np.ndarray          # (n_mortar, 4) int
    mortar_D:     np.ndarray          # (n_mortar, 3, 3)
    # Interface elements (ZT only, empty for EM)
    # Each: [nodeA0, nodeA1, nodeB0, nodeB1], length, is_horiz
    iface_nodes: List[List[int]]      # list of [nA0,nA1,nB0,nB1]
    iface_len:   List[float]
    iface_horiz: List[bool]
    # BC: fixed DOF indices (row: node, col: 0=x,1=y)
    fixed_dofs:  List[int]
    # Top nodes for loading (all nodes on the top edge)
    top_nodes:   List[int]
    # Map from iface element index → Interface object (for post-processing)
    iface_source: List[Interface]     # corresponding Interface from mesh generator
    # Per mortar element: 'bed' or 'head' (EM only, empty for ZT)
    mortar_kind: List[str]


# ============================================================
# Mesh builders
# ============================================================

def _brick_node_indices(bid: int, local: List[int]) -> List[int]:
    """Global node index = 4*bid + local  (ZT convention)."""
    return [4 * bid + l for l in local]


def build_zt_mesh(cfg: WallConfig) -> FEMesh:
    """
    Zero-thickness mesh (Lourenco & Rots 1997 simplified micro-modelling):
    - Each brick expanded by mortar/2 on every face so adjacent expanded bricks
      are face-to-face with zero gap → interface Δu_n = 0 at rest.
    - Interface elements (zero-thickness quads) sit exactly at those shared faces.
    - Each brick → 4 INDEPENDENT nodes (not shared with neighbours) so the
      interface element can capture the relative displacement jump.
    """
    bricks     = place_bricks(cfg)
    interfaces = find_interfaces_kdtree(bricks, cfg)

    n_nodes = 4 * len(bricks)
    nodes   = np.zeros((n_nodes, 2))

    D_brick = plane_stress_D(E_BRICK, NU_BRICK)
    brick_elems = np.zeros((len(bricks), 4), dtype=int)
    brick_D_arr = np.tile(D_brick, (len(bricks), 1, 1))

    mh = cfg.mortar_head
    mb = cfg.mortar_bed
    W  = cfg.wall_width
    H  = cfg.wall_height + mb   # allow top course to extend one mortar height above

    for b in bricks:
        cx, cy = b.center[0], b.center[1]
        lx, ly = b.size[0], b.size[1]
        # *** Expand by mortar/2 and clip to wall boundary ***
        x0 = max(0.0, cx - (lx + mh) / 2.0)
        x1 = min(W,   cx + (lx + mh) / 2.0)
        y0 = max(0.0, cy - (ly + mb) / 2.0)
        y1 = min(H,   cy + (ly + mb) / 2.0)
        base = 4 * b.bid
        nodes[base + 0] = [x0, y0]   # bottom-left
        nodes[base + 1] = [x1, y0]   # bottom-right
        nodes[base + 2] = [x1, y1]   # top-right
        nodes[base + 3] = [x0, y1]   # top-left
        brick_elems[b.bid] = [base, base + 1, base + 2, base + 3]

    # Interface elements
    iface_nodes_list: List[List[int]] = []
    iface_len_list:   List[float]     = []
    iface_horiz_list: List[bool]      = []
    iface_src:        List[Interface] = []

    for ifc in interfaces:
        ba = bricks[ifc.bid_a]
        bb = bricks[ifc.bid_b]
        if ifc.joint_type == 'bed':
            # A=lower, B=upper
            # A top face: local nodes 3(TL),2(TR) — left to right
            # B bot face: local nodes 0(BL),1(BR) — left to right
            nA0 = 4 * ba.bid + 3
            nA1 = 4 * ba.bid + 2
            nB0 = 4 * bb.bid + 0
            nB1 = 4 * bb.bid + 1
            iface_nodes_list.append([nA0, nA1, nB0, nB1])
            iface_len_list.append(float(ifc.area / ba.size[2]))   # overlap_x
            iface_horiz_list.append(True)
        elif ifc.joint_type == 'head':
            # A=left, B=right
            # A right face: local 1(BR),2(TR) — bottom to top
            # B left  face: local 0(BL),3(TL) — bottom to top
            nA0 = 4 * ba.bid + 1
            nA1 = 4 * ba.bid + 2
            nB0 = 4 * bb.bid + 0
            nB1 = 4 * bb.bid + 3
            iface_nodes_list.append([nA0, nA1, nB0, nB1])
            iface_len_list.append(float(ifc.area / ba.size[2]))   # overlap_y
            iface_horiz_list.append(False)
        else:
            continue   # perp joints: skip (single wythe)
        iface_src.append(ifc)

    # Fixed DOFs: bottom edge — all nodes whose y ≈ 0
    tol   = 1e-6
    y_bot = nodes[:, 1].min()
    y_top = nodes[:, 1].max()
    fixed = [2*n + d for n in range(n_nodes)
             if abs(nodes[n, 1] - y_bot) < tol for d in (0, 1)]
    top_nodes = [n for n in range(n_nodes) if abs(nodes[n, 1] - y_top) < tol]

    return FEMesh(
        style='zt',
        nodes=nodes,
        brick_elems=brick_elems,
        brick_D=brick_D_arr,
        mortar_elems=np.empty((0, 4), dtype=int),
        mortar_D=np.empty((0, 3, 3)),
        iface_nodes=iface_nodes_list,
        iface_len=iface_len_list,
        iface_horiz=iface_horiz_list,
        fixed_dofs=fixed,
        top_nodes=top_nodes,
        iface_source=iface_src,
        mortar_kind=[],
    )


def build_em_mesh(cfg: WallConfig) -> FEMesh:
    """
    Explicit-mortar mesh — conforming structured-grid approach.

    Problem with running bond: mortar overlap nodes don't coincide with brick
    face nodes, creating a disconnected mesh.  Fix: generate all x-levels from
    ALL courses' brick boundaries, then subdivide both bricks and mortar into
    thin sub-elements that share nodes at those x-levels.  This is fully
    conforming regardless of bond pattern.
    """
    bricks = place_bricks(cfg)

    by_course: Dict[int, List[Brick]] = {}
    for b in bricks:
        by_course.setdefault(b.course, []).append(b)
    n_courses = max(by_course.keys()) + 1

    bh = cfg.brick_height
    mb = cfg.mortar_bed

    # All unique y-levels (brick + mortar layer boundaries)
    y_set: set = {0.0, cfg.wall_height}
    for ci in range(n_courses + 1):
        y_set.add(ci * (bh + mb))
        if ci < n_courses:
            y_set.add(ci * (bh + mb) + bh)
    y_sorted = sorted(y_set)

    # All unique x-levels (from all brick face positions across all courses)
    x_set: set = {0.0, cfg.wall_width}
    for b in bricks:
        x_set.add(max(0.0,            b.center[0] - b.size[0] / 2.0))
        x_set.add(min(cfg.wall_width, b.center[0] + b.size[0] / 2.0))
    x_sorted = sorted(x_set)

    nm = NodeMap(tol=1e-10)
    D_brick  = plane_stress_D(E_BRICK,  NU_BRICK)
    D_mortar = plane_stress_D(E_MORTAR, NU_MORTAR)
    brick_conn:      List[List[int]]   = []
    mortar_conn:     List[List[int]]   = []
    brick_D_list:    List[np.ndarray]  = []
    mortar_D_list:   List[np.ndarray]  = []
    mortar_kind_list: List[str]        = []

    eps = 1e-9

    def _in_any(xm: float, bs: List[Brick]) -> bool:
        return any(b.center[0] - b.size[0]/2.0 - eps <= xm <=
                   b.center[0] + b.size[0]/2.0 + eps for b in bs)

    for ky in range(len(y_sorted) - 1):
        y0, y1 = y_sorted[ky], y_sorted[ky + 1]
        if y1 - y0 < eps:
            continue
        ym = (y0 + y1) / 2.0

        # Classify this y-band
        cell_kind = None   # 'brick', 'bed_mortar'
        ci_band   = -1
        for ci in range(n_courses):
            yb_lo = ci * (bh + mb)
            yb_hi = yb_lo + bh
            ym_hi = yb_hi + mb
            if yb_lo - eps < ym < yb_hi + eps:
                cell_kind = 'brick';      ci_band = ci; break
            if ci < n_courses - 1 and yb_hi - eps < ym < ym_hi + eps:
                cell_kind = 'bed_mortar'; ci_band = ci; break
        if cell_kind is None:
            continue

        bs_this  = by_course.get(ci_band, [])
        bs_above = by_course.get(ci_band + 1, []) if cell_kind == 'bed_mortar' else []

        for kx in range(len(x_sorted) - 1):
            x0b, x1b = x_sorted[kx], x_sorted[kx + 1]
            if x1b - x0b < eps:
                continue
            xm = (x0b + x1b) / 2.0

            if cell_kind == 'brick':
                if _in_any(xm, bs_this):
                    D, lst = D_brick, brick_conn; dl = brick_D_list
                else:
                    # Check if this gap is a head-mortar zone
                    bs_s = sorted(bs_this, key=lambda b: b.center[0])
                    in_head = False
                    for k in range(len(bs_s) - 1):
                        x_gl = bs_s[k].center[0] + bs_s[k].size[0] / 2.0
                        x_gr = bs_s[k+1].center[0] - bs_s[k+1].size[0] / 2.0
                        if x_gl - eps < xm < x_gr + eps:
                            in_head = True; break
                    if not in_head:
                        continue
                    D, lst, dl = D_mortar, mortar_conn, mortar_D_list
                    mortar_kind_list.append('head')
            else:  # bed_mortar
                if not (_in_any(xm, bs_this) and _in_any(xm, bs_above)):
                    continue
                D, lst, dl = D_mortar, mortar_conn, mortar_D_list
                mortar_kind_list.append('bed')

            n0 = nm.get_or_add(x0b, y0)
            n1 = nm.get_or_add(x1b, y0)
            n2 = nm.get_or_add(x1b, y1)
            n3 = nm.get_or_add(x0b, y1)
            lst.append([n0, n1, n2, n3])
            dl.append(D)

    nodes   = nm.coords()
    n_nodes = len(nodes)
    n_b     = len(brick_conn)
    n_m     = len(mortar_conn)

    brick_elems  = np.array(brick_conn,  dtype=int) if n_b else np.empty((0,4), dtype=int)
    mortar_elems = np.array(mortar_conn, dtype=int) if n_m else np.empty((0,4), dtype=int)
    brick_D_arr  = np.array(brick_D_list)  if n_b else np.empty((0,3,3))
    mortar_D_arr = np.array(mortar_D_list) if n_m else np.empty((0,3,3))

    tol   = 1e-6
    y_bot = nodes[:, 1].min()
    y_top = nodes[:, 1].max()
    fixed = [2*n + d for n in range(n_nodes)
             if abs(nodes[n, 1] - y_bot) < tol for d in (0, 1)]
    top_nodes = [n for n in range(n_nodes) if abs(nodes[n, 1] - y_top) < tol]

    print(f"  EM structured grid: {n_b} brick sub-elems, {n_m} mortar sub-elems, "
          f"{n_nodes} nodes")

    return FEMesh(
        style='em',
        nodes=nodes,
        brick_elems=brick_elems,
        brick_D=brick_D_arr,
        mortar_elems=mortar_elems,
        mortar_D=mortar_D_arr,
        iface_nodes=[],
        iface_len=[],
        iface_horiz=[],
        fixed_dofs=fixed,
        top_nodes=top_nodes,
        iface_source=[],
        mortar_kind=mortar_kind_list,
    )


# ============================================================
# DOF numbering helpers
# ============================================================

def _elem_dofs(conn: np.ndarray) -> np.ndarray:
    """Global DOF indices for a quad element given node connectivity."""
    return np.array([2 * n + d for n in conn for d in (0, 1)], dtype=int)


def _iface_dofs(nodes4: List[int]) -> np.ndarray:
    """Global DOF indices for [nA0,nA1,nB0,nB1]."""
    return np.array([2 * n + d for n in nodes4 for d in (0, 1)], dtype=int)


# ============================================================
# Assembly: tangent stiffness K and internal force f_int
# ============================================================

def assemble(mesh: FEMesh, u: np.ndarray,
             iface_states: List[IFaceState],
             p_iface: IFaceParams,
             mortar_gp_states: Optional[List[List[MortarGPState]]] = None,
             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (K_T, f_int) — tangent stiffness (dense) and internal force.
    For ZT: calls iface_update to update states and get tangent.
    For EM: per-Gauss-point Mohr-Coulomb plasticity for mortar elements.
    """
    n_dof = 2 * len(mesh.nodes)
    K  = np.zeros((n_dof, n_dof))
    fi = np.zeros(n_dof)

    # ---- Brick elements ----
    for e, conn in enumerate(mesh.brick_elems):
        dofs = _elem_dofs(conn)
        xy   = mesh.nodes[conn]
        D    = mesh.brick_D[e]
        ue   = u[dofs]
        K[np.ix_(dofs, dofs)] += q4_stiffness(xy, D, THICK)
        fi[dofs]              += q4_fint(xy, D, ue, THICK)

    # ---- Mortar quad elements (EM only) — per-GP Mohr-Coulomb plasticity ----
    for e, conn in enumerate(mesh.mortar_elems):
        dofs = _elem_dofs(conn)
        xy   = mesh.nodes[conn]
        D    = mesh.mortar_D[e]      # elastic D (3×3)
        ue   = u[dofs]
        kind = mesh.mortar_kind[e]   # 'bed' or 'head'
        G    = D[2, 2]               # shear modulus

        # mortar thickness in normal direction (for kappa2 ↔ GfII scaling)
        if kind == 'bed':
            h_m = float(np.max(xy[:, 1]) - np.min(xy[:, 1]))
        else:
            h_m = float(np.max(xy[:, 0]) - np.min(xy[:, 0]))

        gp_states = (mortar_gp_states[e]
                     if mortar_gp_states is not None else [None] * 4)

        Ke = np.zeros((8, 8))
        fe = np.zeros(8)
        for gp_idx, (xi, eta, w) in enumerate(_GAUSS):
            B, detJ = _q4_B(xi, eta, xy)
            eps      = B @ ue          # [ε_xx, ε_yy, γ_xy]
            sigma_el = D @ eps         # elastic stress trial

            st = gp_states[gp_idx]
            if st is not None and h_m > 1e-12:
                # normal stress component (compressive normal → friction)
                sigma_n  = sigma_el[1] if kind == 'bed' else sigma_el[0]
                gamma_xy = eps[2]
                tau_act, G_ep = mortar_gp_update(p_iface, st, sigma_n,
                                                 gamma_xy, G, h_m)
                D_ep      = D.copy()
                D_ep[2, 2] = G_ep
                sigma_act  = sigma_el.copy()
                sigma_act[2] = tau_act
            else:
                D_ep     = D
                sigma_act = sigma_el

            Ke += (B.T @ D_ep  @ B)         * (detJ * w * THICK)
            fe += (B.T @ sigma_act)          * (detJ * w * THICK)

        K[np.ix_(dofs, dofs)] += Ke
        fi[dofs]               += fe

    # ---- Interface elements (ZT only) ----
    for idx, (nodes4, L, horiz) in enumerate(
            zip(mesh.iface_nodes, mesh.iface_len, mesh.iface_horiz)):
        dofs  = _iface_dofs(nodes4)
        ue    = u[dofs]
        st    = iface_states[idx]

        # Compute displacement jump at element midpoint (xi=0)
        B_mid = _iface_B(0.0, horiz)
        jump  = B_mid @ ue    # [Δu_n, Δu_s]
        un, us = float(jump[0]), float(jump[1])

        # Head joints (not horiz): elastic shear to prevent running-bond tilt cascade.
        # Bed joints (horiz): secant K_ss allows plastic slip accumulation.
        sigma_n, tau, K_nn, K_ss = iface_update(p_iface, st, un, us,
                                                  skip_mode2=(not horiz))

        Ke = iface_stiffness(L, horiz, K_nn, K_ss, THICK)
        K[np.ix_(dofs, dofs)] += Ke

        # Internal force using current (updated) tractions
        fi[dofs] += iface_fint(L, horiz, sigma_n, tau, ue, THICK)

    return K, fi


# ============================================================
# Apply BCs (direct elimination)
# ============================================================

def apply_bcs(K: np.ndarray, f: np.ndarray,
              fixed_dofs: List[int],
              prescribed: Optional[Dict[int, float]] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enforce fixed DOFs via penalty / direct elimination.
    prescribed: optional dict of {dof: value} for non-zero prescribed displacements.
    """
    K2 = K.copy()
    f2 = f.copy()
    pres = prescribed or {}

    for dof in fixed_dofs:
        val = pres.get(dof, 0.0)
        f2 -= K2[:, dof] * val
        K2[dof, :]  = 0.0
        K2[:, dof]  = 0.0
        K2[dof, dof] = 1.0
        f2[dof]      = val
    return K2, f2


# ============================================================
# Newton-Raphson solver (incremental + iterative)
# ============================================================

def solve_wall(mesh: FEMesh, cfg: WallConfig,
               p_iface_override: Optional[IFaceParams] = None,
               sigma_v: float = SIGMA_V,
               u_total: float = U_TOTAL,
               n_steps: int = N_STEPS) -> Dict:
    """
    Incremental-iterative Newton-Raphson solver.
    Loading:
      1. Single elastic step for vertical pre-compression.
      2. Monotonic horizontal displacement ramp at top nodes.

    Optional overrides allow validation runs with custom material parameters
    (e.g. Lourenço 1994 calibrated values for Raijmakers J4D specimen).
    """
    n_dof    = 2 * len(mesh.nodes)
    n_iface  = len(mesh.iface_nodes)
    n_mortar = len(mesh.mortar_elems)
    p_iface  = p_iface_override if p_iface_override is not None else IFaceParams()

    def _fresh_mortar_states():
        return [[MortarGPState() for _ in range(4)] for _ in range(n_mortar)]

    def _copy_mortar_states(src):
        return [[copy.copy(s) for s in gp] for gp in src]

    # ---- Pre-compression (elastic, one step) ----
    saved_states        = [IFaceState() for _ in range(n_iface)]
    saved_mortar_states = _fresh_mortar_states()
    u = np.zeros(n_dof)

    top_area = cfg.wall_width * THICK / max(len(mesh.top_nodes), 1)
    f_precomp = np.zeros(n_dof)
    for n in mesh.top_nodes:
        f_precomp[2 * n + 1] += sigma_v * top_area

    # One elastic solve for pre-compression (no plastic state committed)
    trial = [copy.copy(s) for s in saved_states]
    K0, _ = assemble(mesh, u, trial, p_iface, _fresh_mortar_states())
    K_bc, f_bc = apply_bcs(K0, f_precomp, mesh.fixed_dofs,
                            {d: 0.0 for d in mesh.fixed_dofs})
    u = np.linalg.solve(K_bc, f_bc)
    saved_states        = [IFaceState() for _ in range(n_iface)]
    saved_mortar_states = _fresh_mortar_states()

    # ---- Shear increments ----
    u_history = [0.0]
    F_history = [0.0]
    du_step   = u_total / n_steps

    for step in range(n_steps):
        u_top = (step + 1) * du_step
        presc: Dict[int, float] = {d: 0.0 for d in mesh.fixed_dofs}
        for n in mesh.top_nodes:
            presc[2 * n] = u_top

        for dof, val in presc.items():
            u[dof] = val

        free = [d for d in range(n_dof) if d not in presc]

        converged = False
        du_ref    = 1.0

        for it in range(MAX_ITER):
            # Reset to committed state at start of each Newton iteration
            trial        = [copy.copy(s) for s in saved_states]
            trial_mortar = _copy_mortar_states(saved_mortar_states)
            K, fi = assemble(mesh, u, trial, p_iface, trial_mortar)

            res      = f_precomp - fi
            res_norm = np.linalg.norm(res[free])

            if it == 0:
                du_ref = max(res_norm, 1.0)
            elif res_norm / du_ref < CONV_TOL:
                converged = True
                break

            K_ff = K[np.ix_(free, free)]
            r_f  = res[free]
            try:
                du_f = np.linalg.solve(K_ff, r_f)
            except np.linalg.LinAlgError:
                du_f = np.zeros(len(free))

            u[free] += du_f

        # Commit states from last Newton iteration
        saved_states        = trial
        saved_mortar_states = trial_mortar

        # Horizontal reaction
        trial2   = [copy.copy(s) for s in saved_states]
        trial2_m = _copy_mortar_states(saved_mortar_states)
        _, fi2   = assemble(mesh, u, trial2, p_iface, trial2_m)
        F_react  = sum(fi2[d] for d in mesh.fixed_dofs if d % 2 == 0)

        u_history.append(u_top)
        F_history.append(F_react)

        if step % max(n_steps // 8, 1) == 0 or step == n_steps - 1 or step < 5:
            n_plast = (sum(1 for s in saved_states
                           if s.kappa2 > 0 or s.kappa1 > s.__class__().kappa1) +
                       sum(1 for gp in saved_mortar_states for s in gp if s.kappa2 > 0))
            print(f"  step {step+1:3d}/{n_steps}: u_top={u_top*1e3:.3f} mm  "
                  f"F={F_react/1e3:.3f} kN  plastic={n_plast}  "
                  f"{'OK' if converged else 'NO-CONV'}")

    return {
        'u':                 u,
        'u_history':         np.array(u_history),
        'F_history':         np.array(F_history),
        'iface_states':      saved_states,
        'kappa1':            [s.kappa1 for s in saved_states],
        'kappa2':            [s.kappa2 for s in saved_states],
        'd1':                [s.d1     for s in saved_states],
        'mortar_gp_states':  saved_mortar_states,
        'mesh':              mesh,
    }


# ============================================================
# Post-processing: combined comparison plot
# ============================================================

def plot_comparison(results: Dict[str, Dict], cfg: WallConfig, out_path: str):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#faf9f7')

    # ---- Load-displacement curves (left panel) ----
    ax_ld = fig.add_subplot(1, 3, 1)
    colors = {'zt': '#cc2222', 'em': '#1a52c0'}
    labels = {
        'zt': 'Zero-thickness interfaces\n(Lourenço & Rots 1997)',
        'em': 'Explicit mortar elements',
    }
    for style, res in results.items():
        u_mm = res['u_history'] * 1e3
        F_kN = -res['F_history'] / 1e3   # negate: plot resistance (positive = rightward)
        ax_ld.plot(u_mm, F_kN, color=colors[style], lw=2.0, label=labels[style])

    # Reference lines: analytical friction plateau (full contact)
    F_plateau_kN = abs(SIGMA_V) * cfg.wall_width * THICK * TANPHI / 1e3
    ax_ld.axhline(F_plateau_kN, color='#555', lw=0.8, ls='--',
                  label=f'Friction plateau (100% contact)\n{F_plateau_kN:.1f} kN')

    ax_ld.set_xlabel('Top displacement  u [mm]', fontsize=9)
    ax_ld.set_ylabel('Shear resistance  V [kN]', fontsize=9)
    ax_ld.set_title('Load–Displacement\nComparison', fontsize=10, fontweight='bold')
    ax_ld.set_ylim(bottom=0)
    ax_ld.legend(fontsize=8, framealpha=0.9)
    ax_ld.grid(True, alpha=0.3)

    # ---- Damage maps (middle = ZT, right = EM) ----
    for col, (style, res) in enumerate(results.items(), start=2):
        ax = fig.add_subplot(1, 3, col)
        m  = res['mesh']
        ax.set_facecolor('#f0ede8')

        D_brick = plane_stress_D(E_BRICK, NU_BRICK)
        # Draw bricks
        for conn in m.brick_elems:
            xy = m.nodes[conn]
            xmin, ymin = xy[:, 0].min(), xy[:, 1].min()
            xmax, ymax = xy[:, 0].max(), xy[:, 1].max()
            ax.add_patch(mpatches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=0.3, edgecolor='#4a3520',
                facecolor='#c8945c', zorder=2))

        # Draw mortar (EM only) — colour by plastic damage if available
        mgps     = res.get('mortar_gp_states', [])
        slip_ref = GFII / C0   # characteristic slip for normalization
        cmap_m   = plt.cm.RdYlGn_r
        for e, conn in enumerate(m.mortar_elems):
            xy = m.nodes[conn]
            xmin, ymin = xy[:, 0].min(), xy[:, 1].min()
            xmax, ymax = xy[:, 0].max(), xy[:, 1].max()
            if mgps:
                dmg = min(1.0, max(s.kappa2 for s in mgps[e]) / slip_ref)
                fc  = cmap_m(dmg) if dmg > 0.01 else '#b0a898'
                ec  = '#888'
            else:
                fc, ec = '#b0a898', '#888'
            ax.add_patch(mpatches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=0.3, edgecolor=ec, facecolor=fc, zorder=2))

        # Draw interface damage (ZT only)
        if style == 'zt':
            d1_arr = np.array(res['d1'])
            k2_arr = np.array(res['kappa2'])
            # Normalise damage (0 = intact, 1 = fully open/slid)
            d_mode1 = d1_arr / max(d1_arr.max(), 1e-12)
            d_mode2 = np.minimum(k2_arr / (GFII / C0), 1.0)

            segs_bed  = []; c_bed  = []
            segs_head = []; c_head = []
            for idx, ifc in enumerate(m.iface_source):
                q = ifc.quad
                damage = max(d_mode1[idx], d_mode2[idx])
                if ifc.joint_type == 'bed':
                    segs_bed.append([(q[:,0].min(), ifc.center[1]),
                                     (q[:,0].max(), ifc.center[1])])
                    c_bed.append(damage)
                else:
                    segs_head.append([(ifc.center[0], q[:,1].min()),
                                      (ifc.center[0], q[:,1].max())])
                    c_head.append(damage)

            cmap = plt.cm.RdYlGn_r
            if segs_bed:
                lc = LineCollection(segs_bed, array=np.array(c_bed),
                                    cmap=cmap, norm=plt.Normalize(0, 1),
                                    linewidths=2.5, zorder=4)
                ax.add_collection(lc)
            if segs_head:
                lc = LineCollection(segs_head, array=np.array(c_head),
                                    cmap=cmap, norm=plt.Normalize(0, 1),
                                    linewidths=2.5, zorder=4)
                ax.add_collection(lc)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
            plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01,
                         label='Interface damage\n(0=intact, 1=failed)')

        # EM: colorbar for mortar damage
        if style == 'em' and mgps:
            sm = plt.cm.ScalarMappable(cmap=cmap_m, norm=plt.Normalize(0, 1))
            plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01,
                         label='Mortar damage\n(0=intact, 1=failed)')

        pad = 0.02
        ax.set_xlim(-pad, cfg.wall_width  + pad)
        ax.set_ylim(-pad, cfg.wall_height + pad)
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]', fontsize=8)
        ax.set_ylabel('y [m]', fontsize=8)
        name = labels[style].split('\n')[0]
        ax.set_title(f'Final crack/damage pattern\n{name}',
                     fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)

    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'\nPlot saved → {out_path}')


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--style', choices=['zt', 'em', 'both'], default='both')
    parser.add_argument('--wall_w', type=float, default=0.60)
    parser.add_argument('--wall_h', type=float, default=0.60)
    parser.add_argument('--n_steps', type=int, default=N_STEPS)
    args = parser.parse_args()

    cfg = WallConfig(
        wall_width=args.wall_w, wall_height=args.wall_h,
        brick_length=0.210, brick_height=0.052,
        mortar_bed=0.010, mortar_head=0.010,
        bond='running', n_wythes=1,
    )
    styles = ['zt', 'em'] if args.style == 'both' else [args.style]

    results = {}
    for style in styles:
        print(f'\n{"="*55}')
        print(f' Running style: {style.upper()}  '
              f'({args.wall_w:.2f}m × {args.wall_h:.2f}m)')
        print(f'{"="*55}')
        if style == 'zt':
            mesh = build_zt_mesh(cfg)
        else:
            mesh = build_em_mesh(cfg)
        print(f'  {len(mesh.nodes)} nodes,  '
              f'{len(mesh.brick_elems)} brick elems,  '
              f'{len(mesh.mortar_elems)} mortar elems,  '
              f'{len(mesh.iface_nodes)} interface elems')
        results[style] = solve_wall(mesh, cfg)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, f'urm_fem2d_{"_".join(styles)}.png')
    plot_comparison(results, cfg, out_png)

    # Print summary
    print('\n--- Summary ---')
    for style, res in results.items():
        F_peak = np.max(np.abs(res['F_history']))
        print(f'  {style.upper()}: peak shear force = {F_peak/1e3:.2f} kN')


if __name__ == '__main__':
    main()
