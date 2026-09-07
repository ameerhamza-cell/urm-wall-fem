#!/usr/bin/env python3
"""
urm_wall_mesh.py — Mesh generator for unreinforced masonry walls.

Generates brick and interface meshes for simplified micro-modeling
(Lourenco & Rots 1997 style: expanded bricks, zero-thickness mortar interfaces).

Outputs
-------
  <prefix>_bricks.vtu       -- hex brick elements (ParaView-ready)
  <prefix>_interfaces.vtu   -- quad interface midplane elements with joint-type colour
  <prefix>_mesh.json        -- full connectivity for FEM codes

Interface detection
-------------------
Two algorithms are provided and results are cross-checked:
  - KDTree  : scipy.spatial.KDTree on brick face-centres; O(n log n)
  - AABB    : axis-aligned bounding-box expansion collision; O(n log n) via spatial indexing

Usage
-----
  python urm_wall_mesh.py                      # default 1.2m x 1.0m running-bond wall
  python urm_wall_mesh.py --config my.json     # load WallConfig from JSON file
"""

from __future__ import annotations
import argparse, json, math, sys
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WallConfig:
    # Wall outer dimensions [m]
    wall_width:   float = 1.20
    wall_height:  float = 1.00

    # Brick nominal dimensions [m] (standard Dutch brick)
    brick_length: float = 0.210
    brick_height: float = 0.052
    brick_depth:  float = 0.100  # single-wythe depth

    # Mortar joint thicknesses [m]
    mortar_bed:   float = 0.010   # horizontal (between courses)
    mortar_head:  float = 0.010   # vertical   (within course)
    mortar_perp:  float = 0.010   # perpendicular (through-thickness, multi-wythe)

    # Bond pattern: 'running' | 'stack'
    bond: str = 'running'

    # Number of wythes (layers through wall thickness)
    n_wythes: int = 1

    # Openings: list of [x_left, y_bottom, width, height] in metres
    # A brick is removed if its centre falls inside an opening.
    openings: List[List[float]] = field(default_factory=list)

    # Interface detection algorithm: 'kdtree' | 'aabb' | 'both'
    interface_method: str = 'both'

    @property
    def course_height(self) -> float:
        return self.brick_height + self.mortar_bed

    @property
    def module_width(self) -> float:
        return self.brick_length + self.mortar_head

    @property
    def wythe_depth(self) -> float:
        return self.brick_depth + self.mortar_perp


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Brick:
    bid:    int
    course: int
    wythe:  int
    pos:    int                # column index in course
    center: np.ndarray         # [x, y, z]
    size:   np.ndarray         # [lx, ly, lz]

    def corners(self) -> np.ndarray:
        """Return 8 corner nodes in VTK hex ordering."""
        cx, cy, cz = self.center
        hx, hy, hz = self.size / 2
        return np.array([
            [cx-hx, cy-hy, cz-hz],   # 0
            [cx+hx, cy-hy, cz-hz],   # 1
            [cx+hx, cy+hy, cz-hz],   # 2
            [cx-hx, cy+hy, cz-hz],   # 3
            [cx-hx, cy-hy, cz+hz],   # 4
            [cx+hx, cy-hy, cz+hz],   # 5
            [cx+hx, cy+hy, cz+hz],   # 6
            [cx-hx, cy+hy, cz+hz],   # 7
        ])

    def face_centers(self) -> Dict[str, np.ndarray]:
        """Six face centres keyed by direction."""
        cx, cy, cz = self.center
        hx, hy, hz = self.size / 2
        return {
            'x-': np.array([cx-hx, cy, cz]),
            'x+': np.array([cx+hx, cy, cz]),
            'y-': np.array([cx, cy-hy, cz]),
            'y+': np.array([cx, cy+hy, cz]),
            'z-': np.array([cx, cy, cz-hz]),
            'z+': np.array([cx, cy, cz+hz]),
        }


@dataclass
class Interface:
    bid_a:      int          # lower/left brick index
    bid_b:      int          # upper/right brick index
    joint_type: str          # 'bed' | 'head' | 'perp'
    normal:     np.ndarray   # unit outward normal from a -> b
    area:       float        # overlap area [m²]
    center:     np.ndarray   # midplane centre [m]
    # Four corner nodes of the interface midplane quad (for VTK output)
    quad:       np.ndarray = field(default=None)  # shape (4,3)


# ---------------------------------------------------------------------------
# Brick placement
# ---------------------------------------------------------------------------

def _inside_opening(cx: float, cy: float,
                    openings: List[List[float]]) -> bool:
    """True if brick centre (cx, cy) falls inside any opening rectangle."""
    for (ox, oy, ow, oh) in openings:
        if ox <= cx <= ox+ow and oy <= cy <= oy+oh:
            return True
    return False


def place_bricks(cfg: WallConfig) -> List[Brick]:
    bricks: List[Brick] = []
    bid = 0

    n_courses = math.ceil(cfg.wall_height / cfg.course_height)
    # +2 columns to ensure we cover the full width even after bond offsets
    n_cols = math.ceil(cfg.wall_width / cfg.module_width) + 2

    for wythe in range(cfg.n_wythes):
        cz = wythe * cfg.wythe_depth + cfg.brick_depth / 2

        for course in range(n_courses):
            cy = course * cfg.course_height + cfg.brick_height / 2
            if cy - cfg.brick_height / 2 >= cfg.wall_height:
                break

            # Bond offset: running = half-module on odd courses; stack = none
            if cfg.bond == 'running':
                x_offset = (course % 2) * cfg.module_width / 2
            else:
                x_offset = 0.0

            for pos in range(-1, n_cols + 1):
                cx = pos * cfg.module_width + x_offset

                # Nominal brick extents before clipping
                x_lo = cx - cfg.brick_length / 2
                x_hi = cx + cfg.brick_length / 2

                # Clip to wall boundary — produces half-bricks at the edges
                x_lo = max(x_lo, 0.0)
                x_hi = min(x_hi, cfg.wall_width)

                # Skip bricks entirely outside
                min_frag = 1e-4  # ignore slivers thinner than 0.1 mm
                if x_hi - x_lo < min_frag:
                    continue

                lx_clipped = x_hi - x_lo
                cx_clipped = (x_lo + x_hi) / 2

                # Skip bricks whose (clipped) centre is inside an opening
                if _inside_opening(cx_clipped, cy, cfg.openings):
                    continue

                bricks.append(Brick(
                    bid=bid, course=course, wythe=wythe, pos=pos,
                    center=np.array([cx_clipped, cy, cz]),
                    size=np.array([lx_clipped, cfg.brick_height, cfg.brick_depth]),
                ))
                bid += 1

    print(f"[mesh] placed {len(bricks)} bricks  "
          f"({cfg.n_wythes} wythe(s), {n_courses} courses, bond={cfg.bond})")
    return bricks


# ---------------------------------------------------------------------------
# Interface detection — KDTree method
# ---------------------------------------------------------------------------

def _overlap_1d(a_lo, a_hi, b_lo, b_hi) -> float:
    return max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))


def find_interfaces_kdtree(bricks: List[Brick], cfg: WallConfig) -> List[Interface]:
    """
    Find brick pairs that share a mortar joint using a KDTree on brick centres.

    For each brick pair (i, j), the geometry is checked:
      - bed  joint  : centres differ mainly in y by (hy_i + hy_j + mortar_bed)
                      with overlapping footprints in x and z
      - head joint  : centres differ mainly in x by (hx_i + hx_j + mortar_head)
                      with overlapping footprints in y and z
      - perp joint  : centres differ mainly in z by (hz_i + hz_j + mortar_perp)
                      with overlapping footprints in x and y
    """
    centers = np.array([b.center for b in bricks])
    max_search = (max(cfg.brick_length, cfg.brick_height, cfg.brick_depth) +
                  max(cfg.mortar_bed, cfg.mortar_head, cfg.mortar_perp) + 1e-3)
    tree = KDTree(centers)

    tol = 1e-6  # geometric tolerance [m]
    interfaces: List[Interface] = []
    seen: set = set()

    for i, bi in enumerate(bricks):
        indices = tree.query_ball_point(bi.center, max_search)
        hxi, hyi, hzi = bi.size / 2

        for j in indices:
            if j <= i:
                continue
            key = (i, j)
            if key in seen:
                continue
            bj = bricks[j]
            hxj, hyj, hzj = bj.size / 2
            d = bj.center - bi.center

            # --- bed joint (y direction) ---
            expected_dy = hyi + hyj + cfg.mortar_bed
            if abs(abs(d[1]) - expected_dy) < tol:
                ov_x = _overlap_1d(bi.center[0]-hxi, bi.center[0]+hxi,
                                   bj.center[0]-hxj, bj.center[0]+hxj)
                ov_z = _overlap_1d(bi.center[2]-hzi, bi.center[2]+hzi,
                                   bj.center[2]-hzj, bj.center[2]+hzj)
                if ov_x > tol and ov_z > tol:
                    sign = np.sign(d[1])
                    # Quad at midplane of the mortar joint
                    mid_y  = (bi.center[1] + hyi + bj.center[1] - hyj) / 2
                    x_lo   = max(bi.center[0]-hxi, bj.center[0]-hxj)
                    x_hi   = min(bi.center[0]+hxi, bj.center[0]+hxj)
                    z_lo   = max(bi.center[2]-hzi, bj.center[2]-hzj)
                    z_hi   = min(bi.center[2]+hzi, bj.center[2]+hzj)
                    quad   = np.array([[x_lo, mid_y, z_lo], [x_hi, mid_y, z_lo],
                                       [x_hi, mid_y, z_hi], [x_lo, mid_y, z_hi]])
                    interfaces.append(Interface(
                        bid_a=i, bid_b=j, joint_type='bed',
                        normal=sign * np.array([0., 1., 0.]),
                        area=ov_x * ov_z,
                        center=np.array([(x_lo+x_hi)/2, mid_y, (z_lo+z_hi)/2]),
                        quad=quad,
                    ))
                    seen.add(key)
                    continue

            # --- head joint (x direction) ---
            expected_dx = hxi + hxj + cfg.mortar_head
            if abs(abs(d[0]) - expected_dx) < tol:
                ov_y = _overlap_1d(bi.center[1]-hyi, bi.center[1]+hyi,
                                   bj.center[1]-hyj, bj.center[1]+hyj)
                ov_z = _overlap_1d(bi.center[2]-hzi, bi.center[2]+hzi,
                                   bj.center[2]-hzj, bj.center[2]+hzj)
                if ov_y > tol and ov_z > tol:
                    sign   = np.sign(d[0])
                    mid_x  = (bi.center[0] + hxi + bj.center[0] - hxj) / 2
                    y_lo   = max(bi.center[1]-hyi, bj.center[1]-hyj)
                    y_hi   = min(bi.center[1]+hyi, bj.center[1]+hyj)
                    z_lo   = max(bi.center[2]-hzi, bj.center[2]-hzj)
                    z_hi   = min(bi.center[2]+hzi, bj.center[2]+hzj)
                    quad   = np.array([[mid_x, y_lo, z_lo], [mid_x, y_hi, z_lo],
                                       [mid_x, y_hi, z_hi], [mid_x, y_lo, z_hi]])
                    interfaces.append(Interface(
                        bid_a=i, bid_b=j, joint_type='head',
                        normal=sign * np.array([1., 0., 0.]),
                        area=ov_y * ov_z,
                        center=np.array([mid_x, (y_lo+y_hi)/2, (z_lo+z_hi)/2]),
                        quad=quad,
                    ))
                    seen.add(key)
                    continue

            # --- perp joint (z direction, multi-wythe) ---
            if cfg.n_wythes > 1:
                expected_dz = hzi + hzj + cfg.mortar_perp
                if abs(abs(d[2]) - expected_dz) < tol:
                    ov_x = _overlap_1d(bi.center[0]-hxi, bi.center[0]+hxi,
                                       bj.center[0]-hxj, bj.center[0]+hxj)
                    ov_y = _overlap_1d(bi.center[1]-hyi, bi.center[1]+hyi,
                                       bj.center[1]-hyj, bj.center[1]+hyj)
                    if ov_x > tol and ov_y > tol:
                        sign  = np.sign(d[2])
                        mid_z = (bi.center[2] + hzi + bj.center[2] - hzj) / 2
                        x_lo  = max(bi.center[0]-hxi, bj.center[0]-hxj)
                        x_hi  = min(bi.center[0]+hxi, bj.center[0]+hxj)
                        y_lo  = max(bi.center[1]-hyi, bj.center[1]-hyj)
                        y_hi  = min(bi.center[1]+hyi, bj.center[1]+hyj)
                        quad  = np.array([[x_lo, y_lo, mid_z], [x_hi, y_lo, mid_z],
                                          [x_hi, y_hi, mid_z], [x_lo, y_hi, mid_z]])
                        interfaces.append(Interface(
                            bid_a=i, bid_b=j, joint_type='perp',
                            normal=sign * np.array([0., 0., 1.]),
                            area=ov_x * ov_y,
                            center=np.array([(x_lo+x_hi)/2, (y_lo+y_hi)/2, mid_z]),
                            quad=quad,
                        ))
                        seen.add(key)

    return interfaces


# ---------------------------------------------------------------------------
# Interface detection — AABB expansion method
# ---------------------------------------------------------------------------

def find_interfaces_aabb(bricks: List[Brick], cfg: WallConfig) -> List[Interface]:
    """
    Find interfaces by expanding each brick's AABB by half the relevant mortar
    thickness in each direction, then detecting pairwise AABB overlaps that
    span exactly a face (one axis touching, two axes overlapping).

    Uses a KDTree on brick centres to restrict the candidate pairs to O(1)
    neighbours before the AABB check — giving overall O(n log n) complexity.
    """
    centers = np.array([b.center for b in bricks])
    max_search = (max(cfg.brick_length, cfg.brick_height, cfg.brick_depth) +
                  max(cfg.mortar_bed, cfg.mortar_head, cfg.mortar_perp) + 1e-3)
    tree = KDTree(centers)

    mortars = np.array([cfg.mortar_head, cfg.mortar_bed, cfg.mortar_perp])
    joint_names = ['head', 'bed', 'perp']
    normals_pos = [np.array([1.,0.,0.]), np.array([0.,1.,0.]), np.array([0.,0.,1.])]
    tol = 1e-6

    interfaces: List[Interface] = []
    seen: set = set()

    for i, bi in enumerate(bricks):
        indices = tree.query_ball_point(bi.center, max_search)
        lo_i = bi.center - bi.size / 2
        hi_i = bi.center + bi.size / 2

        for j in indices:
            if j <= i:
                continue
            key = (i, j)
            if key in seen:
                continue
            bj = bricks[j]
            lo_j = bj.center - bj.size / 2
            hi_j = bj.center + bj.size / 2

            # Expand each AABB by half the mortar + epsilon on ALL sides.
            # The epsilon ensures that bricks separated by exactly mortar_thickness
            # produce a STRICTLY POSITIVE overlap (they would otherwise touch at zero).
            eps_exp = 1e-8
            lo_i_exp = lo_i - (mortars / 2 + eps_exp)
            hi_i_exp = hi_i + (mortars / 2 + eps_exp)
            lo_j_exp = lo_j - (mortars / 2 + eps_exp)
            hi_j_exp = hi_j + (mortars / 2 + eps_exp)

            # Pre-filter: skip pairs whose expanded AABBs do not overlap at all.
            overlap = np.minimum(hi_i_exp, hi_j_exp) - np.maximum(lo_i_exp, lo_j_exp)
            if np.any(overlap <= 0):
                continue

            # Identify the interface axis: the axis where the un-expanded AABBs
            # are separated by exactly mortar_thickness (expanded overlap ≈ mortar).
            gap = np.maximum(lo_j - hi_i, lo_i - hi_j)  # positive = gap, negative = penetration
            for axis in range(3):
                if cfg.n_wythes == 1 and axis == 2:
                    continue   # no through-thickness joints for single wythe
                expected_gap = mortars[axis]
                if abs(gap[axis] - expected_gap) < tol:
                    # The other two axes must overlap
                    other = [a for a in range(3) if a != axis]
                    ov = [_overlap_1d(lo_i[a], hi_i[a], lo_j[a], hi_j[a]) for a in other]
                    if ov[0] > tol and ov[1] > tol:
                        sign = 1.0 if bj.center[axis] > bi.center[axis] else -1.0
                        # Midplane coordinates
                        mid_coord = (hi_i[axis] + lo_j[axis]) / 2 if sign > 0 else \
                                    (hi_j[axis] + lo_i[axis]) / 2
                        a0, a1 = other
                        lo_q0 = max(lo_i[a0], lo_j[a0]);  hi_q0 = min(hi_i[a0], hi_j[a0])
                        lo_q1 = max(lo_i[a1], lo_j[a1]);  hi_q1 = min(hi_i[a1], hi_j[a1])
                        cen = np.zeros(3)
                        cen[axis] = mid_coord
                        cen[a0]   = (lo_q0 + hi_q0) / 2
                        cen[a1]   = (lo_q1 + hi_q1) / 2
                        q = np.zeros((4, 3))
                        q[:, axis] = mid_coord
                        q[0, a0] = lo_q0;  q[0, a1] = lo_q1
                        q[1, a0] = hi_q0;  q[1, a1] = lo_q1
                        q[2, a0] = hi_q0;  q[2, a1] = hi_q1
                        q[3, a0] = lo_q0;  q[3, a1] = hi_q1
                        interfaces.append(Interface(
                            bid_a=i, bid_b=j,
                            joint_type=joint_names[axis],
                            normal=sign * normals_pos[axis],
                            area=ov[0] * ov[1],
                            center=cen,
                            quad=q,
                        ))
                        seen.add(key)
                        break

    return interfaces


# ---------------------------------------------------------------------------
# Cross-check: compare two interface lists
# ---------------------------------------------------------------------------

def _iface_key(ifc: Interface) -> Tuple:
    return (ifc.bid_a, ifc.bid_b, ifc.joint_type)


def cross_check_interfaces(a: List[Interface], b: List[Interface],
                           label_a='kdtree', label_b='aabb') -> List[Interface]:
    ka = {_iface_key(i) for i in a}
    kb = {_iface_key(i) for i in b}
    only_a = ka - kb
    only_b = kb - ka
    if only_a or only_b:
        print(f"[warn] interface mismatch: {len(only_a)} only in {label_a}, "
              f"{len(only_b)} only in {label_b}")
    else:
        print(f"[mesh] {label_a} and {label_b} agree: {len(a)} interfaces")
    return a  # return the first list (they agree)


# ---------------------------------------------------------------------------
# VTK .vtu writer  (ASCII, VTK XML unstructured grid)
# ---------------------------------------------------------------------------

def _vtk_header(f, n_points: int, n_cells: int):
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="UnstructuredGrid" version="0.1" '
            'byte_order="LittleEndian">\n')
    f.write('  <UnstructuredGrid>\n')
    f.write(f'    <Piece NumberOfPoints="{n_points}" '
            f'NumberOfCells="{n_cells}">\n')


def _vtk_footer(f):
    f.write('    </Piece>\n')
    f.write('  </UnstructuredGrid>\n')
    f.write('</VTKFile>\n')


def _vtk_points(f, pts: np.ndarray):
    f.write('      <Points>\n')
    f.write('        <DataArray type="Float64" NumberOfComponents="3" '
            'format="ascii">\n')
    for p in pts:
        f.write(f'          {p[0]:.9e} {p[1]:.9e} {p[2]:.9e}\n')
    f.write('        </DataArray>\n')
    f.write('      </Points>\n')


def _vtk_cells(f, conn: List[List[int]], types: List[int]):
    f.write('      <Cells>\n')
    f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
    for c in conn:
        f.write('          ' + ' '.join(map(str, c)) + '\n')
    f.write('        </DataArray>\n')
    f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
    off = 0
    for c in conn:
        off += len(c)
        f.write(f'          {off}\n')
    f.write('        </DataArray>\n')
    f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
    for t in types:
        f.write(f'          {t}\n')
    f.write('        </DataArray>\n')
    f.write('      </Cells>\n')


def _vtk_cell_data(f, arrays: Dict[str, list]):
    if not arrays:
        return
    f.write('      <CellData>\n')
    for name, vals in arrays.items():
        dtype = 'Int32' if isinstance(vals[0], (int, np.integer)) else 'Float64'
        f.write(f'        <DataArray type="{dtype}" Name="{name}" '
                f'format="ascii">\n')
        for v in vals:
            f.write(f'          {v}\n')
        f.write('        </DataArray>\n')
    f.write('      </CellData>\n')


# ---------------------------------------------------------------------------
# Write brick mesh  (nominal size — used by explicit-mortar style)
# ---------------------------------------------------------------------------

def write_bricks_vtu(path: str, bricks: List[Brick]):
    """
    Each brick → one VTK_HEXAHEDRON (type 12) cell.
    Nodes are NOT shared between bricks (duplicate nodes at each brick corner)
    so that each brick is visually independent and can carry independent data.
    Shared-node meshes are the FEM convention; this is the visualisation mesh.
    """
    VTK_HEX = 12
    pts: List[np.ndarray] = []
    conn: List[List[int]] = []
    cell_course = []
    cell_wythe  = []
    cell_bid    = []

    for b in bricks:
        base = len(pts)
        pts.extend(b.corners())
        conn.append(list(range(base, base + 8)))
        cell_course.append(b.course)
        cell_wythe.append(b.wythe)
        cell_bid.append(b.bid)

    with open(path, 'w') as f:
        _vtk_header(f, len(pts), len(bricks))
        _vtk_points(f, pts)
        _vtk_cells(f, conn, [VTK_HEX] * len(bricks))
        _vtk_cell_data(f, {
            'brick_id': cell_bid,
            'course':   cell_course,
            'wythe':    cell_wythe,
        })
        _vtk_footer(f)

    print(f"[vtk] wrote {len(bricks)} brick hex cells → {path}")


# ---------------------------------------------------------------------------
# Write expanded brick mesh (zero-thickness interface style — Lourenco 1997)
# ---------------------------------------------------------------------------

def write_expanded_bricks_vtu(path: str, bricks: List[Brick], cfg: WallConfig):
    """
    Zero-thickness (simplified micro-modelling) style.
    Each brick is expanded by mortar/2 on each face so adjacent bricks share
    a common face with zero gap — the mortar thickness is absorbed into the
    brick.  Interface elements (zero-thickness quads) sit at those shared faces.
    """
    VTK_HEX = 12
    pts:  List[np.ndarray] = []
    conn: List[List[int]]  = []
    cell_course, cell_wythe, cell_bid = [], [], []

    for b in bricks:
        cx, cy, cz = b.center
        lx, ly, lz = b.size
        # Expand by mortar/2; clip to wall limits so boundary bricks don't stick out.
        x0 = max(0.0,             cx - (lx + cfg.mortar_head) / 2)
        x1 = min(cfg.wall_width,  cx + (lx + cfg.mortar_head) / 2)
        y0 = max(0.0,             cy - (ly + cfg.mortar_bed)  / 2)
        y1 = min(cfg.wall_height, cy + (ly + cfg.mortar_bed)  / 2)
        z0 = cz - lz / 2   # no z-expansion for single wythe
        z1 = cz + lz / 2
        base = len(pts)
        pts.extend([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ])
        conn.append(list(range(base, base + 8)))
        cell_course.append(b.course)
        cell_wythe.append(b.wythe)
        cell_bid.append(b.bid)

    with open(path, 'w') as f:
        _vtk_header(f, len(pts), len(bricks))
        _vtk_points(f, pts)
        _vtk_cells(f, conn, [VTK_HEX] * len(bricks))
        _vtk_cell_data(f, {
            'brick_id': cell_bid,
            'course':   cell_course,
            'wythe':    cell_wythe,
        })
        _vtk_footer(f)

    print(f"[vtk] wrote {len(bricks)} expanded brick hex cells → {path}")


# ---------------------------------------------------------------------------
# Write mortar hex mesh (explicit-mortar style)
# ---------------------------------------------------------------------------

def write_mortar_vtu(path: str, interfaces: List[Interface], cfg: WallConfig):
    """
    Explicit-mortar style.
    Each interface midplane quad is extruded by mortar_thickness/2 on each side
    (in the joint-normal direction) to produce a physical mortar hex element.
    Bricks remain at nominal size; mortar hexes fill the gaps.
    """
    VTK_HEX = 12
    mortar_t = {'bed': cfg.mortar_bed, 'head': cfg.mortar_head, 'perp': cfg.mortar_perp}
    type_map  = {'bed': 0, 'head': 1, 'perp': 2}

    pts:  List[np.ndarray] = []
    conn: List[List[int]]  = []
    cell_type, cell_area   = [], []

    for ifc in interfaces:
        half_t = mortar_t[ifc.joint_type] / 2
        n      = ifc.normal               # unit normal from brick_a -> brick_b
        q      = ifc.quad                 # (4,3) midplane corners
        lower  = q - half_t * n           # face toward brick_a
        upper  = q + half_t * n           # face toward brick_b
        base   = len(pts)
        # VTK hex: nodes 0-3 on one face, 4-7 on the other
        pts.extend(lower.tolist() + upper.tolist())
        conn.append(list(range(base, base + 8)))
        cell_type.append(type_map[ifc.joint_type])
        cell_area.append(float(ifc.area))

    with open(path, 'w') as f:
        _vtk_header(f, len(pts), len(interfaces))
        _vtk_points(f, pts)
        _vtk_cells(f, conn, [VTK_HEX] * len(interfaces))
        _vtk_cell_data(f, {
            'joint_type': cell_type,   # 0=bed, 1=head, 2=perp
            'area_m2':    cell_area,
        })
        _vtk_footer(f)

    n_bed  = sum(1 for i in interfaces if i.joint_type == 'bed')
    n_head = sum(1 for i in interfaces if i.joint_type == 'head')
    n_perp = sum(1 for i in interfaces if i.joint_type == 'perp')
    print(f"[vtk] wrote {len(interfaces)} mortar hex cells "
          f"(bed={n_bed}, head={n_head}, perp={n_perp}) → {path}")


# ---------------------------------------------------------------------------
# Write interface mesh (zero-thickness quads)
# ---------------------------------------------------------------------------

def write_interfaces_vtu(path: str, interfaces: List[Interface]):
    """
    Each interface → one VTK_QUAD (type 9) cell at the mortar-joint midplane.
    Cell data: joint_type_id (0=bed, 1=head, 2=perp), area [m²].
    """
    VTK_QUAD = 9
    type_map  = {'bed': 0, 'head': 1, 'perp': 2}
    pts: List[np.ndarray] = []
    conn: List[List[int]] = []
    cell_type = []
    cell_area = []
    cell_bid_a = []
    cell_bid_b = []

    for ifc in interfaces:
        base = len(pts)
        pts.extend(ifc.quad)
        conn.append(list(range(base, base + 4)))
        cell_type.append(type_map[ifc.joint_type])
        cell_area.append(float(ifc.area))
        cell_bid_a.append(ifc.bid_a)
        cell_bid_b.append(ifc.bid_b)

    with open(path, 'w') as f:
        _vtk_header(f, len(pts), len(interfaces))
        _vtk_points(f, pts)
        _vtk_cells(f, conn, [VTK_QUAD] * len(interfaces))
        _vtk_cell_data(f, {
            'joint_type': cell_type,   # 0=bed, 1=head, 2=perp
            'area_m2':    cell_area,
            'brick_a':    cell_bid_a,
            'brick_b':    cell_bid_b,
        })
        _vtk_footer(f)

    n_bed  = sum(1 for i in interfaces if i.joint_type == 'bed')
    n_head = sum(1 for i in interfaces if i.joint_type == 'head')
    n_perp = sum(1 for i in interfaces if i.joint_type == 'perp')
    print(f"[vtk] wrote {len(interfaces)} interface quads "
          f"(bed={n_bed}, head={n_head}, perp={n_perp}) → {path}")


# ---------------------------------------------------------------------------
# JSON export  (FEM connectivity)
# ---------------------------------------------------------------------------

def export_json(path: str,
                bricks: List[Brick],
                interfaces: List[Interface],
                cfg: WallConfig):
    """
    Export full mesh connectivity for use by FEM codes.

    Brick nodes use the EXPANDED brick convention (Lourenco 1997):
      each brick is expanded by mortar/2 in every direction so that adjacent
      bricks are face-to-face with zero gap. The interface elements are then
      zero-thickness at the shared face.

    Node numbering: each brick owns 8 independent node slots (8*n_bricks total).
    A companion 'shared_nodes' list identifies which node slots are geometrically
    coincident and should be connected by interface elements.
    """
    mortar_exp = np.array([cfg.mortar_head / 2,
                           cfg.mortar_bed  / 2,
                           cfg.mortar_perp / 2])

    all_nodes: List[List[float]] = []
    brick_conn: List[List[int]]  = []

    for b in bricks:
        expanded_size = b.size + 2 * mortar_exp
        cx, cy, cz = b.center
        hx, hy, hz = expanded_size / 2
        nodes = [
            [cx-hx, cy-hy, cz-hz], [cx+hx, cy-hy, cz-hz],
            [cx+hx, cy+hy, cz-hz], [cx-hx, cy+hy, cz-hz],
            [cx-hx, cy-hy, cz+hz], [cx+hx, cy-hy, cz+hz],
            [cx+hx, cy+hy, cz+hz], [cx-hx, cy+hy, cz+hz],
        ]
        base = len(all_nodes)
        all_nodes.extend(nodes)
        brick_conn.append(list(range(base, base + 8)))

    # Face node indices within a brick (local 0-7) for each face direction
    FACE_NODES = {
        'y-': [0, 1, 5, 4],   # bottom
        'y+': [3, 2, 6, 7],   # top
        'x-': [0, 3, 7, 4],   # left
        'x+': [1, 2, 6, 5],   # right
        'z-': [0, 1, 2, 3],   # front
        'z+': [4, 5, 6, 7],   # back
    }
    # Which face pair connects a -> b for each joint type
    JOINT_FACES = {
        'bed':  ('y+', 'y-'),
        'head': ('x+', 'x-'),
        'perp': ('z+', 'z-'),
    }

    iface_data: List[dict] = []
    for ifc in interfaces:
        base_a = ifc.bid_a * 8
        base_b = ifc.bid_b * 8
        # If b is in the + direction from a, a's + face connects to b's - face
        d = interfaces[interfaces.index(ifc)].normal
        jt = ifc.joint_type
        face_pos, face_neg = JOINT_FACES[jt]
        # Check which brick is the + side
        if np.dot(d, bricks[ifc.bid_b].center - bricks[ifc.bid_a].center) > 0:
            nodes_a = [base_a + n for n in FACE_NODES[face_pos]]
            nodes_b = [base_b + n for n in FACE_NODES[face_neg]]
        else:
            nodes_a = [base_a + n for n in FACE_NODES[face_neg]]
            nodes_b = [base_b + n for n in FACE_NODES[face_pos]]

        iface_data.append({
            'brick_a':   ifc.bid_a,
            'brick_b':   ifc.bid_b,
            'joint_type': ifc.joint_type,
            'normal':    ifc.normal.tolist(),
            'area':      float(ifc.area),
            'center':    ifc.center.tolist(),
            'nodes_a':   nodes_a,   # node IDs on brick a's face
            'nodes_b':   nodes_b,   # node IDs on brick b's face (coincident with a at t=0)
        })

    out = {
        'config': asdict(cfg),
        'summary': {
            'n_bricks':      len(bricks),
            'n_interfaces':  len(interfaces),
            'n_nodes':       len(all_nodes),
        },
        'nodes':       all_nodes,      # list of [x, y, z]
        'brick_nodes': brick_conn,     # list of 8-node lists per brick
        'interfaces':  iface_data,
    }

    with open(path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"[json] wrote {len(bricks)} bricks, {len(all_nodes)} nodes, "
          f"{len(interfaces)} interfaces → {path}")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def generate_wall(cfg: WallConfig, prefix: str = 'wall',
                  style: str = 'both') -> None:
    """
    style choices
    -------------
    'zero-thickness'  : expanded bricks + zero-thickness quad interfaces
                        (simplified micro-modelling, Lourenco & Rots 1997)
    'explicit-mortar' : nominal bricks + physical mortar hex elements
    'both'            : write all four VTU files
    """
    print(f"\n{'='*60}")
    print(f" URM wall mesh generator  [style: {style}]")
    print(f" Wall: {cfg.wall_width:.3f} m wide × {cfg.wall_height:.3f} m tall")
    print(f" Brick: {cfg.brick_length*1e3:.0f}×{cfg.brick_height*1e3:.0f}×"
          f"{cfg.brick_depth*1e3:.0f} mm,  mortar bed/head/perp: "
          f"{cfg.mortar_bed*1e3:.0f}/{cfg.mortar_head*1e3:.0f}/"
          f"{cfg.mortar_perp*1e3:.0f} mm")
    print(f" Bond: {cfg.bond},  {cfg.n_wythes} wythe(s),  "
          f"{len(cfg.openings)} opening(s)")
    print(f"{'='*60}\n")

    bricks = place_bricks(cfg)

    # Interface detection
    if cfg.interface_method in ('kdtree', 'both'):
        ifaces_kd = find_interfaces_kdtree(bricks, cfg)
        print(f"[kdtree] found {len(ifaces_kd)} interfaces")

    if cfg.interface_method in ('aabb', 'both'):
        ifaces_aabb = find_interfaces_aabb(bricks, cfg)
        print(f"[aabb]   found {len(ifaces_aabb)} interfaces")

    if cfg.interface_method == 'both':
        interfaces = cross_check_interfaces(ifaces_kd, ifaces_aabb)
    elif cfg.interface_method == 'kdtree':
        interfaces = ifaces_kd
    else:
        interfaces = ifaces_aabb

    # ---- zero-thickness style ----
    if style in ('zero-thickness', 'both'):
        write_expanded_bricks_vtu(f'{prefix}_zt_bricks.vtu',     bricks, cfg)
        write_interfaces_vtu(     f'{prefix}_zt_interfaces.vtu', interfaces)

    # ---- explicit-mortar style ----
    if style in ('explicit-mortar', 'both'):
        write_bricks_vtu(f'{prefix}_em_bricks.vtu', bricks)
        write_mortar_vtu(f'{prefix}_em_mortar.vtu', interfaces, cfg)

    export_json(f'{prefix}_mesh.json', bricks, interfaces, cfg)

    files = []
    if style in ('zero-thickness', 'both'):
        files += [f'{prefix}_zt_bricks.vtu', f'{prefix}_zt_interfaces.vtu']
    if style in ('explicit-mortar', 'both'):
        files += [f'{prefix}_em_bricks.vtu', f'{prefix}_em_mortar.vtu']
    files.append(f'{prefix}_mesh.json')
    print(f"\n[done] outputs: {', '.join(files)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', metavar='JSON',
                        help='Load WallConfig from JSON file')
    parser.add_argument('--prefix', default='wall',
                        help='Output file prefix (default: wall)')
    parser.add_argument('--bond',   choices=['running', 'stack'],
                        help='Override bond type')
    parser.add_argument('--width',  type=float, help='Wall width  [m]')
    parser.add_argument('--height', type=float, help='Wall height [m]')
    parser.add_argument('--wythes', type=int,   help='Number of wythes')
    parser.add_argument('--style',
                        choices=['zero-thickness', 'explicit-mortar', 'both'],
                        default='both',
                        help='Mesh style to generate (default: both)')
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            d = json.load(f)
        cfg = WallConfig(**d)
    else:
        cfg = WallConfig()

    if args.bond:   cfg.bond        = args.bond
    if args.width:  cfg.wall_width  = args.width
    if args.height: cfg.wall_height = args.height
    if args.wythes: cfg.n_wythes    = args.wythes

    generate_wall(cfg, prefix=args.prefix, style=args.style)


if __name__ == '__main__':
    main()
