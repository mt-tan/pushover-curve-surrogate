"""Live structure preview: elevation with the actual bay lengths, and the column and beam
sections with longitudinal bars, closed stirrups and tie legs (informational only)."""
from __future__ import annotations

from typing import Dict, List

import matplotlib.patches as mp
import numpy as np
from matplotlib.figure import Figure

from .theme import PANEL, TEXT

C_CONCRETE = '#E3E6EA'
C_BEAM = '#9DC3E6'
C_COLUMN = '#B0B7BF'
C_OUTLINE = '#34495E'
C_BAR = '#212121'
C_HOOP = '#C0392B'
C_GROUND = '#5D6D7E'


def _bays(v: Dict) -> List[float]:
    L = v.get('bay_lengths_display') or v.get('bay_lengths')
    if not L:
        L = [float(v.get('L_mean', 5.0))] * int(v['N_bay'])
    return [float(x) for x in L]


def draw_elevation(ax, v: Dict, pt: float = 9.5) -> None:
    ax.clear(); ax.set_facecolor(PANEL)
    N_story = int(v['N_story']); h1 = float(v['h1']); htyp = float(v['htyp'])
    L = _bays(v); x_cols = np.concatenate([[0.0], np.cumsum(L)]); W = float(x_cols[-1])
    Htot = h1 + (N_story - 1) * htyp
    cw = float(v['col_B']); bh = float(v['beam_h'])
    y_floors = [h1 + i * htyp for i in range(N_story)]
    for x in x_cols:
        ax.add_patch(mp.Rectangle((x - cw / 2, 0), cw, Htot, facecolor=C_COLUMN, edgecolor=C_OUTLINE, lw=0.7))
    for y in y_floors:
        ax.add_patch(mp.Rectangle((-cw / 2, y - bh), W + cw, bh, facecolor=C_BEAM, edgecolor=C_OUTLINE, lw=0.7))
    ax.add_patch(mp.Rectangle((-cw, -0.35), W + 2 * cw, 0.35, facecolor='none', edgecolor=C_GROUND,
                              hatch='///', lw=0.6))
    for i, Li in enumerate(L):                      # bay-length ticks
        ax.annotate('', xy=(x_cols[i], -0.9), xytext=(x_cols[i + 1], -0.9),
                    arrowprops=dict(arrowstyle='<->', lw=0.6, color=C_GROUND, shrinkA=0, shrinkB=0))
        ax.text(0.5 * (x_cols[i] + x_cols[i + 1]), -1.2, f'{Li:.2f}', ha='center', va='top',
                fontsize=pt - 2, color=C_GROUND)
    ax.set_xlim(-1.2, W + 1.2); ax.set_ylim(-2.0, Htot + 0.8)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'{N_story}-storey, {len(L)}-bay frame\n$H_{{\\mathrm{{tot}}}}$ = {Htot:.2f} m,  '
                 f'$\\Sigma L_i$ = {W:.2f} m', fontsize=pt, color=TEXT, pad=4)


def _hoop(ax, x0, y0, w, h, phi_w):
    lw = 1.4 if phi_w < 0.009 else 1.9
    ax.add_patch(mp.FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0,rounding_size=0.015',
                                   facecolor='none', edgecolor=C_HOOP, lw=lw))
    return lw


def draw_column_section(ax, v: Dict, pt: float = 9.5) -> None:
    ax.clear(); ax.set_facecolor(PANEL)
    B = float(v['col_B']); H = float(v.get('col_H') or B); c = float(v['cover'])
    phi_w = float(v['col_phi_etr']); phi_l = float(v['phi_long_col_mm']) / 1000.0
    n = int(v['col_n_long']); nx = int(v['col_n_etr_x']); ny = int(v['col_n_etr_y'])
    s = float(v['col_s_etr'])
    ax.add_patch(mp.Rectangle((0, 0), B, H, facecolor=C_CONCRETE, edgecolor=C_OUTLINE, lw=1.0))
    x0, y0 = c + phi_w / 2, c + phi_w / 2
    lw = _hoop(ax, x0, y0, B - 2 * x0, H - 2 * y0, phi_w)
    # longitudinal bars: n per face, corners shared
    lo, hi_x, hi_y = c + phi_w + phi_l / 2, B - c - phi_w - phi_l / 2, H - c - phi_w - phi_l / 2
    xs = np.linspace(lo, hi_x, max(n, 2)); ys = np.linspace(lo, hi_y, max(n, 2))
    pts = {(float(x), float(lo)) for x in xs} | {(float(x), float(hi_y)) for x in xs} \
        | {(float(lo), float(y)) for y in ys} | {(float(hi_x), float(y)) for y in ys}
    for (x, y) in pts:
        ax.add_patch(mp.Circle((x, y), phi_l / 2, facecolor=C_BAR, edgecolor='white', lw=0.3))
    # tie legs: nx legs parallel to x (spaced in y), ny legs parallel to y (spaced in x)
    for y in np.linspace(y0, H - y0, nx)[1:-1]:
        ax.plot([x0, B - x0], [y, y], color=C_HOOP, lw=lw * 0.8, solid_capstyle='round')
    for x in np.linspace(x0, B - x0, ny)[1:-1]:
        ax.plot([x, x], [y0, H - y0], color=C_HOOP, lw=lw * 0.8, solid_capstyle='round')
    ax.set_xlim(-0.08 * B, 1.08 * B); ax.set_ylim(-0.08 * H, 1.08 * H)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'Column {B * 100:.0f} x {H * 100:.0f} cm,  {n} $\\phi${phi_l * 1000:.0f} per face\n'
                 f'$\\phi${phi_w * 1000:.0f}/{s * 1000:.0f} mm,  legs {nx} x {ny}', fontsize=pt, color=TEXT, pad=4)


def draw_beam_section(ax, v: Dict, pt: float = 9.5) -> None:
    ax.clear(); ax.set_facecolor(PANEL)
    b = float(v['beam_b']); h = float(v['beam_h']); c = float(v['cover'])
    phi_w = float(v['beam_phi_etr']); phi_l = float(v['phi_long_beam_mm']) / 1000.0
    nt = int(v['beam_n_long_top']); nb = int(v['beam_n_long_bot']); s = float(v['beam_s_etr'])
    ax.add_patch(mp.Rectangle((0, 0), b, h, facecolor=C_CONCRETE, edgecolor=C_OUTLINE, lw=1.0))
    x0, y0 = c + phi_w / 2, c + phi_w / 2
    _hoop(ax, x0, y0, b - 2 * x0, h - 2 * y0, phi_w)
    lo, hi = c + phi_w + phi_l / 2, b - c - phi_w - phi_l / 2
    for x in np.linspace(lo, hi, max(nt, 2)):
        ax.add_patch(mp.Circle((x, h - c - phi_w - phi_l / 2), phi_l / 2, facecolor=C_BAR, edgecolor='white', lw=0.3))
    for x in np.linspace(lo, hi, max(nb, 2)):
        ax.add_patch(mp.Circle((x, c + phi_w + phi_l / 2), phi_l / 2, facecolor=C_BAR, edgecolor='white', lw=0.3))
    ax.set_xlim(-0.6 * b, 1.6 * b); ax.set_ylim(-0.08 * h, 1.08 * h)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'Beam {b * 100:.0f} x {h * 100:.0f} cm,  {nt} + {nb} $\\phi${phi_l * 1000:.0f}\n'
                 f'$\\phi${phi_w * 1000:.0f}/{s * 1000:.0f} mm', fontsize=pt, color=TEXT, pad=4)


def render_structure(fig: Figure, v: Dict, pt: float = 9.5) -> None:
    fig.clear(); fig.patch.set_facecolor(PANEL)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.0, 1.0], hspace=0.38,
                          left=0.03, right=0.97, top=0.95, bottom=0.03)
    draw_elevation(fig.add_subplot(gs[0]), v, pt)
    draw_column_section(fig.add_subplot(gs[1]), v, pt)
    draw_beam_section(fig.add_subplot(gs[2]), v, pt)
