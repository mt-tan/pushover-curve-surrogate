"""Capacity-curve assembly (Sections 3.4 and 3.11 of the paper).

* clip_points: the 1.001 monotonic clip of notebook cell 72 (upward clip of the later point).
* multilinear_curve / multilinear_energy: Option A polyline 0 - (dy, Vy) - (dm, Vm) - (du, 0.8 Vm).
* scale_shape / shape_energy: Option B dense-ordinate profile v(s) on 21 stations scaled by
  the predicted (delta_u, Vmax); v(0) = 0 and v(1) = 0.8 are fixed.
* demand_ratios: theta_t = delta_t / H_tot, DCR_delta = delta_t / delta_u (clipped),
  mu_d = delta_t / delta_y (clipped), as in Table 13 of the paper.
* assemble: everything the UI draws for one frame.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

V_END_RATIO = 0.8          # v(1) = 0.8 by the definition of delta_u
CLIP_FACTOR = 1.001        # Section 3.4 upward clip of the later point
EPS_POS = 1e-6
N_ST = 21
S_ST = np.linspace(0.0, 1.0, N_ST)


def _f(x):
    return np.asarray(x, dtype=np.float64)


def clip_points(delta_y, delta_max, delta_u, Vy, Vmax) -> Dict[str, np.ndarray]:
    """Monotonic clip of the predicted characteristic points (notebook cell 72).

    delta_y := max(delta_y, 1e-6); delta_max := max(delta_max, 1.001 delta_y);
    delta_u := max(delta_u, 1.001 delta_max); Vy := max(Vy, 1e-6); Vmax := max(Vmax, Vy).
    The earlier point is never changed. Flags report which values were clipped.
    """
    dy_raw, dm_raw, du_raw = _f(delta_y), _f(delta_max), _f(delta_u)
    Vy_raw, Vm_raw = _f(Vy), _f(Vmax)
    dy = np.maximum(dy_raw, EPS_POS)
    dm = np.maximum(dm_raw, dy * CLIP_FACTOR)
    du = np.maximum(du_raw, dm * CLIP_FACTOR)
    Vy_c = np.maximum(Vy_raw, EPS_POS)
    Vm_c = np.maximum(Vm_raw, Vy_c)
    return {'delta_y': dy, 'delta_max': dm, 'delta_u': du, 'Vy': Vy_c, 'Vmax': Vm_c,
            'clip_delta_max': dm_raw < dy * CLIP_FACTOR,
            'clip_delta_u': du_raw < dm * CLIP_FACTOR,
            'clip_Vmax': Vm_raw < Vy_c}


def multilinear_energy(delta_y, Vy, delta_max, Vmax, delta_u):
    """Area under the multilinear curve 0 - (dy, Vy) - (dm, Vm) - (du, 0.8 Vm) (cell 72)."""
    dy, Vy_, dm, Vm, du = (_f(v) for v in (delta_y, Vy, delta_max, Vmax, delta_u))
    return 0.5 * dy * Vy_ + 0.5 * (Vy_ + Vm) * (dm - dy) + 0.5 * (Vm + V_END_RATIO * Vm) * (du - dm)


def multilinear_curve(delta_y, Vy, delta_max, Vmax, delta_u):
    """Four-point polyline (x, y) of one frame (Option A)."""
    x = np.array([0.0, float(delta_y), float(delta_max), float(delta_u)])
    y = np.array([0.0, float(Vy), float(Vmax), V_END_RATIO * float(Vmax)])
    return x, y


def curve_value(x_curve, y_curve, delta):
    """V read from a curve at displacement delta (beyond the last point: last ordinate)."""
    return float(np.interp(float(delta), _f(x_curve), _f(y_curve), right=float(y_curve[-1])))


def scale_shape(shape, delta_u, Vmax, s_grid=S_ST):
    """Dense-ordinate profile v(s) (Option B) scaled to the physical axes: (delta, V)."""
    shape = _f(shape)
    return _f(s_grid) * float(delta_u), shape * float(Vmax)


def shape_energy(shape, delta_u, Vmax, s_grid=S_ST):
    """Trapezoidal area of one or many scaled profiles (rows = frames)."""
    shape = np.atleast_2d(_f(shape))
    du = np.atleast_1d(_f(delta_u)); Vm = np.atleast_1d(_f(Vmax))
    trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
    return trapz(shape * Vm[:, None], _f(s_grid)[None, :] * du[:, None], axis=1)


def demand_ratios(delta_t, Htot, delta_u_clipped, delta_y_clipped) -> Dict[str, np.ndarray]:
    """theta_t, DCR_delta and mu_d (Table 13; DCR uses the clipped, curve-consistent delta_u)."""
    dt = _f(delta_t)
    return {'theta_t': dt / _f(Htot),
            'DCR_delta': dt / np.maximum(_f(delta_u_clipped), 1e-9),
            'mu_d': dt / np.maximum(_f(delta_y_clipped), 1e-9)}


def assemble(points: Dict[str, float], delta_t: float, V_at_dt: float,
             shape_station: Optional[np.ndarray] = None, shape_pca: Optional[np.ndarray] = None,
             s_grid=S_ST) -> Dict[str, object]:
    """Curves and markers for one frame from the CLIPPED points.

    points: delta_y, Vy, delta_max, Vmax, delta_u (clipped). The demand point is the regressor
    pair (delta_t, V_at_dt); the polyline reading V_curve(delta_t) is returned only as a
    consistency check (Section 3.13).
    """
    x, y = multilinear_curve(points['delta_y'], points['Vy'], points['delta_max'],
                             points['Vmax'], points['delta_u'])
    out = {'multilinear': (x, y),
           'points': {'yield': (points['delta_y'], points['Vy']),
                      'peak': (points['delta_max'], points['Vmax']),
                      'ultimate': (points['delta_u'], V_END_RATIO * points['Vmax'])},
           'V80': V_END_RATIO * points['Vmax'],
           'demand_point': (float(delta_t), float(V_at_dt)),
           'V_curve_at_dt': curve_value(x, y, delta_t),
           'energy_multilinear': float(multilinear_energy(points['delta_y'], points['Vy'],
                                                          points['delta_max'], points['Vmax'],
                                                          points['delta_u']))}
    out['demand_gap_pct'] = ((float(V_at_dt) - out['V_curve_at_dt'])
                             / max(out['V_curve_at_dt'], 1e-9) * 100)
    if shape_station is not None:
        xs, ys = scale_shape(shape_station, points['delta_u'], points['Vmax'], s_grid)
        out['station'] = (xs, ys)
        out['energy_station'] = float(shape_energy(shape_station, points['delta_u'], points['Vmax'], s_grid)[0])
        out['V_station_at_dt'] = curve_value(xs, ys, delta_t)
    if shape_pca is not None:
        xp, yp = scale_shape(shape_pca, points['delta_u'], points['Vmax'], s_grid)
        out['pca'] = (xp, yp)
        out['energy_pca'] = float(shape_energy(shape_pca, points['delta_u'], points['Vmax'], s_grid)[0])
    return out
