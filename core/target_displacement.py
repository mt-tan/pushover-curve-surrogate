"""Derived target displacement: TBEC 2018 Section 5.6.5 single-mode (N2) procedure on the
predicted characteristic points (Section 3.14 of the paper; notebook cell 113).

Exact port of modules/target_displacement.py as re-implemented in the notebook:
* linear first-mode shape phi_i = H_i / H_N with equal storey masses m = (G + 0.3 Q) W_total / g,
  giving Gamma_1 = L1 / M1 and M* = L1^2 / M1 (the dataset also stores the modal quantities of the
  analyses, kp_Gamma1 and kp_M_star_ton, but they are not model inputs, so this route recomputes
  them from the linear mode shape and the raw design inputs);
* modal capacity curve of the ascending branch (0, 0) - (delta_y, V_y) - (delta_max, V_max):
  d* = delta / Gamma_1, a* = V / M* / g, resampled at 200 points;
* equal-area bilinearisation with the secant stiffness at 0.6 a*_peak (FEMA 356 / N2), fallback
  elastic-perfectly-plastic; Te = 2 pi sqrt(d*_y / (a*_y g));
* Ry = Sae(Te) / a*_y, CR of Eq. 5.16 (CR = 1 for Te >= TB, else max(1, (1 + (Ry - 1) TB / Te) / Ry));
* delta_t = Gamma_1 CR Sde(Te).

The learned delta_t regressor is valid for the Nurdagi DD-2 spectrum only; this route accepts any
site through ``site`` (core.spectrum.site_parameters) and is labelled "derived" in the UI.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .spectrum import G_CONST, Sae, site_parameters

_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')


def linear_mode_shape(N_story, h1, htyp, G, Q, W_total, g: float = G_CONST) -> Dict[str, float]:
    """Gamma_1 and M* [kN s2/m] of the linear first mode with equal storey masses (cell 113)."""
    H = h1 + np.arange(int(N_story)) * htyp
    phi = H / H[-1]
    m = (G + 0.3 * Q) * W_total / g
    L1 = m * phi.sum(); M1 = m * (phi ** 2).sum()
    return {'Gamma1': L1 / M1, 'M_star': L1 ** 2 / M1, 'M_total': m * len(phi),
            'mass_participation': (L1 ** 2 / M1) / (m * len(phi)), 'phi': phi, 'H': H}


def derive_delta_t(Vy, delta_y, Vmax, delta_max, site: Dict[str, float], Gamma1: float,
                   M_star: float, g: float = G_CONST) -> Dict[str, float]:
    """One frame: delta_t [m] and the intermediate quantities of the N2 procedure."""
    d_star = np.array([0.0, delta_y, delta_max]) / Gamma1
    a_star = np.array([0.0, Vy, Vmax]) / M_star / g
    dd = np.linspace(0, d_star[-1], 200); aa = np.interp(dd, d_star, a_star)
    a_peak = aa.max(); i_06 = int(np.argmin(np.abs(aa - 0.6 * a_peak)))
    if dd[i_06] <= 0:
        i_06 = 1
    Ke = aa[i_06] / dd[i_06]
    A_real = _trapz(aa, dd)
    b = -2 * Ke * dd[-1]; cc = 2 * Ke * A_real; disc = b * b - 4 * cc
    a_y = None
    bilinear = 'equal-area'
    if disc >= 0:
        for cand in ((-b - np.sqrt(disc)) / 2, (-b + np.sqrt(disc)) / 2):
            if 0 < cand <= a_peak * 1.001:
                a_y = min(cand, a_peak); break
    if a_y is None:
        a_y = a_peak; bilinear = 'elastic-perfectly-plastic fallback'
    d_y = a_y / Ke
    Te = 2 * np.pi * np.sqrt(d_y / (a_y * g))
    SDS, SD1, TB = site['SDS'], site['SD1'], site['TB']
    Sae_Te = float(Sae(Te, site)); Sde_Te = Sae_Te * g * Te ** 2 / (4 * np.pi ** 2)
    Ry = Sae_Te / a_y
    CR = 1.0 if Te >= TB else max(1.0, (1 + (Ry - 1) * TB / Te) / Ry)
    delta_t = Gamma1 * CR * Sde_Te
    return {'delta_t': float(delta_t), 'Te': float(Te), 'Sae_Te_g': Sae_Te, 'Sde_Te': float(Sde_Te),
            'Ry': float(Ry), 'CR': float(CR), 'a_y_star_g': float(a_y), 'd_y_star': float(d_y),
            'a_peak_star_g': float(a_peak), 'Ke_star': float(Ke), 'Gamma1': float(Gamma1),
            'M_star': float(M_star), 'SDS': SDS, 'SD1': SD1, 'TA': site['TA'], 'TB': TB,
            'bilinearisation': bilinear, 'site': site['name']}


def derive_delta_t_frame(N_story, h1, htyp, G, Q, W_total, Vy, delta_y, Vmax, delta_max,
                         soil=None, site: Dict[str, float] | None = None) -> Dict[str, float]:
    """Convenience wrapper: modal parameters from the geometry, then the N2 route."""
    if site is None:
        site = site_parameters(soil)
    modal = linear_mode_shape(N_story, h1, htyp, G, Q, W_total)
    out = derive_delta_t(Vy, delta_y, Vmax, delta_max, site, modal['Gamma1'], modal['M_star'])
    out['mass_participation'] = float(modal['mass_participation'])
    return out


def derive_delta_t_batch(df_like, Vy, delta_y, Vmax, delta_max, sites=None, soil_col='soil_ord'):
    """Vectorised over rows (python loop, as in the notebook); returns (delta_t, Te) arrays.

    ``df_like`` must expose N_story, h1, htyp, G, Q, W_total and ``soil_col`` by column name;
    ``sites`` may give a per-row site dict, else the Nurdagi spectrum of each row's soil class.
    """
    n = len(Vy)
    dt = np.zeros(n); Te = np.zeros(n)
    cache = {}
    for i in range(n):
        if sites is not None:
            site = sites[i]
        else:
            key = int(df_like[soil_col].iloc[i])
            site = cache.setdefault(key, site_parameters(key))
        modal = linear_mode_shape(df_like['N_story'].iloc[i], df_like['h1'].iloc[i],
                                  df_like['htyp'].iloc[i], df_like['G'].iloc[i],
                                  df_like['Q'].iloc[i], df_like['W_total'].iloc[i])
        r = derive_delta_t(Vy[i], delta_y[i], Vmax[i], delta_max[i], site,
                           modal['Gamma1'], modal['M_star'])
        dt[i] = r['delta_t']; Te[i] = r['Te']
    return dt, Te
