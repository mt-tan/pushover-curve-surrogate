"""TBEC 2018 horizontal elastic design spectrum (Section 2.3.4) for the five soil classes.

Two constructions are provided:

* ``site_parameters(soil, Ss, S1)``: the general rule used by the notebook (cell 8) and by the
  reference tool: FS and F1 interpolated in Tables 2.1 and 2.2 at the map coefficients,
  SDS = FS Ss, SD1 = F1 S1, TB = SD1 / SDS, TA = 0.2 TB, TL = 6 s. With the Nurdagi DD-2
  coefficients (Ss = 1.159, S1 = 0.307) this is the spectrum of the derived-delta_t baseline
  (table_derived_delta_t_baseline.csv). Any other Ss / S1 pair gives the site-transfer spectrum.
* ``GENERATION_SITE``: the 3-decimal constants of modules/config.py::SOIL_CONFIG_NURDAGI that the
  OpenSees generation chain used for the ground-truth delta_t (same rule, rounded values).

Sae(T) is in g, Sde(T) = Sae g (T / 2 pi)^2 in m.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

SS_NURDAGI = 1.159          # short-period map spectral acceleration coefficient (AFAD, DD-2)
S1_NURDAGI = 0.307          # 1.0 s map spectral acceleration coefficient
TL_TBDY = 6.0               # transition period (Section 2.3.4)
G_CONST = 9.81
SITE_NAME_DEFAULT = 'Gaziantep Nurdagi, DD-2 (475-year)'

SOIL_CLASSES = ['ZA', 'ZB', 'ZC', 'ZD', 'ZE']
SOIL_TO_INT = {'ZA': 0, 'ZB': 1, 'ZC': 2, 'ZD': 3, 'ZE': 4}
INT_TO_SOIL = {v: k for k, v in SOIL_TO_INT.items()}

# Table 2.1 (FS at Ss breakpoints) and Table 2.2 (F1 at S1 breakpoints)
FS_TABLE_SS = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50])
FS_TABLE = {'ZA': np.array([0.8] * 6), 'ZB': np.array([0.9] * 6),
            'ZC': np.array([1.3, 1.3, 1.2, 1.2, 1.2, 1.2]),
            'ZD': np.array([1.6, 1.4, 1.2, 1.1, 1.0, 1.0]),
            'ZE': np.array([2.4, 1.7, 1.3, 1.1, 0.9, 0.8])}
F1_TABLE_S1 = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
F1_TABLE = {'ZA': np.array([0.8] * 6), 'ZB': np.array([0.8] * 6),
            'ZC': np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.4]),
            'ZD': np.array([2.4, 2.2, 2.0, 1.9, 1.8, 1.7]),
            'ZE': np.array([4.2, 3.3, 2.8, 2.4, 2.2, 2.0])}

# modules/config.py::SOIL_CONFIG_NURDAGI (generation chain, rounded to 3 decimals)
GENERATION_SITE: Dict[str, Dict[str, float]] = {
    'ZA': {'FS': 0.800, 'F1': 0.800, 'SDS': 0.927, 'SD1': 0.246, 'TA': 0.053, 'TB': 0.265},
    'ZB': {'FS': 0.900, 'F1': 0.800, 'SDS': 1.043, 'SD1': 0.246, 'TA': 0.047, 'TB': 0.236},
    'ZC': {'FS': 1.200, 'F1': 1.500, 'SDS': 1.391, 'SD1': 0.461, 'TA': 0.066, 'TB': 0.331},
    'ZD': {'FS': 1.036, 'F1': 1.993, 'SDS': 1.201, 'SD1': 0.612, 'TA': 0.102, 'TB': 0.509},
    'ZE': {'FS': 0.973, 'F1': 2.772, 'SDS': 1.127, 'SD1': 0.851, 'TA': 0.151, 'TB': 0.755},
}


def soil_name(soil) -> str:
    """'ZC', 'zc' or 2 -> 'ZC'."""
    if isinstance(soil, (int, np.integer)) or (isinstance(soil, float) and float(soil).is_integer()):
        return INT_TO_SOIL[int(soil)]
    s = str(soil).strip().upper()
    if s not in SOIL_TO_INT:
        raise ValueError(f'unknown soil class {soil!r}; expected one of {SOIL_CLASSES}')
    return s


def amplification(soil, Ss: float = SS_NURDAGI, S1: float = S1_NURDAGI):
    """(FS, F1) by linear interpolation in Tables 2.1 and 2.2 (flat outside the breakpoints)."""
    s = soil_name(soil)
    FS = float(np.interp(Ss, FS_TABLE_SS, FS_TABLE[s]))
    F1 = float(np.interp(S1, F1_TABLE_S1, F1_TABLE[s]))
    return FS, F1


def site_parameters(soil, Ss: float = SS_NURDAGI, S1: float = S1_NURDAGI,
                    TL: float = TL_TBDY, name: str | None = None) -> Dict[str, float]:
    """Design-spectrum parameters of one site and soil class."""
    s = soil_name(soil)
    FS, F1 = amplification(s, Ss, S1)
    SDS, SD1 = FS * Ss, F1 * S1
    TB = SD1 / SDS
    TA = 0.2 * TB
    is_default = bool(abs(Ss - SS_NURDAGI) < 1e-12 and abs(S1 - S1_NURDAGI) < 1e-12)
    return {'soil': s, 'Ss': float(Ss), 'S1': float(S1), 'FS': FS, 'F1': F1, 'SDS': SDS, 'SD1': SD1,
            'TA': TA, 'TB': TB, 'TL': float(TL), 'is_nurdagi': is_default,
            'name': name or (SITE_NAME_DEFAULT if is_default else f'user site Ss = {Ss:g}, S1 = {S1:g}')}


def generation_site(soil) -> Dict[str, float]:
    """The rounded constants of the generation chain (for reference and audits)."""
    s = soil_name(soil)
    d = dict(GENERATION_SITE[s])
    d.update({'soil': s, 'Ss': SS_NURDAGI, 'S1': S1_NURDAGI, 'TL': TL_TBDY, 'is_nurdagi': True,
              'name': SITE_NAME_DEFAULT + ' (generation constants, 3 decimals)'})
    return d


def Sae(T, site: Dict[str, float]):
    """Elastic spectral acceleration [g] (Eqs. 2.2 to 2.5), scalar or array T."""
    T_arr = np.atleast_1d(np.asarray(T, dtype=float))
    SDS, SD1, TA, TB, TL = site['SDS'], site['SD1'], site['TA'], site['TB'], site['TL']
    out = np.empty_like(T_arr)
    m1 = T_arr <= TA
    m2 = (T_arr > TA) & (T_arr <= TB)
    m3 = (T_arr > TB) & (T_arr <= TL)
    m4 = T_arr > TL
    out[m1] = SDS * (0.4 + 0.6 * T_arr[m1] / TA)
    out[m2] = SDS
    out[m3] = SD1 / T_arr[m3]
    out[m4] = SD1 * TL / (T_arr[m4] * T_arr[m4])
    return out if np.ndim(T) else float(out[0])


def Sde(T, site: Dict[str, float], g: float = G_CONST):
    """Elastic spectral displacement [m]: Sae g (T / 2 pi)^2."""
    T_arr = np.atleast_1d(np.asarray(T, dtype=float))
    out = np.atleast_1d(Sae(T_arr, site)) * g * (T_arr / (2.0 * np.pi)) ** 2
    return out if np.ndim(T) else float(out[0])


def Sae_nurdagi(T, soil):
    """Notebook cell 8 convenience: Sae(T) for the Nurdagi DD-2 spectrum."""
    return Sae(T, site_parameters(soil))


def spectrum_curve(site: Dict[str, float], T_max: float = 4.0, n: int = 600):
    """(T, Sae) samples for plotting, starting at T = 0."""
    T = np.linspace(0.0, T_max, n)
    return T, Sae(T, site)
