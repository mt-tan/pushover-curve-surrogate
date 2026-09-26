"""Applicability screen of Section 3.12 of the paper: shear-governed frames and the bar-buckling proxy.
Port of ``shear_screen`` of notebook cell 109 with the same formulas and constants, except the beam
steel area (see below).

Column: TS 500 shear resistance with the axial-compression term, capacity cap 0.22 fcd b d,
demand V_max / (N_bay + 1) (peak base shear per column). Beam: capacity-design demand
Ve = Vg + 1.4 (Mr_top + Mr_bot) / ln versus the same resistance rule. Buckling proxy: s / phi_l > 6
in the column confinement zone. All quantities vectorised; strengths in MPa, lengths in m,
forces in kN.

Beam steel area: the reinforcement ratios rho_beam_top / rho_beam_bot are defined on the gross
section, rho = n Abar / (b h) (dataset generator and core.features), so As = rho b h. Notebook
cell 109 used As = rho b d; ``beam_as_basis='bd'`` reproduces that run
(table_shear_screen_summary.csv), the default 'bh' reproduces the corrected screen of the paper
(tables_R7/r7_c7_shear_screen_summary.csv).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

SHEAR_RATIO_LIMIT = 1.0
BUCKLING_S_OVER_PHI = 6.0


def _f(x):
    return np.asarray(x, dtype=np.float64)


def shear_screen(f: Dict[str, object], Vmax_kN, beam_as_basis: str = 'bh') -> Dict[str, np.ndarray]:
    """``f`` maps feature names (core.features names) to scalars or arrays; ``Vmax_kN`` is the
    actual (dataset audit) or predicted (deployment) peak base shear. ``beam_as_basis``: 'bh'
    (As = rho b h, consistent with the definition of rho; default) or 'bd' (notebook cell 109)."""
    if beam_as_basis not in ('bh', 'bd'):
        raise ValueError("beam_as_basis must be 'bh' or 'bd'")
    fck, fyk = _f(f['fc_MPa']), _f(f['fy_MPa'])
    fctd = 0.35 * np.sqrt(fck) / 1.5; fcd = fck / 1.5; fywd = fyk / 1.15      # MPa
    # ---- column (square section) ----
    bc, hc, c = _f(f['col_B']), _f(f['col_H']), _f(f['cover'])
    phw_c, s_c = _f(f['col_phi_etr']), _f(f['col_s_etr'])
    phl_c = _f(f['phi_long_col_mm']) / 1000.0
    legs_c = _f(f['col_n_etr_x'])
    d_c = hc - c - phw_c - 0.5 * phl_c
    N_kN = (_f(f['G']) + 0.3 * _f(f['Q'])) * _f(f['L_mean']) * _f(f['N_story'])   # interior column, storey 1
    Ac = bc * hc
    Vc_col = 0.8 * 0.65 * fctd * 1e3 * bc * d_c * (1 + 0.07 * N_kN / (Ac * 1e3))   # kN; N/Ac in MPa
    Asw_c = legs_c * np.pi * phw_c ** 2 / 4
    Vw_col = Asw_c * fywd * 1e3 * d_c / s_c                                          # kN
    Vr_col = np.minimum(Vc_col + Vw_col, 0.22 * fcd * 1e3 * bc * d_c)
    V_dem_col = _f(Vmax_kN) / (_f(f['N_bay']) + 1)
    # ---- beam ----
    bb, hb = _f(f['beam_b']), _f(f['beam_h'])
    phw_b, s_b = _f(f['beam_phi_etr']), _f(f['beam_s_etr'])
    phl_b = _f(f['phi_long_beam_mm']) / 1000.0
    d_b = hb - c - phw_b - 0.5 * phl_b
    depth = hb if beam_as_basis == 'bh' else d_b                 # rho is a gross-section ratio
    As_top = _f(f['rho_beam_top']) * bb * depth; As_bot = _f(f['rho_beam_bot']) * bb * depth
    Mr_top = 0.9 * d_b * As_top * fyk * 1e3; Mr_bot = 0.9 * d_b * As_bot * fyk * 1e3   # kNm
    ln_b = np.maximum(_f(f['L_mean']) - bc, 0.5)
    Vg = (_f(f['G']) + 0.3 * _f(f['Q'])) * _f(f['L_mean']) / 2
    Ve_beam = Vg + 1.4 * (Mr_top + Mr_bot) / ln_b
    Vc_beam = 0.8 * 0.65 * fctd * 1e3 * bb * d_b
    Vw_beam = (2 * np.pi * phw_b ** 2 / 4) * fywd * 1e3 * d_b / s_b
    Vr_beam = np.minimum(Vc_beam + Vw_beam, 0.22 * fcd * 1e3 * bb * d_b)
    out = {'V_dem_col_kN': V_dem_col, 'Vr_col_kN': Vr_col, 'col_shear_ratio': V_dem_col / Vr_col,
           'Ve_beam_kN': Ve_beam, 'Vr_beam_kN': Vr_beam, 'beam_shear_ratio': Ve_beam / Vr_beam,
           's_over_phil_col': s_c / phl_c}
    out['col_shear_governed'] = out['col_shear_ratio'] > SHEAR_RATIO_LIMIT
    out['beam_shear_governed'] = out['beam_shear_ratio'] > SHEAR_RATIO_LIMIT
    out['shear_governed'] = out['col_shear_governed'] | out['beam_shear_governed']
    out['buckling_proxy'] = out['s_over_phil_col'] > BUCKLING_S_OVER_PHI
    return out


def screen_shares(f: Dict[str, object], Vmax_kN, col_is_tbdy_confined,
                  beam_as_basis: str = 'bh') -> Dict[str, float]:
    """Shares [%] in the layout of table_shear_screen_summary.csv."""
    sc = shear_screen(f, Vmax_kN, beam_as_basis)
    return {'column shear-governed (V_dem > Vr)': float(sc['col_shear_governed'].mean() * 100),
            'beam shear-governed (Ve > Vr)': float(sc['beam_shear_governed'].mean() * 100),
            'any shear-governed': float(sc['shear_governed'].mean() * 100),
            'bar-buckling proxy (s/phi_l > 6, column)': float(sc['buckling_proxy'].mean() * 100),
            # key mirrors the row label of the read-only table_shear_screen_summary.csv (data identifier)
            'column TBDY-compliant flag': float((_f(col_is_tbdy_confined) > 0.5).mean() * 100)}
