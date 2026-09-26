"""Derived-feature chain of the TBEC 2018 pushover surrogate.

Pure numpy re-implementation of the generation chain in ``modules/``:

* ``materials.mander_tbdy_confined``   Mander / TBEC Annex 5A confinement (D-67 convention,
  Tan-Atmaca workbook cells T27/U27/V27 for the Eq. 5.4d sum of a_i^2)
* ``transverse_design.sample_column_stirrups`` / ``sample_beam_stirrups``   compliance flags
  (Articles 7.3.4.1 and 7.4.4, Eq. 7.1 minimum A_sh with the 2/3 rule of 7.3.4.1(d))
* ``tbdy_filters.estimate_column_axial_load``   axial-load ratio used by the 2/3 rule
* ``longitudinal_design``   reinforcement ratios from bar diameter and bar count
* ``model_builder.build_2d_frame``   argument mapping and H_tot
* ``lhs_sampling.lhs_to_pilot_params``   raw-parameter to Mander-argument mapping

Every function is vectorised (scalars or equal-length arrays) and uses the same operation
order as the source so that the parity test ``tests/test_feature_parity.py`` reproduces the
stored dataset columns to 1e-9 relative on all clean rows.

Units: m, kN, MPa for strengths (the generation chain works in kPa; only ratios of
strengths enter the confinement formulas, so MPa is used here throughout).
No openseespy import.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

# ----------------------------------------------------------------------------------------
# Constants (modules/config.py, verified against the generation run)
# ----------------------------------------------------------------------------------------
EPS_CO = 0.002                 # unconfined concrete strain at peak
EPS_SU = 0.080                 # bar rupture strain (B420C and B500C, Table 5A.1)
FCE_RATIO = 1.3                # f_ce = 1.3 f_ck (Table 5.1)
FYWE_RATIO = 1.2               # f_ywe = 1.2 f_ywk (Table 5.1)
USE_EXPECTED_STRENGTH_FIBERS = False   # D-49c: Mander curve built with characteristic strengths
EPS_C_GO_BASE = 0.0035
EPS_C_GO_COEF = 0.04
EPS_C_GO_CAP = 0.018           # Eq. 5.4a upper bound
KH_TO_GO_RATIO = 0.75
EPS_C_SH = 0.0025
EPS_S_SH = 0.0075
EPS_S_GO_RATIO = 0.4
LIVE_LOAD_FACTOR = 0.3         # n = 0.3 (BKS 3), W = G + 0.3 Q
AXIAL_MID_COLUMN_FACTOR = 1.3  # tbdy_filters.estimate_column_axial_load
AXIAL_RATIO_2_3_RULE = 0.20    # Article 7.3.4.1(d)
S_MAX_CODE = 0.150             # 150 mm spacing cap (7.3.4.1(a) and 7.4.4)
TOL = 1e-9                     # tolerance used in the generation flags

SOIL_ORDINAL: Dict[str, int] = {'ZA': 0, 'ZB': 1, 'ZC': 2, 'ZD': 3, 'ZE': 4}
SOIL_CLASSES = list(SOIL_ORDINAL.keys())

# ----------------------------------------------------------------------------------------
# Feature order of the regressors (notebook cell 4, INPUT_COLS, D-63 sealed; 45 columns)
# ----------------------------------------------------------------------------------------
FEATURE_ORDER: Sequence[str] = (
    # geometry
    'N_story', 'N_bay', 'L_mean', 'L_max', 'L_min', 'W_total', 'h1', 'htyp', 'Htot',
    'cover', 'col_B', 'col_H', 'beam_b', 'beam_h',
    # longitudinal reinforcement
    'rho_col', 'rho_beam_top', 'rho_beam_bot',
    'phi_long_col_mm', 'phi_long_beam_mm',
    'col_n_long', 'beam_n_long_top', 'beam_n_long_bot',
    # transverse reinforcement, raw descriptors
    'col_phi_etr', 'col_s_etr', 'col_n_etr_x', 'col_n_etr_y',
    'beam_phi_etr', 'beam_s_etr',
    # confinement mechanics (deterministic from the design, inputs of the TBEC 5.8 limits)
    'rho_sh_col', 'rho_sh_beam', 'rho_s_col', 'rho_s_beam',
    'omega_we_col', 'omega_we_beam', 'alpha_se_col', 'alpha_se_beam',
    'fcc_col_MPa', 'fcc_beam_MPa',
    # material, load, site, compliance flags
    'fc_MPa', 'fy_MPa', 'G', 'Q', 'soil_ord',
    'col_is_tbdy_confined', 'beam_is_tbdy_confined',
)
assert len(FEATURE_ORDER) == 45

# Raw (user-facing) layer: Table A1 raw columns. col_H is tied to col_B (square columns).
RAW_INPUTS: Sequence[str] = (
    'N_story', 'N_bay', 'bay_lengths', 'h1', 'htyp',
    'col_B', 'beam_b', 'beam_h', 'cover',
    'phi_long_col_mm', 'col_n_long', 'phi_long_beam_mm', 'beam_n_long_top', 'beam_n_long_bot',
    'col_phi_etr', 'col_s_etr', 'col_n_etr_x', 'col_n_etr_y', 'beam_phi_etr', 'beam_s_etr',
    'fc_MPa', 'fy_MPa', 'G', 'Q', 'soil_class',
)

# Which stored columns are copied from the raw layer and which are derived here.
RAW_FEATURES: Sequence[str] = (
    'N_story', 'N_bay', 'L_mean', 'L_max', 'L_min', 'W_total', 'h1', 'htyp',
    'cover', 'col_B', 'col_H', 'beam_b', 'beam_h',
    'phi_long_col_mm', 'phi_long_beam_mm', 'col_n_long', 'beam_n_long_top', 'beam_n_long_bot',
    'col_phi_etr', 'col_s_etr', 'col_n_etr_x', 'col_n_etr_y', 'beam_phi_etr', 'beam_s_etr',
    'fc_MPa', 'fy_MPa', 'G', 'Q',
)
DERIVED_FEATURES: Sequence[str] = tuple(c for c in FEATURE_ORDER if c not in RAW_FEATURES)
assert len(RAW_FEATURES) + len(DERIVED_FEATURES) == 45

FEATURE_UNITS: Dict[str, str] = {
    'N_story': '-', 'N_bay': '-', 'L_mean': 'm', 'L_max': 'm', 'L_min': 'm', 'W_total': 'm',
    'h1': 'm', 'htyp': 'm', 'Htot': 'm', 'cover': 'm', 'col_B': 'm', 'col_H': 'm',
    'beam_b': 'm', 'beam_h': 'm', 'rho_col': '-', 'rho_beam_top': '-', 'rho_beam_bot': '-',
    'phi_long_col_mm': 'mm', 'phi_long_beam_mm': 'mm', 'col_n_long': '-',
    'beam_n_long_top': '-', 'beam_n_long_bot': '-', 'col_phi_etr': 'm', 'col_s_etr': 'm',
    'col_n_etr_x': '-', 'col_n_etr_y': '-', 'beam_phi_etr': 'm', 'beam_s_etr': 'm',
    'rho_sh_col': '-', 'rho_sh_beam': '-', 'rho_s_col': '-', 'rho_s_beam': '-',
    'omega_we_col': '-', 'omega_we_beam': '-', 'alpha_se_col': '-', 'alpha_se_beam': '-',
    'fcc_col_MPa': 'MPa', 'fcc_beam_MPa': 'MPa', 'fc_MPa': 'MPa', 'fy_MPa': 'MPa',
    'G': 'kN/m', 'Q': 'kN/m', 'soil_ord': '-', 'col_is_tbdy_confined': '0/1',
    'beam_is_tbdy_confined': '0/1',
}
FEATURE_KIND: Dict[str, str] = {c: ('raw' if c in RAW_FEATURES else 'derived') for c in FEATURE_ORDER}
FEATURE_KIND.update({'soil_ord': 'ordinal', 'col_is_tbdy_confined': 'flag',
                     'beam_is_tbdy_confined': 'flag'})
INTEGER_FEATURES = ('N_story', 'N_bay', 'col_n_long', 'beam_n_long_top', 'beam_n_long_bot',
                    'col_n_etr_x', 'col_n_etr_y', 'soil_ord')


def _f(x) -> np.ndarray:
    """float64 array view of a scalar or sequence."""
    return np.asarray(x, dtype=np.float64)


# ----------------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------------
def total_height(h1, htyp, N_story):
    """H_tot = h_1 + (N_story - 1) h_typ (model_builder.build_2d_frame)."""
    return _f(h1) + (_f(N_story) - 1) * _f(htyp)


def bay_statistics(bay_lengths: Iterable[float]) -> Dict[str, float]:
    """L_mean, L_max, L_min and W_total (= sum L_i) of one frame (transverse_design.sample_bay_lengths)."""
    L = _f(list(bay_lengths))
    if L.ndim != 1 or L.size == 0:
        raise ValueError('bay_lengths must be a non-empty 1-D sequence')
    return {'L_mean': float(L.mean()), 'L_max': float(L.max()),
            'L_min': float(L.min()), 'W_total': float(L.sum())}


# ----------------------------------------------------------------------------------------
# Longitudinal reinforcement ratios (longitudinal_design, D-56)
# ----------------------------------------------------------------------------------------
def bar_area(phi_m):
    return np.pi * _f(phi_m) ** 2 / 4.0


def rho_column(col_n_long, phi_long_col_mm, col_B, col_H):
    """rho_col = (4 n_face - 4) A_bar / (B H), n_face = bars per face including corners."""
    Abar = bar_area(_f(phi_long_col_mm) / 1000.0)
    return (4 * _f(col_n_long) - 4) * Abar / (_f(col_B) * _f(col_H))


def rho_beam(n_bars, phi_long_beam_mm, beam_b, beam_h):
    """rho = n A_bar / (b h) for one beam face (top or bottom, single layer)."""
    Abar = bar_area(_f(phi_long_beam_mm) / 1000.0)
    return _f(n_bars) * Abar / (_f(beam_b) * _f(beam_h))


# ----------------------------------------------------------------------------------------
# Mander / TBEC Annex 5A confinement (materials.mander_tbdy_confined, D-67)
# ----------------------------------------------------------------------------------------
def mander_tbdy_confined(B, H, cover, fco_MPa, fyw_MPa,
                         phi_long, n_long_x, n_long_y,
                         phi_etr, s_etr, n_etr_x, n_etr_y,
                         expected: bool = USE_EXPECTED_STRENGTH_FIBERS) -> Dict[str, np.ndarray]:
    """Vectorised copy of materials.mander_tbdy_confined (strengths in MPa, lengths in m).

    Returns a dict of arrays with the same keys as the source (subset used by the app).
    """
    B, H, cover = _f(B), _f(H), _f(cover)
    fco_MPa, fyw_MPa = _f(fco_MPa), _f(fyw_MPa)
    phi_long, phi_etr, s_etr = _f(phi_long), _f(phi_etr), _f(s_etr)
    n_long_x, n_long_y = _f(n_long_x), _f(n_long_y)
    n_etr_x, n_etr_y = _f(n_etr_x), _f(n_etr_y)

    # D-49b/c: model strengths of the Mander curve
    fco_m = fco_MPa * (FCE_RATIO if expected else 1.0)
    fyw_m = fyw_MPa * (FYWE_RATIO if expected else 1.0)

    # stirrup centre-line core
    bo = B - 2 * cover - phi_etr
    ho = H - 2 * cover - phi_etr

    # longitudinal steel and rho_cc
    n_long_total = 2 * n_long_x + 2 * np.maximum(0, n_long_y - 2)
    A_long_each = np.pi * phi_long ** 2 / 4.0
    As_total = n_long_total * A_long_each
    A_core = bo * ho
    rho_cc = As_total / A_core

    # bar centre spacings and the ke sum (Annex 5A, all neighbouring bars)
    aix = np.where(n_long_x >= 2,
                   (bo - phi_etr - phi_long) / np.maximum(n_long_x - 1, 1), 0.0)
    aiy = np.where(n_long_y >= 2,
                   (ho - phi_etr - phi_long) / np.maximum(n_long_y - 1, 1), 0.0)
    Sigma_ai2_ke = (2 * np.maximum(0, n_long_x - 1) * aix ** 2
                    + 2 * np.maximum(0, n_long_y - 1) * aiy ** 2)

    # Eq. 5.4d sum (supported bars only; D-53c/D-67, workbook T27/U27/V27)
    span_x = bo - phi_etr - phi_long
    span_y = ho - phi_etr - phi_long
    Sigma_aix2 = np.where(n_etr_y == 2, span_x ** 2 * 4.0,
                          span_x ** 2 / np.maximum(n_etr_y - 1, 1) * 2.0)
    Sigma_aiy2 = np.where(n_etr_x == 2, span_y ** 2 * 2.0,
                          span_y ** 2 / np.maximum(n_etr_x - 1, 1) * 2.0)
    Sigma_ai2_ase = Sigma_aix2 + Sigma_aiy2

    s_prime = s_etr - phi_etr

    # transverse ratios
    A_etr = np.pi * phi_etr ** 2 / 4.0
    rho_x = (n_etr_x * A_etr) / (s_etr * ho)
    rho_y = (n_etr_y * A_etr) / (s_etr * bo)
    rho_s = rho_x + rho_y

    # effective confinement coefficient ke (full spacing s, D-52a)
    ke_num = (1.0 - Sigma_ai2_ke / (6.0 * bo * ho)) \
        * (1.0 - s_etr / (2.0 * bo)) \
        * (1.0 - s_etr / (2.0 * ho))
    ke = ke_num / (1.0 - rho_cc)
    ke = np.clip(ke, 0.0, 1.0)

    # lateral pressures and Mander strength ratio
    fex = ke * rho_y * fyw_m
    fey = ke * rho_x * fyw_m
    fe = 0.5 * (fex + fey)
    fco_pos = fco_m > 0
    fe_ratio = np.where(fco_pos, fe / np.where(fco_pos, fco_m, 1.0), 0.0)
    lambda_c = np.where(fco_pos,
                        -1.254 + 2.254 * np.sqrt(1.0 + 7.94 * fe_ratio) - 2.0 * fe_ratio,
                        1.0)
    fcc_MPa = lambda_c * fco_m
    eps_cc = EPS_CO * (1.0 + 5.0 * (lambda_c - 1.0))
    Ec_MPa = 5000.0 * np.sqrt(fco_m)
    Esec = np.where(eps_cc > 0, fcc_MPa / np.where(eps_cc > 0, eps_cc, 1.0), 0.0)

    # ultimate strain (D-52b)
    fcc_pos = fcc_MPa > 0
    eps_cu = np.where(fcc_pos,
                      0.0035 + 1.4 * rho_s * fyw_m * EPS_SU / np.where(fcc_pos, fcc_MPa, 1.0),
                      0.005)

    # TBEC Eq. 5.4d alpha_se and Eq. 5.4c omega_we (expected strengths, characteristic inputs)
    alpha_se = (1.0 - Sigma_ai2_ase / (6.0 * bo * ho)) \
        * (1.0 - s_etr / (2.0 * bo)) \
        * (1.0 - s_etr / (2.0 * ho))
    alpha_se = np.clip(alpha_se, 0.0, 1.0)
    fce_MPa = FCE_RATIO * fco_MPa
    fywe_MPa = FYWE_RATIO * fyw_MPa
    rho_sh = np.minimum(rho_x, rho_y)
    omega_we = np.where(fce_MPa > 0,
                        alpha_se * rho_sh * fywe_MPa / np.where(fce_MPa > 0, fce_MPa, 1.0),
                        0.0)

    # TBEC 5.8 strain limits
    eps_c_GO = np.minimum(EPS_C_GO_BASE + EPS_C_GO_COEF * np.sqrt(np.maximum(0.0, omega_we)),
                          EPS_C_GO_CAP)
    eps_c_KH = KH_TO_GO_RATIO * eps_c_GO
    eps_s_GO = EPS_S_GO_RATIO * EPS_SU
    eps_s_KH = KH_TO_GO_RATIO * eps_s_GO

    return {
        'bo_m': bo, 'ho_m': ho, 'A_core_m2': A_core,
        'rho_cc': rho_cc, 'rho_x': rho_x, 'rho_y': rho_y, 'rho_s': rho_s, 'rho_sh': rho_sh,
        's_prime_m': s_prime, 'aix_m': aix, 'aiy_m': aiy,
        'Sigma_ai2_ke_m2': Sigma_ai2_ke, 'Sigma_ai2_ase_m2': Sigma_ai2_ase,
        'ke': ke, 'fex_MPa': fex, 'fey_MPa': fey, 'fe_MPa': fe,
        'lambda_c': lambda_c, 'fcc_MPa': fcc_MPa, 'eps_cc': eps_cc, 'eps_cu': eps_cu,
        'Ec_MPa': Ec_MPa, 'Esec_MPa': Esec,
        'fco_model_MPa': fco_m, 'fyw_model_MPa': fyw_m,
        'fco_char_MPa': fco_MPa, 'fyw_char_MPa': fyw_MPa,
        'alpha_se': alpha_se, 'omega_we': omega_we,
        'fce_MPa': fce_MPa, 'fywe_MPa': fywe_MPa,
        'eps_c_SH': np.full_like(fcc_MPa, EPS_C_SH), 'eps_c_KH': eps_c_KH, 'eps_c_GO': eps_c_GO,
        'eps_s_SH': np.full_like(fcc_MPa, EPS_S_SH),
        'eps_s_KH': np.full_like(fcc_MPa, eps_s_KH), 'eps_s_GO': np.full_like(fcc_MPa, eps_s_GO),
    }


def column_confinement(col_B, col_H, cover, fc_MPa, fy_MPa, phi_long_col_mm, col_n_long,
                       col_phi_etr, col_s_etr, col_n_etr_x, col_n_etr_y):
    """Column Mander call with the generation mapping (model_builder / recompute_confinement_D67b):
    n_long_x = n_long_y = bars per face, f_yw = f_y (D-46)."""
    return mander_tbdy_confined(col_B, col_H, cover, fc_MPa, fy_MPa,
                                _f(phi_long_col_mm) / 1000.0, col_n_long, col_n_long,
                                col_phi_etr, col_s_etr, col_n_etr_x, col_n_etr_y)


def beam_confinement(beam_b, beam_h, cover, fc_MPa, fy_MPa, phi_long_beam_mm,
                     beam_n_long_top, beam_n_long_bot, beam_phi_etr, beam_s_etr):
    """Beam Mander call with the generation mapping: n_long_x = max(n_top, n_bot), n_long_y = 2,
    closed stirrup with 2 x 2 legs (D-45)."""
    n_x = np.maximum(_f(beam_n_long_top), _f(beam_n_long_bot))
    return mander_tbdy_confined(beam_b, beam_h, cover, fc_MPa, fy_MPa,
                                _f(phi_long_beam_mm) / 1000.0, n_x, 2.0,
                                beam_phi_etr, beam_s_etr, 2.0, 2.0)


# ----------------------------------------------------------------------------------------
# TBEC compliance flags (transverse_design, D-54 reporting flags)
# ----------------------------------------------------------------------------------------
def column_axial_ratio_estimate(N_story, N_bay, W_total, G, Q, col_B, col_H, fc_MPa):
    """N_d / (A_c f_ck) of the bottom interior column (tbdy_filters.estimate_column_axial_load)."""
    w_combined = _f(G) + LIVE_LOAD_FACTOR * _f(Q)
    n_axes = _f(N_bay) + 1
    W_total_kN = _f(N_story) * _f(W_total) * w_combined
    Nd_kN = (W_total_kN / n_axes) * AXIAL_MID_COLUMN_FACTOR
    Pn_kN = (_f(col_B) * _f(col_H)) * (_f(fc_MPa) * 1000.0)
    return np.where(Pn_kN > 0, Nd_kN / np.where(Pn_kN > 0, Pn_kN, 1.0), np.inf)


def column_s_max(b_min, phi_l):
    """7.3.4.1(a): s <= min(b_min/3, 6 phi_l, 150 mm)."""
    return np.minimum(np.minimum(_f(b_min) / 3.0, 6.0 * _f(phi_l)), S_MAX_CODE)


def beam_s_max(d_eff, phi_l):
    """7.4.4: s <= min(d/4, 8 phi_l, 150 mm) in the confinement zone."""
    return np.minimum(np.minimum(_f(d_eff) / 4.0, 8.0 * _f(phi_l)), S_MAX_CODE)


def ash_min_column(s, bk, Ac, Ack, fck_kPa, fywk_kPa, axial_ratio):
    """Eq. 7.1 minimum A_sh for one direction (m2), 2/3 factor when N_d <= 0.20 A_c f_ck."""
    r = _f(fck_kPa) / _f(fywk_kPa)
    a1 = 0.30 * _f(s) * _f(bk) * (_f(Ac) / _f(Ack) - 1.0) * r
    a2 = 0.075 * _f(s) * _f(bk) * r
    ash = np.maximum(a1, a2)
    return np.where(_f(axial_ratio) <= AXIAL_RATIO_2_3_RULE, ash * (2.0 / 3.0), ash)


def column_is_tbdy_confined(col_B, col_H, cover, phi_long_col_mm, col_phi_etr, col_s_etr,
                            col_n_etr_x, col_n_etr_y, fc_MPa, fy_MPa, axial_ratio):
    """Article 7.3.4.1 (a, b, d) flag exactly as reported by transverse_design.sample_column_stirrups."""
    B, H, cover = _f(col_B), _f(col_H), _f(cover)
    phi_long = _f(phi_long_col_mm) / 1000.0
    phi_w, s = _f(col_phi_etr), _f(col_s_etr)
    n_x, n_y = _f(col_n_etr_x), _f(col_n_etr_y)
    fck_kPa = _f(fc_MPa) * 1000.0
    fyw_kPa = _f(fy_MPa) * 1000.0
    bo = B - 2.0 * cover - phi_w
    ho = H - 2.0 * cover - phi_w
    Ac = B * H
    Ack = (B - 2.0 * cover) * (H - 2.0 * cover)
    A_etr = np.pi * phi_w ** 2 / 4.0
    s_cap = column_s_max(np.minimum(B, H), phi_long)
    need_x = ash_min_column(s, ho, Ac, Ack, fck_kPa, fyw_kPa, axial_ratio)
    need_y = ash_min_column(s, bo, Ac, Ack, fck_kPa, fyw_kPa, axial_ratio)
    passes = (n_x * A_etr >= need_x) & (n_y * A_etr >= need_y)
    a_x = np.where(n_x > 1, ho / np.maximum(n_x - 1, 1), ho)
    a_y = np.where(n_y > 1, bo / np.maximum(n_y - 1, 1), bo)
    return ((s <= s_cap + TOL) & (a_x <= 25.0 * phi_w + TOL)
            & (a_y <= 25.0 * phi_w + TOL) & passes)


def beam_is_tbdy_confined(beam_h, cover, phi_long_beam_mm, beam_phi_etr, beam_s_etr):
    """Article 7.4.4 flag exactly as reported by transverse_design.sample_beam_stirrups."""
    phi_long = _f(phi_long_beam_mm) / 1000.0
    phi_w = _f(beam_phi_etr)
    d_eff = _f(beam_h) - _f(cover) - phi_w - 0.5 * phi_long
    return _f(beam_s_etr) <= beam_s_max(d_eff, phi_long) + TOL


# ----------------------------------------------------------------------------------------
# Full feature table
# ----------------------------------------------------------------------------------------
def soil_ordinal(soil_class):
    """ZA..ZE -> 0..4 (D-03). Accepts a string, an integer already encoded, or arrays of them."""
    arr = np.asarray(soil_class)
    if arr.dtype.kind in 'iuf':
        return arr.astype(np.int64)
    out = np.empty(arr.shape, dtype=np.int64)
    for idx, v in np.ndenumerate(arr):
        key = str(v).strip().upper()
        if key not in SOIL_ORDINAL:
            raise ValueError(f'unknown soil class {v!r}; expected one of {SOIL_CLASSES}')
        out[idx] = SOIL_ORDINAL[key]
    return out


def compute_features(raw: Mapping[str, object]) -> Dict[str, np.ndarray]:
    """Build the 45 model features (FEATURE_ORDER) from raw columns.

    ``raw`` maps column names to scalars or equal-length arrays. Required keys:
    N_story, N_bay, h1, htyp, col_B, beam_b, beam_h, cover, phi_long_col_mm, col_n_long,
    phi_long_beam_mm, beam_n_long_top, beam_n_long_bot, col_phi_etr, col_s_etr, col_n_etr_x,
    col_n_etr_y, beam_phi_etr, beam_s_etr, fc_MPa, fy_MPa, G, Q, soil_class (or soil_ord),
    and either the four span statistics (L_mean, L_max, L_min, W_total) or bay_lengths
    (a sequence for a single frame). col_H defaults to col_B (square columns).
    Returns a dict of float64 arrays in FEATURE_ORDER.
    """
    r = dict(raw)
    if 'L_mean' not in r:
        if 'bay_lengths' not in r:
            raise KeyError('need L_mean/L_max/L_min/W_total or bay_lengths')
        r.update(bay_statistics(r['bay_lengths']))
    if 'col_H' not in r:
        r['col_H'] = r['col_B']
    if 'soil_ord' not in r:
        r['soil_ord'] = soil_ordinal(r['soil_class'])

    f: Dict[str, np.ndarray] = {}
    for c in RAW_FEATURES:
        f[c] = _f(r[c])
    f['Htot'] = total_height(r['h1'], r['htyp'], r['N_story'])
    f['rho_col'] = rho_column(r['col_n_long'], r['phi_long_col_mm'], r['col_B'], r['col_H'])
    f['rho_beam_top'] = rho_beam(r['beam_n_long_top'], r['phi_long_beam_mm'], r['beam_b'], r['beam_h'])
    f['rho_beam_bot'] = rho_beam(r['beam_n_long_bot'], r['phi_long_beam_mm'], r['beam_b'], r['beam_h'])

    mc = column_confinement(r['col_B'], r['col_H'], r['cover'], r['fc_MPa'], r['fy_MPa'],
                            r['phi_long_col_mm'], r['col_n_long'],
                            r['col_phi_etr'], r['col_s_etr'], r['col_n_etr_x'], r['col_n_etr_y'])
    mb = beam_confinement(r['beam_b'], r['beam_h'], r['cover'], r['fc_MPa'], r['fy_MPa'],
                          r['phi_long_beam_mm'], r['beam_n_long_top'], r['beam_n_long_bot'],
                          r['beam_phi_etr'], r['beam_s_etr'])
    for key in ('rho_sh', 'rho_s', 'omega_we', 'alpha_se'):
        f[f'{key}_col'] = mc[key]
        f[f'{key}_beam'] = mb[key]
    f['fcc_col_MPa'] = mc['fcc_MPa']
    f['fcc_beam_MPa'] = mb['fcc_MPa']

    f['soil_ord'] = _f(r['soil_ord'])
    axial = column_axial_ratio_estimate(r['N_story'], r['N_bay'], r['W_total'], r['G'], r['Q'],
                                        r['col_B'], r['col_H'], r['fc_MPa'])
    f['col_is_tbdy_confined'] = _f(column_is_tbdy_confined(
        r['col_B'], r['col_H'], r['cover'], r['phi_long_col_mm'], r['col_phi_etr'],
        r['col_s_etr'], r['col_n_etr_x'], r['col_n_etr_y'], r['fc_MPa'], r['fy_MPa'], axial))
    f['beam_is_tbdy_confined'] = _f(beam_is_tbdy_confined(
        r['beam_h'], r['cover'], r['phi_long_beam_mm'], r['beam_phi_etr'], r['beam_s_etr']))
    return {c: f[c] for c in FEATURE_ORDER}


def feature_vector(raw: Mapping[str, object]) -> np.ndarray:
    """1 x 45 float64 row for one frame, in FEATURE_ORDER (input of the CatBoost regressors)."""
    f = compute_features(raw)
    return np.array([[float(np.asarray(f[c]).reshape(-1)[0]) for c in FEATURE_ORDER]],
                    dtype=np.float64)


def feature_matrix(raw: Mapping[str, object]) -> np.ndarray:
    """N x 45 float64 matrix for array-valued raw columns (batch use)."""
    f = compute_features(raw)
    n = max(np.asarray(v).size for v in f.values())
    return np.column_stack([np.broadcast_to(np.asarray(f[c], dtype=np.float64), (n,))
                            for c in FEATURE_ORDER])


def confinement_details(raw: Mapping[str, object]) -> Dict[str, Dict[str, np.ndarray]]:
    """Full Mander dictionaries of the column and the beam (for the report and the UI)."""
    r = dict(raw)
    if 'col_H' not in r:
        r['col_H'] = r['col_B']
    col = column_confinement(r['col_B'], r['col_H'], r['cover'], r['fc_MPa'], r['fy_MPa'],
                             r['phi_long_col_mm'], r['col_n_long'],
                             r['col_phi_etr'], r['col_s_etr'], r['col_n_etr_x'], r['col_n_etr_y'])
    beam = beam_confinement(r['beam_b'], r['beam_h'], r['cover'], r['fc_MPa'], r['fy_MPa'],
                            r['phi_long_beam_mm'], r['beam_n_long_top'], r['beam_n_long_bot'],
                            r['beam_phi_etr'], r['beam_s_etr'])
    return {'column': col, 'beam': beam}
