"""Input validation of the tool: TBEC 2018 design filters of the generation chain
(paper Table 2, modules/tbdy_filters.py) and the training envelope (realised [min, max] of the
training pool of the locked split, through feature_spec.json). Every rule is a warning or a note,
never a blocker; the prediction always runs.

Filters ported one to one (same formulas and thresholds):
  min_dimensions        7.3.1.1 / 7.4.1.1 (b_col, h_col >= 0.30 m; b_beam >= 0.25 m; h_beam >= L_max / 12)
  long_feasible         D-56: bar count fits the 2.5 cm clear-spacing rule (nominal 10 mm stirrup)
  reinforcement_ratios  7.3.3.1 / 7.4.3.1 (rho_col in [0.010, 0.040]; beam ratios in [0.0033, 0.020];
                        rho_bot >= 0.5 rho_top; cover in [0.02, 0.05] m)
  material              7.2.5.1 (f_ck in [25, 40] MPa)
  hinge_geometry        2 Lp < clear length (Lp = 0.5 h)
  axial_load_60pct      N_d / (A_c f_ck) <= 0.60 with the generation estimate
  scwb_capacity         7.3.5.1: sum M_c >= 1.2 sum M_b at the most critical interior joint
Extra checks: steel grade in {420, 500}, soil class, bar counts >= 2, tie legs in [2, 4].
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np

from . import features as F

PHI_ETR_NOM = 0.010          # nominal stirrup diameter of the generation spacing check
CLEAR_MIN = 0.025            # 2.5 cm clear bar spacing
AXIAL_LIMIT = 0.60
SCWB_RATIO = 1.2


@dataclass
class RuleResult:
    rule: str
    passed: bool
    message: str
    severity: str = 'warning'      # 'info' | 'warning'
    group: str = 'tbdy'            # 'tbdy' | 'envelope' | 'input'


def _params(raw: Mapping[str, object], feats: Mapping[str, float]) -> Dict[str, float]:
    """Parameter dict in the units of modules/tbdy_filters.py (kPa, m)."""
    p = {k: float(np.asarray(feats[k]).reshape(-1)[0]) for k in F.FEATURE_ORDER}
    p['fc'] = p['fc_MPa'] * 1000.0; p['fck'] = p['fc']; p['fy'] = p['fy_MPa'] * 1000.0
    p['Lbay'] = p['L_mean']
    p['col_phi_long'] = p['phi_long_col_mm'] / 1000.0
    p['beam_phi_long'] = p['phi_long_beam_mm'] / 1000.0
    p['col_n_long_x'] = int(round(p['col_n_long']))
    p['beam_n_long_top'] = int(round(p['beam_n_long_top']))
    p['beam_n_long_bot'] = int(round(p['beam_n_long_bot']))
    return p


# ---------------------------------------------------------------------------- Table 2 filters
def filter_min_dimensions(p):
    if p['col_B'] < 0.30:
        return False, f"bcol = {p['col_B']:.3f} m < 0.30 m (TBEC 7.3.1.1)"
    if p['col_H'] < 0.30:
        return False, f"hcol = {p['col_H']:.3f} m < 0.30 m (TBEC 7.3.1.1)"
    if p['beam_b'] < 0.25:
        return False, f"bbeam = {p['beam_b']:.3f} m < 0.25 m (TBEC 7.4.1.1)"
    L_gov = p.get('L_max', p['Lbay'])
    if p['beam_h'] < L_gov / 12:
        return False, f"hbeam = {p['beam_h']:.3f} m < Lmax/12 = {L_gov / 12:.3f} m (TBEC 7.4.1.1)"
    return True, 'section minimum dimensions satisfied (7.3.1.1, 7.4.1.1)'


def max_bars_per_face(dim, cover, phi_etr, phi_long, clear=CLEAR_MIN):
    bk = dim - 2.0 * cover - 2.0 * phi_etr
    pitch = phi_long + max(clear, phi_long)
    n = int(math.floor((bk - phi_long) / pitch)) + 1
    return max(2, n)


def filter_long_feasible(p):
    n_sp_col = max_bars_per_face(min(p['col_B'], p['col_H']), p['cover'], PHI_ETR_NOM, p['col_phi_long'])
    if p['col_n_long_x'] > n_sp_col:
        return False, (f"column bars per face {p['col_n_long_x']} exceed the 2.5 cm clear-spacing "
                       f"limit {n_sp_col} for φ {p['phi_long_col_mm']:.0f} mm (D-56)")
    n_sp_beam = max_bars_per_face(p['beam_b'], p['cover'], PHI_ETR_NOM, p['beam_phi_long'])
    if max(p['beam_n_long_top'], p['beam_n_long_bot']) > n_sp_beam:
        return False, (f"beam bars per layer exceed the 2.5 cm clear-spacing limit {n_sp_beam} "
                       f"for φ {p['phi_long_beam_mm']:.0f} mm (D-56)")
    return True, f'bar spacing feasible (column limit {n_sp_col}, beam limit {n_sp_beam} bars)'


def filter_reinforcement_ratios(p):
    if not (0.010 <= p['rho_col'] <= 0.040):
        return False, f"ρcol = {p['rho_col']:.4f} outside [0.010, 0.040] (TBEC 7.3.3.1)"
    if not (0.0033 <= p['rho_beam_top'] <= 0.020):
        return False, f"ρbeam,top = {p['rho_beam_top']:.4f} outside [0.0033, 0.020] (TBEC 7.4.3.1)"
    if not (0.0033 <= p['rho_beam_bot'] <= 0.020):
        return False, f"ρbeam,bot = {p['rho_beam_bot']:.4f} outside [0.0033, 0.020] (TBEC 7.4.3.1)"
    if p['rho_beam_bot'] < 0.5 * p['rho_beam_top']:
        return False, (f"ρbeam,bot = {p['rho_beam_bot']:.4f} < 0.5 ρbeam,top = "
                       f"{0.5 * p['rho_beam_top']:.4f} (TBEC 7.4.3.1.2)")
    if not (0.02 <= p['cover'] <= 0.05):
        return False, f"cover = {p['cover'] * 100:.1f} cm outside [2, 5] cm (D-02)"
    return True, 'longitudinal ratios and cover within the code bands (7.3.3.1, 7.4.3.1)'


def filter_material(p):
    fc_MPa = p['fc'] / 1000.0
    if not (25 <= fc_MPa <= 40):
        return False, f"fck = {fc_MPa:.1f} MPa outside [25, 40] (TBEC 7.2.5.1, generation range)"
    return True, 'concrete class within the generation range C25 to C40'


def filter_hinge_geometry(p):
    Lp_col = 0.5 * p['col_H']
    if 2 * Lp_col >= p['h1']:
        return False, (f"2 Lp,col = {2 * Lp_col:.3f} m >= h1 = {p['h1']:.3f} m "
                       f"(column plastic zones overlap)")
    Lp_beam = 0.5 * p['beam_h']
    L_clear_beam = p.get('L_min', p['Lbay']) - p['col_B']
    if 2 * Lp_beam >= L_clear_beam:
        return False, (f"2 Lp,beam = {2 * Lp_beam:.3f} m >= clear span {L_clear_beam:.3f} m "
                       f"(beam plastic zones overlap in the shortest bay)")
    return True, 'plastic-hinge lengths fit the clear lengths'


def estimate_column_axial_load(p):
    w_combined = p['G'] + 0.3 * p['Q']
    n_axes = p['N_bay'] + 1
    W_width = p.get('W_total', p['N_bay'] * p['Lbay'])
    W_total_kN = p['N_story'] * W_width * w_combined
    Nd_kN = (W_total_kN / n_axes) * 1.3
    Ac_m2 = p['col_B'] * p['col_H']
    Pn_kN = Ac_m2 * p['fck']
    n_ratio = Nd_kN / Pn_kN if Pn_kN > 0 else float('inf')
    return Nd_kN, Pn_kN, n_ratio


def filter_axial_load(p, limit=AXIAL_LIMIT):
    Nd, Pn, n_ratio = estimate_column_axial_load(p)
    if n_ratio > limit:
        return False, (f"Nd/(Ac fck) = {n_ratio:.3f} > {limit:.2f} "
                       f"(Nd about {Nd:.0f} kN, Ac fck = {Pn:.0f} kN)")
    return True, f'axial-load ratio {n_ratio:.3f} <= {limit:.2f}'


def _beam_capacities(p):
    b, h, cov = p['beam_b'], p['beam_h'], p['cover']
    fc, fy = p['fc'], p['fy']
    phi = p.get('beam_phi_long', 0.018)
    petr = p.get('beam_phi_etr', 0.010)
    Abar = math.pi * phi ** 2 / 4.0
    As_top = p['beam_n_long_top'] * Abar
    As_bot = p['beam_n_long_bot'] * Abar
    d = h - cov - petr - phi / 2.0

    def Mn(As):
        a = As * fy / (0.85 * fc * b)
        return As * fy * (d - a / 2.0)
    return Mn(As_top), Mn(As_bot)


def _column_capacity(p, N_kN):
    b, h, cov = p['col_B'], p['col_H'], p['cover']
    fc, fy = p['fc'], p['fy']
    phi = p.get('col_phi_long', 0.020)
    petr = p.get('col_phi_etr', 0.010)
    Abar = math.pi * phi ** 2 / 4.0
    As_t = p['col_n_long_x'] * Abar
    dd = h - 2.0 * (cov + petr + phi / 2.0)
    a = max(N_kN, 0.0) / (0.85 * fc * b)
    a = min(a, 0.8 * h)
    Mn = As_t * fy * dd + N_kN * (h / 2.0 - a / 2.0)
    return max(Mn, 0.5 * As_t * fy * dd)


def filter_scwb_capacity(p, ratio_req=SCWB_RATIO):
    Mn_neg, Mn_pos = _beam_capacities(p)
    SMb = Mn_neg + Mn_pos
    if SMb <= 0:
        return True, 'strong column - weak beam: no beam capacity (skipped)'
    Nbase, _, _ = estimate_column_axial_load(p)
    Ns = int(p['N_story'])
    worst = float('inf')
    for L in range(1, max(Ns, 2)):
        N_below = Nbase * (Ns - L + 1) / Ns
        N_above = Nbase * (Ns - L) / Ns
        SMc = _column_capacity(p, N_below) + _column_capacity(p, N_above)
        worst = min(worst, SMc / SMb)
    if worst < ratio_req:
        return False, (f"ΣMc/ΣMb = {worst:.2f} < {ratio_req} at the most critical "
                       f"interior joint (ΣMb = {SMb:.0f} kNm) (TBEC 7.3.5.1)")
    return True, f'strong column - weak beam ratio {worst:.2f} >= {ratio_req} (7.3.5.1)'


TABLE2_FILTERS = [
    ('min_dimensions', filter_min_dimensions),
    ('long_feasible', filter_long_feasible),
    ('reinforcement_ratios', filter_reinforcement_ratios),
    ('material', filter_material),
    ('hinge_geometry', filter_hinge_geometry),
    ('axial_load_60pct', filter_axial_load),
    ('scwb_capacity', filter_scwb_capacity),
]


# ---------------------------------------------------------------------------- envelope
def check_envelope(feats: Mapping[str, float], feature_spec: Mapping) -> List[RuleResult]:
    """Flag features outside the realised [min, max] of the training pool (feature_spec.json)."""
    out = []
    for spec in feature_spec['features']:
        c = spec['name']
        v = float(np.asarray(feats[c]).reshape(-1)[0])
        lo, hi = spec['realised']['min'], spec['realised']['max']
        tol = 1e-9 * max(1.0, abs(hi))
        if v < lo - tol or v > hi + tol:
            name = spec.get('display_unicode') or spec['display']
            unit = f" {spec['unit']}" if spec['unit'] not in ('', '-') else ''
            out.append(RuleResult(f'envelope:{c}', False,
                                  f"{name} = {v:.4g}{unit} outside the training "
                                  f"envelope [{lo:.4g}, {hi:.4g}]", 'warning', 'envelope'))
    return out


# paper notation of the count inputs (plain Unicode, as in assets/labels.json) for the messages
_COUNT_NAMES = {'col_n_long': 'nl,col (column bars per face)', 'beam_n_long_top': 'nl,beam,top (beam top bars)',
                'beam_n_long_bot': 'nl,beam,bot (beam bottom bars)', 'col_n_etr_x': 'nw,col,x (column tie legs, x)',
                'col_n_etr_y': 'nw,col,y (column tie legs, y)'}


def check_inputs(raw: Mapping[str, object]) -> List[RuleResult]:
    out = []
    fy = float(raw['fy_MPa'])
    if fy not in (420.0, 500.0):
        out.append(RuleResult('input:fy', False, f'fy = {fy:g} MPa is not B420C or B500C', 'warning', 'input'))
    for k in ('col_n_long', 'beam_n_long_top', 'beam_n_long_bot'):
        if int(raw[k]) < 2:
            out.append(RuleResult(f'input:{k}', False, f'{_COUNT_NAMES[k]} must be at least 2 bars',
                                  'warning', 'input'))
    for k in ('col_n_etr_x', 'col_n_etr_y'):
        if not 2 <= int(raw[k]) <= 4:
            out.append(RuleResult(f'input:{k}', False,
                                  f'{_COUNT_NAMES[k]} = {int(raw[k])} outside the sampled 2 to 4 legs',
                                  'warning', 'input'))
    if int(raw['beam_n_long_bot']) > int(raw['beam_n_long_top']):
        out.append(RuleResult('input:beam_bars', False, 'bottom bars exceed top bars (not sampled in generation)',
                              'info', 'input'))
    return out


# ---------------------------------------------------------------------------- entry point
def validate(raw: Mapping[str, object], feats: Mapping[str, float],
             feature_spec: Optional[Mapping] = None) -> Dict[str, List[RuleResult]]:
    p = _params(raw, feats)
    tbdy = []
    for name, fn in TABLE2_FILTERS:
        ok, msg = fn(p)
        tbdy.append(RuleResult(name, bool(ok), msg, 'info' if ok else 'warning', 'tbdy'))
    env = check_envelope(feats, feature_spec) if feature_spec is not None else []
    return {'tbdy': tbdy, 'envelope': env, 'input': check_inputs(raw)}


def overall_status(results: Mapping[str, List[RuleResult]]) -> str:
    rules = [r for group in results.values() for r in group]
    if any((not r.passed) and r.severity == 'warning' for r in rules):
        return 'warning'
    return 'ok'
