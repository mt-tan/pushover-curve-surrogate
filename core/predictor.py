"""Headless prediction pipeline of the Pushover Curve Surrogate.

raw inputs -> validator (warnings only) -> 45 features -> eight CatBoost regressors -> clip rule
-> curves (multilinear and dense-ordinate profile) -> demand point (delta_t, V_at_dt regressors)
-> demand ratios -> derived delta_t (TBEC 5.6.5 on the predicted points, any site)
-> per-target local SHAP (CatBoost native ShapValues) -> applicability flags.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

import numpy as np
from catboost import Pool

from . import curve as C
from . import features as F
from . import screen as S
from . import spectrum as SP
from . import target_displacement as TDISP
from . import validator as V
from .model_loader import ModelBundle

DCR_NEAR_ONE_BAND = 0.10        # |DCR_delta - 1| <= 0.10 -> "near 1"; DCR_delta > 1.10 -> "above 1.10"
SHAP_TOP_K = 8


def dcr_state(dcr: float) -> str:
    """Three states of the displacement demand-to-capacity ratio DCR_delta = delta_t / delta_u:
    'ok' (below 0.90), 'near 1' (0.90 to 1.10) or 'exceeded' (above 1.10)."""
    if dcr > 1.0 + DCR_NEAR_ONE_BAND:
        return 'exceeded'
    if dcr >= 1.0 - DCR_NEAR_ONE_BAND:
        return 'near 1'
    return 'ok'


@dataclass
class Flag:
    key: str
    level: str          # 'warning' | 'info' | 'ok'
    label: str
    message: str


@dataclass
class PredictionResult:
    inputs: Dict[str, object] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    Htot: float = 0.0
    targets_raw: Dict[str, float] = field(default_factory=dict)     # regressor outputs, natural units
    targets: Dict[str, float] = field(default_factory=dict)         # after the clip rule
    clips: Dict[str, bool] = field(default_factory=dict)
    curves: Dict[str, object] = field(default_factory=dict)
    demand_point: tuple = (0.0, 0.0)
    ratios: Dict[str, float] = field(default_factory=dict)
    derived: Dict[str, float] = field(default_factory=dict)         # derived delta_t route
    derived_ratios: Dict[str, float] = field(default_factory=dict)
    site: Dict[str, float] = field(default_factory=dict)
    shap: Dict[str, dict] = field(default_factory=dict)
    screen: Dict[str, float] = field(default_factory=dict)
    validation: Dict[str, list] = field(default_factory=dict)
    flags: List[Flag] = field(default_factory=list)
    confinement: Dict[str, dict] = field(default_factory=dict)
    W_seis: float = 0.0
    Sa_T1: float = 0.0

    @property
    def warnings(self) -> List[Flag]:
        return [f for f in self.flags if f.level == 'warning']


def normalise_inputs(user: Mapping[str, object]) -> Dict[str, object]:
    """User dictionary -> raw layer of core.features.compute_features (one frame).

    Accepts either ``bay_lengths`` (sequence) or ``L_bay`` (single length for every bay).
    """
    r = dict(user)
    N_bay = int(r['N_bay'])
    stats = ('L_mean', 'L_max', 'L_min', 'W_total')
    if r.get('bay_lengths') is not None:
        L = [float(v) for v in r['bay_lengths']]
        if len(L) == 1 and N_bay > 1:
            L = L * N_bay
        if len(L) != N_bay:
            raise ValueError(f'{len(L)} bay lengths given for N_bay = {N_bay}')
        r['bay_lengths'] = L
        r.update(F.bay_statistics(L))
    elif r.get('L_bay') is not None:
        L = [float(r['L_bay'])] * N_bay
        r['bay_lengths'] = L
        r.update(F.bay_statistics(L))
    elif all(k in r and r[k] is not None for k in stats):
        r['bay_lengths'] = None          # span statistics given directly (dataset frames)
        for k in stats:
            r[k] = float(r[k])
    else:
        raise KeyError('give bay_lengths, L_bay, or the four span statistics L_mean/L_max/L_min/W_total')
    r['N_story'] = int(r['N_story']); r['N_bay'] = N_bay
    for k in ('col_n_long', 'beam_n_long_top', 'beam_n_long_bot', 'col_n_etr_x', 'col_n_etr_y'):
        r[k] = int(r[k])
    # square columns only (training set): a separate column depth is never accepted
    if r.get('col_H') is not None and abs(float(r['col_H']) - float(r['col_B'])) > 1e-12:
        raise ValueError('columns are square in the training set: hcol must equal bcol; '
                         'a separate column depth is not accepted')
    r['col_H'] = r['col_B']
    r['soil_class'] = SP.soil_name(r['soil_class'])
    r['soil_ord'] = SP.SOIL_TO_INT[r['soil_class']]
    return r


class Predictor:
    def __init__(self, bundle: ModelBundle, shape_layer: str = 'station'):
        """shape_layer: 'station' (default), 'pca' (light) or 'both'."""
        self.bundle = bundle
        self.shape_layer = shape_layer

    # ------------------------------------------------------------------ SHAP
    def explain(self, X: np.ndarray, target: str) -> Dict[str, object]:
        """Local SHAP of one target (transformed space): 45 contributions, base value, check."""
        m = self.bundle.regressors[target]
        sv = m.get_feature_importance(type='ShapValues', data=Pool(np.asarray(X, dtype=float)))
        contrib = sv[0, :-1]; base = float(sv[0, -1])
        pred_t = float(m.predict(np.asarray(X, dtype=float))[0])
        order = np.argsort(-np.abs(contrib))
        names = self.bundle.feature_order
        return {'values': {names[j]: float(contrib[j]) for j in range(len(names))},
                'top': [(names[j], float(contrib[j])) for j in order[:SHAP_TOP_K]],
                'base_value': base, 'prediction_t': pred_t,
                'additivity_gap': float(contrib.sum() + base - pred_t),
                'transform': self.bundle.target_spec['targets'][target]['transform']}

    # ------------------------------------------------------------------ pipeline
    def predict(self, user_inputs: Mapping[str, object], site: Optional[Dict[str, float]] = None,
                shap: bool = True) -> PredictionResult:
        raw = normalise_inputs(user_inputs)
        feats = F.compute_features(raw)
        f1 = {k: float(np.asarray(v).reshape(-1)[0]) for k, v in feats.items()}
        X = F.feature_vector(raw)
        Htot = f1['Htot']
        res = PredictionResult(inputs=raw, features=f1, Htot=Htot)
        res.confinement = {who: {k: float(np.asarray(v).reshape(-1)[0]) for k, v in d.items()}
                           for who, d in F.confinement_details(raw).items()}
        res.validation = V.validate(raw, feats, self.bundle.feature_spec)

        # regressors and clip rule
        t = self.bundle.predict_targets(X, Htot)
        res.targets_raw = {k: float(v[0]) for k, v in t.items()}
        cl = C.clip_points(t['delta_y'], t['delta_max'], t['delta_u'], t['Vy'], t['Vmax'])
        res.targets = dict(res.targets_raw)
        for k in ('delta_y', 'delta_max', 'delta_u', 'Vy', 'Vmax'):
            res.targets[k] = float(cl[k][0])
        res.clips = {k: bool(cl[k][0]) for k in ('clip_delta_max', 'clip_delta_u', 'clip_Vmax')}

        # curves: multilinear and the dense-ordinate profile(s)
        shape_st = shape_pc = None
        if self.bundle.has_shape_layer and self.shape_layer in ('station', 'both'):
            shape_st = self.bundle.predict_shape_station(X)[0]
        if self.bundle.has_shape_layer and self.shape_layer in ('pca', 'both'):
            shape_pc = self.bundle.predict_shape_pca(X)[0]
        pts = {k: res.targets[k] for k in ('delta_y', 'Vy', 'delta_max', 'Vmax', 'delta_u')}
        res.curves = C.assemble(pts, res.targets['delta_t'], res.targets['V_at_dt'],
                                shape_st, shape_pc, self.bundle.s_grid if self.bundle.has_shape_layer else C.S_ST)
        res.demand_point = res.curves['demand_point']

        # demand ratios (learned delta_t)
        rr = C.demand_ratios(res.targets['delta_t'], Htot, res.targets['delta_u'], res.targets['delta_y'])
        res.ratios = {k: float(v) for k, v in rr.items()}

        # derived delta_t (TBEC 5.6.5 on the predicted points; Nurdagi spectrum unless a site is given)
        res.site = site if site is not None else SP.site_parameters(raw['soil_class'])
        res.derived = TDISP.derive_delta_t_frame(
            raw['N_story'], raw['h1'], raw['htyp'], raw['G'], raw['Q'], raw['W_total'],
            res.targets['Vy'], res.targets['delta_y'], res.targets['Vmax'], res.targets['delta_max'],
            site=res.site)
        dr = C.demand_ratios(res.derived['delta_t'], Htot, res.targets['delta_u'], res.targets['delta_y'])
        res.derived_ratios = {k: float(v) for k, v in dr.items()}
        res.derived['V_curve_at_dt'] = C.curve_value(*res.curves['multilinear'], res.derived['delta_t'])
        res.W_seis = (raw['G'] + 0.3 * raw['Q']) * raw['W_total'] * raw['N_story']
        res.Sa_T1 = float(SP.Sae(res.targets['T1'], res.site))

        # applicability screen (predicted Vmax as demand)
        sc = S.shear_screen(f1, res.targets['Vmax'])
        res.screen = {k: (bool(np.asarray(v).reshape(-1)[0]) if np.asarray(v).dtype == bool
                          else float(np.asarray(v).reshape(-1)[0])) for k, v in sc.items()}

        if shap:
            res.shap = {tgt: self.explain(X, tgt) for tgt in self.bundle.targets}

        res.flags = self._flags(res)
        return res

    # ------------------------------------------------------------------ flags
    def _flags(self, r: PredictionResult) -> List[Flag]:
        fl: List[Flag] = []
        sc = r.screen
        if sc['col_shear_governed'] or sc['beam_shear_governed']:
            who = ' and '.join(w for w, k in (('column', 'col_shear_governed'), ('beam', 'beam_shear_governed')) if sc[k])
            fl.append(Flag('shear', 'warning', 'Shear-governed',
                           f'{who} shear demand exceeds the TS 500 resistance (Vdem/Vr column '
                           f'{sc["col_shear_ratio"]:.2f}, beam {sc["beam_shear_ratio"]:.2f}); the ground truth '
                           f'is flexure-governed (Section 3.12)'))
        else:
            fl.append(Flag('shear', 'ok', 'Shear screen',
                           f'flexure-governed (Vdem/Vr column {sc["col_shear_ratio"]:.2f}, beam {sc["beam_shear_ratio"]:.2f})'))
        if sc['buckling_proxy']:
            fl.append(Flag('buckling', 'warning', 'Bar-buckling proxy',
                           f's/φl = {sc["s_over_phil_col"]:.1f} > 6 in the column confinement zone'))
        else:
            fl.append(Flag('buckling', 'ok', 'Bar buckling', f's/φl = {sc["s_over_phil_col"]:.1f} <= 6'))
        for k, lab in (('col_is_tbdy_confined', 'Column confinement 7.3.4.1'),
                       ('beam_is_tbdy_confined', 'Beam confinement 7.4.4')):
            ok = r.features[k] > 0.5
            fl.append(Flag(k, 'ok' if ok else 'info', lab, 'compliant' if ok else 'non-compliant detailing (allowed in the training set)'))
        env = [x for x in r.validation.get('envelope', []) if not x.passed]
        if env:
            fl.append(Flag('envelope', 'warning', 'Outside training envelope',
                           '; '.join(x.message for x in env)))
        else:
            fl.append(Flag('envelope', 'ok', 'Training envelope',
                           'all 45 features inside the ranges of the training pool'))
        tb = [x for x in r.validation.get('tbdy', []) if not x.passed]
        if tb:
            fl.append(Flag('tbdy', 'warning', 'TBEC design filter (Table 2)', '; '.join(x.message for x in tb)))
        else:
            fl.append(Flag('tbdy', 'ok', 'Table 2 filters', 'all seven generation filters satisfied'))
        inp = [x for x in r.validation.get('input', []) if not x.passed]
        if inp:
            fl.append(Flag('input', 'warning', 'Input checks', '; '.join(x.message for x in inp)))
        dcr = r.ratios['DCR_delta']
        state = dcr_state(dcr)
        if state == 'exceeded':
            fl.append(Flag('dcr', 'warning', 'DCRδ above 1.10',
                           f'DCRδ = {dcr:.3f} > {1.0 + DCR_NEAR_ONE_BAND:.2f}: the displacement demand δt '
                           f'exceeds the predicted ultimate displacement δu'))
        elif state == 'near 1':
            fl.append(Flag('dcr', 'warning', 'DCRδ near 1',
                           f'DCRδ = {dcr:.3f} within {DCR_NEAR_ONE_BAND:.2f} of 1: small prediction errors '
                           f'can move the demand point across δu'))
        else:
            fl.append(Flag('dcr', 'ok', 'DCRδ', f'DCRδ = {dcr:.3f} < {1.0 - DCR_NEAR_ONE_BAND:.2f}'))
        if any(r.clips.values()):
            fl.append(Flag('clip', 'info', 'Clip applied',
                           ', '.join(k.replace('clip_', '') for k, v in r.clips.items() if v) + ' adjusted by the 1.001 rule'))
        if not r.site.get('is_nurdagi', True):
            fl.append(Flag('site', 'info', 'Site transfer',
                           'learned δt is valid for the Nurdagi DD-2 spectrum only; use the derived δt'))
        return fl
