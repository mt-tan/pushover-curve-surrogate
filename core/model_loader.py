"""Model artifacts of the tool: eight CatBoost regressors, the curve-shape layer and the specs.

Artifacts live in ``app/assets/models`` and are written by ``export_models.py``:

* ``reg_final_{target}.cbm``      copied from outputs/ml_v11r_regression_R1/models (8 files)
* ``feature_spec.json``           45-feature order, labels, units, realised ranges, kinds
* ``target_spec.json``            targets, dataset columns, transforms, TeX labels, units
* ``station_{j:02d}.cbm``         19 station regressors (j = 1..19; v(0) = 0, v(1) = 0.8 fixed)
* ``pca_coef_{k}.cbm``            5 PCA-coefficient regressors (light alternative)
* ``shape_layer.json``            s grid, mean shape, PCA mean / components / variance, config
* ``provenance.json``             hashes, versions, export date, training report

Target transforms (notebook cell 8): V targets log1p, delta targets log(delta / H_tot), T1 raw.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from catboost import CatBoostRegressor

MODEL_DIR_DEFAULT = Path(__file__).resolve().parents[1] / 'assets' / 'models'
REG_TARGETS: List[str] = ['Vy', 'delta_y', 'Vmax', 'delta_max', 'V_at_dt', 'delta_t', 'delta_u', 'T1']
TARGET_TRANSFORM: Dict[str, str] = {
    'Vy': 'log1p', 'Vmax': 'log1p', 'V_at_dt': 'log1p',
    'delta_y': 'lognormH', 'delta_max': 'lognormH', 'delta_t': 'lognormH', 'delta_u': 'lognormH',
    'T1': 'raw'}
TARGET_TEX: Dict[str, tuple] = {
    'Vy': (r'$V_y$', 'kN'), 'delta_y': (r'$\delta_y$', 'm'),
    'Vmax': (r'$V_{\mathrm{max}}$', 'kN'), 'delta_max': (r'$\delta_{\mathrm{max}}$', 'm'),
    'V_at_dt': (r'$V_{\delta_t}$', 'kN'), 'delta_t': (r'$\delta_t$', 'm'),
    'delta_u': (r'$\delta_u$', 'm'), 'T1': (r'$T_1$', 's')}
TARGET_COLUMN: Dict[str, str] = {
    'Vy': 'kp_Vy_kN', 'delta_y': 'kp_delta_y_m', 'Vmax': 'kp_Vmax_kN', 'delta_max': 'kp_delta_max_m',
    'V_at_dt': 'kp_V_at_delta_t_kN', 'delta_t': 'kp_delta_t_m', 'delta_u': 'kp_delta_u_m',
    'T1': 'kp_T1_s'}


def transform_y(transform: str, y_raw, Htot):
    y_raw = np.asarray(y_raw, dtype=float)
    if transform == 'log1p':
        return np.log1p(np.maximum(y_raw, 0))
    if transform == 'lognormH':
        return np.log(np.maximum(y_raw, 1e-9) / np.asarray(Htot, dtype=float))
    if transform == 'log':
        return np.log(np.maximum(y_raw, 1e-9))
    return y_raw.copy()


def inverse_y(transform: str, y_t, Htot):
    y_t = np.asarray(y_t, dtype=float)
    if transform == 'log1p':
        return np.expm1(y_t)
    if transform == 'lognormH':
        return np.exp(y_t) * np.asarray(Htot, dtype=float)
    if transform == 'log':
        return np.exp(y_t)
    return y_t


def _load_json(path: Path) -> dict:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def _load_cbm(path: Path) -> CatBoostRegressor:
    if not path.exists():
        raise FileNotFoundError(f'missing model file {path}; run export_models.py first')
    m = CatBoostRegressor()
    m.load_model(str(path))
    return m


@dataclass
class ModelBundle:
    model_dir: Path
    feature_spec: dict
    target_spec: dict
    provenance: dict
    regressors: Dict[str, CatBoostRegressor] = field(default_factory=dict)
    shape_spec: Optional[dict] = None
    stations: Dict[int, CatBoostRegressor] = field(default_factory=dict)
    pca_models: List[CatBoostRegressor] = field(default_factory=list)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, model_dir: Path | str | None = None, shape: bool = True) -> 'ModelBundle':
        d = Path(model_dir) if model_dir is not None else MODEL_DIR_DEFAULT
        b = cls(model_dir=d, feature_spec=_load_json(d / 'feature_spec.json'),
                target_spec=_load_json(d / 'target_spec.json'),
                provenance=_load_json(d / 'provenance.json') if (d / 'provenance.json').exists() else {})
        for t in b.targets:
            b.regressors[t] = _load_cbm(d / b.target_spec['targets'][t]['model_file'])
        if shape and (d / 'shape_layer.json').exists():
            b.shape_spec = _load_json(d / 'shape_layer.json')
            for j, fn in zip(b.shape_spec['learned_stations'], b.shape_spec['station_files']):
                b.stations[int(j)] = _load_cbm(d / fn)
            for fn in b.shape_spec['pca']['coefficient_files']:
                b.pca_models.append(_load_cbm(d / fn))
        return b

    # ------------------------------------------------------------------ properties
    @property
    def feature_order(self) -> List[str]:
        return [f['name'] for f in self.feature_spec['features']]

    @property
    def targets(self) -> List[str]:
        return list(self.target_spec['order'])

    @property
    def s_grid(self) -> np.ndarray:
        return np.asarray(self.shape_spec['s_grid'], dtype=float)

    @property
    def has_shape_layer(self) -> bool:
        return bool(self.stations)

    def _check_X(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != len(self.feature_order):
            raise ValueError(f'expected {len(self.feature_order)} features, got {X.shape[1]}')
        return X

    # ------------------------------------------------------------------ regressors
    def predict_transformed(self, X) -> Dict[str, np.ndarray]:
        """Model outputs in the transformed space (log1p V, log(delta / H_tot), raw T1)."""
        X = self._check_X(X)
        return {t: np.asarray(self.regressors[t].predict(X), dtype=float) for t in self.targets}

    def predict_targets(self, X, Htot) -> Dict[str, np.ndarray]:
        """Eight targets in natural units (kN, m, s); Htot is needed for the delta targets."""
        yt = self.predict_transformed(X)
        H = np.asarray(Htot, dtype=float)
        return {t: inverse_y(self.target_spec['targets'][t]['transform'], yt[t], H)
                for t in self.targets}

    # ------------------------------------------------------------------ shape layer
    def mean_shape(self) -> np.ndarray:
        v = np.asarray(self.shape_spec['mean_shape'], dtype=float).copy()
        v[0] = 0.0
        if self.shape_spec['fix_endpoints']:
            v[-1] = self.shape_spec['v_end']
        return v

    def predict_shape_station(self, X) -> np.ndarray:
        """Dense-ordinate profile v(s) on the 21-station grid from the station regressors."""
        X = self._check_X(X)
        n_st = int(self.shape_spec['n_stations'])
        out = np.zeros((X.shape[0], n_st))
        for j, m in self.stations.items():
            out[:, j] = m.predict(X)
        out[:, 0] = 0.0
        if self.shape_spec['fix_endpoints']:
            out[:, -1] = self.shape_spec['v_end']
        return out

    def predict_shape_pca(self, X) -> np.ndarray:
        """Dense-ordinate profile from the K PCA-coefficient regressors."""
        X = self._check_X(X)
        p = self.shape_spec['pca']
        mu = np.asarray(p['mean'], dtype=float)
        comps = np.asarray(p['components'], dtype=float)
        coef = np.column_stack([m.predict(X) for m in self.pca_models])
        out = mu[None, :] + coef @ comps
        out[:, 0] = 0.0
        if self.shape_spec['fix_endpoints']:
            out[:, -1] = self.shape_spec['v_end']
        return out
