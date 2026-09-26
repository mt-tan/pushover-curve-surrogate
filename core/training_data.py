"""Development-time access to the training data of the R1 run (never used by the GUI at run time).

Reproduces the clean row set of ML_V11R_Regression_R1_figstyle.ipynb (cell 10: 73,023 rows, in
parquet order), the locked split (split_lock_seed42.json) and the curve-matrix intersection of
cell 83. Everything under outputs/ is read only.

The data are not distributed with the application. They are looked up in the parent folder of the
application, or in the folder named by the environment variable PCS_DATA_ROOT (the folder that holds
dataset_v4_D67b.parquet and outputs/ml_v11r_regression_R1); tests that need them are skipped with a
message when they are missing (tests/_dataset.require).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .features import FEATURE_ORDER

APP_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('PCS_DATA_ROOT', str(APP_DIR.parent))).resolve()
PARQUET_PATH = ROOT / 'dataset_v4_D67b.parquet'
CURVE_MATRIX_PATH = ROOT / 'curve_matrix_v4.parquet'
ART_DIR = ROOT / 'outputs' / 'ml_v11r_regression_R1'
TBL_DIR = ART_DIR / 'tables'
MODEL_SRC_DIR = ART_DIR / 'models'
CACHE_DIR = ART_DIR / 'cache'
SPLIT_LOCK_JSON = ART_DIR / 'split_lock_seed42.json'
OPTUNA_JSON = ART_DIR / 'optuna_reg_best.json'
STATE_PKL = ART_DIR / 'r1_regression_state.pkl'
LABELS_JSON = ROOT / 'figs_manual' / 'labels.json'

VALID_STATUS = ('post_peak_reached', 'completed')
CLASS_ORDER = ['IO', 'LS', 'CP', 'C']
CAP_RATIO_MAX = 1.2
DY_DRIFT_MAX = 0.02
SOIL_TO_INT = {'ZA': 0, 'ZB': 1, 'ZC': 2, 'ZD': 3, 'ZE': 4}

REG_TARGETS = ['Vy', 'delta_y', 'Vmax', 'delta_max', 'V_at_dt', 'delta_t', 'delta_u', 'T1']
TARGET_COL = {'Vy': 'kp_Vy_kN', 'delta_y': 'kp_delta_y_m', 'Vmax': 'kp_Vmax_kN',
              'delta_max': 'kp_delta_max_m', 'V_at_dt': 'kp_V_at_delta_t_kN',
              'delta_t': 'kp_delta_t_m', 'delta_u': 'kp_delta_u_m', 'T1': 'kp_T1_s'}
TARGET_COLS = [TARGET_COL[t] for t in REG_TARGETS]

# raw columns needed to rebuild the 45 features plus the columns used by the filters
RAW_COLUMNS = ['name', 'sampling_wave', 'N_story', 'N_bay', 'bay_lengths', 'L_mean', 'L_max',
               'L_min', 'W_total', 'h1', 'htyp', 'col_B', 'col_H', 'beam_b', 'beam_h', 'cover',
               'phi_long_col_mm', 'col_n_long', 'phi_long_beam_mm', 'beam_n_long_top',
               'beam_n_long_bot', 'col_phi_etr', 'col_s_etr', 'col_n_etr_x', 'col_n_etr_y',
               'beam_phi_etr', 'beam_s_etr', 'fc_MPa', 'fy_MPa', 'G', 'Q', 'soil_class']
RAW_INPUT_COLUMNS = [c for c in RAW_COLUMNS if c not in ('name', 'sampling_wave', 'bay_lengths')]
FILTER_COLUMNS = ['push_status', 'performance_class', 'Htot', 'rho_col'] + TARGET_COLS

N_ST = 21
S_ST = np.linspace(0.0, 1.0, N_ST)


def sha256_of(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def load_clean_rows(extra_columns: Sequence[str] = ()) -> pd.DataFrame:
    """Clean dataset in notebook (parquet) order with the raw, filter, feature and extra columns."""
    cols = list(dict.fromkeys(RAW_COLUMNS + FILTER_COLUMNS + list(FEATURE_ORDER)
                              + list(extra_columns)))
    cols = [c for c in cols if c != 'soil_ord']
    df = pd.read_parquet(PARQUET_PATH, columns=cols).reset_index(drop=True)
    df['soil_ord'] = df['soil_class'].map(SOIL_TO_INT)

    df = df.loc[df['push_status'].isin(VALID_STATUS)].reset_index(drop=True)
    df = df.loc[df['performance_class'].isin(CLASS_ORDER)].reset_index(drop=True)
    X = df[list(FEATURE_ORDER)].astype(float)
    df = df.loc[~(X.isna().any(axis=1) | np.isinf(X).any(axis=1))].reset_index(drop=True)
    for b in ('col_is_tbdy_confined', 'beam_is_tbdy_confined'):
        df[b] = df[b].astype(float)
    T = df[TARGET_COLS]
    df = df.loc[~(T.isna().any(axis=1) | np.isinf(T).any(axis=1))].reset_index(drop=True)

    m_base = (
        (df['kp_Vmax_kN'] > 0) & (df['kp_Vy_kN'] > 0) &
        (df['kp_Vy_kN'] <= df['kp_Vmax_kN'] * 1.05) &
        (df['kp_V_at_delta_t_kN'] > 0) &
        (df['kp_delta_y_m'] > 0) & (df['kp_delta_max_m'] > 0) &
        (df['kp_delta_u_m'] > 0) & (df['kp_delta_t_m'] > 0) &
        (df['kp_delta_y_m'] <= df['kp_delta_max_m']) &
        (df['kp_delta_max_m'] <= df['kp_delta_u_m'])
    )
    Mp_ub = (0.6 * df['rho_col'] * df['fy_MPa'] + 0.12 * df['fc_MPa']) * 1000 \
        * df['col_B'] * df['col_H'] ** 2
    V_ub = (df['N_bay'] + 1) * 2 * Mp_ub / df['h1']
    m_cap = (df['kp_Vmax_kN'] / V_ub) <= CAP_RATIO_MAX
    m_dyh = (df['kp_delta_y_m'] / df['Htot']) <= DY_DRIFT_MAX
    keep = m_base.values & m_cap.values & m_dyh.values
    return df.loc[keep].reset_index(drop=True)


def expected_clean_count() -> int:
    att = pd.read_csv(TBL_DIR / 'table_attrition.csv')
    return int(att.loc[att['Stage / filter'] == 'FINAL clean dataset', 'Remaining n'].iloc[0])


def split_lock() -> dict:
    with open(SPLIT_LOCK_JSON, encoding='utf-8') as fh:
        return json.load(fh)


def split_hash(lock: dict) -> str:
    """12-character md5 of the locked split, as printed by the notebook (cell 14)."""
    return hashlib.md5(('|'.join(lock['train_names']) + '#' +
                        '|'.join(lock['test_names'])).encode()).hexdigest()[:12]


def feature_hash(cols: Sequence[str] = FEATURE_ORDER) -> str:
    return hashlib.md5(','.join(cols).encode()).hexdigest()[:8]


def raw_mapping(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Raw column arrays for core.features.compute_features."""
    return {c: df[c].to_numpy() for c in RAW_INPUT_COLUMNS}


def stored_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """X exactly as the notebook built it: df[INPUT_COLS].values.astype(float)."""
    return df[list(FEATURE_ORDER)].values.astype(float)


def split_positions(df: pd.DataFrame, lock: dict):
    """Positions (in df order) of the train and test rows, in the order of the lock lists
    (= the order of oof_train / pred_test in r1_regression_state.pkl)."""
    pos = {n: i for i, n in enumerate(df['name'].astype(str))}
    tr_pos = np.array([pos[n] for n in lock['train_names']])
    te_pos = np.array([pos[n] for n in lock['test_names']])
    return tr_pos, te_pos


def curve_intersection(df: pd.DataFrame, lock: dict) -> dict:
    """Curve-matrix join of notebook cell 83.

    Returns Vsh_full (n_clean x 21, NaN where no curve), have, curve_tr and curve_te
    (df positions in df order) and b_sel (index of each curve_te row inside the test list,
    i.e. inside pred_test / cached pred_st arrays).
    """
    cm = pd.read_parquet(CURVE_MATRIX_PATH)
    vcols = [f'v_{j:02d}' for j in range(N_ST)]
    cm = cm.drop_duplicates('name').set_index('name')
    have = df['name'].isin(cm.index).values
    Vsh_full = np.full((len(df), N_ST), np.nan, dtype=float)
    Vsh_full[have] = cm.loc[df.loc[have, 'name'], vcols].values
    tr_pos, te_pos = split_positions(df, lock)
    tr_mask = np.zeros(len(df), dtype=bool); tr_mask[tr_pos] = True
    te_mask = np.zeros(len(df), dtype=bool); te_mask[te_pos] = True
    curve_tr = np.where(have & tr_mask)[0]
    curve_te = np.where(have & te_mask)[0]
    te_pos_index = {p: k for k, p in enumerate(te_pos)}
    b_sel = np.array([te_pos_index[p] for p in curve_te])
    return {'Vsh_full': Vsh_full, 'have': have, 'curve_tr': curve_tr, 'curve_te': curve_te,
            'b_sel': b_sel, 'tr_pos': tr_pos, 'te_pos': te_pos}
