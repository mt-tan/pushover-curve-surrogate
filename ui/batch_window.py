"""Batch prediction: xlsx / csv in (one frame per row, headers in paper notation), xlsx out with
inputs, the 45 features, the eight targets, demand ratios, the derived route and the flags.
Also writes the input template. The headless functions (write_template, read_inputs,
run_batch) are used by the tests."""
from __future__ import annotations

from pathlib import Path
from tkinter import filedialog
from typing import Dict, List, Optional, Tuple

import customtkinter as ctk
import numpy as np
import pandas as pd

from core import features as F
from core import spectrum as SP
from core.predictor import Predictor, dcr_state
from . import theme as T

INPUT_KEYS = ['N_story', 'N_bay', 'bay_lengths', 'h1', 'htyp', 'col_B', 'beam_b', 'beam_h', 'cover',
              'phi_long_col_mm', 'col_n_long', 'phi_long_beam_mm', 'beam_n_long_top', 'beam_n_long_bot',
              'col_phi_etr', 'col_s_etr', 'col_n_etr_x', 'col_n_etr_y', 'beam_phi_etr', 'beam_s_etr',
              'fc_MPa', 'fy_MPa', 'G', 'Q', 'soil_class']
SITE_KEYS = ['Ss', 'S1']
TARGET_KEYS = ['Vy', 'delta_y', 'Vmax', 'delta_max', 'V_at_dt', 'delta_t', 'delta_u', 'T1']
RATIO_KEYS = ['theta_t', 'DCR_delta', 'mu_d']
FLAG_COLUMNS = [('shear', 'shear-governed'), ('buckling', 'bar-buckling proxy'), ('envelope', 'outside training envelope'),
                ('tbdy', 'TBEC design-filter warning'), ('clip', 'clip applied')]
DCR_COLUMN = 'DCRδ state'
DCR_STATE_TEXT = {'ok': 'ok (below 0.90)', 'near 1': 'near 1 (0.90 to 1.10)', 'exceeded': 'above 1.10'}
INT_KEYS = {'N_story', 'N_bay', 'col_n_long', 'beam_n_long_top', 'beam_n_long_bot', 'col_n_etr_x', 'col_n_etr_y'}


def header(key: str) -> str:
    if key in SITE_KEYS:
        return f'{key} (site transfer)'
    return T.with_unit(T.uni(key), T.unit(key))


HEADER_TO_KEY = {header(k): k for k in INPUT_KEYS + SITE_KEYS}


def write_template(path: Path, presets: Optional[List[dict]] = None) -> Path:
    rows = []
    for p in presets or []:
        r = {header(k): (';'.join(f'{x:.4f}' for x in p['inputs'][k]) if k == 'bay_lengths' else p['inputs'][k])
             for k in INPUT_KEYS}
        r[header('Ss')] = ''; r[header('S1')] = ''
        r['id'] = p['id']
        rows.append(r)
    cols = ['id'] + [header(k) for k in INPUT_KEYS + SITE_KEYS]
    df = pd.DataFrame(rows, columns=cols)
    L = T.labels()
    keys = pd.DataFrame([{'header': header(k), 'key': k, 'unit': T.unit(k),
                          'description': L['raw_input_descriptions'].get(k, ''),
                          'allowed': {'phi_long_col_mm': '14, 16, 18, 20, 22', 'phi_long_beam_mm': '12, 14, 16, 18, 20',
                                      'col_phi_etr': '0.008, 0.010', 'beam_phi_etr': '0.008, 0.010',
                                      'col_n_etr_x': '2, 3, 4', 'col_n_etr_y': '2, 3, 4', 'fy_MPa': '420, 500',
                                      'soil_class': 'ZA, ZB, ZC, ZD, ZE',
                                      'bay_lengths': 'one value, or one per bay separated by ;'}.get(k, '')}
                         for k in INPUT_KEYS] +
                        [{'header': header('Ss'), 'key': 'Ss', 'unit': '-', 'description': 'short-period map coefficient (blank = training site 1.159)', 'allowed': ''},
                         {'header': header('S1'), 'key': 'S1', 'unit': '-', 'description': '1.0 s map coefficient (blank = training site 0.307)', 'allowed': ''}])
    notes = pd.DataFrame({'note': [
        'One frame per row on the sheet "inputs"; headers may be the notation shown or the keys of the sheet "keys".',
        'Columns are square (side = bcol). Bar counts are per face (column) or per layer (beam).',
        'Leave Ss and S1 blank for the training site (Gaziantep Nurdagi, DD-2); the learned demand is valid only there.',
        'The example rows are the six presets (locked hold-out frames of the paper).']})
    path = Path(path)
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        df.to_excel(xw, sheet_name='inputs', index=False)
        keys.to_excel(xw, sheet_name='keys', index=False)
        notes.to_excel(xw, sheet_name='notes', index=False)
    return path


def read_inputs(path: Path) -> Tuple[List[Dict[str, object]], List[str]]:
    path = Path(path)
    df = pd.read_csv(path) if path.suffix.lower() == '.csv' else pd.read_excel(path, sheet_name=0)
    colmap = {}
    for c in df.columns:
        cs = str(c).strip()
        if cs in HEADER_TO_KEY:
            colmap[HEADER_TO_KEY[cs]] = c
        elif cs in INPUT_KEYS + SITE_KEYS:
            colmap[cs] = c
    missing = [k for k in INPUT_KEYS if k not in colmap]
    problems = [f'missing column(s): {", ".join(header(k) for k in missing)}'] if missing else []
    if any(str(c).strip().lower().replace(' [m]', '') in ('col_h', 'hcol') for c in df.columns):
        problems.append('a separate column depth (hcol) is not accepted: columns are square, give bcol only')
    rows = []
    if not missing:
        for i, rec in df.iterrows():
            r: Dict[str, object] = {'id': rec['id'] if 'id' in df.columns else f'row {i + 1}'}
            for k in INPUT_KEYS:
                v = rec[colmap[k]]
                if k == 'bay_lengths':
                    r[k] = [float(x) for x in str(v).replace(',', ';').split(';') if str(x).strip()]
                elif k == 'soil_class':
                    r[k] = str(v).strip().upper()
                elif k in INT_KEYS:
                    r[k] = int(float(v))
                else:
                    r[k] = float(v)
            for k in SITE_KEYS:
                v = rec[colmap[k]] if k in colmap else None
                r[k] = None if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == '' else float(v)
            rows.append(r)
    return rows, problems


def run_batch(predictor: Predictor, rows: List[Dict[str, object]], progress=None) -> Dict[str, pd.DataFrame]:
    """Returns {'results': DataFrame, 'features': DataFrame} with notation headers."""
    res_rows, feat_rows = [], []
    for i, r in enumerate(rows):
        rid = r.get('id', i + 1)
        inputs = {k: r[k] for k in INPUT_KEYS}
        site = None
        if r.get('Ss') is not None or r.get('S1') is not None:
            site = SP.site_parameters(inputs['soil_class'], r.get('Ss') or SP.SS_NURDAGI, r.get('S1') or SP.S1_NURDAGI)
        out = {'id': rid}
        for k in INPUT_KEYS:
            out[header(k)] = ';'.join(f'{x:.4f}' for x in inputs[k]) if k == 'bay_lengths' else inputs[k]
        try:
            p = predictor.predict(inputs, site=site, shap=False)
            for t in TARGET_KEYS:
                out[T.with_unit(T.uni(t), T.unit(t))] = p.targets[t]
            for k in RATIO_KEYS:
                out[T.uni(k)] = p.ratios[k]
            out[T.with_unit('δt (derived)', 'm')] = p.derived['delta_t']
            out['θt (derived)'] = p.derived_ratios['theta_t']; out['DCRδ (derived)'] = p.derived_ratios['DCR_delta']
            out['μd (derived)'] = p.derived_ratios['mu_d']
            out[T.with_unit('Te', 's')] = p.derived['Te']; out['CR'] = p.derived['CR']
            out['site of the derived route'] = p.site['name']
            out[T.with_unit('Htot', 'm')] = p.Htot
            out['column TBEC 7.3.4.1'] = 'compliant' if p.features['col_is_tbdy_confined'] > 0.5 else 'non-compliant'
            out['beam TBEC 7.4.4'] = 'compliant' if p.features['beam_is_tbdy_confined'] > 0.5 else 'non-compliant'
            fl = {f.key: f for f in p.flags}
            for key, name in FLAG_COLUMNS:
                f = fl.get(key)
                out[name] = 'yes' if (f is not None and f.level in ('warning', 'info') and key != 'clip') or \
                    (key == 'clip' and f is not None) else 'no'
            out[DCR_COLUMN] = DCR_STATE_TEXT[dcr_state(p.ratios['DCR_delta'])]
            out['warnings'] = ' | '.join(f.message for f in p.warnings)
            out['status'] = 'ok'
            feat = {'id': rid}
            feat.update({T.with_unit(T.uni(k), T.unit(k)): p.features[k] for k in F.FEATURE_ORDER})
            feat_rows.append(feat)
        except Exception as exc:
            out['status'] = f'error: {exc}'
        res_rows.append(out)
        if progress:
            progress(i + 1, len(rows))
    return {'results': pd.DataFrame(res_rows), 'features': pd.DataFrame(feat_rows)}


def write_results(path: Path, tables: Dict[str, pd.DataFrame]) -> Path:
    path = Path(path)
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        tables['results'].to_excel(xw, sheet_name='results', index=False)
        tables['features'].to_excel(xw, sheet_name='features', index=False)
    return path


class BatchWindow(ctk.CTkToplevel):
    def __init__(self, master, predictor: Predictor, presets: Optional[List[dict]] = None):
        super().__init__(master)
        self.predictor = predictor; self.presets = presets or []
        self.title('Batch prediction'); self.geometry('760x420'); self.configure(fg_color=T.BG)
        card = ctk.CTkFrame(self, fg_color=T.PANEL, corner_radius=10, border_width=1, border_color=T.BORDER)
        card.pack(fill='both', expand=True, padx=14, pady=14)
        ctk.CTkLabel(card, text='Batch prediction from a spreadsheet', font=T.font(14, 'bold'),
                     text_color=T.TEXT).pack(anchor='w', padx=18, pady=(16, 6))
        ctk.CTkLabel(card, justify='left', wraplength=690, font=T.font(11), text_color=T.TEXT2, text=(
            'One frame per row with the 25 raw inputs in the notation of the template (sheet "inputs"); optional '
            'Ss and S1 columns switch the derived route to another site. The output workbook holds the inputs, the '
            'eight predicted quantities, the demand ratios, the derived route, the applicability flags (sheet '
            '"results") and the 45 model features (sheet "features").')).pack(anchor='w', padx=18)
        self.in_var, self.out_var = ctk.StringVar(), ctk.StringVar()
        for text, var, cmd in (('Input file', self.in_var, self._browse_in), ('Output file', self.out_var, self._browse_out)):
            row = ctk.CTkFrame(card, fg_color='transparent'); row.pack(fill='x', padx=18, pady=(12, 0))
            ctk.CTkLabel(row, text=text, width=80, anchor='w', font=T.font(11), text_color=T.TEXT).pack(side='left')
            ctk.CTkEntry(row, textvariable=var, width=480, font=T.font(11), fg_color='#FFFFFF', text_color=T.TEXT,
                         border_color='#D0D0D0').pack(side='left', padx=6)
            ctk.CTkButton(row, text='Browse', width=80, command=cmd, fg_color='#E0E0E0', hover_color='#CCCCCC',
                          text_color=T.TEXT, font=T.font(11)).pack(side='left')
        btns = ctk.CTkFrame(card, fg_color='transparent'); btns.pack(fill='x', padx=18, pady=(18, 6))
        ctk.CTkButton(btns, text='Write input template', command=self._template, fg_color='#E0E0E0',
                      hover_color='#CCCCCC', text_color=T.TEXT, font=T.font(11)).pack(side='left', padx=(0, 8))
        self.run_btn = ctk.CTkButton(btns, text='Run batch prediction', command=self._run, fg_color=T.ACCENT,
                                     hover_color=T.ACCENT_DARK, text_color='#FFFFFF', font=T.font(12, 'bold'))
        self.run_btn.pack(side='left', fill='x', expand=True)
        self.status = ctk.CTkLabel(card, text='', font=T.font(11), text_color=T.TEXT2, wraplength=690, justify='left')
        self.status.pack(anchor='w', padx=18, pady=(6, 12))

    def _browse_in(self):
        p = filedialog.askopenfilename(filetypes=[('Spreadsheet', '*.xlsx *.csv')])
        if p:
            self.in_var.set(p); self.out_var.set(str(Path(p).with_name(Path(p).stem + '_predictions.xlsx')))

    def _browse_out(self):
        p = filedialog.asksaveasfilename(defaultextension='.xlsx', filetypes=[('Excel', '*.xlsx')])
        if p:
            self.out_var.set(p)

    def _template(self):
        p = filedialog.asksaveasfilename(defaultextension='.xlsx', initialfile='batch_input_template.xlsx',
                                         filetypes=[('Excel', '*.xlsx')])
        if p:
            write_template(Path(p), self.presets); self.status.configure(text=f'Template written: {p}', text_color=T.OK)

    def _run(self):
        ip, op = self.in_var.get().strip(), self.out_var.get().strip()
        if not ip or not op:
            self.status.configure(text='Choose the input and output files.', text_color=T.WARN); return
        rows, problems = read_inputs(Path(ip))
        if problems:
            self.status.configure(text='; '.join(problems), text_color=T.ERR); return
        self.run_btn.configure(state='disabled'); self.update_idletasks()

        def prog(i, n):
            self.status.configure(text=f'Predicting {i}/{n} ...', text_color=T.TEXT2); self.update_idletasks()
        tables = run_batch(self.predictor, rows, prog)
        write_results(Path(op), tables)
        n_err = int((tables['results']['status'] != 'ok').sum())
        self.status.configure(text=f'Done: {len(rows)} rows written to {op}' + (f' ({n_err} errors)' if n_err else ''),
                              text_color=T.OK if n_err == 0 else T.WARN)
        self.run_btn.configure(state='normal')
