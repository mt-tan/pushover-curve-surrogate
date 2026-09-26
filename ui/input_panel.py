"""Left-hand input panel: the raw layer of Table A1 in six sections with the paper notation,
realised-range hints, live derived quantities, six presets, the collapsible site-transfer box
and the Predict / Reset / Batch buttons."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import customtkinter as ctk
import numpy as np

from core import features as F
from core import spectrum as SP
from core.validator import check_envelope
from . import theme as T

INT_FMT = '{:d}'
SECTIONS = [
    ('Geometry', [('N_story', 'int', None, INT_FMT), ('N_bay', 'int', None, INT_FMT),
                  ('bay_lengths', 'list', None, '{:.2f}'), ('h1', 'float', None, '{:.3f}'),
                  ('htyp', 'float', None, '{:.3f}')]),
    ('Sections', [('col_B', 'float', None, '{:.3f}'), ('beam_b', 'float', None, '{:.3f}'),
                  ('beam_h', 'float', None, '{:.3f}'), ('cover', 'float', None, '{:.4f}')]),
    ('Longitudinal reinforcement', [('phi_long_col_mm', 'choice', ['14', '16', '18', '20', '22'], '{:.0f}'),
                                    ('col_n_long', 'int', None, INT_FMT),
                                    ('phi_long_beam_mm', 'choice', ['12', '14', '16', '18', '20'], '{:.0f}'),
                                    ('beam_n_long_top', 'int', None, INT_FMT), ('beam_n_long_bot', 'int', None, INT_FMT)]),
    ('Transverse reinforcement', [('col_phi_etr', 'choice', ['0.008', '0.010'], '{:.3f}'),
                                  ('col_s_etr', 'float', None, '{:.3f}'),
                                  ('col_n_etr_x', 'choice', ['2', '3', '4'], INT_FMT),
                                  ('col_n_etr_y', 'choice', ['2', '3', '4'], INT_FMT),
                                  ('beam_phi_etr', 'choice', ['0.008', '0.010'], '{:.3f}'),
                                  ('beam_s_etr', 'float', None, '{:.3f}')]),
    ('Materials and loads', [('fc_MPa', 'float', None, '{:.2f}'), ('fy_MPa', 'choice', ['420', '500'], '{:.0f}'),
                             ('G', 'float', None, '{:.2f}'), ('Q', 'float', None, '{:.2f}')]),
    ('Site', [('soil_class', 'choice', SP.SOIL_CLASSES, '{}')]),
]
INT_KEYS = {'N_story', 'N_bay', 'col_n_long', 'beam_n_long_top', 'beam_n_long_bot', 'col_n_etr_x', 'col_n_etr_y'}
ENVELOPE_MAP = {'bay_lengths': ['L_mean', 'L_max', 'L_min', 'W_total'], 'col_B': ['col_B', 'col_H']}
LIVE_DERIVED = ['rho_col', 'rho_beam_top', 'rho_beam_bot', 'omega_we_col', 'alpha_se_col', 'omega_we_beam']
DEFAULTS: Dict[str, object] = {
    'N_story': 4, 'N_bay': 3, 'bay_lengths': [5.0, 5.0, 5.0], 'h1': 3.5, 'htyp': 3.0,
    'col_B': 0.5, 'beam_b': 0.3, 'beam_h': 0.55, 'cover': 0.03,
    'phi_long_col_mm': 18.0, 'col_n_long': 5, 'phi_long_beam_mm': 16.0, 'beam_n_long_top': 4, 'beam_n_long_bot': 3,
    'col_phi_etr': 0.010, 'col_s_etr': 0.10, 'col_n_etr_x': 3, 'col_n_etr_y': 3, 'beam_phi_etr': 0.008,
    'beam_s_etr': 0.12, 'fc_MPa': 30.0, 'fy_MPa': 420.0, 'G': 25.0, 'Q': 10.0, 'soil_class': 'ZC'}


def _fmt_list(L, fmt='{:.3f}') -> str:
    return '; '.join(fmt.format(float(x)) for x in L)


def _parse_list(s: str) -> List[float]:
    s = s.replace(',', ';').replace(' ', ';')
    return [float(x) for x in s.split(';') if x.strip()]


class InputPanel(ctk.CTkFrame):
    def __init__(self, master, examples_dir: Path, feature_spec: Optional[dict] = None,
                 on_change: Optional[Callable[[Dict], None]] = None,
                 on_predict: Optional[Callable[[Dict, Optional[dict], Optional[dict]], None]] = None,
                 on_batch: Optional[Callable[[], None]] = None):
        super().__init__(master, fg_color=T.PANEL, corner_radius=10, border_width=1, border_color=T.BORDER)
        self.examples_dir = examples_dir
        self.feature_spec = feature_spec
        self.on_change, self.on_predict, self.on_batch = on_change, on_predict, on_batch
        self.vars: Dict[str, ctk.StringVar] = {}
        self.widgets: Dict[str, ctk.CTkBaseClass] = {}
        self.dots: Dict[str, ctk.CTkLabel] = {}
        self.dot_tips: Dict[str, T.Tooltip] = {}
        self.live: Dict[str, ctk.CTkLabel] = {}
        self.presets: List[dict] = []
        self.preset: Optional[dict] = None          # active preset (exact values used for prediction)
        self._loading = False
        self._ready = False
        self._load_presets()
        self._build()
        self.reset()
        self._ready = True

    # ------------------------------------------------------------------ build
    def _load_presets(self):
        for p in sorted(self.examples_dir.glob('0*.json')):
            with open(p, encoding='utf-8') as fh:
                self.presets.append(json.load(fh))

    def _build(self):
        head = ctk.CTkFrame(self, fg_color='transparent'); head.pack(fill='x', padx=12, pady=(10, 4))
        ctk.CTkLabel(head, text='Frame inputs (raw layer of Table A1)', font=T.font(13, 'bold'),
                     text_color=T.TEXT).pack(side='left')
        row = ctk.CTkFrame(self, fg_color='transparent'); row.pack(fill='x', padx=12, pady=(0, 6))
        ctk.CTkLabel(row, text='Preset', font=T.font(11), text_color=T.TEXT2).pack(side='left', padx=(0, 6))
        self.preset_var = ctk.StringVar(value='Custom')
        names = ['Custom'] + [f'{i + 1}. {p["title"]}' for i, p in enumerate(self.presets)]
        self.preset_menu = ctk.CTkOptionMenu(row, values=names, variable=self.preset_var, command=self._preset_chosen,
                                             width=300, font=T.font(11), fg_color='#F5F5F5', button_color='#E0E0E0',
                                             button_hover_color='#CCCCCC', text_color=T.TEXT,
                                             dropdown_fg_color='#FFFFFF', dropdown_text_color=T.TEXT)
        self.preset_menu.pack(side='left', fill='x', expand=True)
        self.preset_note = ctk.CTkLabel(self, text='', font=T.font(10), text_color=T.MUTED, wraplength=380,
                                        justify='left', anchor='w')
        self.preset_note.pack(fill='x', padx=12)

        self.scroll = ctk.CTkScrollableFrame(self, fg_color=T.INNER, border_width=1, border_color=T.BORDER)
        self.scroll.pack(fill='both', expand=True, padx=10, pady=(4, 6))
        for title, fields in SECTIONS:
            ctk.CTkLabel(self.scroll, text=title, font=T.font(11, 'bold'), text_color=T.ACCENT_DARK,
                         anchor='w').pack(fill='x', padx=6, pady=(8, 2))
            if title == 'Sections':
                ctk.CTkLabel(self.scroll, text='Columns are square: the depth equals the side entered below '
                                               '(a separate column depth is not accepted).',
                             font=T.font(9), text_color=T.MUTED, anchor='w', wraplength=360,
                             justify='left').pack(fill='x', padx=6, pady=(0, 2))
            for key, kind, choices, fmt in fields:
                self._add_field(self.scroll, key, kind, choices, fmt)
            if title == 'Site':
                self._add_site_transfer(self.scroll)
        self._add_live_strip(self.scroll)

        self.banner = ctk.CTkLabel(self, text='', font=T.font(11), fg_color=T.GREY_BG, text_color=T.TEXT,
                                   corner_radius=8, height=30, wraplength=380)
        self.banner.pack(fill='x', padx=12, pady=(0, 6))
        btns = ctk.CTkFrame(self, fg_color='transparent'); btns.pack(fill='x', padx=12, pady=(0, 10))
        self.predict_btn = ctk.CTkButton(btns, text='Predict', height=40, font=T.font(14, 'bold'),
                                         fg_color=T.ACCENT, hover_color=T.ACCENT_DARK, text_color='#FFFFFF',
                                         command=self._predict_clicked)
        self.predict_btn.pack(fill='x', pady=(0, 6))
        sub = ctk.CTkFrame(btns, fg_color='transparent'); sub.pack(fill='x')
        ctk.CTkButton(sub, text='Reset', command=self.reset, fg_color='#E0E0E0', hover_color='#CCCCCC',
                      text_color=T.TEXT, font=T.font(11)).pack(side='left', expand=True, fill='x', padx=(0, 4))
        ctk.CTkButton(sub, text='Batch (xlsx)', command=lambda: self.on_batch and self.on_batch(),
                      fg_color='#E0E0E0', hover_color='#CCCCCC', text_color=T.TEXT, font=T.font(11)
                      ).pack(side='left', expand=True, fill='x', padx=(4, 0))

    def _hint(self, key: str) -> str:
        if self.feature_spec is None:
            return ''
        spec = {f['name']: f for f in self.feature_spec['features']}
        keys = ENVELOPE_MAP.get(key, [key])
        if key == 'bay_lengths':
            lo, hi = spec['L_min']['realised']['min'], spec['L_max']['realised']['max']
        elif keys[0] in spec:
            lo, hi = spec[keys[0]]['realised']['min'], spec[keys[0]]['realised']['max']
        else:
            return ''
        return f'[{lo:.4g}, {hi:.4g}]'

    def _add_field(self, parent, key, kind, choices, fmt):
        row = ctk.CTkFrame(parent, fg_color='transparent'); row.pack(fill='x', padx=6, pady=2)
        lab = T.MathLabel(row, T.with_unit(T.tex(key), T.unit(key)), pt=10.5, width=150 if kind != 'list' else 52,
                          anchor='w')
        lab.pack(side='left')
        L = T.labels()
        T.Tooltip(lab, L['raw_input_descriptions'].get(key, key) + (
            '. Columns are square: the depth equals this side; a separate column depth is never accepted.'
            if key == 'col_B' else ''))
        var = ctk.StringVar(); self.vars[key] = var
        if kind == 'choice':
            w = ctk.CTkOptionMenu(row, values=choices, variable=var, width=96, font=T.font(11), fg_color='#FFFFFF',
                                  button_color='#E0E0E0', button_hover_color='#CCCCCC', text_color=T.TEXT,
                                  dropdown_fg_color='#FFFFFF', dropdown_text_color=T.TEXT,
                                  command=lambda *_: self._changed(key))
        else:
            w = ctk.CTkEntry(row, textvariable=var, width=228 if kind == 'list' else 96, font=T.font(11),
                             fg_color='#FFFFFF', text_color=T.TEXT, border_color='#D0D0D0')
            var.trace_add('write', lambda *_: self._changed(key))
        w.pack(side='left', padx=(4, 4)); self.widgets[key] = w
        hint = ctk.CTkLabel(row, text=self._hint(key), font=T.font(9), text_color=T.MUTED, anchor='w')
        hint.pack(side='left', padx=(2, 2))
        dot = ctk.CTkLabel(row, text='●', width=14, text_color=T.OK, font=T.font(13, 'bold'))
        dot.pack(side='left'); self.dots[key] = dot
        self.dot_tips[key] = T.Tooltip(dot, 'inside the training envelope')
        if key == 'bay_lengths':
            T.Tooltip(w, 'One length for all bays, or one value per bay separated by semicolons.')

    def _add_site_transfer(self, parent):
        box = ctk.CTkFrame(parent, fg_color='transparent'); box.pack(fill='x', padx=6, pady=(2, 2))
        self.site_open = ctk.BooleanVar(value=False)
        self.site_btn = ctk.CTkButton(box, text='▸  Site transfer (Ss, S1)', anchor='w', fg_color='#EEF1F5',
                                      hover_color='#E2E6EB', text_color=T.TEXT, font=T.font(11),
                                      command=self._toggle_site)
        self.site_btn.pack(fill='x')
        T.Tooltip(self.site_btn, 'Enter the map coefficients of another site; the derived delta_t is then '
                                 'evaluated for that spectrum while the learned demand stays tied to Nurdagi DD-2.')
        self.site_frame = ctk.CTkFrame(box, fg_color='#F7F9FB', border_width=1, border_color=T.BORDER)
        ctk.CTkLabel(self.site_frame, text=f'Training site: Gaziantep Nurdagi, DD-2  (Ss = {SP.SS_NURDAGI}, S1 = {SP.S1_NURDAGI})',
                     font=T.font(10), text_color=T.TEXT2, anchor='w').pack(fill='x', padx=8, pady=(6, 2))
        for key, tex, default in (('Ss', r'$S_S$ (short-period map coefficient)', SP.SS_NURDAGI),
                                  ('S1', r'$S_1$ (1.0 s map coefficient)', SP.S1_NURDAGI)):
            r = ctk.CTkFrame(self.site_frame, fg_color='transparent'); r.pack(fill='x', padx=8, pady=2)
            T.MathLabel(r, tex, pt=10, width=230, anchor='w').pack(side='left')
            var = ctk.StringVar(value=f'{default:g}'); self.vars[key] = var
            e = ctk.CTkEntry(r, textvariable=var, width=80, font=T.font(11), fg_color='#FFFFFF', text_color=T.TEXT,
                             border_color='#D0D0D0')
            e.pack(side='left', padx=4); self.widgets[key] = e
            var.trace_add('write', lambda *_: self._changed('site'))
        ctk.CTkButton(self.site_frame, text='Reset to training site', width=150, fg_color='#E0E0E0',
                      hover_color='#CCCCCC', text_color=T.TEXT, font=T.font(10),
                      command=self._reset_site).pack(anchor='w', padx=8, pady=(2, 6))
        self.site_status = ctk.CTkLabel(self.site_frame, text='', font=T.font(10), text_color=T.INFO, anchor='w',
                                        wraplength=340, justify='left')
        self.site_status.pack(fill='x', padx=8, pady=(0, 6))

    def _add_live_strip(self, parent):
        ctk.CTkLabel(parent, text='Derived (live)', font=T.font(11, 'bold'), text_color=T.ACCENT_DARK,
                     anchor='w').pack(fill='x', padx=6, pady=(8, 2))
        grid = ctk.CTkFrame(parent, fg_color='transparent'); grid.pack(fill='x', padx=6, pady=(0, 6))
        for i, key in enumerate(LIVE_DERIVED):
            r, c = divmod(i, 2)
            cell = ctk.CTkFrame(grid, fg_color='transparent'); cell.grid(row=r, column=c, sticky='w', padx=(0, 12))
            T.MathLabel(cell, T.tex(key), pt=10).pack(side='left')
            lab = ctk.CTkLabel(cell, text='-', font=T.font(11), text_color=T.TEXT, width=70, anchor='w')
            lab.pack(side='left', padx=(4, 0)); self.live[key] = lab
        self.live_flags = ctk.CTkLabel(parent, text='', font=T.font(10), text_color=T.TEXT2, anchor='w',
                                       wraplength=380, justify='left')
        self.live_flags.pack(fill='x', padx=6, pady=(0, 8))

    # ------------------------------------------------------------------ site transfer
    def _toggle_site(self):
        if self.site_open.get():
            self.site_frame.pack_forget(); self.site_open.set(False)
            self.site_btn.configure(text='▸  Site transfer (Ss, S1)')
        else:
            self.site_frame.pack(fill='x', pady=(2, 0)); self.site_open.set(True)
            self.site_btn.configure(text='▾  Site transfer (Ss, S1)')

    def _reset_site(self):
        self.vars['Ss'].set(f'{SP.SS_NURDAGI:g}'); self.vars['S1'].set(f'{SP.S1_NURDAGI:g}')

    def get_site(self) -> Optional[dict]:
        """None for the training site, else the site dict of core.spectrum for the user's Ss / S1."""
        try:
            Ss = float(self.vars['Ss'].get()); S1 = float(self.vars['S1'].get())
        except ValueError:
            return None
        if abs(Ss - SP.SS_NURDAGI) < 1e-12 and abs(S1 - SP.S1_NURDAGI) < 1e-12:
            return None
        return SP.site_parameters(self.vars['soil_class'].get(), Ss, S1)

    # ------------------------------------------------------------------ values
    def set_values(self, values: Dict[str, object]):
        self._loading = True
        try:
            for _, fields in SECTIONS:
                for key, kind, choices, fmt in fields:
                    v = values[key]
                    if kind == 'list':
                        self.vars[key].set(_fmt_list(v, fmt))
                    elif kind == 'choice':
                        self.vars[key].set(str(v) if key == 'soil_class' else
                                           (fmt.format(float(v)) if fmt != INT_FMT else str(int(v))))
                    elif kind == 'int':
                        self.vars[key].set(str(int(v)))
                    else:
                        self.vars[key].set(fmt.format(float(v)))
        finally:
            self._loading = False

    def get_values(self) -> Dict[str, object]:
        """Raw inputs for core.predictor.Predictor.predict (exact preset values while a preset is active)."""
        if self.preset is not None:
            v = dict(self.preset['inputs'])
            v['bay_lengths_display'] = list(v['bay_lengths'])
            v['bay_lengths'] = None
            v.update(self.preset['span_statistics_exact'])
            return v
        v: Dict[str, object] = {}
        for _, fields in SECTIONS:
            for key, kind, choices, fmt in fields:
                s = self.vars[key].get().strip()
                if kind == 'list':
                    v[key] = _parse_list(s)
                elif key == 'soil_class':
                    v[key] = s
                elif key in INT_KEYS:
                    v[key] = int(float(s))
                else:
                    v[key] = float(s)
        if len(v['bay_lengths']) == 1:
            v['bay_lengths'] = v['bay_lengths'] * int(v['N_bay'])
        v['bay_lengths_display'] = list(v['bay_lengths'])
        return v

    def reset(self):
        self.preset = None; self.preset_var.set('Custom'); self.preset_note.configure(text='')
        self.set_values(DEFAULTS)
        self._reset_site()
        self._after_change()

    def load_preset(self, index: int):
        p = self.presets[index]
        self.set_values(p['inputs'])
        self.preset = p
        self.preset_var.set(f'{index + 1}. {p["title"]}')
        self.preset_note.configure(text=p['description'] + '. Exact hold-out values are used until a field is edited.')
        self._after_change()

    def _preset_chosen(self, name: str):
        if name == 'Custom':
            self.preset = None; self.preset_note.configure(text=''); self._after_change(); return
        idx = int(name.split('.')[0]) - 1
        self.load_preset(idx)

    # ------------------------------------------------------------------ change handling
    def _changed(self, key: str):
        if self._loading or not self._ready:
            return
        if key != 'site' and self.preset is not None:
            self.preset = None; self.preset_var.set('Custom')
            self.preset_note.configure(text='Edited: exact hold-out values dropped, the reference column is cleared.')
        self._after_change()

    def _after_change(self):
        try:
            v = self.get_values()
        except (ValueError, KeyError):
            self.banner.configure(text='Some fields cannot be read as numbers.', fg_color='#FADBD8', text_color=T.ERR)
            return
        try:
            feats = F.compute_features(v)
        except Exception as exc:                        # unreadable geometry (e.g. N_bay mismatch)
            self.banner.configure(text=f'Inputs incomplete: {exc}', fg_color='#FADBD8', text_color=T.ERR)
            return
        f1 = {k: float(np.asarray(x).reshape(-1)[0]) for k, x in feats.items()}
        for key, lab in self.live.items():
            lab.configure(text=f'{f1[key]:.4f}' if f1[key] < 1 else f'{f1[key]:.2f}')
        self.live_flags.configure(text=('Column TBEC 7.3.4.1 confinement: ' + ('compliant' if f1['col_is_tbdy_confined'] > 0.5 else 'non-compliant')
                                        + ';  beam 7.4.4: ' + ('compliant' if f1['beam_is_tbdy_confined'] > 0.5 else 'non-compliant')))
        env = check_envelope(feats, self.feature_spec) if self.feature_spec else []
        bad = {r.rule.split(':', 1)[1] for r in env}
        n_out = 0
        for key, dot in self.dots.items():
            keys = ENVELOPE_MAP.get(key, [key])
            out = [k for k in keys if k in bad]
            if out:
                n_out += 1
                dot.configure(text_color=T.WARN)
                self.dot_tips[key].set_text('outside the training envelope: ' + ', '.join(T.uni(k) for k in out))
            else:
                dot.configure(text_color=T.OK); self.dot_tips[key].set_text('inside the training envelope')
        site = self.get_site()
        if site is not None:
            self.site_status.configure(text=f'Site transfer active: SDS = {site["SDS"]:.3f}, SD1 = {site["SD1"]:.3f}, '
                                            f'TB = {site["TB"]:.3f} s. The derived route is highlighted after Predict.')
        else:
            self.site_status.configure(text='Training site active (learned demand valid).')
        if n_out == 0:
            self.banner.configure(text='All inputs inside the training envelope.', fg_color=T.OK_BG, text_color=T.OK)
        else:
            self.banner.configure(text=f'{n_out} input(s) outside the training envelope (prediction allowed, flagged).',
                                  fg_color=T.WARN_BG, text_color=T.WARN)
        if self.on_change:
            self.on_change(v)

    def _predict_clicked(self):
        if self.on_predict:
            try:
                v = self.get_values()
            except (ValueError, KeyError) as exc:
                self.banner.configure(text=f'Cannot read inputs: {exc}', fg_color='#FADBD8', text_color=T.ERR)
                return
            self.on_predict(v, self.get_site(), self.preset)
