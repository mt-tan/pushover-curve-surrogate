"""Results: applicability badges, the numerical table (targets, demand ratios, derived route,
optional reference column for presets) and the per-target local SHAP bar."""
from __future__ import annotations

from typing import Dict, List, Optional

import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core.predictor import Flag, PredictionResult
from . import theme as T

# (key, unit, decimals); TARGET_ROWS / RATIO_ROWS / DERIVED_ROWS are also used by the PDF report
TARGET_ROWS = [('Vy', 'kN', 1), ('delta_y', 'm', 4), ('Vmax', 'kN', 1), ('delta_max', 'm', 4),
               ('V_at_dt', 'kN', 1), ('delta_t', 'm', 4), ('delta_u', 'm', 4), ('T1', 's', 3)]
RATIO_ROWS = [('theta_t', '-', 5), ('DCR_delta', '-', 4), ('mu_d', '-', 3)]
DERIVED_ROWS = [('delta_t_derived', 'm', 4), ('theta_t_derived', '-', 5), ('DCR_delta_derived', '-', 4),
                ('mu_d_derived', '-', 3), ('Te', 's', 3), ('CR', '-', 3)]
DERIVED_TEX = {'delta_t_derived': r'$\delta_t^{\mathrm{derived}}$', 'theta_t_derived': r'$\theta_t^{\mathrm{derived}}$',
               'DCR_delta_derived': r'$DCR_\delta^{\mathrm{derived}}$', 'mu_d_derived': r'$\mu_d^{\mathrm{derived}}$',
               'Te': r'$T_e$', 'CR': r'$C_R$',
               # short forms of the demand ratios (notation of the paper); the definition is in the tooltip
               'theta_t': r'$\theta_t$', 'DCR_delta': r'$DCR_\delta$', 'mu_d': r'$\mu_d$'}
RATIO_DEF = {'theta_t': 'θt = δt / Htot (roof drift ratio)', 'DCR_delta': 'DCRδ = δt / δu '
             '(clipped, curve-consistent δu; Table 13)', 'mu_d': 'μd = δt / δy (displacement ductility demand)'}
LEARNED_NOTE = 'Nurdagi DD-2 spectrum (training site)'
DERIVED_NOTE = 'TBEC 5.6.5 on the predicted curve'
# UI table layout: three labelled sections
UI_SECTIONS = [
    ('points', 'Characteristic points and period', [('Vy', 'kN', 1), ('delta_y', 'm', 4), ('Vmax', 'kN', 1),
                                                    ('delta_max', 'm', 4), ('delta_u', 'm', 4), ('T1', 's', 3)]),
    ('learned', f'Demand point, learned: {LEARNED_NOTE}', [('delta_t', 'm', 4), ('V_at_dt', 'kN', 1),
                                                           ('theta_t', '-', 5), ('DCR_delta', '-', 4), ('mu_d', '-', 3)]),
    ('derived', f'Derived route, {DERIVED_NOTE} (any site)', [('delta_t_derived', 'm', 4), ('DCR_delta_derived', '-', 4),
                                                              ('Te', 's', 3)]),
]
SITE_DEPENDENT = {'delta_t', 'V_at_dt', 'theta_t', 'DCR_delta', 'mu_d'}

EXPLAIN: Dict[str, str] = {
    'shear': 'The shear screen compares the peak base-shear demand per column and the capacity-design beam '
             'shear with the TS 500 resistance; the training ground truth is flexure-governed, so a '
             'shear-governed frame lies outside the validated domain.',
    'buckling': 'Stirrup spacing above six bar diameters makes bar buckling plausible, which the fibre model '
                'of the training analyses does not represent.',
    'col_is_tbdy_confined': 'Column confinement zone satisfies TBEC 7.3.4.1 (spacing, leg spacing, Eq. 7.1); '
                            'non-compliant detailing was sampled on purpose and is inside the training domain.',
    'beam_is_tbdy_confined': 'Beam confinement zone satisfies the TBEC 7.4.4 spacing cap; non-compliant '
                             'detailing is inside the training domain.',
    'envelope': 'At least one of the 45 features lies outside the realised range of the training pool '
                '(51,116 frames; the locked hold-out is excluded), where the tree ensemble can only '
                'extrapolate by clamping.',
    'tbdy': 'One of the seven generation-time design filters (Table 2) is violated, so no comparable frame '
            'was ever analysed.',
    'dcr': 'DCRδ = δt / δu has three states: ok below 0.90, near 1 from 0.90 to 1.10 (small prediction '
           'errors can move the demand point across δu) and above 1.10 (the demand exceeds the predicted '
           'ultimate displacement).',
    'clip': 'The 1.001 monotonic clip rule of Section 3.4 raised a later characteristic point so that the '
            'curve stays ordered.',
    'site': 'The learned δt and V(δt) were trained for the Nurdagi DD-2 spectrum only; for another '
            'site use the derived route.',
    'input': 'An input is outside the discrete choices used in generation.',
}
BADGE_STYLE = {'ok': (T.OK_BG, T.OK), 'warning': (T.WARN_BG, T.WARN), 'info': (T.INFO_BG, T.INFO)}


def fmt(v: float, dec: int) -> str:
    return '-' if v is None or not np.isfinite(v) else f'{v:,.{dec}f}'


# ---------------------------------------------------------------------------- badges
class Badges(ctk.CTkFrame):
    """Pill badges that wrap into as many rows as needed."""

    def __init__(self, master, max_width: int = 690):
        super().__init__(master, fg_color='transparent')
        self.max_width = max_width
        self._items: List[ctk.CTkLabel] = []
        self.placeholder = ctk.CTkLabel(self, text='Applicability badges appear here after Predict',
                                        text_color=T.MUTED, font=T.font(11))
        self.placeholder.grid(row=0, column=0, sticky='w', padx=4)

    def show(self, flags: List[Flag]) -> None:
        for w in self._items:
            w.destroy()
        self._items.clear()
        self.placeholder.grid_forget()
        from customtkinter.windows.widgets.scaling import ScalingTracker
        s = ScalingTracker.get_widget_scaling(self)
        self.update_idletasks()
        limit = (self.winfo_width() if self.winfo_width() > 100 else int(self.max_width * s)) - int(24 * s)
        x = 0; row = 0; col = 0
        import tkinter.font as tkfont
        for f in flags:
            bg, fg = BADGE_STYLE.get(f.level, (T.GREY_BG, T.TEXT2))
            state = {'ok': 'ok', 'warning': 'warning', 'info': 'note'}[f.level]
            weight = 'bold' if f.level == 'warning' else 'normal'
            text = f'{f.label}: {state}'
            lab = ctk.CTkLabel(self, text=text, fg_color=bg, text_color=fg, corner_radius=10,
                               font=T.font(11, weight), padx=8, pady=2)
            # measure with the font at its scaled pixel size (CTkFont.measure uses the unscaled size)
            meas = tkfont.Font(family=T.FONT_FAMILY, size=-int(round(11 * s)), weight=weight)
            w = meas.measure(text) + int(28 * s)
            if x + w > limit and col > 0:
                row += 1; col = 0; x = 0
            lab.grid(row=row, column=col, padx=3, pady=2, sticky='w')
            x += w; col += 1
            T.Tooltip(lab, f'{f.message}. {EXPLAIN.get(f.key, "")}'.replace('..', '.'))
            self._items.append(lab)


# ---------------------------------------------------------------------------- table
class ResultsTable(ctk.CTkFrame):
    REF_HEADER = 'reference (locked hold-out)'

    def __init__(self, master):
        super().__init__(master, fg_color=T.PANEL, corner_radius=10, border_width=1, border_color=T.BORDER)
        ctk.CTkLabel(self, text='Predicted quantities', font=T.font(13, 'bold'), text_color=T.TEXT).pack(
            anchor='w', padx=12, pady=(6, 0))
        self.grid_frame = ctk.CTkScrollableFrame(self, fg_color=T.PANEL, corner_radius=0)
        self.grid_frame.pack(fill='both', expand=True, padx=4, pady=(0, 4))
        for c, w in ((0, 110), (1, 28), (2, 84), (3, 92)):
            self.grid_frame.grid_columnconfigure(c, minsize=T.px(self, w), weight=0)
        self.cells: Dict[str, Dict[str, ctk.CTkLabel]] = {}
        self.tips: Dict[str, T.Tooltip] = {}
        self.section_labels: Dict[str, ctk.CTkLabel] = {}
        self._row = 0
        self._header()
        for sec_key, title, rows in UI_SECTIONS:
            self._section(sec_key, title)
            for k, u, _ in rows:
                self._add_row(k, DERIVED_TEX.get(k, T.tex(k)), u)

    def _header(self):
        for c, txt in enumerate(('Quantity', 'Unit', 'Predicted', 'reference\n(locked hold-out)')):
            ctk.CTkLabel(self.grid_frame, text=txt, font=T.font(10, 'bold'), text_color=T.TEXT2, justify='right',
                         anchor='sw' if c < 2 else 'se').grid(row=self._row, column=c, sticky='ews', padx=4)
        self._row += 1

    def _section(self, key: str, title: str):
        lab = ctk.CTkLabel(self.grid_frame, text=title, font=T.font(10, 'bold'), text_color=T.ACCENT_DARK, anchor='w',
                           wraplength=318, justify='left', height=20)
        lab.grid(row=self._row, column=0, columnspan=4, sticky='w', padx=4, pady=(3, 0))
        self.section_labels[key] = lab
        self._row += 1

    def _add_row(self, key: str, tex: str, unit: str):
        cell = {}
        cell['label'] = T.MathLabel(self.grid_frame, tex, pt=9.5, anchor='w', height=20)
        cell['label'].grid(row=self._row, column=0, sticky='w', padx=(10, 4))
        cell['unit'] = ctk.CTkLabel(self.grid_frame, text=unit if unit != '-' else '', font=T.font(10),
                                    text_color=T.TEXT2, anchor='w', height=20)
        cell['unit'].grid(row=self._row, column=1, sticky='w', padx=4)
        cell['pred'] = ctk.CTkLabel(self.grid_frame, text='-', font=T.font(11), text_color=T.TEXT, anchor='e', height=20)
        cell['pred'].grid(row=self._row, column=2, sticky='e', padx=4)
        cell['ref'] = ctk.CTkLabel(self.grid_frame, text='-', font=T.font(11), text_color=T.TEXT2, anchor='e', height=20)
        cell['ref'].grid(row=self._row, column=3, sticky='e', padx=4)
        self.cells[key] = cell
        self.tips[key] = T.Tooltip(cell['pred'], '')
        self._row += 1

    def show(self, r: PredictionResult, preset: Optional[dict] = None, site_transfer: bool = False) -> None:
        vals = dict(r.targets); vals.update(r.ratios)
        vals.update({'delta_t_derived': r.derived['delta_t'], 'theta_t_derived': r.derived_ratios['theta_t'],
                     'DCR_delta_derived': r.derived_ratios['DCR_delta'], 'mu_d_derived': r.derived_ratios['mu_d'],
                     'Te': r.derived['Te'], 'CR': r.derived['CR']})
        ref = (preset or {}).get('reference', {}) if preset else {}
        for sec_key, title, rows in UI_SECTIONS:
            for k, u, dec in rows:
                c = self.cells[k]
                c['pred'].configure(text=fmt(vals[k], dec))
                c['ref'].configure(text=fmt(ref[k], dec) if k in ref else '-')
                grey = site_transfer and k in SITE_DEPENDENT
                hi = site_transfer and sec_key == 'derived'
                c['pred'].configure(text_color=T.MUTED if grey else (T.ACCENT_DARK if hi else T.TEXT),
                                    font=T.font(11, 'bold' if hi else 'normal'))
                c['label'].set_math(DERIVED_TEX.get(k, T.tex(k)),
                                    color=T.MUTED if grey else (T.ACCENT_DARK if hi else T.TEXT))
                if grey:
                    tip = ('Greyed: the learned value is valid for the Nurdagi DD-2 spectrum (training site) only; '
                           'the site-transfer result is the derived route below.')
                elif k in ('delta_t', 'V_at_dt'):
                    tip = 'Learned by the dedicated regressor for the Nurdagi DD-2 spectrum (training site).'
                elif k == 'delta_t_derived':
                    tip = (f'TBEC 5.6.5 single-mode procedure on the predicted points with the {r.site["name"]} '
                           f'spectrum: Te = {r.derived["Te"]:.3f} s, Sae = {r.derived["Sae_Te_g"]:.3f} g, '
                           f'Ry = {r.derived["Ry"]:.3f}, CR = {r.derived["CR"]:.3f}, Γ1 = {r.derived["Gamma1"]:.3f}; '
                           f'θt = {r.derived_ratios["theta_t"]:.5f}, μd = {r.derived_ratios["mu_d"]:.3f}.')
                elif k in RATIO_DEF:
                    tip = RATIO_DEF[k]
                else:
                    tip = ''
                self.tips[k].set_text(tip)
        self.section_labels['learned'].configure(
            text=f'Demand point, learned: {LEARNED_NOTE}' + ('  [greyed: site transfer active]' if site_transfer else ''),
            text_color=T.MUTED if site_transfer else T.ACCENT_DARK)
        self.section_labels['derived'].configure(
            text=f'Derived route, {DERIVED_NOTE}: {r.site["name"]}' if site_transfer else
                 f'Derived route, {DERIVED_NOTE} (any site)')


# ---------------------------------------------------------------------------- SHAP
def draw_shap(ax, r: PredictionResult, target: str, pt: float = 9.5) -> None:
    T.style_axes(ax, pt)
    s = r.shap[target]
    top = list(reversed(s['top']))
    names = [T.tex(n) for n, _ in top]; vals = [v for _, v in top]
    colors = [T.C_SHAP_POS if v >= 0 else T.C_SHAP_NEG for v in vals]
    ypos = np.arange(len(top))
    ax.barh(ypos, vals, color=colors, edgecolor='white', lw=0.5, height=0.7)
    ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=pt)
    ax.axvline(0, color='#444444', lw=0.7)
    span = max(abs(v) for v in vals) if vals else 1.0
    for y, v in zip(ypos, vals):
        ax.text(v + (0.03 * span if v >= 0 else -0.03 * span), y, f'{v:+.3f}', va='center',
                ha='left' if v >= 0 else 'right', fontsize=pt - 1.5, color=T.TEXT2)
    ax.set_xlim(-1.4 * span, 1.4 * span)
    unit = {'log1p': 'log(1 + V)', 'lognormH': r'log($\delta$ / $H_{\mathrm{tot}}$)', 'raw': 's'}[s['transform']]
    ax.set_xlabel(f'SHAP contribution [{unit}]')
    ax.set_title(f'{T.tex(target)}: base {s["base_value"]:+.3f}, prediction {s["prediction_t"]:+.3f}',
                 fontsize=pt + 0.5, color=T.TEXT)


class ShapPanel(ctk.CTkFrame):
    def __init__(self, master, targets: List[str], figsize=(3.0, 2.7)):
        super().__init__(master, fg_color=T.PANEL, corner_radius=10, border_width=1, border_color=T.BORDER)
        head = ctk.CTkFrame(self, fg_color='transparent'); head.pack(fill='x', padx=10, pady=(6, 0))
        ctk.CTkLabel(head, text='Local SHAP (top 8)', font=T.font(13, 'bold'), text_color=T.TEXT).pack(side='left')
        self.targets = targets
        self.names = {T.uni(t): t for t in targets}
        self.sel = ctk.StringVar(value=T.uni('Vmax'))
        self.menu = ctk.CTkOptionMenu(head, values=list(self.names), variable=self.sel, width=96,
                                      command=lambda *_: self._redraw(), font=T.font(11),
                                      fg_color='#F5F5F5', button_color='#E0E0E0', button_hover_color='#CCCCCC',
                                      text_color=T.TEXT, dropdown_fg_color='#FFFFFF', dropdown_text_color=T.TEXT)
        self.menu.pack(side='right')
        T.Tooltip(self.menu, 'Target of the local explanation (CatBoost native TreeSHAP, transformed space).')
        self.fig = Figure(figsize=figsize, dpi=T.fig_dpi(), facecolor=T.PANEL)
        self.ax = self.fig.add_subplot(111); T.style_axes(self.ax)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.canvas = T.make_canvas(self.fig, self); self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=(2, 4))
        self.result: Optional[PredictionResult] = None

    def show(self, r: PredictionResult) -> None:
        self.result = r; self._redraw()

    def _redraw(self) -> None:
        if self.result is None or not self.result.shap:
            return
        self.ax.clear()
        target = self.names[self.sel.get()]
        draw_shap(self.ax, self.result, target, pt=8.5)
        self.fig.tight_layout(pad=0.4)
        self.canvas.draw_idle()


# ---------------------------------------------------------------------------- panel
class ResultPanel(ctk.CTkFrame):
    def __init__(self, master, targets: List[str]):
        super().__init__(master, fg_color='transparent')
        self.badges = Badges(self)
        self.badges.pack(fill='x', padx=2, pady=(0, 4))
        body = ctk.CTkFrame(self, fg_color='transparent'); body.pack(fill='both', expand=True)
        body.grid_columnconfigure(0, weight=0, minsize=T.px(body, 392)); body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)
        self.table = ResultsTable(body); self.table.grid(row=0, column=0, sticky='nsew', padx=(0, 4))
        self.shap = ShapPanel(body, targets); self.shap.grid(row=0, column=1, sticky='nsew', padx=(4, 0))

    def show(self, r: PredictionResult, preset: Optional[dict] = None, site_transfer: bool = False) -> None:
        self.badges.show(r.flags)
        self.table.show(r, preset, site_transfer)
        self.shap.show(r)
