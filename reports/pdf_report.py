"""PDF report (fpdf2, DejaVu Sans from matplotlib): inputs, features, results (with the
locked hold-out reference for presets), the capacity-curve figure, the SHAP figures, the
applicability flags and the provenance of the model artifacts.

Notation: every quantity label is rendered from the same mathtext strings as the Tk widgets
(assets/labels.json through ui.theme.render_math) and placed as a small transparent image, so
subscripts, superscripts and Greek letters appear exactly as in the paper. The flag and
validation messages are text with real subscripts and superscripts (reports/richtext.py, fpdf2
char_vpos); headers and plain words stay as text.
"""
from __future__ import annotations

import io
import itertools
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.font_manager as fm
from fpdf import FPDF
from matplotlib.figure import Figure

from core import features as F
from core.predictor import PredictionResult
from reports.richtext import rich_width, write_rich
from ui import theme as T
from ui.curve_panel import draw_curve
from ui.result_panel import DERIVED_TEX, RATIO_ROWS, TARGET_ROWS, draw_shap, fmt

FONT = 'DejaVu'
INPUT_KEYS = ['N_story', 'N_bay', 'bay_lengths', 'h1', 'htyp', 'col_B', 'beam_b', 'beam_h', 'cover',
              'phi_long_col_mm', 'col_n_long', 'phi_long_beam_mm', 'beam_n_long_top', 'beam_n_long_bot',
              'col_phi_etr', 'col_s_etr', 'col_n_etr_x', 'col_n_etr_y', 'beam_phi_etr', 'beam_s_etr',
              'fc_MPa', 'fy_MPa', 'G', 'Q', 'soil_class']
IMG_COLOR = '#1E1E1E'
IMG_H_FACTOR = 1.08          # label strip height relative to the row height (strip has headroom)


def _dejavu(variant: str) -> str:
    name = {'': 'DejaVuSans.ttf', 'B': 'DejaVuSans-Bold.ttf', 'I': 'DejaVuSans-Oblique.ttf'}[variant]
    for f in fm.fontManager.ttflist:
        if Path(f.fname).name == name:
            return f.fname
    return str(Path(matplotlib.get_data_path()) / 'fonts' / 'ttf' / name)


class _PDF(FPDF):
    def __init__(self, title: str, tool: str = 'Pushover Curve Surrogate - TBEC 2018'):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.doc_title = title; self.tool = tool
        for v in ('', 'B', 'I'):
            self.add_font(FONT, v, _dejavu(v))
        self.set_margins(16, 16, 16); self.set_auto_page_break(auto=True, margin=16)

    def header(self):
        self.set_font(FONT, 'B', 12); self.set_text_color(20, 20, 20)
        self.cell(0, 7, self.doc_title, new_x='LMARGIN', new_y='NEXT')
        self.set_font(FONT, '', 8); self.set_text_color(100, 100, 100)
        self.multi_cell(0, 4, f'Generated {datetime.now():%Y-%m-%d %H:%M}   |   research and preliminary-screening '
                              f'output; not a substitute for nonlinear analysis and engineering judgement',
                        align='L', new_x='LMARGIN', new_y='NEXT')
        self.ln(2)

    def footer(self):
        self.set_y(-11); self.set_font(FONT, 'I', 7.5); self.set_text_color(120, 120, 120)
        self.cell(0, 5, f'{self.tool}   page {self.page_no()}', align='C')


# ---------------------------------------------------------------------------- notation helpers
def _img(tex: str, pt: float = 9.0):
    return T.render_math(tex, pt=pt, color=IMG_COLOR)


def _label(pdf: FPDF, tex: str, w_mm: float, row_h: float, pt: float = 9.0) -> None:
    """Mathtext label image in a cell of width w_mm at the cursor; advances x by w_mm."""
    img = _img(tex, pt)
    h = row_h * IMG_H_FACTOR
    w = img.width / img.height * h
    if w > w_mm - 1.0:
        h *= (w_mm - 1.0) / w; w = w_mm - 1.0
    x, y = pdf.get_x(), pdf.get_y()
    pdf.image(img, x=x + 0.6, y=y + (row_h - h) / 2.0, h=h)
    pdf.set_x(x + w_mm)


def _line(pdf: FPDF, tex: str, pt: float = 8.5, row_h: float = 4.8, max_w: float = 178.0) -> None:
    """One full-width line rendered from mathtext (scaled down if wider than the text width)."""
    if pdf.get_y() + row_h > pdf.page_break_trigger:
        pdf.add_page()
    img = _img(tex, pt)
    h = row_h * IMG_H_FACTOR
    w = img.width / img.height * h
    if w > max_w:
        h *= max_w / w; w = max_w
    pdf.image(img, x=pdf.l_margin + 0.6, y=pdf.get_y() + (row_h - h) / 2.0, h=h)
    pdf.ln(row_h)


def _title(pdf: FPDF, text: str):
    pdf.set_font(FONT, 'B', 10.5); pdf.set_text_color(31, 79, 134)
    pdf.cell(0, 6, text, new_x='LMARGIN', new_y='NEXT')
    pdf.set_draw_color(180, 180, 180); pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(1.5)


def _text_cells(pdf: FPDF, values: List[str], widths: List[float], row_h: float, align: List[str],
                size: float = 8.2, bold: bool = False, color=(30, 30, 30)):
    pdf.set_font(FONT, 'B' if bold else '', size); pdf.set_text_color(*color)
    for w, v, a in zip(widths, values, align):
        pdf.cell(w, row_h, str(v), align=a)


def _figure_png(draw, figsize, dpi=200) -> io.BytesIO:
    fig = Figure(figsize=figsize, dpi=dpi, facecolor='white')
    draw(fig)
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=dpi, facecolor='white'); buf.seek(0)
    return buf


def _tex_unit(key: str, tex: Optional[str] = None) -> str:
    return T.with_unit(tex if tex is not None else T.tex(key), T.unit(key))


# ---------------------------------------------------------------------------- report
def generate_report(result: PredictionResult, output_path: Path, preset: Optional[dict] = None,
                    provenance: Optional[dict] = None, site_transfer: bool = False, show_pca: bool = True,
                    title_tool: Optional[str] = None) -> Path:
    r = result
    tool = title_tool or 'Pushover Curve Surrogate - TBEC 2018'
    pdf = _PDF(f'{tool}: capacity-curve prediction report', tool)
    pdf.add_page()
    sec = itertools.count(1)
    RH = 4.6

    # ---- 1. inputs (two side-by-side label / value tables)
    _title(pdf, f'{next(sec)}. Frame inputs' + (f'   (preset: {preset["title"]}; hold-out frame {preset["source"]["name"]})'
                                                if preset else ''))
    inp = r.inputs
    items = []
    for k in INPUT_KEYS:
        v = inp.get(k)
        if k == 'bay_lengths':
            v = ';'.join(f'{x:.2f}' for x in (inp.get('bay_lengths') or inp.get('bay_lengths_display') or []))
            if not v:
                v = f'{inp["L_mean"]:.2f} mean; {inp["L_min"]:.2f} to {inp["L_max"]:.2f}'
        elif isinstance(v, float):
            v = f'{v:.4g}'
        items.append((_tex_unit(k), str(v)))
    half = (len(items) + 1) // 2
    y0 = pdf.get_y()
    for col, x_left in ((items[:half], pdf.l_margin), (items[half:], pdf.l_margin + 92)):
        pdf.set_y(y0)
        for tex, val in col:
            pdf.set_x(x_left)
            _label(pdf, tex, 40, RH)
            _text_cells(pdf, [val], [50], RH, ['L'], size=8.2)
            pdf.ln(RH)
        y_end = pdf.get_y()
        y_col_end = y_end
    pdf.set_y(y0 + half * RH + 1)
    site = r.site
    _line(pdf, f'Site of the derived route: {site["name"]};  $S_{{DS}}$ = {site["SDS"]:.3f}, '
               f'$S_{{D1}}$ = {site["SD1"]:.3f}, $T_B$ = {site["TB"]:.3f} s, $T_L$ = {site["TL"]:.0f} s', pt=8.0)
    if site_transfer:
        pdf.set_font(FONT, 'I', 8); pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 4.2, 'Site transfer active: the learned demand is valid for the Nurdagi DD-2 spectrum only; '
                               'the derived route is the site-transfer result.', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(1.5)

    # ---- 2. results
    _title(pdf, f'{next(sec)}. Predicted quantities' + ('   (reference = OpenSees values of the locked hold-out frame)' if preset else ''))
    ref = preset['reference'] if preset else {}
    W_ = [40, 58, 30, 36, 14]; AL = ['L', 'L', 'R', 'R', 'L']; RH2 = 4.4
    _text_cells(pdf, ['Quantity', 'Note', 'Predicted', 'Reference', ''], W_, RH2, AL, size=8.2, bold=True, color=(60, 60, 60))
    pdf.ln(RH2)
    vals = dict(r.targets); vals.update(r.ratios)
    rows = []
    for k, u, d in TARGET_ROWS:
        note = 'Nurdagi DD-2 spectrum (training site)' if k in ('delta_t', 'V_at_dt') else ''
        rows.append((_tex_unit(k), note, fmt(vals[k], d), fmt(ref[k], d) if k in ref else '-',
                     'clipped' if r.clips.get(f'clip_{k}') else ''))
    for k, u, d in RATIO_ROWS:
        rows.append((DERIVED_TEX[k], 'learned demand', fmt(vals[k], d), '-', ''))
    dv = {'delta_t_derived': r.derived['delta_t'], 'theta_t_derived': r.derived_ratios['theta_t'],
          'DCR_delta_derived': r.derived_ratios['DCR_delta'], 'mu_d_derived': r.derived_ratios['mu_d'],
          'Te': r.derived['Te'], 'CR': r.derived['CR']}
    units = {'delta_t_derived': 'm', 'Te': 's'}
    for k, d in (('delta_t_derived', 4), ('theta_t_derived', 5), ('DCR_delta_derived', 4), ('mu_d_derived', 3), ('Te', 3), ('CR', 3)):
        rows.append((T.with_unit(DERIVED_TEX[k], units.get(k, '')),
                     'TBEC 5.6.5 on the predicted curve' if k == 'delta_t_derived' else '', fmt(dv[k], d), '-', ''))
    for tex, note, pred, refv, clip in rows:
        _label(pdf, tex, W_[0], RH2)
        _text_cells(pdf, [note, pred, refv, clip], W_[1:], RH2, AL[1:], size=8.2)
        pdf.ln(RH2)
    _line(pdf, f'$H_{{\\mathrm{{tot}}}}$ = {r.Htot:.2f} m,  $W$ = {r.W_seis:.1f} kN,  $S_{{ae}}(T_1)$ = {r.Sa_T1:.4f} g;  '
               f'derived route: $S_{{ae}}(T_e)$ = {r.derived["Sae_Te_g"]:.3f} g, $R_y$ = {r.derived["Ry"]:.3f}, '
               f'$\\Gamma_1$ = {r.derived["Gamma1"]:.3f}', pt=8.0)
    pdf.ln(1.5)

    # ---- curve figure (page 1 if it fits)
    fig_w_in, fig_h_in = 7.2, 2.75
    curve = _figure_png(lambda fig: (draw_curve(fig.add_subplot(111), r, ref or None, show_pca, pt=9), fig.tight_layout(pad=0.4)),
                        (fig_w_in, fig_h_in))
    h_img = 178 * fig_h_in / fig_w_in
    if pdf.get_y() + 9 + h_img > pdf.page_break_trigger:
        pdf.add_page()
    _title(pdf, f'{next(sec)}. Capacity curve')
    pdf.image(curve, w=178)
    pdf.ln(2)

    # ---- flags (label and message with real subscripts and superscripts, reports/richtext.py)
    if pdf.get_y() > pdf.h - pdf.b_margin - 60:
        pdf.add_page()
    _title(pdf, f'{next(sec)}. Applicability flags and validation')
    LAB_W, FH = 50.0, 4.4
    left = pdf.l_margin
    for f in r.flags:
        lev = {'ok': 'ok', 'warning': 'warning', 'info': 'note'}[f.level]
        if pdf.get_y() + FH > pdf.page_break_trigger:
            pdf.add_page()
        y0 = pdf.get_y()
        pdf.set_text_color(*{'ok': (30, 132, 73), 'warning': (185, 119, 14), 'note': (36, 113, 163)}[lev])
        lab = f'{f.label}: {lev}'
        size = 8.0
        while size > 6.0 and rich_width(pdf, lab, FONT, size) > LAB_W - 1.5:
            size -= 0.25
        pdf.set_xy(left, y0)
        write_rich(pdf, lab, FH, FONT, size)
        pdf.set_text_color(40, 40, 40)
        pdf.set_left_margin(left + LAB_W)                   # hanging indent for the message
        pdf.set_xy(left + LAB_W, y0)
        write_rich(pdf, f.message, FH, FONT, 8.0)
        pdf.set_left_margin(left)
        pdf.ln(FH)
    pdf.ln(2)

    # ---- SHAP figures
    pdf.add_page()
    _title(pdf, f'{next(sec)}. Local explanation: top-8 SHAP contributions per target (CatBoost TreeSHAP, transformed space)')
    if r.shap:
        def draw_all(fig):
            axes = fig.subplots(4, 2)
            for ax, t in zip(axes.ravel(), r.shap.keys()):
                draw_shap(ax, r, t, pt=7)
            fig.tight_layout(pad=0.5)
        pdf.image(_figure_png(draw_all, (7.2, 9.0)), w=178)
    pdf.ln(2)

    # ---- features (45, three columns of label image + value)
    pdf.add_page()
    _title(pdf, f'{next(sec)}. Derived model features (45, the exact input vector of the regressors)')
    frows = [(_tex_unit(k), f'{r.features[k]:.6g}') for k in F.FEATURE_ORDER]
    third = (len(frows) + 2) // 3
    cols = [frows[:third], frows[third:2 * third], frows[2 * third:]]
    y0 = pdf.get_y(); RH3 = 4.3
    for ci, col in enumerate(cols):
        pdf.set_y(y0)
        for tex, val in col:
            pdf.set_x(pdf.l_margin + ci * 60)
            _label(pdf, tex, 36, RH3, pt=8.5)
            _text_cells(pdf, [val], [22], RH3, ['R'], size=7.5)
            pdf.ln(RH3)
    pdf.set_y(y0 + third * RH3 + 3)
    c = r.confinement
    _line(pdf, f'Confinement (Mander / TBEC Annex 5A): column $k_e$ = {c["column"]["ke"]:.3f}, '
               f'$f_{{cc}}$ = {c["column"]["fcc_MPa"]:.2f} MPa, $\\varepsilon_{{cu}}$ = {c["column"]["eps_cu"]:.4f}, '
               f'$\\varepsilon_c^{{\\mathrm{{GO}}}}$ = {c["column"]["eps_c_GO"]:.4f};  beam $k_e$ = {c["beam"]["ke"]:.3f}, '
               f'$f_{{cc}}$ = {c["beam"]["fcc_MPa"]:.2f} MPa, $\\varepsilon_{{cu}}$ = {c["beam"]["eps_cu"]:.4f}', pt=8.0)
    pdf.ln(2)

    # ---- provenance
    _title(pdf, f'{next(sec)}. Provenance')
    p = provenance or {}
    prov_rows = []
    if p:
        prov_rows += [('dataset', f'{p["dataset"]["file"]}  sha256 {p["dataset"]["sha256"][:16]}...  '
                                  f'({p["dataset"]["n_rows"]} rows, {p["dataset"]["n_clean"]} clean)'),
                      ('split lock', f'{p["split_lock"]["file"]}  sha256 {p["split_lock"]["sha256"][:16]}...  '
                                     f'(seed {p["split_lock"]["seed"]}, hash {p["split_lock"]["split_hash_md5_12"]})'),
                      ('export date', p['export_date']),
                      ('packages', ', '.join(f'{k} {v}' for k, v in p['packages'].items()))]
        for t, rr in p['regressors'].items():
            prov_rows.append((f'regressor {T.tex(t)}', f'{rr["file"]}  sha256 {rr["sha256"][:16]}...  ({rr["transform"]})'))
        sl = p.get('shape_layer', {})
        if sl:
            prov_rows.append(('shape layer', f'{len(sl["station_files"])} station + {len(sl["pca_files"])} PCA models, '
                                             f'{sl["rows_train"]} training rows, verified = {sl["verified"]}'))
    if preset:
        prov_rows.append(('preset source', f'{preset["source"]["name"]}: {preset["source"]["set"]}, class {preset["source"]["performance_class"]}'))
    prov_rows.append(('report', f'{datetime.now():%Y-%m-%d %H:%M:%S}'))
    for lab, val in prov_rows:
        if pdf.get_y() + 4.2 > pdf.page_break_trigger:
            pdf.add_page()
        _label(pdf, lab, 32, 4.2, pt=8.0)
        _text_cells(pdf, [val], [146], 4.2, ['L'], size=7.5)
        pdf.ln(4.2)

    output_path = Path(output_path)
    pdf.output(str(output_path))
    return output_path
