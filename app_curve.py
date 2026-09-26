"""Pushover Curve Surrogate - TBEC 2018: desktop entry point.

    python app_curve.py

Left: raw inputs of Table A1 with the paper notation, presets, live structure preview.
Right: capacity curve, predicted quantities (with the locked hold-out reference for presets),
local SHAP, applicability badges. Batch window and PDF report from the toolbar.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import customtkinter as ctk
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.figure import Figure  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
from core.model_loader import MODEL_DIR_DEFAULT, ModelBundle  # noqa: E402
from core.predictor import PredictionResult, Predictor  # noqa: E402
from reports.pdf_report import generate_report  # noqa: E402
from ui import theme as T  # noqa: E402
from ui.batch_window import BatchWindow  # noqa: E402
from ui.curve_panel import CurvePanel  # noqa: E402
from ui.input_panel import InputPanel  # noqa: E402
from ui.result_panel import ResultPanel  # noqa: E402
from ui.structure_canvas import render_structure  # noqa: E402

APP_TITLE = 'Pushover Curve Surrogate - TBEC 2018'
APP_VERSION = '1.0.0'
EXAMPLES_DIR = APP_DIR / 'assets' / 'examples'


class CurveApp(ctk.CTk):
    TITLE = APP_TITLE
    SUBTITLE = 'capacity curve, demand point and applicability from 25 raw design inputs'
    STRUCT_FIGSIZE = (2.3, 7.2)

    def __init__(self, model_dir: Path = MODEL_DIR_DEFAULT, load_models: bool = True):
        super().__init__()
        ctk.set_appearance_mode('light'); ctk.set_default_color_theme('blue')
        from customtkinter.windows.widgets.scaling import ScalingTracker
        T.set_ui_scale(ScalingTracker.get_widget_scaling(self))
        self.title(self.TITLE); self.geometry('1400x880'); self.minsize(1200, 760); self.configure(fg_color=T.BG)
        self.model_dir = Path(model_dir)
        self.bundle: Optional[ModelBundle] = None
        self.predictor: Optional[Predictor] = None
        self.last_result: Optional[PredictionResult] = None
        self.last_preset: Optional[dict] = None
        self.last_site_transfer = False
        feature_spec = None
        if (self.model_dir / 'feature_spec.json').exists():
            feature_spec = json.load(open(self.model_dir / 'feature_spec.json', encoding='utf-8'))
        self._build_bar()
        self._build_layout(feature_spec)
        if load_models:
            self.after(100, self.load_models)

    # ------------------------------------------------------------------ layout
    def _build_bar(self):
        bar = ctk.CTkFrame(self, fg_color=T.PANEL, height=44, corner_radius=0)
        bar.pack(fill='x', side='top')
        ctk.CTkLabel(bar, text=self.TITLE, font=T.font(15, 'bold'), text_color=T.TEXT).pack(side='left', padx=14)
        ctk.CTkLabel(bar, text=self.SUBTITLE, font=T.font(11), text_color=T.MUTED).pack(side='left', padx=4)
        self.export_btn = ctk.CTkButton(bar, text='Export PDF report', width=150, command=self.export_pdf,
                                        fg_color=T.OK, hover_color='#166638', text_color='#FFFFFF', state='disabled',
                                        font=T.font(12))
        self.export_btn.pack(side='right', padx=(6, 14), pady=6)
        ctk.CTkButton(bar, text='Batch (xlsx)', width=100, command=self.open_batch, fg_color='#E0E0E0',
                      hover_color='#CCCCCC', text_color=T.TEXT, font=T.font(12)).pack(side='right', padx=4, pady=6)
        ctk.CTkButton(bar, text='About', width=70, command=self._about, fg_color='#E0E0E0', hover_color='#CCCCCC',
                      text_color=T.TEXT, font=T.font(12)).pack(side='right', padx=4, pady=6)

    def _build_layout(self, feature_spec):
        self.status = ctk.CTkLabel(self, text='Starting', anchor='w', fg_color=T.PANEL, text_color=T.TEXT2,
                                   font=T.font(11), height=24)
        self.status.pack(fill='x', side='bottom')          # packed before the body so it keeps its height
        body = ctk.CTkFrame(self, fg_color=T.BG); body.pack(fill='both', expand=True)
        body.grid_columnconfigure(0, weight=0, minsize=T.px(body, 412))
        body.grid_columnconfigure(1, weight=0, minsize=T.px(body, 248))
        body.grid_columnconfigure(2, weight=1)
        body.grid_rowconfigure(0, weight=1)
        self.input_panel = InputPanel(body, EXAMPLES_DIR, feature_spec, on_change=self._on_change,
                                      on_predict=self._on_predict, on_batch=self.open_batch)
        self.input_panel.grid(row=0, column=0, sticky='nsew', padx=(8, 4), pady=8)

        self.mid = ctk.CTkFrame(body, fg_color='transparent')
        self.mid.grid(row=0, column=1, sticky='nsew', padx=4, pady=8)
        sf = ctk.CTkFrame(self.mid, fg_color=T.PANEL, corner_radius=10, border_width=1, border_color=T.BORDER)
        sf.pack(fill='both', expand=True)
        self.struct_frame = sf
        ctk.CTkLabel(sf, text='Structure preview', font=T.font(13, 'bold'), text_color=T.TEXT).pack(anchor='w', padx=12, pady=(8, 2))
        self.struct_fig = Figure(figsize=self.STRUCT_FIGSIZE, dpi=T.fig_dpi(), facecolor=T.PANEL)
        self.struct_canvas = T.make_canvas(self.struct_fig, sf)
        self.struct_canvas.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=(0, 8))

        right = ctk.CTkFrame(body, fg_color='transparent'); right.grid(row=0, column=2, sticky='nsew', padx=(4, 8), pady=8)
        right.grid_rowconfigure(0, weight=0); right.grid_rowconfigure(1, weight=1); right.grid_columnconfigure(0, weight=1)
        self.curve_panel = CurvePanel(right); self.curve_panel.grid(row=0, column=0, sticky='nsew', pady=(0, 6))
        targets = ['Vy', 'delta_y', 'Vmax', 'delta_max', 'V_at_dt', 'delta_t', 'delta_u', 'T1']
        self.result_panel = ResultPanel(right, targets); self.result_panel.grid(row=1, column=0, sticky='nsew')
        self._render_structure(self.input_panel.get_values())

    # ------------------------------------------------------------------ models
    def load_models(self):
        try:
            self.set_status('Loading models ...')
            self.update_idletasks()
            self.bundle = ModelBundle.load(self.model_dir, shape=True)
            self.predictor = Predictor(self.bundle, shape_layer='both')
            self.set_status(f'Models loaded from {self.model_dir} '
                            f'({len(self.bundle.regressors)} regressors, {len(self.bundle.stations)} station models, '
                            f'{len(self.bundle.pca_models)} PCA models). Ready.', 'ok')
        except Exception as exc:
            self.predictor = None
            self.set_status(f'Model loading failed: {exc}. Run export_models.py first.', 'err')
            traceback.print_exc()

    # ------------------------------------------------------------------ callbacks
    def _on_change(self, values: dict):
        if 'struct_fig' not in self.__dict__:          # input panel initialises before the preview
            return
        try:
            self._render_structure(values)
        except Exception:
            traceback.print_exc()

    def _render_structure(self, values: dict):
        render_structure(self.struct_fig, values, pt=8.5)
        self.struct_canvas.draw_idle()

    def _on_predict(self, values: dict, site: Optional[dict], preset: Optional[dict]):
        if self.predictor is None:
            messagebox.showwarning('Models not loaded', 'Run export_models.py first (see README.md).'); return
        try:
            self.set_status('Predicting ...'); self.update_idletasks()
            r = self.predictor.predict(values, site=site, shap=True)
            self.show_result(r, preset, site is not None and not site.get('is_nurdagi', True))
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror('Prediction failed', str(exc)); self.set_status(f'Prediction failed: {exc}', 'err')

    def show_result(self, r: PredictionResult, preset: Optional[dict], site_transfer: bool):
        self.last_result, self.last_preset, self.last_site_transfer = r, preset, site_transfer
        self.curve_panel.show(r, preset['reference'] if preset else None)
        self.result_panel.show(r, preset, site_transfer)
        self.export_btn.configure(state='normal')
        n_w = len(r.warnings)
        self.set_status(f'Prediction done: Vmax = {r.targets["Vmax"]:.0f} kN, δu = {r.targets["delta_u"]:.3f} m, '
                        f'δt = {r.targets["delta_t"]:.4f} m (learned), {r.derived["delta_t"]:.4f} m (derived); '
                        f'{n_w} warning(s).', 'ok' if n_w == 0 else 'warn')

    def predict_preset(self, index: int) -> PredictionResult:
        """Programmatic path used by the smoke test: load preset i and predict."""
        self.input_panel.load_preset(index)
        self.update_idletasks()
        self._on_predict(self.input_panel.get_values(), self.input_panel.get_site(), self.input_panel.preset)
        self.update_idletasks()
        return self.last_result

    # ------------------------------------------------------------------ exports
    def export_pdf(self, path: Optional[Path] = None) -> Optional[Path]:
        if self.last_result is None:
            return None
        if path is None:
            p = filedialog.asksaveasfilename(defaultextension='.pdf', filetypes=[('PDF', '*.pdf')],
                                             initialfile='pushover_curve_report.pdf')
            if not p:
                return None
            path = Path(p)
        try:
            out = generate_report(self.last_result, Path(path), preset=self.last_preset,
                                  provenance=self.bundle.provenance if self.bundle else None,
                                  site_transfer=self.last_site_transfer, show_pca=self.curve_panel.pca_var.get())
            self.set_status(f'PDF report saved: {out}', 'ok')
            return out
        except Exception as exc:
            traceback.print_exc(); self.set_status(f'PDF export failed: {exc}', 'err')
            messagebox.showerror('Export failed', str(exc)); return None

    def open_batch(self):
        if self.predictor is None:
            messagebox.showwarning('Models not loaded', 'Run export_models.py first.'); return
        BatchWindow(self, self.predictor, self.input_panel.presets)

    def _about(self):
        prov = self.bundle.provenance if self.bundle else {}
        messagebox.showinfo('About', f'{APP_TITLE}\nVersion {APP_VERSION}\n\n'
                            'Surrogate of the TBEC 2018 pushover capacity curve of planar RC frames: eight CatBoost '
                            'regressors for the characteristic points and the period, a station-wise dense-ordinate '
                            'shape layer, the learned demand point for the Nurdagi DD-2 spectrum and a derived '
                            'TBEC 5.6.5 route for other sites.\n\n'
                            f'Model export: {prov.get("export_date", "n/a")}\n'
                            f'Dataset: {prov.get("dataset", {}).get("file", "n/a")} '
                            f'({prov.get("dataset", {}).get("n_clean", "n/a")} clean frames)\n\n'
                            'Research and preliminary-screening tool; the results do not replace a nonlinear '
                            'analysis and engineering judgement.')

    def set_status(self, text: str, level: str = 'info'):
        self.status.configure(text=text, text_color={'info': T.TEXT2, 'ok': T.OK, 'warn': T.WARN, 'err': T.ERR}[level])


def create_app(load_models: bool = True) -> CurveApp:
    return CurveApp(load_models=load_models)


def main():
    app = create_app()
    app.mainloop()


if __name__ == '__main__':
    main()
