"""Capacity-curve panel: multilinear polyline (dashed), station-wise dense-ordinate profile
(solid), optional PCA profile (dotted), the four characteristic points, the regressor demand
point, the 0.8 Vmax line and, for presets, the OpenSees reference points (hollow markers)."""
from __future__ import annotations

from typing import Dict, Optional

import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core.predictor import PredictionResult
from . import theme as T


def draw_curve(ax, r: PredictionResult, reference: Optional[Dict[str, float]] = None,
               show_pca: bool = False, pt: float = 10.0) -> None:
    T.style_axes(ax, pt)
    t = r.targets
    x, y = r.curves['multilinear']
    ax.plot(x, y, '--', color=T.C_MULTI, lw=1.8, label='multilinear (four points)', zorder=3)
    if 'station' in r.curves:
        ax.plot(*r.curves['station'], '-', color=T.C_STATION, lw=2.2,
                label='station profile (dense ordinates)', zorder=4)
    if show_pca and 'pca' in r.curves:
        ax.plot(*r.curves['pca'], ':', color=T.C_PCA, lw=2.0, label='PCA profile (light layer)', zorder=4)
    V80 = r.curves['V80']
    ax.axhline(V80, color=T.MUTED, lw=0.9, ls=(0, (5, 3)), zorder=2, label=r'$0.8\,V_{\mathrm{max}}$ line')
    ax.plot([t['delta_y']], [t['Vy']], 'o', color=T.C_YIELD, ms=8, mec='white', mew=0.8, zorder=6,
            label=r'yield $(\delta_y,\ V_y)$')
    ax.plot([t['delta_max']], [t['Vmax']], 's', color=T.C_PEAK, ms=8, mec='white', mew=0.8, zorder=6,
            label=r'peak $(\delta_{\mathrm{max}},\ V_{\mathrm{max}})$')
    ax.plot([t['delta_u']], [V80], 'D', color=T.C_ULT, ms=7.5, mec='white', mew=0.8, zorder=6,
            label=r'ultimate $(\delta_u,\ 0.8\,V_{\mathrm{max}})$')
    dt, Vdt = r.demand_point
    ax.plot([dt], [Vdt], '*', color=T.C_DEMAND, ms=15, mec='white', mew=0.8, zorder=7,
            label=r'demand $(\delta_t,\ V_{\delta_t})$, regressors')
    ax.axvline(dt, color=T.MUTED, lw=0.8, ls=':', zorder=2)
    xmax = t['delta_u']; ymax = t['Vmax']
    if reference:
        rf = reference
        ax.plot([rf['delta_y'], rf['delta_max'], rf['delta_u'], rf['delta_t']],
                [rf['Vy'], rf['Vmax'], 0.8 * rf['Vmax'], rf['V_at_dt']],
                linestyle='none', marker='o', mfc='none', mec=T.C_REF, mew=1.4, ms=9, zorder=5,
                label='reference (locked hold-out), OpenSees')
        xmax = max(xmax, rf['delta_u']); ymax = max(ymax, rf['Vmax'])
    # the legend lives in a free band above the curve (3 rows, 3 columns)
    ax.set_xlim(0, 1.08 * xmax); ax.set_ylim(0, 1.62 * ymax)
    ax.set_xlabel(r'Roof displacement $\delta$ [m]'); ax.set_ylabel(r'Base shear $V$ [kN]')
    ax.set_title(f'Predicted capacity curve   ($T_1$ = {t["T1"]:.3f} s,  $H_{{\\mathrm{{tot}}}}$ = {r.Htot:.2f} m)',
                 fontsize=pt + 0.5, color=T.TEXT)
    ax.legend(loc='upper center', ncol=3, fontsize=pt - 2, facecolor='white', edgecolor=T.BORDER, framealpha=0.95,
              handlelength=1.8, columnspacing=0.9, borderaxespad=0.3, handletextpad=0.5)


class CurvePanel(ctk.CTkFrame):
    def __init__(self, master, figsize=(6.9, 2.95)):
        super().__init__(master, fg_color=T.PANEL, corner_radius=10, border_width=1, border_color=T.BORDER)
        head = ctk.CTkFrame(self, fg_color='transparent')
        head.pack(fill='x', padx=12, pady=(8, 0))
        ctk.CTkLabel(head, text='Capacity curve', font=T.font(13, 'bold'), text_color=T.TEXT).pack(side='left')
        self.pca_var = ctk.BooleanVar(value=False)
        self.pca_box = ctk.CTkCheckBox(head, text='PCA layer (light alternative)', variable=self.pca_var,
                                       command=self._redraw, font=T.font(11), text_color=T.TEXT2,
                                       checkbox_width=18, checkbox_height=18)
        self.pca_box.pack(side='right')
        T.Tooltip(self.pca_box, 'Adds the five-coefficient PCA reconstruction of the curve shape next to the '
                                'station-wise profile (Table 12 of the paper).')
        self.fig = Figure(figsize=figsize, dpi=T.fig_dpi(), facecolor=T.PANEL)
        self.ax = self.fig.add_subplot(111)
        T.style_axes(self.ax)
        self.ax.text(0.5, 0.5, 'Press Predict to draw the capacity curve', ha='center', va='center',
                     transform=self.ax.transAxes, color=T.MUTED, fontsize=11)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.canvas = T.make_canvas(self.fig, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=8, pady=(2, 8))
        self.result: Optional[PredictionResult] = None
        self.reference: Optional[Dict[str, float]] = None

    def show(self, result: PredictionResult, reference: Optional[Dict[str, float]] = None) -> None:
        self.result = result; self.reference = reference
        self._redraw()

    def _redraw(self) -> None:
        if self.result is None:
            return
        self.ax.clear()
        draw_curve(self.ax, self.result, self.reference, self.pca_var.get(), pt=9.5)
        self.fig.tight_layout(pad=0.6)
        self.canvas.draw_idle()
