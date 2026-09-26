"""Shared UI theme: colours, fonts, scaling, notation labels rendered from mathtext, tooltips.

Notation policy: every symbol shown in a Tk widget comes from assets/labels.json. Labels with
LaTeX mathtext are rendered by matplotlib into small transparent images (cached) so the paper
notation appears exactly as in the manuscript; plain-Unicode forms (Greek letters, no code
names) are used where an image cannot be placed (tooltips, spreadsheet headers, PDF text).
"""
from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import customtkinter as ctk
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

APP_DIR = Path(__file__).resolve().parents[1]
LABELS_PATH = APP_DIR / 'assets' / 'labels.json'

# ------------------------------------------------------------------ colours
BG = '#F4F5F7'
PANEL = '#FFFFFF'
INNER = '#FAFBFC'
BORDER = '#E0E0E0'
TEXT = '#212121'
TEXT2 = '#555555'
MUTED = '#8A8F98'
ACCENT = '#2E6DB4'
ACCENT_DARK = '#1F4F86'
OK = '#1E8449'
WARN = '#B9770E'
WARN_BG = '#FCF3CF'
ERR = '#C0392B'
INFO = '#2471A3'
INFO_BG = '#D6EAF8'
OK_BG = '#D5F5E3'
GREY_BG = '#ECEFF1'
# curve palette (same symbol, same colour everywhere)
C_MULTI = '#4477AA'
C_STATION = '#CC3311'
C_PCA = '#AA3377'
C_YIELD = '#228833'
C_PEAK = '#EE7733'
C_ULT = '#BB5566'
C_DEMAND = '#000000'
C_REF = '#8A8F98'
C_SHAP_POS = '#2E6DB4'
C_SHAP_NEG = '#CC3311'

FONT_FAMILY = 'Segoe UI'
BASE_PT = 12               # Tk base font size (points at 1x)
MATH_PT = 11.5             # mathtext label size (points at 1x)

_SCALE = 1.0               # widget scaling used for figure DPI (set by the app at start-up)


def set_ui_scale(s: float) -> None:
    global _SCALE
    _SCALE = float(s)


def ui_scale() -> float:
    return _SCALE


def fig_dpi(base: int = 100) -> int:
    """Figure DPI at logical (96 dpi) size. HiDPI is handled by matplotlib's Tk backend itself
    (device pixel ratio from 'tk scaling'), so the widget scaling must NOT be applied here."""
    return int(base)


def font(size: int = BASE_PT, weight: str = 'normal') -> ctk.CTkFont:
    return ctk.CTkFont(family=FONT_FAMILY, size=size, weight=weight)


def px(widget, logical: float) -> int:
    """Logical (96 dpi) pixels -> physical pixels for Tk options that customtkinter does not scale
    (grid minsize, canvas sizes)."""
    from customtkinter.windows.widgets.scaling import ScalingTracker
    return int(round(logical * ScalingTracker.get_widget_scaling(widget)))


def device_ratio(widget) -> float:
    """Device-pixel ratio that matplotlib's Tk backend derives from 'tk scaling'."""
    return round(float(widget.tk.call('tk', 'scaling')) / (96.0 / 72.0), 2)


def make_canvas(fig, master):
    """FigureCanvasTkAgg whose figure renders at base 100 dpi x the customtkinter widget scaling.

    customtkinter scales its widgets itself (the Tk interpreter keeps 'tk scaling' near 1.33 on
    Windows), so matplotlib's own device-pixel ratio stays 1.0 and figures would render at 1x
    inside 1.5x or 2x widgets. The figure dpi is therefore set from the widget scaling, and the
    backend's <Map> update is made a no-op by presetting the ratio it would compute.
    """
    from customtkinter.windows.widgets.scaling import ScalingTracker
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    s = ScalingTracker.get_widget_scaling(master)
    fig.set_dpi(fig_dpi() * s)
    canvas = FigureCanvasTkAgg(fig, master=master)
    canvas._device_pixel_ratio = device_ratio(master)
    return canvas


# ------------------------------------------------------------------ labels
@lru_cache(maxsize=1)
def labels() -> Dict[str, dict]:
    with open(LABELS_PATH, encoding='utf-8') as fh:
        return json.load(fh)


def tex(key: str) -> str:
    """Mathtext of a feature, target or derived quantity (falls back to the Unicode form)."""
    L = labels()
    return L['tex'].get(key) or L['target_tex'].get(key) or L['unicode'].get(key, key)


def uni(key: str) -> str:
    """Plain-Unicode notation (Greek letters, no code-style names)."""
    return labels()['unicode'].get(key, key)


def unit(key: str) -> str:
    L = labels()
    return L['feature_units'].get(key) or L['raw_input_units'].get(key) or L['target_units'].get(key, '')


def with_unit(text: str, u: str) -> str:
    return f'{text} [{u}]' if u and u != '-' else text


@lru_cache(maxsize=512)
def render_math(text: str, pt: float = MATH_PT, color: str = TEXT, oversample: int = 3) -> Image.Image:
    """Render a mathtext string to a transparent RGBA image at ``oversample`` x the 96-dpi size.

    Every label is drawn on a strip of fixed height (baseline at a fixed position) so that
    labels line up in tables regardless of descenders or superscripts.
    """
    dpi = 96 * oversample
    strip_in = pt * 1.6 / 72.0                      # fixed strip height in inches
    fig = Figure(figsize=(6.0, strip_in), dpi=dpi)
    fig.patch.set_alpha(0.0)
    t = fig.text(0.0, 0.30, text, fontsize=pt, color=color, ha='left', va='baseline',
                 fontfamily='DejaVu Sans', transform=fig.transFigure)
    canvas = FigureCanvasAgg(fig)
    renderer = canvas.get_renderer()
    bb = t.get_window_extent(renderer)
    width_in = bb.width / dpi + 0.04
    from matplotlib.transforms import Bbox
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, transparent=True,
                bbox_inches=Bbox([[0.0, 0.0], [width_in, strip_in]]), pad_inches=0)
    buf.seek(0)
    return Image.open(buf).convert('RGBA')


def math_image(text: str, pt: float = MATH_PT, color: str = TEXT) -> ctk.CTkImage:
    img = render_math(text, pt, color)
    w, h = img.size
    logical = (w / 3, h / 3)           # oversample = 3 -> logical size at 96 dpi
    return ctk.CTkImage(light_image=img, dark_image=img, size=logical)


class MathLabel(ctk.CTkLabel):
    """A CTkLabel that shows a mathtext string as an image (paper notation in Tk)."""

    def __init__(self, master, text: str, pt: float = MATH_PT, color: str = TEXT, **kw):
        self._math_text = text; self._pt = pt; self._color = color
        super().__init__(master, text='', image=math_image(text, pt, color), **kw)

    def set_math(self, text: str, color: Optional[str] = None) -> None:
        if color is not None:
            self._color = color
        self._math_text = text
        self.configure(image=math_image(text, self._pt, self._color))


# ------------------------------------------------------------------ tooltip
class Tooltip:
    """One-sentence hover text for any widget."""

    def __init__(self, widget, text: str, delay_ms: int = 350, wrap: int = 380):
        self.widget = widget; self.text = text; self.delay = delay_ms; self.wrap = wrap
        self._after = None; self._tip = None
        widget.bind('<Enter>', self._schedule, add='+')
        widget.bind('<Leave>', self._hide, add='+')
        widget.bind('<ButtonPress>', self._hide, add='+')

    def set_text(self, text: str) -> None:
        self.text = text

    def _schedule(self, _e=None):
        self._cancel()
        self._after = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after is not None:
            try:
                self.widget.after_cancel(self._after)
            except Exception:
                pass
            self._after = None

    def _show(self):
        if self._tip is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self._tip = ctk.CTkToplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f'+{x}+{y}')
        self._tip.attributes('-topmost', True)
        ctk.CTkLabel(self._tip, text=self.text, fg_color='#FFFBE6', text_color=TEXT,
                     corner_radius=6, wraplength=self.wrap, justify='left', font=font(11),
                     padx=8, pady=5).pack()

    def _hide(self, _e=None):
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


def style_axes(ax, pt: float = 10.0):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color('#C8CCD1'); s.set_linewidth(0.8)
    ax.tick_params(colors='#333333', labelsize=pt)
    ax.xaxis.label.set_color('#333333'); ax.yaxis.label.set_color('#333333')
    ax.xaxis.label.set_size(pt + 0.5); ax.yaxis.label.set_size(pt + 0.5)
    ax.grid(alpha=0.25, color='#888888', linestyle=':')
    matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
