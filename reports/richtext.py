"""Paper notation in PDF text: plain-Unicode messages (flags, validation) -> text runs with real
subscripts and superscripts (fpdf2 ``char_vpos``), base letters in italics and indices upright as in
the manuscript.

The symbol table is built from assets/labels.json (the same 'unicode' -> 'tex' pairs the widgets
use) plus the few symbols that only occur in messages (shear screen, validation filters). A symbol
is replaced only as a whole token, so ordinary words are never touched.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

APP_DIR = Path(__file__).resolve().parents[1]
LABELS_PATH = APP_DIR / 'assets' / 'labels.json'

Segment = Tuple[str, str, bool]          # (text, vpos 'LINE' | 'SUB' | 'SUP', italic)

GREEK = {'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε', 'varepsilon': 'ε',
         'theta': 'θ', 'lambda': 'λ', 'mu': 'μ', 'pi': 'π', 'rho': 'ρ', 'sigma': 'σ', 'phi': 'φ',
         'omega': 'ω', 'Gamma': 'Γ', 'Delta': 'Δ', 'Sigma': 'Σ', 'Phi': 'Φ', 'Omega': 'Ω'}

# symbols of the messages that are not labels of a feature or target
EXTRA = {'Vdem': r'$V_{\mathrm{dem}}$', 'Vr': r'$V_{\mathrm{r}}$', 'φl': r'$\phi_{\mathrm{l}}$',
         'Lp,col': r'$L_{\mathrm{p,col}}$', 'Lp,beam': r'$L_{\mathrm{p,beam}}$', 'Nd': r'$N_{\mathrm{d}}$',
         'Ac': r'$A_{\mathrm{c}}$', 'ΣMc': r'$\Sigma M_{\mathrm{c}}$', 'ΣMb': r'$\Sigma M_{\mathrm{b}}$',
         'DCRδ': r'$DCR_{\delta}$', 'θt': r'$\theta_{t}$', 'μd': r'$\mu_{d}$', 'Ss': r'$S_{\mathrm{S}}$',
         'S1': r'$S_{1}$', 'Sae': r'$S_{ae}$'}


@lru_cache(maxsize=1)
def symbol_table() -> List[Tuple[str, str]]:
    """(unicode form, mathtext) pairs, longest first; single letters and non-math labels skipped."""
    L = json.load(open(LABELS_PATH, encoding='utf-8'))
    pairs = {}
    for k, u in L['unicode'].items():
        t = L['tex'].get(k) or L['target_tex'].get(k) or ''
        if '$' not in t or len(u) < 2 or ' ' in u or '=' in t:
            continue
        pairs[u] = t
    pairs.update(EXTRA)
    return sorted(pairs.items(), key=lambda kv: -len(kv[0]))


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == '_'


# ---------------------------------------------------------------------------- mathtext subset
_TOKEN = re.compile(r'\\mathrm\{([^{}]*)\}|\\([A-Za-z]+)|([_^])|(\{)|(\})|(.)', re.S)


def tex_segments(tex: str) -> List[Segment]:
    """Segments of a mathtext string ($...$ parts; the rest is plain text). Supports the subset
    used by the labels: \\mathrm{}, Greek commands, _x, _{...}, ^x, ^{...}, \\, (thin space)."""
    out: List[Segment] = []
    parts = re.split(r'(\$[^$]*\$)', tex)
    for part in parts:
        if not part:
            continue
        if not (part.startswith('$') and part.endswith('$')):
            out.append((part, 'LINE', False))
            continue
        body = part[1:-1]
        stack = ['LINE']              # vertical position of the current brace level
        pending = None                # '_' or '^' waiting for its argument
        for m in _TOKEN.finditer(body):
            rm, cmd, script, lb, rb, ch = m.groups()
            if script:
                pending = 'SUB' if script == '_' else 'SUP'
                continue
            if lb:
                stack.append(pending or stack[-1]); pending = None
                continue
            if rb:
                if len(stack) > 1:
                    stack.pop()
                continue
            vpos = pending or stack[-1]
            pending = None
            if rm is not None:
                out.append((rm, vpos, False))
            elif cmd is not None:
                if cmd == ',':
                    continue
                out.append((GREEK.get(cmd, ''), vpos, False))
            elif ch is not None:
                if ch == ' ':
                    continue
                italic = ch.isascii() and ch.isalpha() and vpos == 'LINE'
                out.append((ch, vpos, italic))
    return _merge(out)


def _merge(segs: List[Segment]) -> List[Segment]:
    merged: List[Segment] = []
    for s in segs:
        if not s[0]:
            continue
        if merged and merged[-1][1:] == s[1:]:
            merged[-1] = (merged[-1][0] + s[0],) + s[1:]
        else:
            merged.append(s)
    return merged


def rich_segments(text: str) -> List[Segment]:
    """Plain-Unicode text (or text with $...$ mathtext parts) -> segments with real indices."""
    if '$' in text:
        return tex_segments(text)
    table = symbol_table()
    out: List[Segment] = []
    i, n, plain = 0, len(text), []
    while i < n:
        hit = None
        if i == 0 or not _is_word_char(text[i - 1]):
            for u, t in table:
                j = i + len(u)
                if text.startswith(u, i) and (j == n or not _is_word_char(text[j])):
                    hit = (u, t); break
        if hit:
            if plain:
                out.append((''.join(plain), 'LINE', False)); plain = []
            out.extend(tex_segments(hit[1]))
            i += len(hit[0])
        else:
            plain.append(text[i]); i += 1
    if plain:
        out.append((''.join(plain), 'LINE', False))
    return _merge(out)


def write_rich(pdf, text: str, h: float, family: str, size: float, bold: bool = False) -> None:
    """Write ``text`` at the cursor with fpdf2 ``write`` (wraps at the right margin)."""
    for seg, vpos, italic in rich_segments(text):
        style = ('B' if bold else '') + ('I' if italic else '')
        pdf.set_font(family, style if style != 'BI' else 'B', size)
        pdf.char_vpos = vpos
        pdf.write(h, seg)
    pdf.char_vpos = 'LINE'


def rich_width(pdf, text: str, family: str, size: float, bold: bool = False) -> float:
    """Width [user units] of ``text`` as written by write_rich."""
    w = 0.0
    for seg, vpos, italic in rich_segments(text):
        style = ('B' if bold else '') + ('I' if italic else '')
        pdf.set_font(family, style if style != 'BI' else 'B', size * (0.7 if vpos != 'LINE' else 1.0))
        w += pdf.get_string_width(seg)
    pdf.set_font(family, 'B' if bold else '', size)
    return w
