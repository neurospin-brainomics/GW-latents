"""Shared slide style for the Champollion figures (16:9, large type). Used by make_champ_figs_v2.py and 222_champ_figs_jz.py."""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

EUC, GEO, ORC = "#4fb592", "#5b8fd9", "#ec8a62"          # softened slots 3, 1, 2 (validated all-pairs; always direct-labelled)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
NULL, GRID, BG, BAND = "#bdbbb4", "#ebeae6", "#fcfcfb", "#f3f2ee"
METRIC_COL = {"euclidean": EUC, "geodesic": GEO, "orc": ORC}
METRIC_NAME = {"euclidean": "Euclidean", "geodesic": "Geodesic", "orc": "Ollivier–Ricci"}
BLUES = LinearSegmentedColormap.from_list("blues", [BG, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ORANGES = LinearSegmentedColormap.from_list("oranges", [BG, "#fbd9c9", "#f4a27f", "#eb6834", "#b8461c", "#7a2c10"])
DIVERGE = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f19a98", "#b8302f"])
W, H = 13.33, 7.5                                          # 16:9 slide


def _register_fonts():
    """use Inter if its .ttf files sit next to this file (or in ./fonts); otherwise matplotlib falls back to DejaVu Sans"""
    from pathlib import Path
    from matplotlib import font_manager
    here = Path(__file__).resolve().parent
    for f in list(here.glob("Inter-*.ttf")) + list((here / "fonts").glob("Inter-*.ttf")):
        try: font_manager.fontManager.addfont(str(f))
        except Exception: pass


def apply():
    _register_fonts()
    mpl.rcParams.update({
        "font.family": ["Inter", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 17, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "axes.labelsize": 17, "xtick.labelsize": 15, "ytick.labelsize": 15, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.0,
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "savefig.dpi": 200,
        "axes.titleweight": "semibold", "axes.titlesize": 20, "axes.titlelocation": "left", "axes.titlepad": 12,
        "legend.frameon": False, "legend.fontsize": 15, "lines.solid_capstyle": "round"})
    import logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def headline(fig, title, sub=None, y=0.975):
    """slide-style title block, left-aligned on the figure"""
    fig.text(0.04, y, title, fontsize=28, weight="bold", ha="left", va="top", color=INK)
    if sub:
        fig.text(0.04, y - 0.075, sub, fontsize=17, ha="left", va="top", color=INK2)


def save(fig, path):
    fig.savefig(path, dpi=200)
    import matplotlib.pyplot as plt; plt.close(fig)
