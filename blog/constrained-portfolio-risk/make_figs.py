"""Generate the SVG figures for the constrained-portfolio-risk blog post.

Usage: python3 make_figs.py img

Numbers are copied from the raw outputs of the paper's reproducibility package:
  SIM_DF     <- data/sim_df.csv        (Table 2 / Figure 1 of the paper; 400 samples per row, n = 250)
  PHASE      <- data/phase_p400.csv    (Figure 2 of the paper; p = 400, 100 samples per point)
  TE_NASDAQ  <- data/te_nasdaq_pgn.csv (Figure 3(b) of the paper; medians over monthly windows)
Palette follows blog/assets/style.css: ink #172033, muted #667085, navy #00205b, royal blue #4169e1.
"""
import sys
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "img")
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, NAVY, BLUE = "#172033", "#667085", "#00205b", "#4169e1"
TEAL, ORANGE, GREY = "#0f766e", "#c2410c", "#8a93a6"
GRID, AXIS, REF, SOFT, PANEL = "#ece9e3", "#c9c4bb", "#8a93a6", "#eef3ff", "#f7f6f3"
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"

# market, p, mean d/m, mean R_in/R_out, mean tau^2/R_out, s.e. of tau^2/R_out
SIM_DF = [
    ("iid", 125, 0.4195, 0.3373, 1.0029, 0.0070),
    ("iid", 250, 0.6785, 0.1050, 1.0262, 0.0107),
    ("iid", 375, 0.8583, 0.0200, 1.0046, 0.0165),
    ("ar1", 125, 0.3202, 0.4639, 1.0056, 0.0065),
    ("ar1", 250, 0.5088, 0.2397, 0.9978, 0.0081),
    ("ar1", 500, 0.7672, 0.0537, 0.9994, 0.0127),
    ("ar1", 700, 0.9191, 0.0068, 1.0185, 0.0247),
    ("f1pos", 125, 0.1119, 0.7904, 1.0022, 0.0051),
    ("f1pos", 500, 0.1841, 0.6736, 1.0124, 0.0052),
    ("f1pos", 1000, 0.2408, 0.5747, 0.9980, 0.0054),
    ("f1mixed", 125, 0.3161, 0.4679, 1.0016, 0.0059),
    ("f1mixed", 250, 0.5196, 0.2309, 1.0024, 0.0078),
    ("f1mixed", 500, 0.7817, 0.0466, 0.9958, 0.0140),
    ("f1mixed", 625, 0.8858, 0.0130, 1.0243, 0.0221),
    ("f3pos", 125, 0.1138, 0.7878, 1.0033, 0.0054),
    ("f3pos", 500, 0.1849, 0.6610, 0.9954, 0.0055),
    ("f3pos", 1000, 0.2431, 0.5739, 1.0029, 0.0064),
    ("f3mixed", 125, 0.3134, 0.4719, 1.0020, 0.0063),
    ("f3mixed", 250, 0.5179, 0.2345, 1.0120, 0.0081),
    ("f3mixed", 500, 0.7813, 0.0478, 1.0093, 0.0125),
]

# market, statistical dimension delta, m, share of samples with zero in-sample variance
PHASE = [
    ("iid", 199.381, 100, 1.000), ("iid", 199.381, 140, 1.000), ("iid", 199.381, 160, 1.000),
    ("iid", 199.381, 179, 0.970), ("iid", 199.381, 199, 0.500), ("iid", 199.381, 219, 0.020),
    ("iid", 199.381, 239, 0.000), ("iid", 199.381, 259, 0.000), ("iid", 199.381, 299, 0.000),
    ("iid", 199.381, 359, 0.000),
    ("ar1", 120.571, 60, 1.000), ("ar1", 120.571, 84, 1.000), ("ar1", 120.571, 96, 0.990),
    ("ar1", 120.571, 109, 0.900), ("ar1", 120.571, 121, 0.440), ("ar1", 120.571, 133, 0.060),
    ("ar1", 120.571, 145, 0.010), ("ar1", 120.571, 157, 0.000), ("ar1", 120.571, 181, 0.000),
    ("ar1", 120.571, 217, 0.000),
    ("f1pos", 17.077, 9, 1.000), ("f1pos", 17.077, 12, 0.870), ("f1pos", 17.077, 14, 0.750),
    ("f1pos", 17.077, 15, 0.600), ("f1pos", 17.077, 17, 0.460), ("f1pos", 17.077, 19, 0.300),
    ("f1pos", 17.077, 20, 0.220), ("f1pos", 17.077, 22, 0.090), ("f1pos", 17.077, 26, 0.030),
    ("f1pos", 17.077, 31, 0.000),
    ("f1mixed", 130.392, 65, 1.000), ("f1mixed", 130.392, 91, 1.000), ("f1mixed", 130.392, 104, 1.000),
    ("f1mixed", 130.392, 117, 0.890), ("f1mixed", 130.392, 130, 0.520), ("f1mixed", 130.392, 143, 0.070),
    ("f1mixed", 130.392, 156, 0.000), ("f1mixed", 130.392, 170, 0.000), ("f1mixed", 130.392, 196, 0.000),
    ("f1mixed", 130.392, 235, 0.000),
    ("f3pos", 17.918, 9, 0.990), ("f3pos", 17.918, 13, 0.940), ("f3pos", 17.918, 14, 0.790),
    ("f3pos", 17.918, 16, 0.730), ("f3pos", 17.918, 18, 0.470), ("f3pos", 17.918, 20, 0.210),
    ("f3pos", 17.918, 22, 0.190), ("f3pos", 17.918, 23, 0.090), ("f3pos", 17.918, 27, 0.020),
    ("f3pos", 17.918, 32, 0.000),
]

# config, median floor (% a year), target (% a year), naive TE/target, df-calibrated TE/target
TE_NASDAQ = [
    ("126_252_random", 3.95, 2.0, 3.227, 2.122), ("126_252_random", 3.95, 4.0, 2.116, 1.128),
    ("126_252_random", 3.95, 6.0, 1.754, 1.020), ("126_252_random", 3.95, 8.0, 1.513, 1.032),
    ("126_504_random", 5.32, 2.0, 3.628, 2.716), ("126_504_random", 5.32, 4.0, 2.186, 1.363),
    ("126_504_random", 5.32, 6.0, 1.721, 1.030), ("126_504_random", 5.32, 8.0, 1.523, 0.967),
    ("252_500_random", 2.95, 2.0, 2.594, 1.566), ("252_500_random", 2.95, 3.0, 2.085, 1.092),
    ("252_500_random", 2.95, 4.0, 1.775, 0.993), ("252_500_random", 2.95, 6.0, 1.496, 1.028),
    ("252_500_random", 2.95, 8.0, 1.384, 1.053),
]

LANG = {
    "en": {
        "f1_title": "Long-only weights for three stocks form a triangle",
        "f1_sub": ["The optimizer lands on one face of it. Free directions on that face: d = holdings − 1."],
        "f1_vertex": "a corner: one stock held, d = 0",
        "f1_edge": "an edge: two held, d = 1",
        "f1_inside": "the inside: all three held, d = 2",
        "f1_pt": "optimizer lands here",
        "f1_pin": "weight of C pinned at 0",
        "f1_free": "free to slide: 1 direction",
        "f2_title": "Before correction the ratio follows (1 − d/m)²; after it, the ratio sits at 1",
        "f2_sub": ["20 simulated markets, n = 250 days, p = 125 to 1,000 stocks, 400 samples each.",
                   "Ratio to the true out-of-sample variance; bars are ±2 standard errors."],
        "f2_x": "share of free directions, d/m",
        "f2_corr": "corrected: R_in / (1 − d/m)²",
        "f2_unc": "uncorrected: R_in",
        "f2_curve": "(1 − d/m)²",
        "f3_title": "The portfolio breaks when m falls below the statistical dimension δ",
        "f3_sub": ["Long-only, p = 400 stocks, five markets, 100 samples per point.",
                   "Share of samples whose minimum in-sample variance is exactly zero."],
        "f3_x": "days over statistical dimension, m/δ",
        "f3_names": {"iid": "independent, δ = 199", "ar1": "AR(1), δ = 121", "f1mixed": "one factor, mixed signs, δ = 130",
                     "f1pos": "one factor, positive, δ = 17", "f3pos": "three factors, positive, δ = 18"},
        "f4_title": "On NASDAQ data the calibrated portfolio tracks max(target, floor)",
        "f4_sub": ["More stocks than days. Median realized tracking error over target, monthly windows.",
                   "Solid: df calibration. Dashed: in-sample cap set equal to the target."],
        "f4_x": "target over median floor",
        "f4_ref": "max(1, floor / target)",
        "f4_names": {"126_252_random": "n = 126, p = 252", "126_504_random": "n = 126, p = 504",
                     "252_500_random": "n = 252, p = 500"},
        "f4_solid": "df calibration", "f4_dashed": "cap = target",
    },
    "zh": {
        "f1_title": "三只股票、禁止卖空：所有可行权重构成一个三角形",
        "f1_sub": ["优化器落在三角形的某个面上；这个面上的自由方向数 d = 持仓数 − 1。"],
        "f1_vertex": "顶点：只持有一只，d = 0",
        "f1_edge": "边：持有两只，d = 1",
        "f1_inside": "内部：三只都持有，d = 2",
        "f1_pt": "优化器落在这里",
        "f1_pin": "C 的权重被钉在 0",
        "f1_free": "只能沿边滑动：1 个方向",
        "f2_title": "校正前，比值沿着 (1 − d/m)² 走；校正后，贴在 1 上",
        "f2_sub": ["20 个模拟市场，n = 250 天，p = 125 到 1,000 只股票，每个 400 次。",
                   "纵轴是与真实样本外方差之比；误差棒为 ±2 个标准误。"],
        "f2_x": "自由方向占比 d/m",
        "f2_corr": "校正后：R_in / (1 − d/m)²",
        "f2_unc": "未校正：R_in",
        "f2_curve": "(1 − d/m)²",
        "f3_title": "天数 m 一旦低于统计维数 δ，组合就会崩成样本内零风险",
        "f3_sub": ["禁止卖空，p = 400 只股票，五个市场，每个点 100 次。",
                   "纵轴：样本内最小方差恰好为零的样本占比。"],
        "f3_x": "天数与统计维数之比 m/δ",
        "f3_names": {"iid": "独立资产，δ = 199", "ar1": "AR(1)，δ = 121", "f1mixed": "单因子、载荷有正有负，δ = 130",
                     "f1pos": "单因子、载荷全正，δ = 17", "f3pos": "三因子、载荷全正，δ = 18"},
        "f4_title": "纳斯达克数据上，校准后的实际跟踪误差 ≈ max(目标, 地板)",
        "f4_sub": ["股票比天数多。月度窗口的中位数：实际跟踪误差 ÷ 目标。",
                   "实线：df 校准；虚线：直接把样本内上限设成目标。"],
        "f4_x": "目标 ÷ 地板中位数",
        "f4_ref": "max(1, 地板 / 目标)",
        "f4_names": {"126_252_random": "n = 126, p = 252", "126_504_random": "n = 126, p = 504",
                     "252_500_random": "n = 252, p = 500"},
        "f4_solid": "df 校准", "f4_dashed": "上限 = 目标",
    },
}


def svg(w, h, body, label):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-label="{escape(label)}" font-family="{FONT}">'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>{body}</svg>\n')


def t(x, y, s, size=12, fill=INK, anchor="start", weight=400, extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{extra}>{escape(s)}</text>')


def title_block(title, subtitle_lines):
    out = [t(24, 34, title, 16, INK, weight=650)]
    for i, line in enumerate(subtitle_lines):
        out.append(t(24, 56 + 18 * i, line, 12, MUTED))
    return "".join(out)


def axes(x0, x1, y0, y1, X, Y, xticks, yticks, xfmt, yfmt, xlabel, ylabel=None):
    b = []
    for v in yticks:
        b.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        b.append(t(x0 - 8, Y(v) + 4.5, yfmt(v), 12, MUTED, "end"))
    for v in xticks:
        b.append(t(X(v), y1 + 20, xfmt(v), 12, MUTED, "middle"))
    b.append(f'<line x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}" stroke="{AXIS}" stroke-width="1"/>')
    b.append(t((x0 + x1) / 2, y1 + 42, xlabel, 12.5, MUTED, "middle"))
    if ylabel:
        b.append(t(x0 - 40, (y0 + y1) / 2, ylabel, 12.5, MUTED, "middle",
                   extra=f' transform="rotate(-90 {x0 - 40} {(y0 + y1) / 2})"'))
    return b


# ---------------------------------------------------------------- figure 1: the face
def fig_face(L):
    W, H = 640, 470
    A, B, C = (320, 110), (520, 420), (120, 420)
    b = [title_block(L["f1_title"], L["f1_sub"])]
    b.append(f'<polygon points="{A[0]},{A[1]} {B[0]},{B[1]} {C[0]},{C[1]}" fill="{SOFT}" stroke="{AXIS}" stroke-width="1.4"/>')
    # highlighted edge A-B
    b.append(f'<line x1="{A[0]}" y1="{A[1]}" x2="{B[0]}" y2="{B[1]}" stroke="{BLUE}" stroke-width="4" stroke-linecap="round"/>')
    for (x, y), name in ((A, "A"), (B, "B"), (C, "C")):
        b.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{NAVY}"/>')
    b.append(t(A[0], A[1] - 12, "A", 14, NAVY, "middle", 650))
    b.append(t(B[0] + 12, B[1] + 5, "B", 14, NAVY, "start", 650))
    b.append(t(C[0] - 12, C[1] + 5, "C", 14, NAVY, "end", 650))
    # point on edge A-B
    px, py = A[0] + 0.55 * (B[0] - A[0]), A[1] + 0.55 * (B[1] - A[1])
    b.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7.5" fill="{BLUE}" stroke="#ffffff" stroke-width="2"/>')
    # sliding arrows along edge
    ux, uy = (B[0] - A[0]), (B[1] - A[1])
    n = (ux ** 2 + uy ** 2) ** 0.5
    ux, uy = ux / n, uy / n
    b.append('<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
             f'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{BLUE}"/></marker></defs>')
    off = 18
    nx, ny = -uy, ux  # inward normal, so the arrow sits inside the triangle and clear of the labels
    sx, sy = px + nx * off - ux * 46, py + ny * off - uy * 46
    ex, ey = px + nx * off + ux * 46, py + ny * off + uy * 46
    b.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{BLUE}" stroke-width="1.6" '
             'marker-start="url(#ar)" marker-end="url(#ar)"/>')
    b.append(t(px + 40, py - 2, L["f1_pt"], 12.5, INK, "start", 650))
    b.append(t(px + 40, py + 16, L["f1_free"], 12, MUTED))
    b.append(t(px + 40, py + 33, L["f1_pin"], 12, MUTED))
    # region labels
    b.append(t(320, 345, L["f1_inside"], 12.5, INK, "middle"))
    b.append(t(320, 446, L["f1_edge"], 12.5, INK, "middle"))
    b.append(t(98, 110, L["f1_vertex"], 12.5, INK, "start"))
    b.append(f'<path d="M{150} {116}L{A[0] - 10} {A[1] + 2}" stroke="{AXIS}" stroke-width="1.2" fill="none"/>')
    return svg(W, H, "".join(b), L["f1_title"])


# ---------------------------------------------------------------- figure 2: accuracy
def fig_accuracy(L):
    W, H = 640, 520
    x0, x1, y0, y1 = 72, 612, 104, 440
    lo, hi = 0.0, 1.12
    X = lambda v: x0 + v / 0.95 * (x1 - x0)
    Y = lambda v: y1 - (v - lo) / (hi - lo) * (y1 - y0)
    b = [title_block(L["f2_title"], L["f2_sub"])]
    b += axes(x0, x1, y0, y1, X, Y, [0, 0.2, 0.4, 0.6, 0.8], [0, 0.25, 0.5, 0.75, 1.0],
              lambda v: f"{v:.1f}", lambda v: f"{v:.2f}".rstrip("0").rstrip(".") if v not in (0,) else "0", L["f2_x"])
    b.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(1):.1f}" y2="{Y(1):.1f}" stroke="{AXIS}" stroke-width="1.2"/>')
    pts = " ".join(f"{X(v):.1f},{Y((1 - v) ** 2):.1f}" for v in [i / 100 for i in range(0, 96)])
    b.append(f'<polyline points="{pts}" fill="none" stroke="{REF}" stroke-width="1.4" stroke-dasharray="2 4"/>')
    for _, _, dm, rin, tau, se in SIM_DF:
        b.append(f'<circle cx="{X(dm):.1f}" cy="{Y(rin):.1f}" r="5" fill="#ffffff" stroke="{MUTED}" stroke-width="1.8"/>')
        b.append(f'<line x1="{X(dm):.1f}" x2="{X(dm):.1f}" y1="{Y(tau - 2 * se):.1f}" y2="{Y(tau + 2 * se):.1f}" '
                 f'stroke="{BLUE}" stroke-width="1.4"/>')
        b.append(f'<circle cx="{X(dm):.1f}" cy="{Y(tau):.1f}" r="5" fill="{BLUE}"/>')
    # legend
    lx, ly = 340, 168
    b.append(f'<circle cx="{lx}" cy="{ly}" r="5" fill="{BLUE}"/>' + t(lx + 12, ly + 4, L["f2_corr"], 12.5, INK))
    b.append(f'<circle cx="{lx}" cy="{ly + 22}" r="5" fill="#ffffff" stroke="{MUTED}" stroke-width="1.8"/>'
             + t(lx + 12, ly + 26, L["f2_unc"], 12.5, INK))
    b.append(f'<line x1="{lx - 8}" x2="{lx + 6}" y1="{ly + 44}" y2="{ly + 44}" stroke="{REF}" stroke-width="1.4" '
             'stroke-dasharray="2 4"/>' + t(lx + 12, ly + 48, L["f2_curve"], 12.5, INK))
    return svg(W, H, "".join(b), L["f2_title"])


# ---------------------------------------------------------------- figure 3: phase transition
def fig_phase(L):
    W, H = 640, 520
    x0, x1, y0, y1 = 72, 612, 104, 440
    xlo, xhi = 0.45, 1.85
    X = lambda v: x0 + (v - xlo) / (xhi - xlo) * (x1 - x0)
    Y = lambda v: y1 - v * (y1 - y0)
    b = [title_block(L["f3_title"], L["f3_sub"])]
    b += axes(x0, x1, y0, y1, X, Y, [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8], [0, 0.25, 0.5, 0.75, 1.0],
              lambda v: f"{v:.1f}", lambda v: f"{v:.2f}".rstrip("0").rstrip(".") if v else "0", L["f3_x"])
    b.append(f'<line x1="{X(1):.1f}" x2="{X(1):.1f}" y1="{y0}" y2="{y1}" stroke="{AXIS}" stroke-width="1.2" '
             'stroke-dasharray="4 4"/>')
    b.append(t(X(1) + 6, y0 + 12, "m = δ", 12, MUTED))
    colors = {"iid": NAVY, "ar1": BLUE, "f1mixed": TEAL, "f1pos": ORANGE, "f3pos": GREY}
    order = ["iid", "ar1", "f1mixed", "f1pos", "f3pos"]
    for k in order:
        rows = [(m / d, s) for mk, d, m, s in PHASE if mk == k]
        pts = " ".join(f"{X(a):.1f},{Y(s):.1f}" for a, s in rows)
        b.append(f'<polyline points="{pts}" fill="none" stroke="{colors[k]}" stroke-width="2"/>')
        for a, s in rows:
            b.append(f'<circle cx="{X(a):.1f}" cy="{Y(s):.1f}" r="3.2" fill="{colors[k]}"/>')
    lx, ly = 372, 160
    for i, k in enumerate(order):
        y = ly + 21 * i
        b.append(f'<line x1="{lx - 18}" x2="{lx}" y1="{y}" y2="{y}" stroke="{colors[k]}" stroke-width="2.4"/>')
        b.append(t(lx + 8, y + 4, L["f3_names"][k], 12, INK))
    return svg(W, H, "".join(b), L["f3_title"])


# ---------------------------------------------------------------- figure 4: tracking error on NASDAQ
def fig_te(L):
    W, H = 640, 520
    x0, x1, y0, y1 = 72, 612, 104, 440
    xlo, xhi, ylo, yhi = 0.3, 3.0, 0.8, 3.8
    X = lambda v: x0 + (v - xlo) / (xhi - xlo) * (x1 - x0)
    Y = lambda v: y1 - (v - ylo) / (yhi - ylo) * (y1 - y0)
    b = [title_block(L["f4_title"], L["f4_sub"])]
    b += axes(x0, x1, y0, y1, X, Y, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
              lambda v: f"{v:.1f}", lambda v: f"{v:.1f}", L["f4_x"])
    xs = [xlo + i * (xhi - xlo) / 200 for i in range(201)]
    pts = " ".join(f"{X(v):.1f},{Y(min(yhi, max(1.0, 1.0 / v))):.1f}" for v in xs)
    b.append(f'<polyline points="{pts}" fill="none" stroke="{REF}" stroke-width="1.4" stroke-dasharray="2 4"/>')
    colors = {"126_252_random": NAVY, "126_504_random": BLUE, "252_500_random": TEAL}
    for k, col in colors.items():
        rows = [(tg / fl, nv, df) for c, fl, tg, nv, df in TE_NASDAQ if c == k]
        for idx, dash in ((2, ""), (1, ' stroke-dasharray="6 5"')):
            pts = " ".join(f"{X(r[0]):.1f},{Y(r[idx]):.1f}" for r in rows)
            b.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"{dash}/>')
            for r in rows:
                if idx == 2:
                    b.append(f'<circle cx="{X(r[0]):.1f}" cy="{Y(r[2]):.1f}" r="3.6" fill="{col}"/>')
                else:
                    b.append(f'<circle cx="{X(r[0]):.1f}" cy="{Y(r[1]):.1f}" r="3.6" fill="#ffffff" stroke="{col}" '
                             'stroke-width="1.6"/>')
    lx, ly = 400, 150
    for i, (k, col) in enumerate(colors.items()):
        y = ly + 21 * i
        b.append(f'<line x1="{lx - 18}" x2="{lx}" y1="{y}" y2="{y}" stroke="{col}" stroke-width="2.4"/>')
        b.append(t(lx + 8, y + 4, L["f4_names"][k], 12, INK))
    y = ly + 21 * 3 + 6
    b.append(f'<line x1="{lx - 18}" x2="{lx}" y1="{y}" y2="{y}" stroke="{INK}" stroke-width="2"/>'
             + t(lx + 8, y + 4, L["f4_solid"], 12, INK))
    b.append(f'<line x1="{lx - 18}" x2="{lx}" y1="{y + 21}" y2="{y + 21}" stroke="{INK}" stroke-width="2" '
             'stroke-dasharray="6 5"/>' + t(lx + 8, y + 25, L["f4_dashed"], 12, INK))
    b.append(f'<line x1="{lx - 18}" x2="{lx}" y1="{y + 42}" y2="{y + 42}" stroke="{REF}" stroke-width="1.4" '
             'stroke-dasharray="2 4"/>' + t(lx + 8, y + 46, L["f4_ref"], 12, INK))
    return svg(W, H, "".join(b), L["f4_title"])


for lang, L in LANG.items():
    suffix = "" if lang == "en" else "-zh"
    (OUT / f"figure-001{suffix}.svg").write_text(fig_face(L), encoding="utf-8")
    (OUT / f"figure-002{suffix}.svg").write_text(fig_accuracy(L), encoding="utf-8")
    (OUT / f"figure-003{suffix}.svg").write_text(fig_phase(L), encoding="utf-8")
    (OUT / f"figure-004{suffix}.svg").write_text(fig_te(L), encoding="utf-8")
print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
