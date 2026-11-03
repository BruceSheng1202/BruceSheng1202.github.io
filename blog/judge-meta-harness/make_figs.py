"""Generate SVG figures 004-008 for the judge-meta-harness blog post.

All numbers below were verified against the run logs (see the fact-check notes).
Palette follows blog/assets/style.css: ink #172033, muted #667085, navy #00205b,
royal blue #4169e1, warm line #e8e5df.
"""
import sys
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, NAVY, BLUE = "#172033", "#667085", "#00205b", "#4169e1"
GRID, AXIS, DOT, REF, SOFT = "#ece9e3", "#c9c4bb", "#bdb7ad", "#8a93a6", "#eef3ff"
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"


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


# ---------------------------------------------------------------- figure 4
def fig_loop():
    W, H = 760, 470
    bw, bh = 212, 74
    c1, c2, c3 = 136, 380, 624
    rA, rB = 140, 296
    b = [title_block("One round of the meta-harness search",
                     ["A candidate becomes the new reference only after passing two paired gates."])]
    b.append('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{AXIS}"/></marker></defs>')

    def box(cx, cy, head, l1, l2, gate=False):
        x, y = cx - bw / 2, cy - bh / 2
        fill, stroke, sw = (SOFT, BLUE, 1.8) if gate else ("#ffffff", AXIS, 1.2)
        col = INK if gate else MUTED
        return (f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
                + t(cx, cy - 12, head, 13.5, NAVY if gate else INK, "middle", 650)
                + t(cx, cy + 8, l1, 11.5, col, "middle")
                + t(cx, cy + 25, l2, 11.5, col, "middle"))

    def line(d):
        return f'<path d="{d}" fill="none" stroke="{AXIS}" stroke-width="1.4" marker-end="url(#arr)"/>'

    b.append(line(f"M{c1 + bw/2} {rA}H{c2 - bw/2 - 2}"))
    b.append(line(f"M{c2 + bw/2} {rA}H{c3 - bw/2 - 2}"))
    b.append(line(f"M{c3} {rA + bh/2}V{rB - bh/2 - 2}"))
    b.append(line(f"M{c3 - bw/2} {rB}H{c2 + bw/2 + 2}"))
    b.append(line(f"M{c2 - bw/2} {rB}H{c1 + bw/2 + 2}"))
    b.append(line(f"M{c1} {rB - bh/2}V{rA + bh/2 + 2}"))
    b.append(line(f"M{c1} {rB + bh/2}V{388}"))

    b.append(box(c1, rA, "Proposer", "reads past code and per-row traces", "writes 4 candidates per round"))
    b.append(box(c2, rA, "Static checks", "import allowlist, leak scan", "mock contract tests, ≤ 2 repairs"))
    b.append(box(c3, rA, "Train-set eval", "2,000 rows, traces sent back", "score does not decide"))
    b.append(box(c3, rB, "Dev gate", "1,000 rotating rows, paired", "ΔMF1 ≥ +0.3, F1⁻ not lower", True))
    b.append(box(c2, rB, "Validation gate", "all 7,404 rows, paired", "95% CI lower bound > 0, F1⁻ not lower", True))
    b.append(box(c1, rB, "New reference", "accepted only if both gates pass", "next round starts here"))

    b.append(t(c3 - 10, (rA + rB) / 2 + 4, "timing and failure-rate check", 11.5, MUTED, "end"))
    b.append(t((c2 + c3) / 2, rB - 10, "pass", 11.5, MUTED, "middle"))
    b.append(t((c1 + c2) / 2, rB - 10, "pass", 11.5, MUTED, "middle"))
    b.append(t(c1 + 12, (rA + rB) / 2 + 4, "next round", 11.5, MUTED))
    b.append(t(c1 + 12, 366, "at the deadline", 11.5, MUTED))
    b.append(t(736, 366, "fail either gate: rejected, logged to history", 11.5, MUTED, "end"))

    b.append(f'<rect x="24" y="390" width="712" height="58" rx="9" fill="#f7f6f3" stroke="{AXIS}" stroke-width="1.2"/>')
    b.append(t(380, 413, "Final test, run once", 13.5, INK, "middle", 650))
    b.append(t(380, 432, "best 2 accepted candidates + 3 seeds on the 14,728 test rows; paired CIs against the seed reference",
               11.5, MUTED, "middle"))
    return svg(W, H, "".join(b), "One round of the meta-harness search")


# ---------------------------------------------------------------- dot plots
def dot_plot(title, subtitle, rows, ref, ref_label, lo, hi, ticks, label_w=300, name_x=None, aria=""):
    W = 760
    x0, x1 = label_w + 18, 712
    y0, step = 104 + 18 * (len(subtitle) - 1), 27
    n = len(rows)
    yEnd = y0 + (n - 1) * step + 16
    H = int(yEnd + 52)
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    b = [title_block(title, subtitle)]
    for tk in ticks:
        b.append(f'<line x1="{X(tk):.1f}" x2="{X(tk):.1f}" y1="{y0 - 16}" y2="{yEnd}" stroke="{GRID}"/>')
        b.append(t(X(tk), yEnd + 18, f"{tk}", 11.5, MUTED, "middle"))
    b.append(t(x1, yEnd + 38, "MF1 (%)", 11.5, MUTED, "end"))
    b.append(f'<line x1="{X(ref):.1f}" x2="{X(ref):.1f}" y1="{y0 - 16}" y2="{yEnd}" stroke="{REF}" stroke-dasharray="4 4"/>')
    b.append(t(X(ref) + 6, y0 - 22, ref_label, 11.5, MUTED))
    nx = name_x if name_x is not None else label_w
    for i, r in enumerate(rows):
        y = y0 + i * step
        kind, v = r["kind"], r["v"]
        strong = kind in ("harness", "calibrated")
        b.append(t(nx, y + 4, r["name"], 12.5, INK, "end", 650 if strong else 400))
        if kind in ("harness", "calibrated"):
            b.append(f'<circle cx="{X(v):.1f}" cy="{y}" r="6.5" fill="{BLUE}"/>')
        elif kind == "label":
            b.append(f'<circle cx="{X(v):.1f}" cy="{y}" r="5.6" fill="#ffffff" stroke="{MUTED}" stroke-width="2"/>')
        elif kind == "jev":
            b.append(f'<circle cx="{X(v):.1f}" cy="{y}" r="5.6" fill="#ffffff" stroke="{NAVY}" stroke-width="2.4"/>')
        else:
            b.append(f'<circle cx="{X(v):.1f}" cy="{y}" r="6.5" fill="{DOT}"/>')
        left = v < ref
        b.append(t(X(v) + (-12 if left else 12), y + 4, f"{v:.2f}", 11.5, INK if strong else MUTED,
                   "end" if left else "start", 600 if strong else 400))
    return svg(W, H, "".join(b), aria or title)


def fig_same_rows():
    rows = [
        dict(name="Gemma-4-E4B + harness (Fable)", kind="harness", v=72.58),
        dict(name="Gemma-4-E4B + harness (Astra)", kind="harness", v=70.17),
        dict(name="Qwen3.8-27B-abliterated", kind="raw", v=69.14),
        dict(name="Qwen3.8-27B", kind="raw", v=69.03),
        dict(name="Qwen3.8-27B-uncensored", kind="raw", v=68.77),
        dict(name="Gemma4-31B", kind="raw", v=68.56),
        dict(name="Gemma-4-E4B + seed harness", kind="harness", v=68.35),
        dict(name="Jev 1.13 (decision model)", kind="jev", v=68.18),
        dict(name="Claude Sonnet 5", kind="raw", v=67.89),
        dict(name="Gemma4-12B", kind="raw", v=67.77),
        dict(name="Qwen3-14B", kind="raw", v=64.87),
        dict(name="Qwen3.5-4B", kind="raw", v=62.98),
        dict(name="Gemma-4-E4B, bare", kind="raw", v=61.66),
    ]
    return dot_plot("Same 14,728 test rows: a 4B judge with a harness beats every bare judge",
                    ["MF1 per physician label. Bare judges use the official grader prompt; harnesses also use the",
                     "7,379 labeled calibration rows. Ring = decision model (no generated text)."],
                    rows, 69.14, "best bare judge", 58, 76, [60, 64, 68, 72, 76], label_w=272)


def fig_ablation():
    rows = [
        dict(name="Full r07b: two views + calibration + prior", kind="calibrated", v=72.58),
        dict(name="Compact-contrast view + calibration", kind="calibrated", v=72.53),
        dict(name="Two views + calibration, no lexical prior", kind="calibrated", v=72.19),
        dict(name="Grading-notes view + calibration", kind="calibrated", v=71.92),
        dict(name="Compact-contrast view, own verdict", kind="view", v=70.27),
        dict(name="Grading-notes view, own verdict", kind="view", v=69.04),
        dict(name="Lexical classifier only (no LLM)", kind="label", v=66.60),
        dict(name="Constant verdict per rubric (no LLM)", kind="label", v=63.14),
        dict(name="Bare judge (official prompt)", kind="raw", v=61.11),
    ]
    return dot_plot("Reading better gets to 69; calibrating on labels adds 2–3 more",
                    ["MF1 on the same test rows, recomputed from r07b's per-row logs with no new inference.",
                     "Blue = calibrated on the 2,000 training labels. Hollow = no LLM call at all."],
                    rows, 69.14, "best bare judge 69.14", 58, 76, [60, 64, 68, 72, 76], label_w=300)


# ---------------------------------------------------------------- figure 7
def fig_gates():
    data = {
        "v2 (Astra)": [2.34, 2.02, 0.88],
        "v3 (Astra)": [5.07, 0.90, 0.70],
        "v4 (Astra)": [1.55, 2.20, 1.82],
        "v4 (Fable)": [5.53, 4.54, 4.23],
    }
    series = ["proposer-visible set", "validation", "test"]
    fills = ["#dde3f0", "#9aa6bd", BLUE]
    W, H = 760, 372
    left, gw, bw, gap, yBase, hiV = 76, 162, 36, 8, 318, 6
    Y = lambda v: yBase - v / hiV * 200
    b = [title_block("A fixed dev subset turns noise into gains; under v4, validation gains hold on test",
                     ["MF1 gain of each run's final reference over prior_best_v1 in the same run (points)."])]
    for k, (s, f) in enumerate(zip(series, fills)):
        x = 300 + k * 150
        b.append(f'<rect x="{x}" y="78" width="12" height="12" rx="2" fill="{f}"/>')
        b.append(t(x + 18, 88, s, 11.5, MUTED))
    for tk in [0, 2, 4, 6]:
        b.append(f'<line x1="{left}" x2="724" y1="{Y(tk):.1f}" y2="{Y(tk):.1f}" stroke="{AXIS if tk == 0 else GRID}"/>')
        b.append(t(left - 8, Y(tk) + 4, "0" if tk == 0 else f"+{tk}", 11.5, MUTED, "end"))
    for gi, (g, vals) in enumerate(data.items()):
        gx = left + gi * gw + 22
        for si, v in enumerate(vals):
            x = gx + si * (bw + gap)
            b.append(f'<rect x="{x}" y="{Y(v):.1f}" width="{bw}" height="{yBase - Y(v):.1f}" fill="{fills[si]}"/>')
            b.append(t(x + bw / 2, Y(v) - 6, f"{v:.2f}", 11, INK if si == 2 else MUTED, "middle", 600 if si == 2 else 400))
        b.append(t(gx + (3 * bw + 2 * gap) / 2, yBase + 22, g, 12.5, INK, "middle"))
    b.append(t(24, H - 8, "Visible set: v2/v3 fixed 600-row dev, v4 2,000-row train. Validation: v2 fixed 1,600-row gate, others all 7,404 rows. v3 shown at r36a.",
               10.5, MUTED))
    return svg(W, H, "".join(b), "Gate gains across four searches")


# ---------------------------------------------------------------- figure 8
def fig_traj():
    steps = {
        "Astra": [(0, 67.12, "seed"), (15, 68.71, "r15d"), (29, 69.32, "r29a")],
        "Fable": [(0, 66.94, "seed"), (1, 69.07, "r01a"), (3, 70.24, "r03a"), (6, 70.90, "r06a"), (7, 71.48, "r07b")],
    }
    ends = {"Astra": dict(rounds=60, accepted=2, test=70.17), "Fable": dict(rounds=16, accepted=4, test=72.58)}
    col = {"Fable": BLUE, "Astra": "#9aa3b5"}
    W, H = 760, 372
    x0, x1, yB = 70, 700, 320
    X = lambda r: x0 + r / 60 * (x1 - x0)
    Y = lambda v: yB - (v - 66) / 7 * 196
    b = [title_block("Fable reached 71.5 validation MF1 in 7 rounds; Astra stopped at 69.3 after 60",
                     ["Same protocol and seeds. Dots = accepted references; diamonds = final reference on test."])]
    for tk in [66, 68, 70, 72]:
        b.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(tk):.1f}" y2="{Y(tk):.1f}" stroke="{GRID}"/>')
        b.append(t(x0 - 8, Y(tk) + 4, f"{tk}", 11.5, MUTED, "end"))
    b.append(t(x0, 98, "Validation MF1 of the reference (%)", 11.5, MUTED))
    b.append(f'<line x1="{x0}" x2="{x1}" y1="{yB}" y2="{yB}" stroke="{AXIS}"/>')
    for tk in range(0, 61, 10):
        b.append(t(X(tk), yB + 18, f"{tk}", 11.5, MUTED, "middle"))
    b.append(t(x1, yB + 38, "search round", 11.5, MUTED, "end"))
    for name in ["Astra", "Fable"]:
        s, e, c = steps[name], ends[name], col[name]
        d = f"M{X(s[0][0]):.1f} {Y(s[0][1]):.1f}"
        for r, v, _ in s[1:]:
            d += f"H{X(r):.1f}V{Y(v):.1f}"
        d += f"H{X(e['rounds']):.1f}"
        b.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2.6"/>')
        for r, v, lab in s:
            b.append(f'<circle cx="{X(r):.1f}" cy="{Y(v):.1f}" r="4.8" fill="{c}"/>')
        r, v, lab = s[-1]
        b.append(t(X(r), Y(v) - 11, lab, 11.5, MUTED, "middle"))
        tx, ty = X(e["rounds"]), Y(e["test"])
        b.append(f'<path d="M{tx:.1f} {ty - 6.5:.1f}l6.5 6.5l-6.5 6.5l-6.5 -6.5z" fill="#ffffff" stroke="{c}" stroke-width="2.2"/>')
        far = e["rounds"] > 30
        b.append(t(tx + (-12 if far else 12), ty + (-10 if far else 4), f"test {e['test']:.2f}", 11.5, INK,
                   "end" if far else "start", 600))
        last_v = s[-1][1]
        if far:
            b.append(t(tx, Y(last_v) + 22, f"{name}: {e['rounds']} rounds, {e['accepted']} accepted", 12.5, INK, "end", 650))
        else:
            b.append(t(tx + 12, Y(last_v) + 4, f"{name}: {e['rounds']} rounds, {e['accepted']} accepted", 12.5, INK, "start", 650))
    return svg(W, H, "".join(b), "Validation MF1 of the reference across search rounds")


figs = {
    "figure-004.svg": fig_loop(),
    "figure-005.svg": fig_same_rows(),
    "figure-006.svg": fig_ablation(),
    "figure-007.svg": fig_gates(),
    "figure-008.svg": fig_traj(),
}
for name, s in figs.items():
    (OUT / name).write_text(s, encoding="utf-8")
    print("wrote", OUT / name, len(s))
