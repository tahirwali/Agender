"""Draw the methodology diagram and the graphical abstract of the paper.

    python draw_diagrams.py OUTPUT_FOLDER

The methodology diagram is 16 cm wide, black lines, Times 9 pt. The graphical abstract keeps the
2.5 : 1 shape asked for by Elsevier (at least 1328 x 531 pixels) and uses the colours of the result
figures: orange for the privacy-aware QUBO, blue for the attacker and for the QUBO without the
privacy term, grey for everything else. Its numbers are the Kaggle results quoted in the paper.

Every label is checked against the box it sits in; a label that runs over an edge is reported.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

CM = 1 / 2.54
FILL = "#e6e6e6"  # data and settings; the steps stay white
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, LIGHT = "#2b2b2b", "#8a8a85", "#c3c2b7"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


class Diagram:
    def __init__(self, width_cm, height_cm):
        self.fig = plt.figure(figsize=(width_cm * CM, height_cm * CM))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width_cm)
        self.ax.set_ylim(0, height_cm)
        self.ax.axis("off")
        self.width, self.height = width_cm, height_cm
        self.fits = []  # (text, x0, y0, x1, y1) the text must stay inside

    def box(self, x, y, w, h, text, fill="white", heading=None, edge="black", size=None):
        """Rounded box with its lower-left corner at (x, y) cm. An optional bold heading sits at the top."""
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15",
                                         facecolor=fill, edgecolor=edge, linewidth=1.0))
        top = y + h
        if heading:
            t = self.ax.text(x + w / 2, top - 0.1, heading, ha="center", va="top", fontweight="bold", size=size)
            self.fits.append((t, x, y, x + w, top))
            top -= 0.42
        if text:
            t = self.ax.text(x + w / 2, (y + top) / 2, text, ha="center", va="center", linespacing=1.25,
                             size=size)
            self.fits.append((t, x, y, x + w, top))
        return {"left": (x, y + h / 2), "right": (x + w, y + h / 2), "top": (x + w / 2, y + h),
                "bottom": (x + w / 2, y), "x": x, "y": y, "w": w, "h": h}

    def frame(self, x, y, w, h, title):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.2",
                                         facecolor="none", edgecolor="black", linewidth=1.0,
                                         linestyle=(0, (4, 2))))
        self.label(x + w / 2, y + h - 0.3, title, fontstyle="italic", fontweight="bold")

    def label(self, x, y, text, inside=None, **kw):
        kw.setdefault("ha", "center")
        t = self.ax.text(x, y, text, va="center", linespacing=1.25, **kw)
        self.fits.append((t, *(inside or (0, 0, self.width, self.height))))
        return t

    def arrow(self, start, end, dashed=False, color="black"):
        self.ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, color=color,
                                          linewidth=1.0, linestyle=(0, (3, 2)) if dashed else "-",
                                          shrinkA=0, shrinkB=0))

    def elbow(self, points):
        xs, ys = zip(*points)
        self.ax.plot(xs[:-1], ys[:-1], color="black", linewidth=1.0)
        self.arrow(points[-2], points[-1])

    def save(self, stem):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        to_cm = self.ax.transData.inverted()
        for text, x0, y0, x1, y1 in self.fits:
            (a, b), (c, d) = to_cm.transform(text.get_window_extent(renderer).get_points())
            margin = 0.06
            if a < x0 + margin or c > x1 - margin or b < y0 + margin or d > y1 - margin:
                print(f"  does not fit in {stem.name}: {text.get_text()[:50]!r}")
        self.fig.savefig(stem.with_suffix(".pdf"))
        self.fig.savefig(stem.with_suffix(".png"))
        plt.close(self.fig)


def stack(top, heights, gap):
    """Lower edges of boxes placed from `top` downwards."""
    ys = []
    for h in heights:
        ys.append(top - h)
        top -= h + gap
    return ys


def methodology_figure(out):
    g = Diagram(16, 17.6)

    # the four datasets
    sets = [
        ("LAT", "11,034 subscribers\n24 attributes\ngender, 6 age groups\ntask: data demand"),
        ("Copenhagen", "582 students\n15 attributes from raw\ncall and SMS logs\ngender"),
        ("Iranian Churn", "2,850 customers\n10 attributes\n5 age groups\ntask: churn"),
        ("IBM Telco", "7,043 customers\n17 attributes\ngender (simulated)\nnegative control"),
    ]
    for i, (name, text) in enumerate(sets):
        g.box(0.1 + 4.0 * i, 15.3, 3.8, 2.2, text, fill=FILL, heading=name)
    g.label(8.0, 14.85, "cleaning rules fixed before any model; five repeats with seeds 42, 7, 123, 2024, 99",
            fontstyle="italic")

    # the four parts of every split
    parts = [("Training 60%", 7.0), ("Validation 10%", 2.6), ("Calibration 10%", 2.6), ("Test 20%", 3.6)]
    x = 0.1
    edges = []
    for name, w in parts:
        g.box(x, 13.2, w, 0.8, name, fill=FILL if name != "Test 20%" else "white")
        edges.append((x, x + w))
        x += w
    known = (edges[0][0] + edges[2][1]) / 2
    g.ax.plot([edges[0][0] + 0.1, edges[2][1] - 0.1], [14.15, 14.15], color="black", linewidth=0.8)
    g.label(known, 14.4, "labels known to the partner")
    g.ax.plot([edges[3][0] + 0.1, edges[3][1] - 0.1], [14.15, 14.15], color="black", linewidth=0.8)
    g.label((edges[3][0] + edges[3][1]) / 2, 14.4, "labels hidden")

    # the two parts of the method
    top, bottom = 12.7, 2.22
    g.frame(0.1, bottom, 7.75, top - bottom, "Part 1: leakage audit")
    g.frame(8.15, bottom, 7.75, top - bottom, "Part 2: release selection")
    # one line from the split into both parts
    g.ax.plot([8.0, 8.0], [13.2, 12.95], color="black", linewidth=1.0)
    g.ax.plot([3.95, 12.05], [12.95, 12.95], color="black", linewidth=1.0)
    g.arrow((3.95, 12.95), (3.95, top))
    g.arrow((12.05, 12.95), (12.05, top))

    lx, rx, w = 0.35, 8.4, 7.25
    col_top = 12.05
    # Part 1
    hs = [2.0, 1.36, 2.06, 1.36]
    y1, y2, y3, y4 = stack(col_top, hs, 0.93)
    attackers = g.box(lx, y1, w, hs[0],
                      "logistic regression, LightGBM, CatBoost, MLP,\nmulti-task network (LAT only)\n"
                      "encodings and models fitted on the training part;\nsettings chosen by validation log-loss",
                      heading="Five attackers for each private label")
    ref = g.box(lx, y2, 3.5, hs[1], "logistic regression on\nshuffled training labels", fill=FILL,
                heading="Reference attacker")
    chosen = g.box(lx + 3.75, y2, 3.5, hs[1], "the one with the best\nvalidation AUC", heading="Chosen attacker")
    g.arrow((chosen["top"][0], y1), chosen["top"])
    g.arrow((ref["top"][0], y1), ref["top"])
    sets_box = g.box(lx, y3, w, hs[2],
                     "one threshold $\\hat{q}_y$ per class from the calibration part\n"
                     "$C(x) = \\{\\, y : 1 - \\hat{p}(y \\mid x) \\leq \\hat{q}_y \\,\\}$\n"
                     "promised coverage 95%, 90% and 80%",
                     heading="Class-conditional conformal sets")
    g.arrow(ref["bottom"], (ref["bottom"][0], y3 + hs[2]))
    g.arrow(chosen["bottom"], (chosen["bottom"][0], y3 + hs[2]))
    exposure = g.box(lx, y4, w, hs[3],
                     "gender: correct single-class sets minus the reference\n"
                     "age: mean set size, sets with at most two groups",
                     heading="Exposure of each test subscriber")
    g.arrow(sets_box["bottom"], exposure["top"])

    # Part 2
    hs = [1.71, 1.92, 1.71, 2.06, 1.36]
    r1, r2, r3, r4, r5 = stack(col_top, hs, 0.2)
    info = g.box(rx, r1, w, hs[0],
                 "$U_i$ task relevance, $Q_{ij}$ redundancy, $P_i$ private\n"
                 "information, $P_{ij}$ pair synergy; Miller–Madow entropies",
                 heading="Information terms on the training part")
    qubo = g.box(rx, r2, w, hs[1],
                 "$E(\\mathbf{x}) = -\\alpha\\, \\Sigma_i U_i x_i + \\beta\\, \\Sigma_{i<j} Q_{ij} x_i x_j"
                 " + \\lambda\\, (\\Sigma_i x_i - k)^2$\n"
                 "$+\\, \\gamma\\, (\\Sigma_i P_i x_i + \\Sigma_{i<j} P_{ij} x_i x_j)$,"
                 "  $\\gamma \\in \\{0, 0.5, 1, 2, 4, 8, 16\\}$",
                 heading="Privacy-aware QUBO")
    solver = g.box(rx, r3, w, hs[2],
                   "6 restarts of $2 \\times 10^4$ moves, $T$ from 5 to 0.01; every\n"
                   "second move a swap; checked against all $2^{24}$ subsets",
                   heading="Simulated annealing with swap moves")
    candidates = g.box(rx, r4, w, hs[3],
                       "QUBO with and without the privacy term,\n"
                       "release everything, remove the most revealing,\n"
                       "coarsen values, add noise",
                       fill=FILL, heading="Candidate releases of each defence")
    choose = g.box(rx, r5, w, hs[4],
                   "at least 90% of validation utility,\nthen the lowest validation leakage",
                   heading="Release chosen on validation")
    for a, b in ((info, qubo), (qubo, solver), (solver, candidates), (candidates, choose)):
        g.arrow(a["bottom"], b["top"])

    # scoring on the test part
    test = g.box(0.1, 0.1, 15.8, 1.85,
                 "leakage $L(R)$: mean of $(\\mathrm{AUC}_t(R) - 0.5)\\,/\\,(\\mathrm{AUC}_t(\\mathrm{all}) - 0.5)$"
                 " over the private labels $t$\n"
                 "utility kept: task score of $R$ divided by the task score of the full release\n"
                 "full audit of each chosen release; means over five repeats with 95% $t$ intervals;"
                 " paired differences between defences",
                 heading="Test part, scored once per repeat")
    g.arrow(exposure["bottom"], (exposure["bottom"][0], 1.95))
    g.arrow(choose["bottom"], (choose["bottom"][0], 1.95))
    g.save(out / "fig_methodology")


def bars(g, x, y, rows, scale, size=8.5):
    """Labelled horizontal bars from the top row at y downwards: (label, value, colour)."""
    for label, value, colour in rows:
        g.ax.text(x, y, label, ha="left", va="bottom", size=size, color=INK)
        g.ax.add_patch(Rectangle((x, y - 0.32), value * scale, 0.28, facecolor=colour, edgecolor="none"))
        g.ax.text(x + value * scale + 0.08, y - 0.18, f"{value:.1f}%", ha="left", va="center", size=size,
                  color=INK)
        y -= 0.78


def graphical_abstract(out):
    g = Diagram(16, 6.4)
    top, bottom = 6.3, 1.05
    hw = 4.85
    xs = [0.1, 5.575, 11.05]
    titles = ["What the operator shares", "Audit: who is exposed?", "What to release instead"]
    for x, title in zip(xs, titles):
        g.box(x, bottom, hw, top - bottom, "", heading=title, size=10)
    for x in xs[:2]:
        g.arrow((x + hw + 0.08, 3.7), (x + hw + 0.545, 3.7))

    # panel 1: a usage table with the identifiers removed
    x0 = xs[0] + 0.3
    cols = ["name", "recharge", "calls", "MB", "sessions"]
    widths = [0.75, 1.0, 0.7, 0.7, 1.0]
    rows = [["", "35", "112", "640", "58"], ["", "10", "41", "95", "17"], ["", "50", "208", "1210", "96"],
            ["", "20", "77", "380", "33"]]
    ytop = 5.45
    cx = x0
    for c, cw in zip(cols, widths):
        g.ax.add_patch(Rectangle((cx, ytop - 0.42), cw, 0.42, facecolor=FILL, edgecolor=MUTED, linewidth=0.6))
        g.ax.text(cx + cw / 2, ytop - 0.21, c, ha="center", va="center", size=7.5, color=INK)
        for r, row in enumerate(rows):
            yr = ytop - 0.42 * (r + 2)
            g.ax.add_patch(Rectangle((cx, yr), cw, 0.42, facecolor="white", edgecolor=LIGHT, linewidth=0.6))
            value = row[cols.index(c)]
            if c == "name":
                g.ax.plot([cx + 0.15, cx + cw - 0.15], [yr + 0.21, yr + 0.21], color=MUTED, linewidth=1.2)
            else:
                g.ax.text(cx + cw / 2, yr + 0.21, value, ha="center", va="center", size=7.5, color=INK)
        cx += cw
    g.label(xs[0] + hw / 2, 2.75, "names and numbers removed,\nusage aggregates released to a partner",
            size=8.5, inside=(xs[0], bottom, xs[0] + hw, top))
    g.label(xs[0] + hw / 2, 1.6, "Does the partner learn\nage and gender anyway?", size=8.5, fontstyle="italic",
            inside=(xs[0], bottom, xs[0] + hw, top))

    # panel 2: conformal sets and the share of exposed subscribers
    x0 = xs[1] + 0.3
    chips = [("{F}", "revealed", BLUE), ("{F, M}", "not revealed", None), ("{M}", "revealed", BLUE)]
    for i, (text, note, colour) in enumerate(chips):
        cx = x0 + 1.47 * i
        g.ax.add_patch(FancyBboxPatch((cx, 4.8), 1.3, 0.45, boxstyle="round,pad=0,rounding_size=0.12",
                                      facecolor="white", edgecolor=colour or MUTED, linewidth=1.2))
        g.ax.text(cx + 0.65, 5.025, text, ha="center", va="center", size=8.5, color=INK)
        g.ax.text(cx + 0.65, 4.55, note, ha="center", va="center", size=7.5, color=INK)
    g.label(xs[1] + hw / 2, 4.0, "a conformal set holds the true gender\nfor 90% of subscribers in each class",
            size=8.5, inside=(xs[1], bottom, xs[1] + hw, top))
    bars(g, x0, 3.2, [("chosen attacker", 27.3, BLUE),
                       ("attacker that knows nothing", 8.7, LIGHT)], scale=0.11)
    g.label(xs[1] + hw / 2, 1.5,"LAT, full release: test subscribers\nwith a single, correct gender",
            size=8, inside=(xs[1], bottom, xs[1] + hw, top))

    # panel 3: gender exposure of the chosen releases
    x0 = xs[2] + 0.3
    bars(g, x0, 5.25, [("release everything", 18.5, LIGHT), ("add noise", 10.2, LIGHT),
                       ("QUBO aimed at the partner's task", 3.2, BLUE),
                       ("QUBO with the privacy term", 3.9, ORANGE)], scale=0.17)
    g.label(xs[2] + hw / 2, 1.65, "gender exposure beyond the reference\nprivacy term: 4 attributes, 94% utility,"
            "\nage AUC from 0.699 to 0.618", size=8, inside=(xs[2], bottom, xs[2] + hw, top))

    g.label(8.0, 0.5, "Iranian Churn: the age group leaks strongly (AUC 0.964).   "
            "IBM Telco, simulated customers: no gender leak found.", size=8.5)
    g.save(out / "graphical_abstract")


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    folder.mkdir(parents=True, exist_ok=True)
    methodology_figure(folder)
    graphical_abstract(folder)
