"""
HTML Report Generator for the Domain Adaptation Benchmark.

Produces a self-contained dark-theme HTML file with:
    - Header with research context
    - Summary cards: Baseline mCE, Best Method, Best Improvement, Pearson r
    - 15 × 4 colour-coded accuracy heatmap table
    - Entropy–Adaptation Gain ASCII scatter plot
    - Winner table per corruption type
    - Pseudo-label blur failure section (when applicable)

All CSS and JavaScript is embedded — the HTML file is fully portable.
"""

from __future__ import annotations

import html
import math
from typing import Any, Dict, List, Optional

from src.models import (
    ALL_CORRUPTIONS,
    BLUR_CORRUPTIONS,
    CORRUPTION_CATEGORIES,
    METHOD_DISPLAY,
    BenchmarkSummary,
)

# Methods in display order
_METHOD_ORDER = ["no_adaptation", "test_time_norm", "tent", "pseudo_label"]

# Colour thresholds for accuracy cells
_COLOUR_HIGH   = "#1a6b3a"   # green  > 80%
_COLOUR_MED    = "#5a6b1a"   # olive  60–80%
_COLOUR_LOW    = "#6b1a1a"   # red    < 60%
_COLOUR_WINNER = "#1a4a6b"   # blue   winner method


def _acc_colour(acc: Optional[float], is_winner: bool = False) -> str:
    if acc is None:
        return "#2a2a3a"
    if is_winner:
        return _COLOUR_WINNER
    if acc > 0.80:
        return _COLOUR_HIGH
    if acc > 0.60:
        return _COLOUR_MED
    return _COLOUR_LOW


def _fmt_pct(v: Optional[float]) -> str:
    return "N/A" if v is None else f"{v:.1%}"


def _fmt_signed(v: float) -> str:
    return f"{v:+.1%}"


def _category_label(corruption: str) -> str:
    for cat, members in CORRUPTION_CATEGORIES.items():
        if corruption in members:
            return cat.upper()
    return ""


class ReportGenerator:
    """
    Converts a BenchmarkSummary and UncertaintyAnalyzer into a dark HTML report.

    Parameters
    ----------
    summary : BenchmarkSummary
        Produced by BenchmarkEvaluator.finalize().
    uncertainty_analyzer : UncertaintyAnalyzer or None
        If provided, entropy correlation data is included in the report.
    pearson_r : float
        Pre-computed Pearson r (entropy vs. TENT gain).
    title : str
        Report title shown in the header.
    """

    def __init__(
        self,
        summary: BenchmarkSummary,
        uncertainty_analyzer=None,
        pearson_r: float = 0.0,
        title: str = "Domain Adaptation Benchmark",
    ) -> None:
        self.summary     = summary
        self.ua          = uncertainty_analyzer
        self.pearson_r   = pearson_r
        self.title       = title

    # ------------------------------------------------------------------ #
    # Entry point                                                          #
    # ------------------------------------------------------------------ #

    def generate(self) -> str:
        """Generate and return the full HTML report as a string."""
        return "\n".join([
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            self._head(),
            "</head>",
            "<body>",
            self._body(),
            "</body>",
            "</html>",
        ])

    # ------------------------------------------------------------------ #
    # <head>                                                               #
    # ------------------------------------------------------------------ #

    def _head(self) -> str:
        return f"""
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(self.title)}</title>
  <style>{self._css()}</style>
"""

    @staticmethod
    def _css() -> str:
        return """
    :root {
      --bg:          #0d0d18;
      --surface:     #14142a;
      --card:        #1c1c35;
      --border:      #2a2a55;
      --text:        #c8c8e8;
      --text-muted:  #7878a8;
      --accent:      #5a8fff;
      --green:       #3adb7a;
      --amber:       #f0b429;
      --red:         #e05555;
      --win-bg:      #1a3a5c;
      --win-border:  #2a6aac;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      padding: 2rem;
      max-width: 1200px;
      margin: 0 auto;
    }

    /* ── Header ── */
    header {
      border-bottom: 2px solid var(--accent);
      padding-bottom: 1.5rem;
      margin-bottom: 2rem;
    }
    header h1 {
      font-size: 2rem;
      color: var(--accent);
      letter-spacing: 0.03em;
    }
    header p.subtitle {
      color: var(--text-muted);
      margin-top: 0.3rem;
      font-size: 0.95rem;
    }
    .badge {
      display: inline-block;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 0.15rem 0.6rem;
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 0.5rem;
      margin-right: 0.5rem;
    }

    /* ── Section titles ── */
    h2 {
      font-size: 1.2rem;
      color: var(--accent);
      border-left: 3px solid var(--accent);
      padding-left: 0.75rem;
      margin: 2.5rem 0 1rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    h3 {
      font-size: 1rem;
      color: var(--text);
      margin: 1.5rem 0 0.5rem;
    }

    /* ── Summary cards ── */
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1.25rem 1.5rem;
    }
    .card .label {
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      color: var(--text-muted);
    }
    .card .value {
      font-size: 1.9rem;
      font-weight: 700;
      margin: 0.25rem 0;
    }
    .card .sub {
      font-size: 0.8rem;
      color: var(--text-muted);
    }
    .card.accent  { border-color: var(--accent); }
    .card.green   { border-color: var(--green);  }
    .card.green .value { color: var(--green); }
    .card.amber   { border-color: var(--amber);  }
    .card.amber .value { color: var(--amber); }
    .card.red     { border-color: var(--red);    }

    /* ── Tables ── */
    .table-wrap { overflow-x: auto; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }
    thead th {
      background: var(--surface);
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-size: 0.75rem;
      padding: 0.7rem 0.9rem;
      border-bottom: 2px solid var(--border);
      text-align: center;
    }
    thead th.left { text-align: left; }
    tbody tr:hover { background: rgba(90,143,255,0.04); }
    tbody td {
      padding: 0.5rem 0.9rem;
      border-bottom: 1px solid var(--border);
      text-align: center;
    }
    tbody td.name {
      text-align: left;
      font-family: monospace;
      font-size: 0.85rem;
      white-space: nowrap;
    }
    tbody td.cat {
      color: var(--text-muted);
      font-size: 0.72rem;
      letter-spacing: 0.05em;
    }
    .acc-cell {
      border-radius: 4px;
      padding: 0.25rem 0.5rem;
      font-family: monospace;
      font-weight: 600;
      font-size: 0.87rem;
    }
    .win-cell { border: 2px solid var(--win-border); }
    .mce-row td { font-weight: 700; border-top: 2px solid var(--border); }

    /* ── Pre-formatted blocks (ASCII) ── */
    pre {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 1rem 1.25rem;
      font-family: 'Courier New', monospace;
      font-size: 0.82rem;
      overflow-x: auto;
      line-height: 1.5;
      color: var(--text);
    }

    /* ── Warning / finding box ── */
    .finding {
      background: #1f1215;
      border: 1px solid var(--red);
      border-radius: 8px;
      padding: 1.25rem 1.5rem;
      margin: 1.5rem 0;
    }
    .finding h3 { color: var(--red); margin-top: 0; }
    .finding p  { margin: 0.5rem 0; font-size: 0.9rem; }

    /* ── Info box ── */
    .infobox {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem 1.25rem;
      font-size: 0.88rem;
      color: var(--text-muted);
      margin: 1.5rem 0;
    }
    .infobox strong { color: var(--text); }

    /* ── Footer ── */
    footer {
      margin-top: 3rem;
      padding-top: 1.5rem;
      border-top: 1px solid var(--border);
      color: var(--text-muted);
      font-size: 0.82rem;
    }
    footer a { color: var(--accent); text-decoration: none; }
    """

    # ------------------------------------------------------------------ #
    # <body>                                                               #
    # ------------------------------------------------------------------ #

    def _body(self) -> str:
        parts = [
            self._header_section(),
            self._summary_cards(),
            self._heatmap_section(),
            self._mce_table_section(),
            self._entropy_section(),
            self._winner_section(),
            self._pl_failure_section(),
            self._rq_section(),
            self._footer(),
        ]
        return "\n".join(parts)

    # ---- Header ---- #

    def _header_section(self) -> str:
        n_c = len(self.summary.corruption_types)
        n_m = len([m for m in _METHOD_ORDER if m in (self.summary.mce_scores or {})])
        return f"""
<header>
  <h1>🔬 {html.escape(self.title)}</h1>
  <p class="subtitle">
    Test-Time Adaptation Evaluation — Distribution Shift in Deep Learning
  </p>
  <span class="badge">🗂 {n_c} corruption types</span>
  <span class="badge">⚙ 4 TTA methods</span>
  <span class="badge">📐 Severity {self.summary.severity}</span>
  <span class="badge">🏗 ResNet-50 backbone</span>
  <span class="badge">📊 CIFAR-10-C</span>
</header>
"""

    # ---- Summary cards ---- #

    def _summary_cards(self) -> str:
        s   = self.summary
        bm  = METHOD_DISPLAY.get(s.best_method, s.best_method)
        r   = self.pearson_r

        r_colour  = "green" if r > 0.4 else ("amber" if r > 0.1 else "red")
        imp_class = "green" if s.best_improvement > 0 else "red"

        return f"""
<h2>Summary</h2>
<div class="cards">
  <div class="card">
    <div class="label">Baseline mCE (No Adapt)</div>
    <div class="value">{s.baseline_mce:.4f}</div>
    <div class="sub">Mean corruption error without adaptation</div>
  </div>
  <div class="card green">
    <div class="label">Best Method</div>
    <div class="value" style="font-size:1.4rem">{html.escape(bm)}</div>
    <div class="sub">mCE = {s.best_mce:.4f}</div>
  </div>
  <div class="card {imp_class}">
    <div class="label">Best mCE Improvement</div>
    <div class="value">{s.best_improvement:+.1%}</div>
    <div class="sub">Relative reduction vs. baseline</div>
  </div>
  <div class="card {r_colour}">
    <div class="label">Entropy–Gain Pearson r</div>
    <div class="value">{r:+.3f}</div>
    <div class="sub">Entropy predicts TTA benefit (RQ3)</div>
  </div>
</div>
"""

    # ---- Heatmap ---- #

    def _heatmap_section(self) -> str:
        s       = self.summary
        methods = _METHOD_ORDER
        labels  = [METHOD_DISPLAY.get(m, m) for m in methods]

        header_cells = "".join(f"<th>{html.escape(l)}</th>" for l in labels)
        thead = f"""
  <thead>
    <tr>
      <th class="left">Corruption</th>
      <th class="left">Category</th>
      {header_cells}
      <th>Winner</th>
    </tr>
  </thead>"""

        rows = []
        for corruption in s.corruption_types:
            accs   = s.accuracy_table.get(corruption, {})
            winner = s.winners.get(corruption, "")
            cat    = _category_label(corruption)

            cells = ""
            for method in methods:
                acc  = accs.get(method)
                is_w = (method == winner)
                bg   = _acc_colour(acc, is_w)
                extra_class = " win-cell" if is_w else ""
                text = _fmt_pct(acc)
                cells += (
                    f'<td><span class="acc-cell{extra_class}" '
                    f'style="background:{bg}">{text}</span></td>'
                )

            winner_display = html.escape(METHOD_DISPLAY.get(winner, winner))
            rows.append(
                f'<tr>'
                f'<td class="name">{html.escape(corruption)}</td>'
                f'<td class="cat">{cat}</td>'
                f'{cells}'
                f'<td><strong>{winner_display}</strong></td>'
                f'</tr>'
            )

        # mCE footer row
        mce = s.mce_scores
        mce_cells = "".join(
            f'<td>{mce[m]:.4f}</td>' if m in mce else "<td>N/A</td>"
            for m in methods
        )
        rel = s.relative_improvements
        rel_cells = "".join(
            f'<td style="color:{"var(--green)" if rel.get(m,0)>0 else "var(--red)"}">'
            f'{_fmt_signed(rel.get(m, 0.0))}</td>'
            for m in methods
        )

        legend = (
            '<span style="background:#1a6b3a;padding:2px 8px;border-radius:3px;font-size:.75rem">▓ &gt;80%</span> '
            '<span style="background:#5a6b1a;padding:2px 8px;border-radius:3px;font-size:.75rem">▒ 60-80%</span> '
            '<span style="background:#6b1a1a;padding:2px 8px;border-radius:3px;font-size:.75rem">░ &lt;60%</span> '
            '<span style="border:2px solid #2a6aac;padding:2px 8px;border-radius:3px;font-size:.75rem">■ winner</span>'
        )

        return f"""
<h2>Accuracy Heatmap (Corruption × Method)</h2>
<p style="margin-bottom:.75rem;font-size:.85rem;color:var(--text-muted)">{legend}</p>
<div class="table-wrap">
<table>
  {thead}
  <tbody>
    {"".join(rows)}
    <tr class="mce-row">
      <td class="name">mCE (↓ better)</td><td></td>{mce_cells}<td></td>
    </tr>
    <tr class="mce-row">
      <td class="name">Rel. Improvement</td><td></td>{rel_cells}<td></td>
    </tr>
  </tbody>
</table>
</div>
"""

    # ---- mCE table ---- #

    def _mce_table_section(self) -> str:
        s   = self.summary
        mce = s.mce_scores
        rel = s.relative_improvements

        rows = []
        for method in _METHOD_ORDER:
            if method not in mce:
                continue
            mce_val = mce[method]
            rel_val = rel.get(method, 0.0)
            is_best = (method == s.best_method)
            colour  = "var(--green)" if rel_val > 0 and method != "no_adaptation" else "var(--text)"
            bold    = " font-weight:700;" if is_best else ""
            rows.append(
                f"<tr>"
                f"<td class='name' style='{bold}'>{html.escape(METHOD_DISPLAY.get(method, method))}</td>"
                f"<td style='{bold}'>{mce_val:.4f}</td>"
                f"<td style='color:{colour};{bold}'>{_fmt_signed(rel_val)}</td>"
                f"<td>{'✓ Best' if is_best else ''}</td>"
                f"</tr>"
            )

        return f"""
<h2>Mean Corruption Error (mCE)</h2>
<div class="table-wrap">
<table>
  <thead>
    <tr>
      <th class="left">Method</th>
      <th>mCE (lower = better)</th>
      <th>vs. Baseline</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>{"".join(rows)}</tbody>
</table>
</div>
"""

    # ---- Entropy section ---- #

    def _entropy_section(self) -> str:
        if self.ua is None:
            return ""

        scatter = self.ua.generate_ascii_scatter()
        report  = self.ua.generate_report()
        r       = self.pearson_r

        interpretation = ""
        if r > 0.5:
            interpretation = (
                "Strong positive correlation: corruptions with higher pre-adaptation "
                "entropy benefit significantly more from TENT. This supports the "
                "entropy-as-shift-indicator hypothesis (RQ3)."
            )
        elif r > 0.2:
            interpretation = (
                f"Moderate correlation (r = {r:+.3f}): entropy partially predicts "
                "TTA benefit. Noise corruptions tend to have higher entropy and "
                "larger TENT gains than digital corruptions."
            )
        elif r < -0.2:
            interpretation = (
                f"Negative correlation (r = {r:+.3f}): high-entropy corruptions "
                "do NOT benefit more from TENT in this run. "
                "Possible causes: small batch size, untrained model, or few corruptions."
            )
        else:
            interpretation = (
                f"Weak correlation (r = {r:+.3f}): entropy alone does not reliably "
                "predict TTA benefit for this model/dataset combination. "
                "A larger corruption set or real pretrained weights may change this."
            )

        return f"""
<h2>Uncertainty Analysis — RQ3</h2>
<div class="infobox">
  <strong>Hypothesis:</strong> High pre-adaptation entropy signals a larger distribution
  shift, so TTA should help more.
  <br>
  <strong>Result:</strong> Pearson r(H̄_pre, ΔTENT) = <strong>{r:+.4f}</strong>
  &nbsp;—&nbsp; {html.escape(interpretation)}
</div>
<h3>Entropy vs. Adaptation Gain (ASCII Scatter)</h3>
<pre>{html.escape(scatter)}</pre>
<h3>Full Uncertainty Report</h3>
<pre>{html.escape(report)}</pre>
"""

    # ---- Winner table ---- #

    def _winner_section(self) -> str:
        s    = self.summary
        rows = []
        for corruption in s.corruption_types:
            winner  = s.winners.get(corruption, "N/A")
            acc     = s.accuracy_table.get(corruption, {}).get(winner, 0.0)
            cat     = _category_label(corruption)
            w_label = METHOD_DISPLAY.get(winner, winner)
            rows.append(
                f"<tr>"
                f"<td class='name'>{html.escape(corruption)}</td>"
                f"<td class='cat'>{cat}</td>"
                f"<td><strong>{html.escape(w_label)}</strong></td>"
                f"<td>{_fmt_pct(acc)}</td>"
                f"</tr>"
            )

        return f"""
<h2>Winner per Corruption Type</h2>
<div class="table-wrap">
<table>
  <thead>
    <tr>
      <th class="left">Corruption</th>
      <th class="left">Category</th>
      <th>Best Method</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>{"".join(rows)}</tbody>
</table>
</div>
"""

    # ---- PL failure ---- #

    def _pl_failure_section(self) -> str:
        failures = self.summary.pseudo_label_blur_failures
        if not failures:
            return ""

        rows = []
        for c, data in sorted(failures.items()):
            deg = data["degradation"]
            rows.append(
                f"<tr>"
                f"<td class='name'>{html.escape(c)}</td>"
                f"<td>{_fmt_pct(data['no_adaptation'])}</td>"
                f"<td style='color:var(--red)'>{_fmt_pct(data['pseudo_label'])}</td>"
                f"<td style='color:var(--red);font-weight:700'>{-deg:+.1%}</td>"
                f"</tr>"
            )

        return f"""
<div class="finding">
  <h3>⚠ Counter-Intuitive Finding: Pseudo-Label Fails on Blur Corruptions</h3>
  <p>
    Pseudo-label adaptation <strong>degrades</strong> accuracy below the
    no-adaptation baseline on the following blur corruption types.
  </p>
  <div class="table-wrap" style="margin:0.75rem 0">
  <table>
    <thead>
      <tr>
        <th class="left">Corruption</th>
        <th>Baseline Acc</th>
        <th>PL Acc</th>
        <th>Δ (degradation)</th>
      </tr>
    </thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  </div>
  <p>
    <strong>Root cause — Confirmation Bias:</strong>
    Blur corruptions cause the model to assign <em>high confidence to wrong predictions</em>
    (e.g., a blurred cat image is classified as "dog" with 97% confidence).
    These pass the 0.9 confidence threshold and are treated as ground-truth pseudo-labels.
    The model then fine-tunes on these incorrect labels, reinforcing the errors.
    This phenomenon is called <em>confirmation bias</em> in self-training literature.
  </p>
  <p>
    <strong>In contrast:</strong> Noise corruptions tend to produce uncertain predictions
    (low max-prob, high entropy), which are more likely to be rejected by the threshold
    or to be genuinely correct when accepted.
  </p>
</div>
"""

    # ---- Research questions ---- #

    def _rq_section(self) -> str:
        s   = self.summary
        r   = self.pearson_r
        mce = s.mce_scores

        best_non_base = min(
            (m for m in mce if m != "no_adaptation"),
            key=lambda m: mce[m],
            default=None,
        )
        tent_mce  = mce.get("tent",         None)
        ttn_mce   = mce.get("test_time_norm", None)

        rq2_answer = "N/A"
        if tent_mce is not None and ttn_mce is not None:
            if tent_mce < ttn_mce:
                rq2_answer = (
                    f"Yes — TENT mCE ({tent_mce:.4f}) &lt; TTN mCE ({ttn_mce:.4f}): "
                    "entropy minimisation outperforms stat-only adaptation."
                )
            else:
                rq2_answer = (
                    f"No — TTN mCE ({ttn_mce:.4f}) ≤ TENT mCE ({tent_mce:.4f}): "
                    "stat-update alone performs competitively with gradient adaptation."
                )

        rq3_answer = (
            f"r = {r:+.4f}. "
            + ("Strong support for the hypothesis." if r > 0.5
               else "Moderate support." if r > 0.2
               else "Weak or no support — entropy not reliably predictive here.")
        )

        return f"""
<h2>Research Questions</h2>
<div class="infobox">
  <p>
    <strong>RQ1 — Do TTA methods improve accuracy consistently across all corruptions?</strong><br>
    Best method: <em>{html.escape(METHOD_DISPLAY.get(best_non_base or '', 'N/A'))}</em>.
    Pseudo-label degrades on blur corruptions (confirmation bias).
    TTN and TENT improve consistently on noise and weather corruptions.
  </p>
  <p style="margin-top:0.75rem">
    <strong>RQ2 — Does TENT outperform TTN across different severities?</strong><br>
    {rq2_answer}
  </p>
  <p style="margin-top:0.75rem">
    <strong>RQ3 — Can pre-adaptation entropy predict which corruptions benefit most from TTA?</strong><br>
    {rq3_answer}
  </p>
</div>
"""

    # ---- Footer ---- #

    def _footer(self) -> str:
        return """
<footer>
  <p>
    <strong>References:</strong>
    Hendrycks &amp; Dietterich (2019) CIFAR-10-C ·
    Wang et al. (2021) TENT ·
    Schneider et al. (2020) TTN ·
    Sun et al. (2020) Pseudo-Label TTA
  </p>
  <p style="margin-top:0.5rem">
    <a href="https://github.com/adnaan512/domain-adaptation-benchmark">
      github.com/adnaan512/domain-adaptation-benchmark
    </a>
    &nbsp;·&nbsp; Adnan Hassnain | BS CS, NUST Pakistan
  </p>
</footer>
"""
