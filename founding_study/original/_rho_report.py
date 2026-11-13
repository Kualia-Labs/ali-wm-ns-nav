"""Render /tmp/rho_grid.json into a styled PDF report (gap-vs-rho / H2b)."""
import json, asyncio, html

with open("/tmp/rho_grid.json") as f:
    D = json.load(f)

cfg = D["config"]
rows = D["rows"]
verdict = D["h2b_verdict"]
mono = D["monotonicity"]

H2B = ("H2b: The explicit-over-implicit advantage is near-zero at low rho and grows "
       "monotonically with rho, peaking under per-step change.")
HYP = ("Given identical raw interaction history, an agent that explicitly computes multi-scale "
       "(micro/mezo/macro) self-observation features adapts better in non-stationary navigation "
       "than an external-state-only agent and a recurrent LSTM that integrates implicitly; the "
       "explicit-over-implicit advantage grows monotonically with the change rate rho.")

# ---- build gap-vs-rho SVG line chart ----
pts_b = [(r["rho"], r["gap_treat_baseline"]) for r in rows if r["gap_treat_baseline"] is not None]
pts_c = [(r["rho"], r["gap_treat_control"]) for r in rows if r["gap_treat_control"] is not None]
all_y = [y for _, y in pts_b + pts_c] or [0.0]
ymin, ymax = min(all_y + [0.0]), max(all_y + [0.0])
if ymax == ymin:
    ymax = ymin + 1.0
pad = (ymax - ymin) * 0.15 or 0.1
ymin -= pad; ymax += pad
W, Hgt, ml, mr, mt, mb = 720, 360, 70, 30, 30, 50
pw, ph = W - ml - mr, Hgt - mt - mb


def sx(rho):
    return ml + rho * pw  # rho in [0,1]


def sy(v):
    return mt + (ymax - v) / (ymax - ymin) * ph


def polyline(pts, color):
    if not pts:
        return ""
    pl = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
    dots = "".join(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>' for x, y in pts)
    return f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{pl}"/>{dots}'


# axes + gridlines
yticks = 5
grid = ""
for i in range(yticks + 1):
    v = ymin + (ymax - ymin) * i / yticks
    y = sy(v)
    grid += f'<line x1="{ml}" y1="{y:.1f}" x2="{W-mr}" y2="{y:.1f}" stroke="#eee"/>'
    grid += f'<text x="{ml-8}" y="{y+4:.1f}" font-size="11" fill="#666" text-anchor="end">{v:.2f}</text>'
xs = ""
for rho in cfg["rhos"]:
    x = sx(rho)
    xs += f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{Hgt-mb}" stroke="#f4f4f4"/>'
    xs += f'<text x="{x:.1f}" y="{Hgt-mb+18}" font-size="11" fill="#666" text-anchor="middle">{rho:g}</text>'
zero_line = ""
if ymin < 0 < ymax:
    zy = sy(0.0)
    zero_line = f'<line x1="{ml}" y1="{zy:.1f}" x2="{W-mr}" y2="{zy:.1f}" stroke="#bbb" stroke-dasharray="4 3"/>'

svg = f'''<svg viewBox="0 0 {W} {Hgt}" xmlns="http://www.w3.org/2000/svg" width="100%">
  {grid}{xs}{zero_line}
  <line x1="{ml}" y1="{mt}" x2="{ml}" y2="{Hgt-mb}" stroke="#333"/>
  <line x1="{ml}" y1="{Hgt-mb}" x2="{W-mr}" y2="{Hgt-mb}" stroke="#333"/>
  {polyline(pts_b, "#c0392b")}
  {polyline(pts_c, "#2980b9")}
  <text x="{ml+pw/2}" y="{Hgt-8}" font-size="12" fill="#333" text-anchor="middle">change rate &#961;</text>
  <text x="18" y="{mt+ph/2}" font-size="12" fill="#333" text-anchor="middle" transform="rotate(-90 18 {mt+ph/2})">reward gap</text>
</svg>'''

# ---- tables ----
def cell(v, pct=False):
    if v is None:
        return "N/A"
    return f"{v*100:.1f}%" if pct else f"{v:.3f}"

main_rows = ""
for r in rows:
    main_rows += (
        f"<tr><td>{r['rho']:g}</td><td>{cell(r['baseline'])}</td><td>{cell(r['control'])}</td>"
        f"<td>{cell(r['treatment'])}</td><td class='gap'>{cell(r['gap_treat_baseline'])}</td>"
        f"<td class='gap'>{cell(r['gap_treat_control'])}</td></tr>"
    )

res = D["results"]
per_variant = ""
for role in ("baseline", "control", "treatment"):
    cells = "".join(
        f"<td>{cell(res.get(role,{}).get(str(rho) if str(rho) in res.get(role,{}) else rho,{}).get('mean'))}</td>"
        for rho in cfg["rhos"]
    )
    per_variant += f"<tr><td class='role'>{role}</td>{cells}</tr>"
pv_head = "".join(f"<th>&#961;={rho:g}</th>" for rho in cfg["rhos"])

verdict_color = {"supported": "#1e7e34", "contradicted": "#b21f2d",
                 "partial / non-monotonic": "#b8860b"}.get(verdict, "#555")

doc = f'''<!doctype html><html><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 18mm 16mm; }}
* {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
body {{ color:#1a1a1a; font-size:12.5px; line-height:1.5; }}
h1 {{ font-size:20px; margin:0 0 2px; }}
h2 {{ font-size:14px; margin:18px 0 6px; border-bottom:2px solid #eee; padding-bottom:3px; }}
.sub {{ color:#666; font-size:11px; margin-bottom:14px; }}
.box {{ background:#f7f8fa; border:1px solid #e5e7eb; border-radius:6px; padding:10px 12px; font-size:11.5px; }}
table {{ border-collapse:collapse; width:100%; margin:8px 0; font-size:11.5px; }}
th,td {{ border:1px solid #ddd; padding:5px 7px; text-align:center; }}
th {{ background:#2c3e50; color:#fff; font-weight:600; }}
td.gap {{ font-weight:700; background:#fff7ec; }}
td.role {{ text-align:left; font-weight:600; text-transform:capitalize; }}
.verdict {{ font-size:14px; font-weight:700; color:{verdict_color}; }}
.legend span {{ display:inline-block; margin-right:16px; font-size:11.5px; }}
.dot {{ display:inline-block; width:11px; height:11px; border-radius:50%; vertical-align:middle; margin-right:5px; }}
.note {{ color:#777; font-size:10.5px; margin-top:6px; }}
</style></head><body>
<h1>&#961;-Sweep Report &mdash; Explicit-vs-Implicit Self-Observation under Non-Stationarity</h1>
<div class="sub">Project #18 &middot; QTI Continual Non-Stationary Navigation &middot; Hypothesis H2b &middot; PPO, {cfg['timesteps']:,} steps, {cfg['n_seeds']} seed, {cfg['eval_episodes']} eval episodes</div>

<div class="box"><b>Hypothesis (H2b):</b> {html.escape(H2B)}<br><br>
<b>Main hypothesis:</b> {html.escape(HYP)}</div>

<h2>Key result: explicit-vs-implicit gap vs change rate &#961;</h2>
<div class="legend">
  <span><i class="dot" style="background:#c0392b"></i>gap = treatment &minus; baseline (explicit vs external-only)</span>
  <span><i class="dot" style="background:#2980b9"></i>gap = treatment &minus; control (macro-scale contribution)</span>
</div>
{svg}

<h2>Gap-vs-&#961; table</h2>
<table>
<tr><th>&#961;</th><th>baseline<br>(ext-only)</th><th>control<br>(micro+mezo)</th><th>treatment<br>(full)</th><th>gap<br>(treat&minus;base)</th><th>gap<br>(treat&minus;ctrl)</th></tr>
{main_rows}
</table>
<div class="note">Reward shaping, dynamics and termination are byte-identical across the three siblings, so observation content is the only manipulated variable; mean-reward gaps are therefore directly comparable.</div>

<h2>Mean reward per variant (eval, deterministic)</h2>
<table>
<tr><th>variant</th>{pv_head}</tr>
{per_variant}
</table>

<h2>H2b verdict</h2>
<p class="verdict">H2b: {verdict.upper()}</p>
<p>{html.escape(mono)}</p>

<h2>Notes &amp; caveats</h2>
<ul>
<li>This is a fast, uniform-budget look ({cfg['timesteps']:,} steps, single seed) intended to populate the gap-vs-&#961; curve immediately. A definitive 100k-step, 2-seed sweep is running in the background for the final paper.</li>
<li>The "implicit" recurrent baseline (Recurrent PPO / PPO-LSTM, H2) is evaluated in the full sweep; this report's gap uses the external-state-only sibling as the explicit-vs-non-explicit reference.</li>
<li>Total compute for this report: {D.get('total_time_sec',0):.0f} s.</li>
</ul>
</body></html>'''

with open("/tmp/rho_report.html", "w") as f:
    f.write(doc)


async def render():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page()
        await pg.set_content(doc, wait_until="networkidle")
        await pg.pdf(path="/tmp/rho_report.pdf", format="A4", print_background=True)
        await b.close()
    print("PDF written: /tmp/rho_report.pdf")

asyncio.run(render())
