import re, json, base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sae_utils import SAEWrapper, load_image, threshold_mask, delta_for_scale

st.set_page_config(page_title="RadiSpect — Report Hover-to-Overlay", layout="wide")

# ---------- Paths ----------
PAIRS = Path("data/pairs.csv")
RUNS_DIR = Path("models/sae_final")
LABELS_PATH = Path("models/latent_labels.json")
PASSLIST_PATH = Path("models/gate_pass_images.txt")
CLICKABLE_PATH = Path("models/clickable_images.txt")
CLICKMAP_PATH  = Path("models/clickable_map.json")

# ---------- Pillow for PNGs ----------
try:
    from PIL import Image
except ImportError:
    st.error("Please install Pillow:  pip install pillow")
    st.stop()

# ---------- Loaders ----------
@st.cache_data(show_spinner=False)
def list_runs():
    if not RUNS_DIR.exists(): return []
    return sorted([str(p) for p in RUNS_DIR.iterdir() if p.is_dir()])

@st.cache_data(show_spinner=False)
def load_pairs():
    df = pd.read_csv(PAIRS)
    return df[["image","report"]].copy()

@st.cache_data(show_spinner=False)
def load_labels():
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text())
    return {}

@st.cache_data(show_spinner=False)
def load_passlist():
    if PASSLIST_PATH.exists():
        return set(l.strip() for l in PASSLIST_PATH.read_text().splitlines() if l.strip())
    return set()

@st.cache_data(show_spinner=False)
def load_clickable():
    s = set()
    m = {}
    if CLICKABLE_PATH.exists():
        s = set(l.strip() for l in CLICKABLE_PATH.read_text().splitlines() if l.strip())
    if CLICKMAP_PATH.exists():
        m = json.loads(CLICKMAP_PATH.read_text())
    return s, m

# ---------- Utils ----------
def to_base64_png_uint8(arr_uint8):
    buf = BytesIO()
    Image.fromarray(arr_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def base_to_png64(base01):
    img = (np.clip(base01,0,1)*255).astype(np.uint8)
    return to_base64_png_uint8(img)

def overlay_from_mask_png64(mask, alpha=0.35):
    h,w = mask.shape
    rgba = np.zeros((h,w,4), dtype=np.uint8)
    m = mask.astype(bool)
    rgba[m,0] = 255   # R
    rgba[m,3] = int(alpha*255)
    return to_base64_png_uint8(rgba)

def escape_html(s:str) -> str:
    return (s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
             .replace('"',"&quot;").replace("'","&#39;"))

# ---------- UI ----------
runs = list_runs()
if not runs:
    st.warning("No runs found under models/sae_final/."); st.stop()
run = st.sidebar.selectbox("SAE run", runs, index=0)

pairs = load_pairs()
labels_map = load_labels()
passset = load_passlist()
clickable_set, clickable_map = load_clickable()

# Enforce passlist and clickable-only to guarantee spans that map to masks
if not passset:
    st.warning("Missing models/gate_pass_images.txt. Run eval_gates.py first."); st.stop()
pairs = pairs[pairs["image"].isin(passset)]
if pairs.empty:
    st.warning("No images intersect pass list."); st.stop()

only_clickable = st.sidebar.checkbox("Only scans with clickable spans", value=True)
if only_clickable:
    if not clickable_set:
        st.warning("Missing clickable list. Run app/build_clickable_list.py or uncheck this filter.")
        st.stop()
    pairs = pairs[pairs["image"].isin(clickable_set)]
    if pairs.empty:
        st.warning("No scans meet clickable criteria."); st.stop()

# Mask threshold controls
use_pct = st.sidebar.checkbox("Use percentile threshold for mask", value=True)
thr = st.sidebar.slider("Fixed threshold", 0.05, 0.90, 0.30, 0.05, disabled=use_pct)
pct = st.sidebar.slider("Percentile", 50, 99, 92, 1, disabled=not use_pct)

# Choose image
q = st.sidebar.text_input("Filter images:", "")
subset = pairs[pairs["image"].str.contains(q, case=False, na=False)] if q else pairs
if subset.empty:
    st.warning("No images after filters."); st.stop()
img_path = st.sidebar.selectbox("Image", subset["image"].tolist(), index=0)
report_text = pairs[pairs["image"]==img_path]["report"].iloc[0]

if st.button("Analyze", type="primary", use_container_width=True):
    # Load model & image
    w = SAEWrapper(run)
    x = load_image(img_path, size=w.cfg["img_size"])
    base = x.numpy()[0,0]  # [0,1]
    base64_png = base_to_png64(base)

    # Build list of latents we will show overlays for = those referenced by clickable_map
    if img_path not in clickable_map or not clickable_map[img_path]:
        st.warning("No clickable spans mapped for this image. Pick another.")
        st.stop()

    # Prepare overlays for every referenced latent (dedup)
    needed = []
    seen = set()
    for rec in clickable_map[img_path]:
        j = int(rec["latent"])
        if j in seen: continue
        seen.add(j)
        d = delta_for_scale(w, x, j, 0.0)
        m = threshold_mask(d, thr=thr, use_percentile=use_pct, p=pct)
        overlay64 = overlay_from_mask_png64(m)
        lbl = labels_map.get(str(j),{}).get("label", f"Latent {j}")
        needed.append({"j": j, "label": lbl, "overlay64": overlay64})

    # Build HTML spans from precomputed map (ordered by start)
    spans=[]
    for rec in clickable_map[img_path]:
        j = int(rec["latent"]); lbl = rec["label"]
        for (s,e) in rec["spans"]:
            spans.append((s,e,j,lbl))
    spans.sort(key=lambda t: t[0])

    # Precompute dynamic chunks (avoid complex f-string expressions)
    overlay_imgs_html = "".join(
        [f"<img class='overlay' id='ov-{r['j']}' src='data:image/png;base64,{r['overlay64']}' style='opacity:0;'>" for r in needed]
    )
    legend_items_html = "".join(
        [f"<div class='legend-item'><span class='swatch' data-j='{r['j']}'></span>{escape_html(r['label'])} (j={r['j']})</div>" for r in needed]
    )
    overlay_setters_js = "".join(
        [f"overlays.set('{r['j']}', document.getElementById('ov-{r['j']}'));\n" for r in needed]
    )

    # Report text with span elements
    report_html_parts=[]; i=0
    for (s,e,j,lbl) in spans:
        report_html_parts.append(escape_html(report_text[i:s]))
        frag = escape_html(report_text[s:e])
        report_html_parts.append(f"<span class='chip' data-j='{j}' title='{escape_html(lbl)}'>{frag}</span>")
        i=e
    report_html_parts.append(escape_html(report_text[i:]))
    report_html = "".join(report_html_parts)

    # Compose final HTML
    html = f"""
<div class="wrap">
  <div class="left">
    <div class="report">{report_html}</div>
    <div class="hint">Hover a highlighted phrase to reveal its evidence overlay. Click to lock/unlock.</div>
  </div>
  <div class="right">
    <div class="viewer">
      <img class="base" src="data:image/png;base64,{base64_png}">
      {overlay_imgs_html}
    </div>
    <div class="legend">
      {legend_items_html}
    </div>
  </div>
</div>

<style>
.wrap {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; font-family: ui-sans-serif, system-ui; }}
.report {{ white-space: pre-wrap; line-height: 1.45; font-size: 0.95rem; }}
.chip {{ background: #fff3a3; border-radius: 4px; padding: 0 2px; cursor: pointer; }}
.hint {{ color: #6b7280; font-size: 0.85rem; margin-top: 8px; }}
.viewer {{ position: relative; width: 100%; }}
.viewer img {{ width: 100%; height: auto; display: block; }}
.overlay {{ position: absolute; top: 0; left: 0; transition: opacity 120ms ease-in-out; }}
.legend {{ margin-top: 10px; color:#374151; font-size:0.85rem; }}
.legend-item {{ display:flex; align-items:center; gap:8px; margin-bottom:4px; }}
.swatch {{ width:14px; height:14px; display:inline-block; background: rgba(255,0,0,0.35); border:1px solid rgba(0,0,0,0.1); border-radius:2px; }}
</style>

<script>
(function() {{
  const overlays = new Map();
  {overlay_setters_js}

  function show(j) {{
    overlays.forEach((el, key) => {{ if (key !== j) el.style.opacity = 0; }});
    const el = overlays.get(j);
    if (el) el.style.opacity = 1;
  }}
  function hide(j) {{
    const el = overlays.get(j);
    if (el) el.style.opacity = 0;
  }}
  function toggle(j) {{
    const el = overlays.get(j);
    if (!el) return;
    el.style.opacity = (el.style.opacity === '1') ? 0 : 1;
  }}

  // Hover + click on report spans
  document.querySelectorAll('.chip').forEach(span => {{
    const j = span.getAttribute('data-j');
    span.addEventListener('mouseenter', () => show(j));
    span.addEventListener('mouseleave', () => hide(j));
    span.addEventListener('click', () => toggle(j));
  }});

  // Also allow clicking legend swatches
  document.querySelectorAll('.swatch').forEach(sw => {{
    const j = sw.getAttribute('data-j');
    sw.addEventListener('click', () => toggle(j));
  }});
}})();
</script>
"""
    st.components.v1.html(html, height=700, scrolling=True)
else:
    st.info("Pick an image and click **Analyze**.")