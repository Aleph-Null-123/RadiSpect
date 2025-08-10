import re, json, html as _html
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sae_utils import (
    SAEWrapper, load_image, topk_latents_by_activation,
    threshold_mask, delta_for_scale,
)

st.set_page_config(page_title="RadiSpect — Safe Viewer (Report Click-to-Overlay)", layout="wide")

# --------- Paths ----------
PAIRS = Path("data/pairs.csv")
RUNS_DIR = Path("models/sae_final")
LABELS_PATH = Path("models/latent_labels.json")
PASSLIST_PATH = Path("models/gate_pass_images.txt")
GATE_EVAL_CSV = Path("models/gate_eval.csv")
CLICKABLE_PATH = Path("models/clickable_images.txt")
CLICKMAP_PATH  = Path("models/clickable_map.json")

# --------- Loaders ----------
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
def load_gate_eval():
    if GATE_EVAL_CSV.exists():
        df = pd.read_csv(GATE_EVAL_CSV)
        if {"image","latent","passed"}.issubset(df.columns):
            df["latent"] = df["latent"].astype(int)
            return df
    return None

@st.cache_data(show_spinner=False)
def load_clickable():
    s = set()
    m = {}
    if CLICKABLE_PATH.exists():
        s = set(l.strip() for l in CLICKABLE_PATH.read_text().splitlines() if l.strip())
    if CLICKMAP_PATH.exists():
        m = json.loads(CLICKMAP_PATH.read_text())
    return s, m

# --------- Matching helpers (fallback if no precomputed map) ----------
NEG_WORDS = r"(no|without|absent|free of|negative for|no evidence of|denies)"
NEG_WINDOW_TOKENS = 6

def _token_positions(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

def _negated(text, span_start):
    toks = _token_positions(text)
    idx = 0
    for i,(s,e) in enumerate(toks):
        if s <= span_start < e: idx = i; break
    lo = max(0, idx-NEG_WINDOW_TOKENS)
    window = text[toks[lo][0]:span_start] if toks else text[:span_start]
    return re.search(NEG_WORDS, window, flags=re.IGNORECASE) is not None

def find_positive_spans(report: str, label: str):
    rx = re.compile("(" + re.escape(label) + ")", flags=re.IGNORECASE)
    hits=[]
    for m in rx.finditer(report):
        s,e = m.start(), m.end()
        if not _negated(report, s): hits.append((s,e))
    # merge overlaps
    hits = sorted(hits)
    merged=[]
    for s,e in hits:
        if not merged: merged.append([s,e]); continue
        ps,pe = merged[-1]
        if s <= pe: merged[-1][1] = max(pe,e)
        else: merged.append([s,e])
    return [(s,e) for s,e in merged]

def highlight_spans_html(text: str, spans):
    if not spans:
        return "<div style='white-space:pre-wrap; font-family: ui-sans-serif, system-ui;'>{}</div>".format(
            _html.escape(text)
        )
    parts=[]; i=0
    for s,e in spans:
        parts.append(_html.escape(text[i:s]))
        parts.append(f"<mark style='background:#fff3a3'>{_html.escape(text[s:e])}</mark>")
        i=e
    parts.append(_html.escape(text[i:]))
    return "<div style='white-space:pre-wrap; font-family: ui-sans-serif, system-ui;'>{}</div>".format("".join(parts))

# --------- Latent selection logic ----------
def choose_latents_for_clickable(w, x, img_path, labels_map, gate_df, clickable_map, k_display, prefer_labeled=True):
    """
    Prioritize latents from clickable_map[img] (these are guaranteed to have positive report spans).
    If fewer than k_display, fill from precomputed passing latents (gate_df).
    If still fewer, fill from live top-k (no QA).
    Returns list of (j, label_or_None).
    """
    chosen = []
    chosen_set = set()

    # 1) from clickable map
    if img_path in clickable_map:
        for rec in clickable_map[img_path]:
            j = int(rec["latent"])
            lbl = labels_map.get(str(j), {}).get("label", None)
            if prefer_labeled and not lbl: 
                continue  # but they should be labeled already
            if j not in chosen_set:
                chosen.append((j, lbl))
                chosen_set.add(j)
                if len(chosen) >= k_display: return chosen

    # 2) fill from gate_df passing latents (for this image)
    if gate_df is not None:
        rows = gate_df[(gate_df["image"] == img_path) & (gate_df["passed"] == 1)]
        for _, r in rows.iterrows():
            j = int(r["latent"])
            if j in chosen_set: continue
            lbl = labels_map.get(str(j), {}).get("label", None)
            if prefer_labeled and not lbl: 
                continue
            chosen.append((j, lbl))
            chosen_set.add(j)
            if len(chosen) >= k_display: return chosen

    # 3) fallback to live top-k
    _, z = w.reconstruct(x)
    idx, _ = topk_latents_by_activation(z, k=k_display)
    for j in idx:
        j = int(j)
        if j in chosen_set: continue
        lbl = labels_map.get(str(j), {}).get("label", None)
        chosen.append((j, lbl))
        chosen_set.add(j)
        if len(chosen) >= k_display: break

    return chosen[:k_display]

# --------- UI ----------
runs = list_runs()
if not runs:
    st.warning("No runs found under models/sae_final/.")
    st.stop()
run = st.sidebar.selectbox("SAE run", runs, index=0)

pairs = load_pairs()
labels_map = load_labels()
passset = load_passlist()
gate_df = load_gate_eval()
clickable_set, clickable_map = load_clickable()

# enforce passlist-only to avoid empty findings
if not passset:
    st.warning("Missing or empty models/gate_pass_images.txt. Run eval_gates.py to generate it.")
    st.stop()
pairs = pairs[pairs["image"].isin(passset)]
if pairs.empty:
    st.warning("No images in pass list match pairs.csv.")
    st.stop()

# NEW: only show scans that will have at least one clickable finding
only_clickable = st.sidebar.checkbox("Only scans with clickable report spans", value=True)
if only_clickable:
    if clickable_set:
        pairs = pairs[pairs["image"].isin(clickable_set)]
        if pairs.empty:
            st.warning("No scans meet the clickable criteria. Uncheck the filter or rebuild the clickable list.")
            st.stop()
    else:
        st.warning("Clickable list missing. Run app/build_clickable_list.py, or uncheck this filter.")
        st.stop()

# controls
k_display = st.sidebar.slider("How many findings to show", 1, 6, 3)
use_pct = st.sidebar.checkbox("Use percentile threshold for mask", value=True)
thr = st.sidebar.slider("Fixed threshold", 0.05, 0.90, 0.30, 0.05, disabled=use_pct)
pct = st.sidebar.slider("Percentile", 50, 99, 92, 1, disabled=not use_pct)
prefer_labeled = st.sidebar.checkbox("Prefer labeled latents", value=True)

# image chooser
q = st.sidebar.text_input("Filter images:", "")
subset = pairs[pairs["image"].str.contains(q, case=False, na=False)] if q else pairs
img_path = st.sidebar.selectbox("Image", subset["image"].tolist(), index=0)
report_text = pairs[pairs["image"]==img_path]["report"].iloc[0] if not pairs.empty else ""

# analyze
if st.button("Analyze", type="primary", use_container_width=True):
    # clear previous selection so nothing shows until you click
    st.session_state.pop("RS_SEL", None)

    # load model & image
    w = SAEWrapper(run)
    x = load_image(img_path, size=w.cfg["img_size"])
    base = x.numpy()[0,0]

    # choose latents prioritized by clickable map
    chosen_pairs = choose_latents_for_clickable(
        w=w, x=x, img_path=img_path, labels_map=labels_map,
        gate_df=gate_df, clickable_map=clickable_map,
        k_display=k_display, prefer_labeled=prefer_labeled
    )

    # gather z and build masks (ablation)
    _, z = w.reconstruct(x)
    items = []  # (j, label, zval, mask)
    for j, lbl in chosen_pairs:
        zval = float(z[0, j].item())
        d_ablate = delta_for_scale(w, x, j, 0.0)
        m = threshold_mask(d_ablate, thr=thr, use_percentile=use_pct, p=pct)
        items.append((j, lbl, zval, m))

    # --- Build spans/chips: use precomputed click map if available; fallback otherwise ---
    spans=[]; span_map=[]
    if img_path in clickable_map and clickable_map[img_path]:
        for rec in clickable_map[img_path]:
            j = int(rec["latent"])
            lbl = rec["label"]
            # only add chips for latents we actually loaded masks for
            if j not in {it[0] for it in items}: 
                continue
            for (s,e) in rec["spans"]:
                spans.append((s,e))
                span_map.append({"start":s,"end":e,"label":lbl,"latent":j})
        # sort by order in text
        order = np.argsort([s for (s,_) in spans])
        spans = [spans[i] for i in order]
        span_map = [span_map[i] for i in order]
    else:
        # fallback: compute spans live from labels we have
        by_label = {}
        for (j,lbl,zv,_m) in items:
            if not lbl: continue
            if (lbl not in by_label) or (zv > by_label[lbl][1]):
                by_label[lbl] = (j, zv)
        for lbl,(j,_zv) in by_label.items():
            sps = find_positive_spans(report_text, lbl)
            for (s,e) in sps:
                spans.append((s,e))
                span_map.append({"start":s,"end":e,"label":lbl,"latent":j})
        order = np.argsort([s for (s,_) in spans])
        spans = [spans[i] for i in order]
        span_map = [span_map[i] for i in order]

    st.session_state["RS"] = dict(
        run=run, img=img_path, base=base, items=items,
        report=report_text, spans=spans, span_map=span_map,
        use_pct=use_pct, thr=float(thr), pct=int(pct)
    )

# render
if "RS" in st.session_state and st.session_state["RS"]["img"] == img_path and st.session_state["RS"]["run"] == run:
    S = st.session_state["RS"]
    st.subheader(Path(img_path).name)

    # layout: left = report; right = image
    c1, c2 = st.columns([1,1])

    # --- LEFT: Original report with highlights + clickable chips ---
    with c1:
        if S["spans"]:
            html = highlight_spans_html(S["report"], S["spans"])
            st.markdown(html, unsafe_allow_html=True)
            st.caption("Click a highlighted finding below to reveal its evidence overlay.")
        else:
            st.text(S["report"])
            st.caption("No positive spans matched our latent labels in this report.")

        # chips in reading order
        st.write("**Clickable findings from report:**")
        for i, info in enumerate(S["span_map"]):
            j = info["latent"]; lbl = info["label"]
            excerpt = S["report"][info["start"]:info["end"]]
            if st.button(f"{i+1}. {lbl} — “{excerpt}”", key=f"span_{i}"):
                st.session_state["RS_SEL"] = j

    # --- RIGHT: Image with overlay ONLY after a click ---
    with c2:
        sel_j = st.session_state.get("RS_SEL", None)
        base = S["base"].astype(np.float32)
        rgb = np.stack([base, base, base], axis=-1)
        vis = rgb.copy()

        cap = Path(S['img']).name
        if sel_j is not None:
            # find the mask for sel_j
            for (j,lbl,zv,m) in S["items"]:
                if j == sel_j:
                    vis[m] = 0.65*vis[m] + 0.35*np.array([1,0,0], dtype=np.float32)
                    cap = f"{Path(S['img']).name} — {(lbl or f'Latent {j}')} (j={j}, z={zv:.3f})"
                    break

        st.image(vis, use_container_width=True, caption=cap)

        # optional latent buttons (no auto-select)
        st.write("**Latents (manual):**")
        for (j,lbl,zv,_m) in S["items"]:
            tag = lbl or f"Latent {j}"
            if st.button(f"{tag}  (j={j}, z={zv:.3f})", key=f"lat_{j}"):
                st.session_state["RS_SEL"] = j

        st.button("Clear selection", on_click=lambda: st.session_state.pop("RS_SEL", None))
else:
    st.info("Pick an image and click **Analyze**.")
