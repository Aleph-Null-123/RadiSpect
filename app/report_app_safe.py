import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sae_utils import (
    SAEWrapper,
    load_image,
    topk_latents_by_activation,
    threshold_mask,
    delta_for_scale,
)

st.set_page_config(page_title="RadiSpect — Safe Viewer (no QA errors)", layout="wide")

# --------- Paths ----------
PAIRS = Path("data/pairs.csv")
RUNS_DIR = Path("models/sae_final")
LABELS_PATH = Path("models/latent_labels.json")
PASSLIST_PATH = Path("models/gate_pass_images.txt")
GATE_EVAL_CSV = Path("models/gate_eval.csv")

# --------- Helpers ----------
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
        # basic sanity
        need = {"image","latent","passed"}
        if need.issubset(set(df.columns)):
            # normalize latent to int
            df["latent"] = df["latent"].astype(int)
            return df
    return None

def ensure_nonempty_latents(w, x, img_path, want_labeled_only, labels_map, pre_rows, k_display):
    """
    Choose a non-empty list of (latent j, zval, label, Ebest, mono) for this image.
    Preference order:
      1) precomputed pass latents (possibly filtered by labeled_only)
      2) if empty -> allow unlabeled precomputed pass latents
      3) if still empty -> fall back to live top-k by activation (no QA)
    """
    # 1) precomputed passing latents for this image
    cand = []
    if pre_rows is not None and len(pre_rows):
        # prefer labeled if requested
        for _, row in pre_rows.iterrows():
            j = int(row["latent"])
            lbl = labels_map.get(str(j), {}).get("label", None)
            if want_labeled_only and not lbl:
                continue
            cand.append((j, lbl, float(row.get("Ebest", np.nan)), float(row.get("mono_corr", np.nan))))
        if not cand:
            # 2) allow unlabeled if labeled-only filtering erased all
            for _, row in pre_rows.iterrows():
                j = int(row["latent"])
                lbl = labels_map.get(str(j), {}).get("label", None)
                cand.append((j, lbl, float(row.get("Ebest", np.nan)), float(row.get("mono_corr", np.nan))))
    # 3) fallback: top-k by activation if still empty
    if not cand:
        _, z = w.reconstruct(x)
        idx, _ = topk_latents_by_activation(z, k=k_display)
        cand = [(int(j), labels_map.get(str(int(j)), {}).get("label", None), np.nan, np.nan) for j in idx]

    # cap to k_display
    return cand[:k_display]

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

# Enforce passlist-only to avoid empty findings
if not passset:
    st.warning("Missing or empty models/gate_pass_images.txt. Run eval_gates.py to generate it.")
    st.stop()

# Filter images to pass list
pairs = pairs[pairs["image"].isin(passset)]
if pairs.empty:
    st.warning("No images in pass list match pairs.csv.")
    st.stop()

# Controls (kept minimal, no live QA)
k_display = st.sidebar.slider("How many findings to show", 1, 6, 3)
use_pct = st.sidebar.checkbox("Use percentile threshold for mask", value=True)
thr = st.sidebar.slider("Fixed threshold", 0.05, 0.90, 0.30, 0.05, disabled=use_pct)
pct = st.sidebar.slider("Percentile", 50, 99, 92, 1, disabled=not use_pct)
labeled_only = st.sidebar.checkbox("Prefer labeled latents", value=True)

# Image chooser
q = st.sidebar.text_input("Filter images:", "")
subset = pairs[pairs["image"].str.contains(q, case=False, na=False)] if q else pairs
img_path = st.sidebar.selectbox("Image", subset["image"].tolist(), index=0)

# Analyze button
if st.button("Analyze", type="primary", use_container_width=True):
    # Load model and image
    w = SAEWrapper(run)
    x = load_image(img_path, size=w.cfg["img_size"])
    base = x.numpy()[0,0]  # original

    # Get precomputed passing latents for this image (if available)
    pre_rows = None
    if gate_df is not None:
        pre_rows = gate_df[(gate_df["image"] == img_path) & (gate_df["passed"] == 1)]

    chosen = ensure_nonempty_latents(
        w, x, img_path,
        want_labeled_only=labeled_only,
        labels_map=labels_map,
        pre_rows=pre_rows,
        k_display=k_display
    )

    # Gather z values for display
    _, z = w.reconstruct(x)
    items = []
    for j, lbl, Ebest, mono in chosen:
        zval = float(z[0, j].item())
        # Build mask using ablation delta, no live QA
        d_ablate = delta_for_scale(w, x, j, 0.0)
        m = threshold_mask(d_ablate, thr=thr, use_percentile=use_pct, p=pct)
        items.append((j, lbl, zval, m, Ebest, mono))

    # Always non-empty by construction
    st.session_state["SAFE"] = dict(
        run=run, img=img_path, base=base, items=items,
        use_pct=use_pct, thr=float(thr), pct=int(pct)
    )

# Render
if "SAFE" in st.session_state and st.session_state["SAFE"]["img"] == img_path and st.session_state["SAFE"]["run"] == run:
    S = st.session_state["SAFE"]
    st.subheader(Path(img_path).name)

    options=[]
    for (j, lbl, zval, _m, Ebest, mono) in S["items"]:
        tag = lbl or f"Latent {j}"
        meta = []
        if not np.isnan(Ebest): meta.append(f"E*={Ebest:.3f}")
        if not np.isnan(mono):  meta.append(f"mono={mono:.2f}")
        meta_str = ", ".join(meta) if meta else "precomputed/fallback"
        options.append(f"{tag}  (j={j}, z={zval:.3f}; {meta_str})")

    choice = st.radio("Findings (click to overlay)", options, index=0)
    sel_i = options.index(choice)
    sel_j, sel_lbl, sel_z, sel_mask, sel_E, sel_mono = S["items"][sel_i]

    # overlay on original
    base = S["base"].astype(np.float32)
    rgb = np.stack([base, base, base], axis=-1)
    vis = rgb.copy()
    vis[sel_mask] = 0.65*vis[sel_mask] + 0.35*np.array([1,0,0], dtype=np.float32)

    cap = f"{Path(img_path).name} — {(sel_lbl or f'Latent {sel_j}')}"
    if not np.isnan(sel_E) or not np.isnan(sel_mono):
        cap += f"  [E*={sel_E if not np.isnan(sel_E) else 0:.3f}, mono={sel_mono if not np.isnan(sel_mono) else 0:.2f}]"
    st.image(vis, use_container_width=True, caption=cap)

    st.button("Clear highlight", on_click=lambda: st.session_state.pop("SAFE"))
else:
    st.info("Pick an image and click **Analyze**.")
