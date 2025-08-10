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
from qa_utils import mask_energy, monotonicity

st.set_page_config(page_title="RadiSpect — Click-to-Evidence (QA-gated)", layout="wide")

PAIRS = "data/pairs.csv"
RUNS_DIR = Path("models/sae_final")
LABELS_PATH = Path("models/latent_labels.json")
PASSLIST_PATH = Path("models/gate_pass_images.txt")
DEMO_JSON = Path("models/demo_cases.json")

# ------------------------ helpers ------------------------

@st.cache_data(show_spinner=False)
def load_pairs():
    df = pd.read_csv(PAIRS)
    return df[["image", "report"]].copy()

@st.cache_data(show_spinner=False)
def list_runs():
    if not RUNS_DIR.exists():
        return []
    return sorted([str(p) for p in RUNS_DIR.iterdir() if p.is_dir()])

@st.cache_data(show_spinner=False)
def load_labels():
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text())
    return {}

@st.cache_data(show_spinner=False)
def load_passlist():
    if PASSLIST_PATH.exists():
        lines = [l.strip() for l in PASSLIST_PATH.read_text().splitlines() if l.strip()]
        return set(lines)
    return set()

@st.cache_data(show_spinner=False)
def load_demo_cases():
    if DEMO_JSON.exists():
        data = json.loads(DEMO_JSON.read_text())
        images = [c["image"] for c in data.get("cases", [])]
        cases_by_image = {c["image"]: c for c in data.get("cases", [])}
        return images, cases_by_image
    return [], {}

def energy_topq(delta: np.ndarray, q: float = 0.10) -> float:
    """Mean of the top-q% hottest pixels."""
    d = np.asarray(delta, np.float32).ravel()
    keep = max(1, int(len(d) * max(0.001, min(0.5, q))))
    thresh = np.partition(d, len(d) - keep)[len(d) - keep]
    sel = d[d >= thresh]
    return float(sel.mean()) if sel.size else 0.0

# ------------------------ sidebar ------------------------

runs = list_runs()
if not runs:
    st.warning("No runs found under models/sae_final/.")
    st.stop()

run = st.sidebar.selectbox("SAE run", runs, index=0)

use_curated = st.sidebar.checkbox("Use curated demo cases (demo_cases.json)", value=True)
only_pass = st.sidebar.checkbox("Show only QA-passing images (gate_pass_images.txt)", value=not use_curated)

k = st.sidebar.slider("Top-k latents", 1, 6, 3)
use_pct = st.sidebar.checkbox("Use percentile threshold for mask", value=True)
thr = st.sidebar.slider("Fixed threshold", 0.05, 0.90, 0.30, 0.05, disabled=use_pct)
pct = st.sidebar.slider("Percentile", 50, 99, 92, 1, disabled=not use_pct)

# QA gates (match upgraded evaluator defaults)
energy_metric = st.sidebar.selectbox("Energy metric", ["topq", "mean", "masked"], index=0)
topq = st.sidebar.slider("Top-q fraction", 0.01, 0.20, 0.10, 0.01, disabled=(energy_metric != "topq"))
mask_pct_for_energy = st.sidebar.slider("Masked energy percentile", 80, 99, 92, 1, disabled=(energy_metric != "masked"))
energy_gate = st.sidebar.slider("Energy gate", 0.00, 0.20, 0.03, 0.01)
mono_gate = st.sidebar.slider("Monotonicity gate (corr)", 0.0, 1.0, 0.60, 0.05)

show_labeled_only = st.sidebar.checkbox("Show only labeled latents", value=True)

# ------------------------ image list (curated/pass/all) ------------------------

pairs = load_pairs()
labels_map = load_labels()
demo_images, demo_cases = load_demo_cases()
passset = load_passlist()

if use_curated and demo_images:
    pairs = pairs[pairs["image"].isin(demo_images)]
elif only_pass and passset:
    pairs = pairs[pairs["image"].isin(passset)]

q = st.sidebar.text_input("Filter images:", "")
subset = pairs[pairs["image"].str.contains(q, case=False, na=False)] if q else pairs
if subset.empty:
    st.warning("No images to show with current filters.")
    st.stop()

img_path = st.sidebar.selectbox("Image", subset["image"].tolist(), index=0)

# ------------------------ analyze ------------------------

if st.button("Analyze", type="primary", use_container_width=True):
    w = SAEWrapper(run)
    x = load_image(img_path, size=w.cfg["img_size"])
    base = x.numpy()[0, 0]  # ORIGINAL grayscale [0,1]

    # Choose latents: curated (if available), otherwise top-k by activation
    curated_latents = []
    if use_curated and img_path in demo_cases:
        curated_latents = [int(d["j"]) for d in demo_cases[img_path].get("latents", [])]

    if curated_latents:
        # build z for info, but enforce curated indices
        _, z = w.reconstruct(x)
        vals = [float(z[0, j].item()) for j in curated_latents]
        idx = curated_latents
    else:
        _, z = w.reconstruct(x)
        idx, vals = topk_latents_by_activation(z, k=k)

    # Build masks & QA per latent
    items = []  # (j, label_or_None, z, mask, qa_dict)
    alpha_scales = [0.0, 1.25, 1.5]
    steps_for_corr = [0.0, 0.25, 0.50]

    for j, v in zip(idx, vals):
        j = int(j)
        name = labels_map.get(str(j), {}).get("label", None)
        if show_labeled_only and not name:
            continue

        # QA energies across scales
        Es = []
        deltas_for_mask = None
        for s in alpha_scales:
            d = delta_for_scale(w, x, j, s)
            if s == 0.0:
                deltas_for_mask = d  # use ablation heatmap for masking
            if energy_metric == "mean":
                Es.append(mask_energy(d))
            elif energy_metric == "masked":
                # energy of masked hottest pixels at chosen percentile
                t = np.percentile(d, mask_pct_for_energy)
                m = d >= t
                Es.append(float(d[m].mean()) if m.any() else 0.0)
            else:
                Es.append(energy_topq(d, q=topq))

        Ebest = float(max(Es))
        mono = float(monotonicity(steps_for_corr, Es))
        passes = (Ebest >= energy_gate) and (mono >= mono_gate)

        # make a mask (ablation delta), using current UI threshold mode
        m = threshold_mask(deltas_for_mask, thr=thr, use_percentile=use_pct, p=pct)

        qa = {"Ebest": Ebest, "mono": mono, "passes": passes}
        items.append((j, name, float(v), m, qa))

    # Keep only QA-passing latents (clean demo)
    items = [it for it in items if it[4]["passes"]]
    if not items:
        st.warning("No findings passed QA on this image. Try relaxing gates or pick another image.")
        st.stop()

    st.session_state["A"] = dict(
        run=run,
        img=img_path,
        base=base,
        items=items,  # already filtered by label/QA as requested
        use_pct=use_pct, thr=float(thr), pct=int(pct),
        energy_metric=energy_metric, topq=float(topq), mask_pct_for_energy=int(mask_pct_for_energy),
        energy_gate=float(energy_gate), mono_gate=float(mono_gate),
    )

# ------------------------ render ------------------------

if "A" in st.session_state and st.session_state["A"]["img"] == img_path and st.session_state["A"]["run"] == run:
    A = st.session_state["A"]
    st.subheader(Path(img_path).name)

    # Compose options list (1:1 with masks)
    options = []
    for (j, name, zval, _mask, qa) in A["items"]:
        tag = name or f"Latent {j}"
        options.append(f"🟢 {tag}  (j={j}, z={zval:.3f}, E*={qa['Ebest']:.3f}, mono={qa['mono']:.2f})")

    choice = st.radio("Findings (click to overlay)", options, index=0)
    sel_i = options.index(choice)
    sel_j, sel_name, sel_z, sel_mask, sel_qa = A["items"][sel_i]

    # Overlay on ORIGINAL
    base = A["base"].astype(np.float32)
    rgb = np.stack([base, base, base], axis=-1)
    vis = rgb.copy()
    vis[sel_mask] = 0.65 * vis[sel_mask] + 0.35 * np.array([1, 0, 0], dtype=np.float32)

    cap = f"{Path(img_path).name} — {(sel_name or f'Latent {sel_j}')}  [E*={sel_qa['Ebest']:.3f}, mono={sel_qa['mono']:.2f}]"
    st.image(vis, use_container_width=True, caption=cap)

    st.button("Clear highlight", on_click=lambda: st.session_state.pop("A"))
else:
    st.info("Pick an image and click **Analyze**.")
