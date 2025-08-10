import numpy as np, pandas as pd
import streamlit as st
from pathlib import Path
from sae_utils import SAEWrapper, load_image, topk_latents_by_activation

st.set_page_config(page_title="RadiSpect — Click-to-Evidence (Threshold Tuning)", layout="wide")
PAIRS = "data/pairs.csv"
RUNS_DIR = Path("models/sae_final")

@st.cache_data(show_spinner=False)
def list_runs(root: str):
    d = Path(root)
    if not d.exists(): return []
    return sorted([str(p) for p in d.iterdir() if p.is_dir()])

@st.cache_data(show_spinner=False)
def load_pairs(pairs_path: str):
    df = pd.read_csv(pairs_path)
    return df[["image","report"]].copy()

def threshold_from_mode(delta: np.ndarray, mode: str, thr: float, pct: int, area_pct: int) -> np.ndarray:
    """delta in [0,1], returns boolean mask"""
    d = np.asarray(delta, dtype=np.float32).ravel()
    if mode == "Fixed":
        t = thr
    elif mode == "Percentile":
        t = np.percentile(d, pct)
    else:  # Top-X% area
        keep = max(1, int(len(d) * (area_pct/100.0)))
        # threshold = value at (len-keep)-th order statistic
        t = np.partition(d, len(d)-keep)[len(d)-keep]
    return (delta >= t)

def mask_energy(delta: np.ndarray) -> float:
    return float(np.mean(delta))

def mask_area(mask: np.ndarray) -> float:
    return float(mask.mean())  # fraction in [0,1]

# ---- UI ----
st.sidebar.title("RadiSpect — Threshold Tuning")
runs = list_runs(str(RUNS_DIR))
if not runs:
    st.warning(f"No runs found under {RUNS_DIR}/")
    st.stop()

run = st.sidebar.selectbox("SAE run", runs, index=0)
pairs = load_pairs(PAIRS)

q = st.sidebar.text_input("Filter images:", "")
subset = pairs[pairs["image"].str.contains(q, case=False, na=False)] if q else pairs
img_path = st.sidebar.selectbox("Image", subset["image"].tolist(), index=0)

k = st.sidebar.slider("Top-k latents", 1, 6, 3)

# Masking controls
mode = st.sidebar.radio("Mask mode", ["Fixed", "Percentile", "Top-X% area"], index=1)
thr = st.sidebar.slider("Fixed threshold", 0.05, 0.90, 0.30, 0.05, disabled=(mode!="Fixed"))
pct = st.sidebar.slider("Percentile", 50, 99, 92, 1, disabled=(mode!="Percentile"))
area_pct = st.sidebar.slider("Top-X% pixels", 1, 20, 8, 1, disabled=(mode!="Top-X% area"))

energy_gate = st.sidebar.slider("Energy gate (min mean Δ)", 0.00, 0.20, 0.05, 0.01,
                                help="Mask hidden if mean heatmap energy below this value")

analyze = st.sidebar.button("Analyze", type="primary", use_container_width=True)

# ---- Analyze ----
if analyze:
    w = SAEWrapper(run)
    x = load_image(img_path, size=w.cfg["img_size"])
    base = x.numpy()[0,0]  # ORIGINAL image in [0,1]
    # encode once
    _, z = w.reconstruct(x)
    idx, vals = topk_latents_by_activation(z, k=k)

    # store raw heatmaps (NOT binarized) so we can re-threshold interactively
    deltas = []
    for j in idx:
        delta, _ = w.ablation_heatmap(x, j)  # torch->np ok; values in [0,1]
        deltas.append(delta.numpy() if hasattr(delta, "numpy") else np.asarray(delta))

    st.session_state["A"] = dict(run=run, img=img_path, idx=idx, vals=[float(v) for v in vals],
                                 base=base, deltas=deltas, mode=mode, thr=thr, pct=pct, area_pct=area_pct,
                                 energy_gate=energy_gate)

# ---- Display ----
if "A" in st.session_state and st.session_state["A"]["img"] == img_path and st.session_state["A"]["run"] == run:
    A = st.session_state["A"]
    # update live knobs in case the user tweaks them without re-analyze
    A["mode"], A["thr"], A["pct"], A["area_pct"], A["energy_gate"] = mode, thr, pct, area_pct, energy_gate

    st.subheader(Path(img_path).name)

    # Findings list
    options = [f"Latent {j} (z={A['vals'][i]:.3f})" for i, j in enumerate(A["idx"])]
    choice = st.radio("Findings (click to overlay)", options, index=0)
    sel_i = options.index(choice)
    sel_j = A["idx"][sel_i]
    delta = A["deltas"][sel_i]

    E = mask_energy(delta)
    mask = threshold_from_mode(delta, A["mode"], A["thr"], A["pct"], A["area_pct"])

    # Energy gate
    if E < A["energy_gate"]:
        st.warning(f"Evidence too weak for Latent {sel_j} (mean Δ={E:.3f} < {A['energy_gate']:.3f}). Mask hidden.")
        show_mask = False
    else:
        show_mask = True

    # Overlay on ORIGINAL image
    base = A["base"].astype(np.float32)
    rgb = np.stack([base, base, base], axis=-1)
    vis = rgb.copy()
    if show_mask:
        vis[mask] = 0.65*vis[mask] + 0.35*np.array([1,0,0], dtype=np.float32)

    st.image(vis, use_container_width=True,
             caption=f"{Path(img_path).name} — Latent {sel_j} | mode={A['mode']} | E={E:.3f} | area={mask_area(mask):.3f}")

    # Quick stats panel
    st.markdown(
        f"**Stats:** mean Δ = `{E:.4f}`  ·  mask area = `{mask_area(mask):.3f}`  ·  "
        f"top-k latents = `{len(A['idx'])}`  ·  z = `{A['vals'][sel_i]:.3f}`"
    )

    st.button("Clear highlight", on_click=lambda: st.session_state.pop("A"))
else:
    st.info("Pick an image and click **Analyze**.")
