import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from sae_utils import (
    SAEWrapper,
    load_image,
    topk_latents_by_activation,
    delta_for_scale,
    threshold_mask,
)

st.set_page_config(page_title="RadiSpect", layout="wide")

PAIRS       = Path("data/pairs.csv")
RUNS_DIR    = Path("models/sae-tuned")
LABELS_PATH = Path("models/latent_labels.json")
GATE_EVAL   = Path("models/gate_eval.csv")

CLICKABLE_TXT = Path("models/clickable_images.txt")
CLICKABLE_MAP = Path("models/clickable_map.json")


SAFE_SPANS_FILES = [
    "439_IM-2078-1001.dcm.png",  # (gran, false atel)
    "455_IM-2086-4004.dcm.png",  # (atel)
    "1034_IM-0028-1001.dcm.png", # (risky — pneumonia suspected)
]
CLINICIAN_FILES = [
    "165_IM-0427-1001.dcm.png",  # (gran, false atel)
    "455_IM-2086-4004.dcm.png",  # (atel)
    "903_IM-2409-1001.dcm.png",  # (gran)
]


@st.cache_data(show_spinner=False)
def list_runs():
    if not RUNS_DIR.exists(): return []
    return sorted([str(p) for p in RUNS_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")])

@st.cache_data(show_spinner=False)
def load_pairs():
    df = pd.read_csv(PAIRS)
    return df[["image","report"]].copy()

@st.cache_data(show_spinner=False)
def load_labels():
    return json.loads(LABELS_PATH.read_text()) if LABELS_PATH.exists() else {}

@st.cache_data(show_spinner=False)
def load_gate():
    if not GATE_EVAL.exists(): return None
    df = pd.read_csv(GATE_EVAL)
    if "latent" in df.columns:
        df["latent"] = df["latent"].astype(int)
    return df

@st.cache_data(show_spinner=False)
def load_clickables():
    s = set(l.strip() for l in CLICKABLE_TXT.read_text().splitlines() if l.strip()) if CLICKABLE_TXT.exists() else set()
    m = json.loads(CLICKABLE_MAP.read_text()) if CLICKABLE_MAP.exists() else {}
    return s, m

def name_to_path(df_pairs: pd.DataFrame, wanted: list[str]) -> list[str]:
    """Map base filenames to full relative paths from pairs.csv."""
    base_to_path = {Path(p).name: p for p in df_pairs["image"].tolist()}
    found = [base_to_path[n] for n in wanted if n in base_to_path]
    missing = [n for n in wanted if n not in base_to_path]
    if missing:
        st.warning("Missing in pairs.csv: " + ", ".join(missing))
    return found

def overlay_on(base01: np.ndarray, mask: np.ndarray):
    rgb = np.stack([base01, base01, base01], axis=-1).astype(np.float32)
    vis = rgb.copy()
    vis[mask] = 0.65*vis[mask] + 0.35*np.array([1,0,0], dtype=np.float32)
    return vis

def fmt_label_from_j(labels_map: dict, j: int) -> str:
    lbl = (labels_map.get(str(j),{}) or {}).get("label")
    return lbl if lbl else f"Latent {j}"

def sanitize_float(v):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f): return None
        return f
    except Exception:
        return None

runs = list_runs()
if not runs:
    st.error("No runs found under models/sae-tuned/.")
    st.stop()
run = st.sidebar.selectbox("SAE run", runs, index=0)

use_pct = st.sidebar.checkbox("Use percentile threshold", value=True)
thr = st.sidebar.slider("Fixed threshold", 0.05, 0.90, 0.30, 0.05, disabled=use_pct)
pct = st.sidebar.slider("Percentile", 50, 99, 80, 1, disabled=not use_pct)

pairs = load_pairs()
labels_map = load_labels()
gate_df = load_gate()
clickable_set, clickable_map = load_clickables()

safe_spans_options = name_to_path(pairs, SAFE_SPANS_FILES)
clinician_options  = name_to_path(pairs, CLINICIAN_FILES)

# ---------------- Tabs ----------------
tab1, tab2 = st.tabs(["Safe Spans (grounded-in-text)", "Clinician View (unseen review)"])

with tab1:
    st.subheader("Safe Spans — click a labeled finding to show its evidence mask")
    if not safe_spans_options:
        st.warning("No curated Safe Spans images found.")
    else:
        img_path = st.selectbox("Study", safe_spans_options, index=0, key="safe_img")

        # Load once for both lists (report spans + latent browser)
        w = SAEWrapper(run)
        x = load_image(img_path, size=w.cfg["img_size"])
        base = x.numpy()[0, 0]
        # z for top-activation ranking
        try:
            _, z = w.reconstruct(x)
        except Exception:
            z = None

        # Helpers
        def label_of(j: int) -> str:
            lbl = (labels_map.get(str(j), {}) or {}).get("label")
            return lbl if lbl else f"Latent {j}"

        left, right = st.columns([1, 1])

        with left:
            # Original report
            rep = pairs[pairs["image"] == img_path]["report"].iloc[0]
            st.markdown("**Original report**")
            st.text(rep)

            # Report-linked safe spans (from clickable_map)
            spans = clickable_map.get(img_path, [])
            st.markdown("**Findings from report (click to overlay)**")
            if not spans:
                st.info("No safe spans stored for this image. (Run build_clickable_list.py)")
            else:
                seen_span = set()
                ordered = []
                for rec in spans:
                    j = int(rec["latent"])
                    if j not in seen_span:
                        seen_span.add(j)
                        ordered.append(j)
                for j in ordered:
                    lbl = label_of(j)
                    if st.button(f"{lbl} (j={j})", key=f"safe_{j}"):
                        st.session_state["SAFE_SEL"] = j

            st.divider()

            # Latent Browser (labeled-only)
            st.markdown("**Latent browser (labeled)**")
            k_lat = st.slider("Top-k labeled activations (for this image)", 1, 12, 3, key="safe_k_lat")

            # A) QA-passed, labeled latents for this image (from gate_eval)
            qa_list = []
            if gate_df is not None and not gate_df.empty:
                rows = gate_df[(gate_df["image"] == img_path) & (gate_df["passed"] == 1)].copy()
                if "Ebest" in rows.columns:
                    rows = rows.sort_values("Ebest", ascending=False)
                for _, r in rows.iterrows():
                    j = int(r["latent"])
                    # labeled-only
                    if str(j) in labels_map and (labels_map[str(j)] or {}).get("label"):
                        qa_list.append(j)
                # de-dup while preserving order
                qa_list = list(dict.fromkeys(qa_list))

            with st.expander("QA-passed latents (labeled)"):
                if not qa_list:
                    st.caption("No QA-passed labeled latents recorded for this image.")
                else:
                    for j in qa_list:
                        lbl = label_of(j)
                        if st.button(f"✅ {lbl} (j={j})", key=f"qa_{j}"):
                            st.session_state["SAFE_SEL"] = j

            # B) Top-activation, labeled candidates (not necessarily QA-passed)
            with st.expander("Top-activation labeled candidates"):
                if z is None:
                    st.caption("Activation vector unavailable.")
                else:
                    idx, _ = topk_latents_by_activation(z, k=max(k_lat * 2, k_lat + 3))
                    listed = set(qa_list)  # avoid duplicates from QA list
                    shown = 0
                    for j_ in idx:
                        j = int(j_)
                        if j in listed:
                            continue
                        lbl = (labels_map.get(str(j), {}) or {}).get("label")
                        if not lbl:
                            continue  # labeled-only
                        if st.button(f"🟡 {lbl} (j={j})", key=f"cand_{j}"):
                            st.session_state["SAFE_SEL"] = j
                        listed.add(j)
                        shown += 1
                        if shown >= k_lat:
                            break
                    if shown == 0:
                        st.caption("No additional labeled candidates to show.")

            # Clear
            st.button("Clear highlight", key="safe_clear", on_click=lambda: st.session_state.pop("SAFE_SEL", None))

        with right:
            sel_j = st.session_state.get("SAFE_SEL", None)
            mask = np.zeros_like(base, dtype=bool)
            cap = Path(img_path).name
            if sel_j is not None:
                d = delta_for_scale(w, x, sel_j, 0.0)
                mask = threshold_mask(d, thr=thr, use_percentile=use_pct, p=pct)
                cap += " — " + label_of(sel_j) + f" (j={sel_j})"
            st.image(overlay_on(base, mask), use_container_width=True, caption=cap)



with tab2:
    st.subheader("Clinician View — propose grounded findings for unseen studies")
    if not clinician_options:
        st.warning("No curated Clinician images found.")
    else:
        img_path = st.selectbox("Study", clinician_options, index=0, key="clin_img")
        k_show = st.sidebar.slider("How many suggested findings", 1, 6, 3, key="k_show")

        if st.button("Analyze", type="primary", use_container_width=True, key="clin_an"):
            st.session_state.pop("CLIN_SEL", None)

            w = SAEWrapper(run)
            x = load_image(img_path, size=w.cfg["img_size"])
            base = x.numpy()[0,0]
            _, z = w.reconstruct(x)


            def get_label(j:int):
                return (labels_map.get(str(j),{}) or {}).get("label")

            choices=[]; seen=set()

            # 1) clickable for this image
            if img_path in clickable_map and clickable_map[img_path]:
                order=[]; seen_click=set()
                for rec in clickable_map[img_path]:
                    j=int(rec["latent"])
                    if j not in seen_click:
                        seen_click.add(j); order.append(j)
                for j in order:
                    lbl=get_label(j)
                    if not lbl: continue
                    zval=float(z[0,j].item())
                    eb=mn=float("nan")
                    if gate_df is not None:
                        rr=gate_df[(gate_df["image"]==img_path)&(gate_df["latent"]==j)]
                        if len(rr):
                            r=rr.iloc[0]; eb=sanitize_float(r.get("Ebest")); mn=sanitize_float(r.get("mono_corr"))
                    choices.append((j,lbl,zval,eb,mn,"clickable")); seen.add(j)
                    if len(choices)>=k_show: break

            # 2) QA-passed sorted by E*
            if len(choices)<k_show and gate_df is not None:
                rows=gate_df[(gate_df["image"]==img_path)&(gate_df["passed"]==1)].copy()
                if "Ebest" in rows.columns:
                    rows=rows.sort_values("Ebest", ascending=False)
                for _,r in rows.iterrows():
                    j=int(r["latent"])
                    if j in seen: continue
                    lbl=get_label(j)
                    if not lbl: continue
                    zval=float(z[0,j].item())
                    eb=sanitize_float(r.get("Ebest")); mn=sanitize_float(r.get("mono_corr"))
                    choices.append((j,lbl,zval,eb,mn,"qa")); seen.add(j)
                    if len(choices)>=k_show: break

            # 3) Fill with labeled high-activation candidates
            if len(choices)<k_show:
                idx,_=topk_latents_by_activation(z, k=max(k_show*2,k_show+3))
                for j_ in idx:
                    j=int(j_)
                    if j in seen: continue
                    lbl=get_label(j)
                    if not lbl: continue
                    zval=float(z[0,j].item())
                    choices.append((j,lbl,zval,None,None,"candidate")); seen.add(j)
                    if len(choices)>=k_show: break

            # Build masks
            items=[]
            for (j,lbl,zv,eb,mn,src) in choices:
                d = delta_for_scale(w, x, j, 0.0)
                m = threshold_mask(d, thr=thr, use_percentile=use_pct, p=pct)
                items.append({"j":j, "label":lbl, "z":zv, "Ebest":eb, "mono":mn, "mask":m, "src":src})

            st.session_state["CLIN"] = dict(run=run, img=img_path, base=base, items=items, report_text="")

        if "CLIN" in st.session_state and st.session_state["CLIN"]["img"]==img_path and st.session_state["CLIN"]["run"]==run:
            C = st.session_state["CLIN"]
            left, right = st.columns([1,1])

            with left:
                st.subheader(Path(img_path).name)
                sel_j = st.session_state.get("CLIN_SEL", None)
                mask = np.zeros_like(C["base"], dtype=bool)
                cap = Path(img_path).name
                if sel_j is not None:
                    for it in C["items"]:
                        if it["j"]==sel_j:
                            mask = it["mask"]; 
                            cap += " — " + it["label"] + f" (j={sel_j})"
                            break
                st.image(overlay_on(C["base"], mask), use_container_width=True, caption=cap)

                st.markdown("**Proposed findings (click to preview)**")
                for it in C["items"]:
                    cols = st.columns([0.22, 0.58, 0.20])
                    if cols[0].button("Show", key=f"clin_show_{it['j']}"):
                        st.session_state["CLIN_SEL"] = it["j"]
                    cols[1].write(f"{it['label']} · z={it['z']:.3f}")

            with right:
                st.subheader("Report editor")
                if "report_text" not in C:
                    C["report_text"] = ""
                report_text = st.text_area("Type your report here:", value=C["report_text"], height=360, key=f"rep_{Path(C['img']).stem}")

                if st.button("Insert ALL proposed findings as bullets", key="clin_ins_all"):
                    bullets = "\n".join(f"- {it['label']}." for it in C["items"])
                    prefix = report_text.rstrip()
                    report_text = (prefix + ("\n" if prefix else "") + bullets).strip()
                    st.session_state["CLIN"]["report_text"] = report_text

                if st.button("Save", type="primary", key="clin_save"):
                    missing = [it["label"] for it in C["items"] if it["label"].lower() not in report_text.lower()]
                    if missing:
                        n=len(missing); plural="finding" if n==1 else "findings"
                        st.error(f"We found {n} {plural} you did not include in your report: " + ", ".join(missing) + ". Saved anyway.")
                    else:
                        st.success("Saved.")

