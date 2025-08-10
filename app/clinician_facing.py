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

st.set_page_config(page_title="RadiSpect — Clinician Console", layout="wide")

# ---------------- Paths ----------------
PAIRS = Path("data/pairs.csv")
RUNS_DIR = Path("models/sae-tuned")
LABELS_PATH = Path("models/latent_labels.json")
GATE_EVAL = Path("models/gate_eval.csv")
PASSLIST = Path("models/gate_pass_images.txt")
TRIAGE_CSV = Path("models/triage_queue.csv")
TRIAGE_JSON = Path("models/triage_index.json")

DRAFT_DIR = Path("models/drafts")
PROV_DIR  = Path("models/provenance")

# Clickable provenance (built by app/build_clickable_list.py)
CLICKABLE_PATH = Path("models/clickable_images.txt")
CLICKMAP_PATH  = Path("models/clickable_map.json")

# --------------- Loaders ----------------
@st.cache_data(show_spinner=False)
def load_clickable():
    s = set()
    m = {}
    if CLICKABLE_PATH.exists():
        s = set(l.strip() for l in CLICKABLE_PATH.read_text().splitlines() if l.strip())
    if CLICKMAP_PATH.exists():
        m = json.loads(CLICKMAP_PATH.read_text())
    return s, m

@st.cache_data(show_spinner=False)
def list_runs():
    if not RUNS_DIR.exists(): return []
    # hide hidden dirs like ".DS_Store" etc.
    return sorted([str(p) for p in RUNS_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")])

@st.cache_data(show_spinner=False)
def load_pairs():
    df = pd.read_csv(PAIRS)
    return df[["image", "report"]].copy()

@st.cache_data(show_spinner=False)
def load_labels():
    return json.loads(LABELS_PATH.read_text()) if LABELS_PATH.exists() else {}

@st.cache_data(show_spinner=False)
def load_gate_eval_df():
    if not GATE_EVAL.exists(): return None
    df = pd.read_csv(GATE_EVAL)
    if "latent" in df.columns:
        df["latent"] = df["latent"].astype(int)
    return df

@st.cache_data(show_spinner=False)
def load_passlist():
    if not PASSLIST.exists(): return set()
    return set(l.strip() for l in PASSLIST.read_text().splitlines() if l.strip())

@st.cache_data(show_spinner=False)
def load_triage():
    if TRIAGE_CSV.exists() and TRIAGE_JSON.exists():
        q = pd.read_csv(TRIAGE_CSV)
        j = json.loads(TRIAGE_JSON.read_text())
        return q, j
    return None, None

# --------------- Utils ----------------
def overlay_on(base01: np.ndarray, mask: np.ndarray):
    rgb = np.stack([base01, base01, base01], axis=-1).astype(np.float32)
    vis = rgb.copy()
    vis[mask] = 0.65 * vis[mask] + 0.35 * np.array([1, 0, 0], dtype=np.float32)
    return vis

def fmt_label_from_item(it: dict) -> str:
    return str(it["label"]) if it.get("label") else f"Latent {it['j']}"

def fmt_label_from_j(labels_map: dict, j: int) -> str:
    lbl = (labels_map.get(str(j), {}) or {}).get("label")
    return lbl if lbl else f"Latent {j}"

def safe_excerpt(text: str, s: int, e: int, pad: int = 30) -> str:
    left = max(0, s - pad)
    right = min(len(text), e + pad)
    frag = text[left:right].replace("\n", " ")
    return f" “…{frag}…”"

def sanitize_for_json(obj):
    """Make structure strictly JSON-valid (no NaN/Inf/np types)."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if obj is None:
        return None
    if isinstance(obj, (int, str)):
        return obj
    return str(obj)

def write_json_strict(path: Path, data: dict):
    data_clean = sanitize_for_json(data)
    path.write_text(json.dumps(data_clean, indent=2, ensure_ascii=False, allow_nan=False))

def load_draft(image_path: str) -> str:
    p = DRAFT_DIR / f"{Path(image_path).stem}.md"
    return p.read_text() if p.exists() else ""

def save_draft(image_path: str, text: str, findings: list):
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    PROV_DIR.mkdir(parents=True, exist_ok=True)
    (DRAFT_DIR / f"{Path(image_path).stem}.md").write_text(text)
    prov = {
        "image": image_path,
        "findings": findings,  # list of {latent,label,z,Ebest,mono,mentioned_in_report}
        "note": "Each finding corresponds 1:1 to a latent with a visible mask.",
    }
    write_json_strict(PROV_DIR / f"{Path(image_path).stem}__draft.json", prov)

def label_is_mentioned(report_text: str, label: str) -> bool:
    if not label: return False
    return label.lower() in (report_text or "").lower()

# --------------- Sidebar ----------------
mode = "Start Review"   # this file is focused on Start Review; Flag Review block remains below
runs = list_runs()
if not runs:
    st.warning("No runs found under models/sae-tuned/.")
    st.stop()
run = st.sidebar.selectbox("SAE run", runs, index=0)

use_pct = st.sidebar.checkbox("Use percentile threshold", value=True)
thr = st.sidebar.slider("Fixed threshold", 0.05, 0.90, 0.30, 0.05, disabled=use_pct)
pct = st.sidebar.slider("Percentile", 50, 99, 80, 1, disabled=not use_pct)

labels_map = load_labels()
pairs = load_pairs()
clickable_set, clickable_map = load_clickable()
gate_df = load_gate_eval_df()

# ----------- START REVIEW -----------
if mode == "Start Review":
    st.title("Start Review — propose grounded findings, type the report")

    # Filter to pass-list ∩ clickable to guarantee good demo cases with provenance
    passset = load_passlist()
    cand = pairs.copy()
    if passset:
        cand = cand[cand["image"].isin(passset)]
    if clickable_set:
        cand = cand[cand["image"].isin(clickable_set)]
    else:
        st.warning("No clickable provenance list found at models/clickable_images.txt. Run app/build_clickable_list.py first.")
        st.stop()

    img_path = st.selectbox("Study", cand["image"].tolist(), index=0)

    k_show = st.sidebar.slider("How many suggested findings", 1, 6, 3)
    #prefer_labeled = st.sidebar.checkbox("Prefer labeled latents", value=True)

    if st.button("Load study", type="primary", use_container_width=True):
        st.session_state.pop("SEL_LAT", None)

        w = SAEWrapper(run)
        x = load_image(img_path, size=w.cfg["img_size"])
        base = x.numpy()[0, 0]
        _, z = w.reconstruct(x)

        # ---- Build choices in three tiers: clickable -> QA -> labeled candidates ----
        def add_choice(buf, seen, j, lbl, zval, ebest=float("nan"), mono=float("nan"), source="qa"):
            buf.append((int(j), lbl, float(zval), float(ebest), float(mono), source))
            seen.add(int(j))
        
        def get_label(j: int):
            return (labels_map.get(str(j), {}) or {}).get("label")
        
        choices, seen = [], set()
        
        # 1) Clickable-mapped latents first (label + positive span)
        if img_path in clickable_map and clickable_map[img_path]:
            order, seen_click = [], set()
            for rec in clickable_map[img_path]:
                j = int(rec["latent"])
                if j not in seen_click:
                    seen_click.add(j); order.append(j)
            for j in order[:k_show]:
                lbl = get_label(j)
                if not lbl:  # labeled-only
                    continue
                zval = float(z[0, j].item())
                ebest = mono = float("nan")
                if gate_df is not None:
                    rr = gate_df[(gate_df["image"] == img_path) & (gate_df["latent"] == j)]
                    if len(rr):
                        r = rr.iloc[0]
                        ebest = float(r.get("Ebest", float("nan")))
                        mono  = float(r.get("mono_corr", float("nan")))
                add_choice(choices, seen, j, lbl, zval, ebest, mono, source="clickable")
        
        # 2) QA-passed latents (sorted by Ebest), labeled-only
        if len(choices) < k_show and gate_df is not None:
            rows = gate_df[(gate_df["image"] == img_path) & (gate_df["passed"] == 1)].copy()
            if "Ebest" in rows.columns:
                rows = rows.sort_values("Ebest", ascending=False)
            for _, r in rows.iterrows():
                j = int(r["latent"])
                if j in seen: 
                    continue
                lbl = get_label(j)
                if not lbl:  # labeled-only
                    continue
                zval = float(z[0, j].item())
                ebest = float(r.get("Ebest", float("nan")))
                mono  = float(r.get("mono_corr", float("nan")))
                add_choice(choices, seen, j, lbl, zval, ebest, mono, source="qa")
                if len(choices) >= k_show:
                    break
        
        # 3) Fill to k with top-activation **labeled** candidates (uncertified)
        if len(choices) < k_show:
            idx, _ = topk_latents_by_activation(z, k=max(k_show*2, k_show+3))
            for j_ in idx:
                j = int(j_)
                if j in seen:
                    continue
                lbl = get_label(j)
                if not lbl:  # labeled-only
                    continue
                zval = float(z[0, j].item())
                add_choice(choices, seen, j, lbl, zval, source="candidate")
                if len(choices) >= k_show:
                    break
        
        # If we couldn't reach k_show, let the user know how many labeled suggestions we found
        if len(choices) < k_show:
            st.info(f"Showing {len(choices)} labeled finding(s); no more labeled latents available for this study.")


        # build masks now so clicks are instant
        items = []
        for (j, lbl, zval, ebest, mono, src) in choices:
            d = delta_for_scale(w, x, j, 0.0)  # ablation
            m = threshold_mask(d, thr=thr, use_percentile=use_pct, p=pct)
            items.append({"j": j, "label": lbl, "z": zval, "Ebest": ebest, "mono": mono, "mask": m, "src": src})

        st.session_state["SR"] = dict(run=run, img=img_path, base=base, items=items)
        st.session_state["__report_text"] = ""   # start blank on new analysis

    if "SR" in st.session_state and st.session_state["SR"]["img"] == img_path and st.session_state["SR"]["run"] == run:
        S = st.session_state["SR"]
        left, right = st.columns([1, 1])

        with left:
            st.subheader(Path(img_path).name)
            sel_j = st.session_state.get("SEL_LAT", None)
            if sel_j is not None:
                mask = np.zeros_like(S["base"], dtype=bool)
                for it in S["items"]:
                    if it["j"] == sel_j:
                        mask = it["mask"]; break
                vis = overlay_on(S["base"], mask)
                cap = f"{Path(img_path).name} — j={sel_j}"
            else:
                vis = overlay_on(S["base"], np.zeros_like(S["base"], dtype=bool))
                cap = Path(img_path).name
            st.image(vis, use_container_width=True, caption=cap)

            st.markdown("**Proposed findings (click to preview mask):**")
            for it in S["items"]:
                lbl_text = fmt_label_from_item(it)
                cols = st.columns([0.22, 0.58, 0.20])
                if cols[0].button("Show", key=f"show_{it['j']}"):
                    st.session_state["SEL_LAT"] = it["j"]
                cols[1].write(f"{lbl_text}  ·  z={it['z']:.3f}")
                # badge = "✅ **QA**" if it.get("src") in ("qa","clickable") else "🟡 _candidate_"
                #cols[2].markdown(badge)

        with right:
            st.subheader("Report editor")

            def _label(it): return fmt_label_from_item(it)

            # always blank unless user adds text
            if "__report_text" not in st.session_state:
                st.session_state["__report_text"] = ""
            report_text = st.text_area(
                "Type your report here:",
                value=st.session_state["__report_text"],
                height=360,
                key=f"rep_{Path(S['img']).stem}",
            )

            if st.button("Insert ALL proposed findings as bullets"):
                bullets = "\n".join([f"- {_label(it)}." for it in S["items"]])
                prefix = report_text.rstrip()
                new_text = (prefix + ("\n" if prefix else "") + bullets).strip()
                st.session_state["__report_text"] = new_text
                report_text = new_text

            if st.button("Save draft + provenance", type="primary"):
                findings = []
                missing_labels = []
                for it in S["items"]:
                    lbl = _label(it)
                    mentioned = label_is_mentioned(report_text, lbl)
                    if not mentioned:
                        missing_labels.append(lbl)
                    findings.append({
                        "latent": int(it["j"]),
                        "label": lbl,
                        "z": float(it["z"]),
                        "Ebest": float(it["Ebest"]) if not (isinstance(it["Ebest"], float) and math.isnan(it["Ebest"])) else None,
                        "mono": float(it["mono"]) if not (isinstance(it["mono"], float) and math.isnan(it["mono"])) else None,
                        "source": it.get("src", "qa"),
                        "mentioned_in_report": bool(mentioned),
                    })

                save_draft(S["img"], report_text, findings)
                if missing_labels:
                    n = len(missing_labels)
                    plural = "finding" if n==1 else "findings"
                    st.error("We found " + str(n) + f" {plural} you did not include in your report: " + ", ".join(missing_labels))
                    st.info("Draft and provenance were saved. You can edit the report and save again.")
                else:
                    st.success("Saved draft and provenance JSON. All proposed findings are present in the report text.")

# ----------- FLAG REVIEW (unchanged from your last version) -----------
else:
    st.title("Flag Review — evidence ↔ text mismatch checks")

    queue, tri = load_triage()
    if queue is None:
        st.warning("No triage files. Run:  python app/triage_build.py")
        st.stop()

    st.sidebar.write("Queue filters")
    max_sev = int(queue["severity_score"].max()) if not queue.empty else 0
    min_sev = st.sidebar.slider("Min severity", 0, max_sev, min(1, max_sev))
    require_evid = st.sidebar.checkbox("Require evidence-only flags", value=False)
    require_text = st.sidebar.checkbox("Require text-only flags", value=False)

    df = queue.copy()
    df = df[df["severity_score"] >= min_sev]
    if require_evid:
        df = df[df["n_evidence_only"] > 0]
    if require_text:
        df = df[df["n_text_only"] > 0]

    st.subheader("Flagged studies")
    show_cols = ["image", "severity_score", "n_evidence_only", "n_text_only", "n_aligned", "evidence_only", "text_only"]
    st.dataframe(df[show_cols].reset_index(drop=True), use_container_width=True)

    if df.empty:
        st.info("No studies match filters.")
        st.stop()

    img_path = st.selectbox("Study", df["image"].tolist(), index=0)

    if st.button("Load study", type="primary", use_container_width=True):
        st.session_state.pop("SEL_LAT", None)
        w = SAEWrapper(run)
        x = load_image(img_path, size=w.cfg["img_size"])
        base = x.numpy()[0, 0]
        st.session_state["FR"] = dict(run=run, img=img_path, base=base)

    if "FR" in st.session_state and st.session_state["FR"]["img"] == img_path and st.session_state["FR"]["run"] == run:
        F = st.session_state["FR"]
        left, right = st.columns([1, 1])

        with left:
            existing = load_draft(img_path) or pairs[pairs["image"] == img_path]["report"].iloc[0]
            report_text = st.text_area("Report (editable)", value=existing, height=360, key=f"rep_edit_{Path(img_path).stem}")

            info = tri.get(img_path, {"aligned": [], "evidence_only": [], "text_only": []})
            st.markdown("**Flags**")
            st.caption("👀 evidence-only (present on image, missing in text); ⚠️ text-only (in text, not supported by mask); ✅ aligned.")

            for tag, icon, keyp in [("evidence_only", "👀", "e"), ("aligned", "✅", "a")]:
                items = info.get(tag, [])
                if items:
                    st.write(f"**{icon} {tag.replace('_', ' ').title()}**")
                for rec in items:
                    j = int(rec["latent"])
                    lbl = rec.get("label") or fmt_label_from_j(labels_map, j)
                    if st.button(f"{icon} {lbl} (j={j})", key=f"{keyp}_{j}"):
                        st.session_state["SEL_LAT"] = j

            texts = info.get("text_only", [])
            if texts:
                st.write("**⚠️ Text-only**")
                rep0 = pairs[pairs["image"] == img_path]["report"].iloc[0]
                for rec in texts:
                    lbl = rec["label"]
                    spans = rec.get("spans", [])
                    if spans:
                        s, e = spans[0]
                        excerpt = safe_excerpt(rep0, s, e)
                        st.caption(f"⚠️ {lbl}{excerpt}")
                    else:
                        st.caption(f"⚠️ {lbl}")

            if st.button("Save edited report"):
                st.warning("Saving report only — findings/provenance not changed in Flag Review.")
                save_draft(img_path, report_text, findings=[])
                st.success("Saved edited report.")

        with right:
            sel_j = st.session_state.get("SEL_LAT", None)
            vis = overlay_on(F["base"], np.zeros_like(F["base"], dtype=bool))
            cap = Path(img_path).name
            if sel_j is not None:
                w = SAEWrapper(run)
                x = load_image(img_path, size=w.cfg["img_size"])
                d = delta_for_scale(w, x, sel_j, 0.0)
                m = threshold_mask(d, thr=thr, use_percentile=use_pct, p=pct)
                vis = overlay_on(F["base"], m)
                lbl = fmt_label_from_j(labels_map, sel_j)
                cap = f"{Path(img_path).name} — {lbl} (j={sel_j})"

            st.image(vis, use_container_width=True, caption=cap)
            if st.button("Clear selection"):
                st.session_state.pop("SEL_LAT", None)