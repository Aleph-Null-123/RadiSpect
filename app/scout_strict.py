import json, argparse
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

from sae_utils import SAEWrapper, load_image, topk_latents_by_activation, delta_for_scale
# reuse simple stats
def iou(a,b):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a,b).sum(); uni = np.logical_or(a,b).sum()
    return float(inter/(uni+1e-8))

def mask_from_percentile(delta: np.ndarray, pct: int=92):
    d = np.asarray(delta, np.float32)
    t = np.percentile(d, pct)
    return (d >= t)

def energy_topq(delta: np.ndarray, q: float=0.10) -> float:
    d = np.asarray(delta, np.float32).ravel()
    keep = max(1, int(len(d)*q))
    t = np.partition(d, len(d)-keep)[len(d)-keep]
    return float(d[d>=t].mean()) if keep>0 else 0.0

def mask_area_frac(mask: np.ndarray) -> float:
    return float(mask.mean())

def bbox_compactness(mask: np.ndarray) -> float:
    # ratio: mask area / bounding-box area (1.0 = perfectly compact)
    ys, xs = np.where(mask)
    if ys.size == 0: return 0.0
    h = ys.max()-ys.min()+1; w = xs.max()-xs.min()+1
    return float(mask.sum()/(h*w+1e-8))

def monotonicity(steps, energies) -> float:
    s = np.asarray(steps, float); e = np.asarray(energies, float)
    if s.std()<1e-8 or e.std()<1e-8: return 0.0
    return float(np.corrcoef(s,e)[0,1])

def overlay_png(base01: np.ndarray, mask: np.ndarray, out_path: Path):
    base = np.stack([base01, base01, base01], axis=-1).astype(np.float32)
    vis = base.copy()
    vis[mask] = 0.65*vis[mask] + 0.35*np.array([1,0,0], dtype=np.float32)
    Image.fromarray((np.clip(vis,0,1)*255).astype(np.uint8)).save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="models/sae_final/<RUN_DIR>")
    ap.add_argument("--pairs", default="data/pairs.csv")
    ap.add_argument("--labels_json", default="models/latent_labels.json")
    ap.add_argument("--k", type=int, default=3, help="top-k latents per image")
    ap.add_argument("--N", type=int, default=5, help="how many images to keep")
    # strict thresholds (tune if too few found)
    ap.add_argument("--q", type=float, default=0.10, help="top-q fraction for energy")
    ap.add_argument("--mono_gate", type=float, default=0.85)
    ap.add_argument("--emin_gate", type=float, default=0.05)   # on best-of-scales energy
    ap.add_argument("--area_min", type=float, default=0.01)    # 1% image
    ap.add_argument("--area_max", type=float, default=0.12)    # 12% image
    ap.add_argument("--stab_pcts", type=int, nargs="+", default=[90,92,94])
    ap.add_argument("--stab_iou_min", type=float, default=0.50)
    ap.add_argument("--overlap_penalty_iou", type=float, default=0.20)
    ap.add_argument("--require_label", action="store_true", default=True)
    ap.add_argument("--save_cache", action="store_true", help="save overlays to models/demo_cache/")
    ap.add_argument("--out_json", default="models/demo_cases.json")
    args = ap.parse_args()

    # load labels if available
    labels_map = {}
    p = Path(args.labels_json)
    if p.exists():
        labels_map = json.loads(p.read_text())

    df = pd.read_csv(args.pairs)[["image","report"]]
    paths = df["image"].tolist()

    w = SAEWrapper(args.run)

    # per-image scoring
    cases = []
    scales = [0.0, 1.25, 1.5]
    step_for_corr = [0.0, 0.25, 0.5]

    for img_path in paths:
        try:
            x = load_image(img_path, size=w.cfg["img_size"])
        except Exception as e:
            continue
        base = x.numpy()[0,0]

        _, z = w.reconstruct(x)
        idx, vals = topk_latents_by_activation(z, k=args.k)

        latents = []
        for j, zv in zip(idx, vals):
            lbl = labels_map.get(str(int(j)), {}).get("label", "")
            if args.require_label and not lbl:
                continue

            # energy across scales (use top-q% metric)
            Es = []
            deltas = []
            for s in scales:
                d = delta_for_scale(w, x, int(j), s)
                deltas.append(d)
                Es.append( energy_topq(d, q=args.q) )
            Ebest = float(max(Es))
            mono  = float(monotonicity(step_for_corr, Es))
            if Ebest < args.emin_gate or mono < args.mono_gate:
                continue

            # mask stability across thresholds at baseline (1.0 scale)
            d1 = deltas[1]  # scale ~ 1.* (baseline-ish)
            masks = [mask_from_percentile(d1, p) for p in args.stab_pcts]
            # area and compactness on the middle threshold
            m_mid = masks[len(masks)//2]
            area = mask_area_frac(m_mid)
            if not (args.area_min <= area <= args.area_max):
                continue
            compact = bbox_compactness(m_mid)

            stabs = []
            for a in range(len(masks)):
                for b in range(a+1, len(masks)):
                    stabs.append(iou(masks[a], masks[b]))
            stab_min = float(min(stabs)) if stabs else 0.0
            if stab_min < args.stab_iou_min:
                continue

            latents.append({
                "j": int(j),
                "label": lbl,
                "z": float(zv),
                "Ebest": Ebest,
                "mono": mono,
                "area": area,
                "compact": compact,
                "mask": m_mid  # for overlap calc & optional caching
            })

        if len(latents) == 0:
            continue

        # distinctness: penalize overlap between selected latents
        # (greedy pick best trio)
        latents = sorted(latents, key=lambda r: (2*r["Ebest"] + r["mono"] + 0.5*r["compact"]), reverse=True)
        chosen = []
        for L in latents:
            if len(chosen) >= args.k: break
            ok = True
            for C in chosen:
                if iou(L["mask"], C["mask"]) > args.overlap_penalty_iou:
                    ok = False; break
            if ok: chosen.append(L)

        if len(chosen)==0:
            continue

        img_score = sum(2*c["Ebest"] + c["mono"] + 0.5*c["compact"] for c in chosen)
        cases.append({
            "image": img_path,
            "score": float(img_score),
            "latents": [{"j":c["j"], "label":c["label"], "Ebest":c["Ebest"], "mono":c["mono"], "area":c["area"], "compact":c["compact"]} for c in chosen]
        })

        # optional cache of overlays (middle mask)
        if args.save_cache:
            outdir = Path("models/demo_cache")/Path(img_path).stem
            outdir.mkdir(parents=True, exist_ok=True)
            for c in chosen:
                overlay_png(base, c["mask"], outdir/f"latent_{c['j']:03d}.png")

    # pick top N
    cases = sorted(cases, key=lambda r: -r["score"])[:args.N]
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"run": args.run, "cases": cases}, indent=2))
    print(f"Saved {len(cases)} curated cases → {args.out_json}")
    if cases:
        print("Top case:", cases[0]["image"], "score=", round(cases[0]["score"],3))
