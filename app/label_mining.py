import re, json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd

# --- tiny clinical vocab (you can edit) ---
VOCAB = {
  "pleural effusion": ["pleural effusion","effusion"],
  "pneumothorax": ["pneumothorax","ptx"],
  "cardiomegaly": ["cardiomegaly","enlarged heart","enlarged cardiac"],
  "consolidation": ["consolidation","airspace opacity","alveolar opacity"],
  "atelectasis": ["atelectasis","collapse"],
  "pulmonary edema": ["pulmonary edema","interstitial edema","edema"],
  "pneumonia": ["pneumonia"],
  "lines/tubes": ["line","lines","tube","tubes","catheter","et tube","endotracheal","ng tube","enteric"],
  "cardiac device": ["pacemaker","pacer","icd","leads","cardiac device","defibrillator"],
  "fracture": ["fracture"],
  "emphysema": ["emphysema","hyperinflation"],
  "nodule/mass": ["nodule","mass"],
  "pleural thickening": ["pleural thickening","pleural disease"],
  "normal": ["no acute","unremarkable","normal study"]
}

def _norm(s:str)->str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9/ \-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _contains_any(text:str, terms:list[str])->bool:
    t = _norm(text)
    for q in terms:
        if q in t:
            return True
    return False

def mine_labels(pairs_csv:str, z_npy:str, paths_csv:str,
                topK:int=50, min_support:int=8,
                out_json:str="models/latent_labels.json"):
    """
    For each latent j:
      - take topK images by z[:,j]
      - count vocabulary matches in their reports
      - assign the highest-support label if support >= min_support
    Saves { "<j>": {"label": str, "support": int} } to out_json
    """
    pairs = pd.read_csv(pairs_csv)  # must have columns: image, report
    if "image" not in pairs or "report" not in pairs:
        raise ValueError("pairs.csv must have columns: image, report")

    # load run activations aligned with paths
    z = np.load(z_npy)               # (N, m)
    paths = pd.read_csv(paths_csv)["image"].tolist()
    if z.shape[0] != len(paths):
        raise ValueError(f"z_npy rows ({z.shape[0]}) != paths ({len(paths)})")

    # build image->report map
    img2rep = dict(zip(pairs["image"], pairs["report"]))
    reps = [img2rep.get(p, "") for p in paths]

    m = z.shape[1]
    out = {}
    for j in range(m):
        order = np.argsort(-z[:, j])
        picked_idx = [i for i in order[:topK] if 0 <= i < len(reps)]
        picked_reports = [reps[i] for i in picked_idx]

        # score vocab
        best_label, best_count = None, 0
        for label, terms in VOCAB.items():
            c = sum(1 for r in picked_reports if _contains_any(r, terms))
            if c > best_count:
                best_label, best_count = label, c

        if best_label is not None and best_count >= min_support:
            out[str(j)] = {"label": best_label, "support": int(best_count)}

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} labels to {out_json}")

if __name__ == "__main__":
    # simple CLI: auto-pick a run under models/sae_final if not specified
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs.csv")
    ap.add_argument("--run", default=None, help="Path to a run dir containing z_val.npy and paths_val.csv")
    ap.add_argument("--topK", type=int, default=50)
    ap.add_argument("--min_support", type=int, default=8)
    ap.add_argument("--out", default="models/latent_labels.json")
    args = ap.parse_args()

    run_dir = args.run
    if run_dir is None:
        root = Path("models/sae_final")
        runs = [p for p in root.iterdir() if p.is_dir()]
        if not runs:
            sys.exit("No runs under models/sae_final/. Provide --run.")
        run_dir = str(runs[0])
        print("Auto-selected run:", run_dir)

    mine_labels(args.pairs, f"{run_dir}/z_val.npy", f"{run_dir}/paths_val.csv",
                topK=args.topK, min_support=args.min_support, out_json=args.out)
