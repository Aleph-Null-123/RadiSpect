import re, json
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd

VOCAB = {
  # core findings
  "pleural effusion": ["pleural effusion"],
  "pneumothorax": ["pneumothorax","ptx"],
  "cardiomegaly": ["cardiomegaly","enlarged heart","enlarged cardiac silhouette","enlarged cardiomediastinal silhouette"],
  "consolidation": ["consolidation","airspace opacity","air space opacity","airspace disease","air space disease","alveolar opacity","lobar consolidation"],
  "atelectasis": ["atelectasis","atelectatic","volume loss"],
  "pulmonary edema": ["pulmonary edema","interstitial edema","vascular congestion"],
  "pneumonia": ["pneumonia"],
  "granuloma": ["granuloma","calcified granuloma"],
  "nodule/mass": ["pulmonary nodule","nodule","mass"],
  "emphysema/hyperinflation": ["emphysema","hyperinflation"],
  "hiatal hernia": ["hiatal hernia"],
  "fracture": ["fracture"],

  "lines/tubes": ["endotracheal tube","tracheostomy","chest tube","picc","port","mediport","enteric tube","ng tube","feeding tube","catheter","central venous catheter","internal jugular","subclavian"],
  "sternotomy/cabg": ["sternotomy","median sternotomy","cabg"],
}

NEG_TRIGGERS = [r"\bno\b", r"\bwithout\b", r"\babsent\b", r"\bfree of\b", r"\bnegative for\b", r"\bruled out\b"]
PSEUDO_NEG = [r"\bno (?:significant )?change\b", r"\bunchanged\b", r"\bstable\b"]

def _norm(s:str)->str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9/ \-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _pos_neg_mention(text:str, phrase:str, window_tokens:int=6):
    """
    Count positive/negative hits of `phrase` in `text`.
    A hit is NEG if a negation trigger appears within `window_tokens` tokens before phrase,
    unless a pseudo-negation is closer.
    Returns (pos_count, neg_count) for that phrase.
    """
    t = _norm(text)
    if phrase not in t: return (0,0)

    toks = t.split()
    ptoks = phrase.split()
    L = len(ptoks)
    pos = neg = 0
    for i in range(len(toks)-L+1):
        if toks[i:i+L] == ptoks:
            start = max(0, i-window_tokens)
            ctx = " ".join(toks[start:i+L])
            pseudo = any(re.search(p, ctx) for p in PSEUDO_NEG)
            has_neg = any(re.search(n, ctx) for n in NEG_TRIGGERS)
            if has_neg and not pseudo:
                neg += 1
            else:
                pos += 1
    return (pos, neg)

def _background_rate(pairs:pd.DataFrame, vocab:dict):
    """
    For each label, compute how often ANY positive mention appears across all reports.
    Used for enrichment scoring to avoid generic boilerplate winning.
    """
    rates = {}
    texts = pairs["report"].astype(str).tolist()
    N = len(texts)
    for label, phrases in vocab.items():
        pos_any = 0
        for s in texts:
            hit = False
            for ph in phrases:
                p, n = _pos_neg_mention(s, ph)
                if p > 0:
                    hit = True; break
            if hit: pos_any += 1
        rates[label] = pos_any / max(1, N)
    return rates

def mine_labels(pairs_csv:str, z_npy:str, paths_csv:str,
                topK:int=50, min_support:int=8,
                min_enrichment:float=0.05, min_pos_ratio:float=0.6,
                out_json:str="models/latent_labels.json"):
    """
    For each latent j:
      1) Take topK images by activation z[:,j].
      2) For each candidate label: count positive vs negated mentions across those reports.
      3) Score = enrichment = (pos/topK) - background_rate[label].
      4) Keep the best label if all gates pass:
         - pos >= min_support
         - pos/(pos+neg) >= min_pos_ratio
         - enrichment >= min_enrichment
    Save: { "<j>": {"label": str, "support": int, "pos": int, "neg": int, "enrichment": float} }
    """
    pairs = pd.read_csv(pairs_csv)  # columns: image, report
    if "image" not in pairs or "report" not in pairs:
        raise ValueError("pairs.csv must have columns: image, report")

    z = np.load(z_npy)               # (N, m)
    paths = pd.read_csv(paths_csv)["image"].tolist()
    if z.shape[0] != len(paths):
        raise ValueError(f"z_npy rows ({z.shape[0]}) != paths ({len(paths)})")

    img2rep = dict(zip(pairs["image"], pairs["report"]))
    reps = [img2rep.get(p, "") for p in paths]

    bg = _background_rate(pairs, VOCAB)
    m = z.shape[1]
    out = {}
    for j in range(m):
        order = np.argsort(-z[:, j])
        picked = [i for i in order[:topK] if 0 <= i < len(reps)]
        if not picked: continue

        best_label, best_score, best_stats = None, -1e9, None

        for label, phrases in VOCAB.items():
            pos = neg = 0
            for i in picked:
                r = reps[i]
                pos_hit = neg_hit = 0
                for ph in phrases:
                    p, n = _pos_neg_mention(r, ph)
                    pos_hit += p; neg_hit += n
                # one vote per report per label:
                if pos_hit > 0 and not (neg_hit > 0 and pos_hit == 0):
                    pos += 1
                elif pos_hit == 0 and neg_hit > 0:
                    neg += 1

            total = pos + neg
            if total == 0: 
                continue
            pos_ratio = pos / max(1, total)
            enrich = (pos / len(picked)) - bg.get(label, 0.0)

            if pos >= min_support and pos_ratio >= min_pos_ratio and enrich >= min_enrichment:
                score = enrich
                if score > best_score:
                    best_score = score
                    best_label = label
                    best_stats = dict(pos=pos, neg=neg, enrichment=float(enrich))

        if best_label is not None:
            out[str(j)] = {
                "label": best_label,
                "support": int(best_stats["pos"]),
                "pos": int(best_stats["pos"]),
                "neg": int(best_stats["neg"]),
                "enrichment": best_stats["enrichment"]
            }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} labels to {out_json}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs.csv")
    ap.add_argument("--run", required=True, help="Path to a run dir containing z_val.npy and paths_val.csv")
    ap.add_argument("--topK", type=int, default=50)
    ap.add_argument("--min_support", type=int, default=8)
    ap.add_argument("--min_enrichment", type=float, default=0.05)
    ap.add_argument("--min_pos_ratio", type=float, default=0.6)
    ap.add_argument("--out", default="models/latent_labels.json")
    args = ap.parse_args()

    mine_labels(args.pairs, f"{args.run}/z_val.npy", f"{args.run}/paths_val.csv",
                topK=args.topK, min_support=args.min_support,
                min_enrichment=args.min_enrichment, min_pos_ratio=args.min_pos_ratio,
                out_json=args.out)