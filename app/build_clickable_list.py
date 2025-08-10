import re, json
from pathlib import Path
import pandas as pd

PAIRS = Path("data/pairs.csv")
GATE  = Path("models/gate_eval.csv")
LABELS= Path("models/latent_labels.json")
OUT_TXT = Path("models/clickable_images.txt")
OUT_JSON= Path("models/clickable_map.json")

# simple synonym list (extend as needed)
SYN = {
    "pleural effusion": [r"pleural effusion", r"effusion"],
    "cardiomegaly": [r"cardiomegaly", r"enlarg(?:ed|ement) (?:cardiac|cardiomediastinal) (?:silhouette|contours?)"],
    "pneumothorax": [r"pneumothorax"],
    "consolidation": [r"consolidation", r"airspace opacity"],
    "opacity": [r"opacity", r"opacities"],
    "atelectasis": [r"atelectasis"],
    "lines/tubes": [r"(?:lines?|tubes?)", r"endotracheal tube", r"ng tube", r"picc", r"central line"],
    "pacemaker": [r"pacemaker", r"cardiac device", r"icd", r"pacer leads?"],
    "edema": [r"edema"],
    "effusion": [r"effusion"],
}
NEG_WORDS = r"(no|without|absent|free of|negative for|no evidence of|denies)"
NEG_WINDOW_TOKENS = 6

def token_spans(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
def negated(text, start, end):
    toks = token_spans(text)
    idx = 0
    for i,(s,e) in enumerate(toks):
        if s <= start < e: idx = i; break
    lo = max(0, idx-NEG_WINDOW_TOKENS)
    window = text[toks[lo][0]:start] if toks else text[:start]
    return re.search(NEG_WORDS, window, flags=re.IGNORECASE) is not None

def positive_spans(report, label):
    pats = [re.escape(label)]
    for k,vals in SYN.items():
        if k.lower() == label.lower(): pats += vals
    rx = re.compile("(" + "|".join(pats) + ")", flags=re.IGNORECASE)
    hits=[]
    for m in rx.finditer(report):
        s,e = m.start(), m.end()
        if not negated(report, s, e): hits.append((s,e))
    # merge overlaps
    hits = sorted(hits)
    merged=[]
    for s,e in hits:
        if not merged: merged.append([s,e]); continue
        ps,pe = merged[-1]
        if s <= pe: merged[-1][1] = max(pe,e)
        else: merged.append([s,e])
    return [(s,e) for s,e in merged]

def main():
    assert PAIRS.exists() and GATE.exists(), "Missing pairs.csv or gate_eval.csv"
    pairs = pd.read_csv(PAIRS)[["image","report"]]
    gate  = pd.read_csv(GATE)
    gate = gate[(gate.get("passed",0)==1)]
    gate["latent"] = gate["latent"].astype(int)
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}

    clickable = []
    mapping = {}
    for img, rep in pairs[["image","report"]].itertuples(index=False):
        rows = gate[gate["image"]==img]
        if rows.empty: continue
        found = []
        for _,r in rows.iterrows():
            j = int(r["latent"])
            lbl = labels.get(str(j),{}).get("label","").strip()
            if not lbl: continue
            spans = positive_spans(rep, lbl)
            if spans:
                found.append({"latent": j, "label": lbl, "spans": spans})
        if found:
            clickable.append(img)
            mapping[img] = found

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(clickable))
    OUT_JSON.write_text(json.dumps(mapping, indent=2))
    print(f"Clickable images: {len(clickable)}  -> {OUT_TXT}")
    print(f"Clickable map    : {len(mapping)} images  -> {OUT_JSON}")

if __name__ == "__main__":
    main()