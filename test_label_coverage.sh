python - <<'PY'
import json, numpy as np, pandas as pd
from pathlib import Path
RUN = Path("models/sae_final/m384_lam0.0005_lr0.001_bs8_e12_seed42__ft1")
z = np.load(RUN/"z_val.npy")              # (N, m)
labels = set(map(int, json.load(open("models/latent_labels.json")).keys()))
k = 3
topk = np.argpartition(-z, kth=k-1, axis=1)[:,:k]
hit = sum(any(int(j) in labels for j in row) for row in topk)
print(f"Coverage: {hit}/{z.shape[0]} images ({100*hit/z.shape[0]:.1f}%) with >1 labeled latent in top-{k}")
PY
