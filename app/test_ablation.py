import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sae_utils import SAEWrapper, load_image, topk_latents_by_activation, threshold_mask

PAIRS = Path("data/pairs.csv")
RUNS  = Path("models/sae_runs")
OUT   = Path("models/ablation_previews")

K = 3                     # top-k latents per image
USE_PERCENTILE = False    # set True to use percentile threshold
THR = 0.30                # fixed threshold if not percentile
PCT = 90                  # percentile if USE_PERCENTILE=True

def pick_run() -> str:
    lb_mse = RUNS / "leaderboard_by_mse.csv"
    if lb_mse.exists():
        df = pd.read_csv(lb_mse)
        run = df.sort_values("val_mse").iloc[0]["run"]
        return str(RUNS / run)
    dirs = [p for p in RUNS.iterdir() if p.is_dir()]
    if not dirs: raise SystemExit("No runs in models/sae_runs yet.")
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(dirs[0])

def overlay_and_save(base_gray, masks, out_png, title):
    base = base_gray.astype(np.float32)
    rgb = np.stack([base, base, base], axis=-1)
    colors = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
    vis = rgb.copy()
    for i, m in enumerate(masks):
        c = colors[i % 3].reshape(1,1,3)
        vis[m] = 0.65*vis[m] + 0.35*c
    vis = np.clip(vis, 0, 1)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,6)); plt.imshow(vis); plt.axis("off"); plt.title(title)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0); plt.close()

def main():
    run_dir = pick_run()
    print("Using run:", run_dir)
    w = SAEWrapper(run_dir)

    df = pd.read_csv(PAIRS)
    # test three images (start, middle, end)
    idxs = [0, max(0, len(df)//2 - 1), len(df)-1][:3]

    for i in idxs:
        img_path = df.loc[i, "image"]
        x = load_image(img_path, size=w.cfg["img_size"])
        xhat, z = w.reconstruct(x)
        top_idx, top_vals = topk_latents_by_activation(z, k=K)

        masks = []
        for j in top_idx:
            delta, _ = w.ablation_heatmap(x, j)
            m = threshold_mask(delta, thr=THR, use_percentile=USE_PERCENTILE, p=PCT)
            masks.append(m)

        stem = Path(img_path).stem
        title = f"{Path(img_path).name} | top-k={top_idx} | z={np.round(top_vals,3)}"
        out_png = OUT / f"{stem}__top{K}.png"
        delta, base = w.ablation_heatmap_with_base(x, j)
        
        overlay_and_save(base.numpy() if hasattr(base, "numpy") else base, masks, OUT/..., title)
        print(f"Saved {out_png}")

if __name__ == "__main__":
    main()