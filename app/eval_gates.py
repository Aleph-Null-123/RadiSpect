import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

from sae_utils import SAEWrapper, load_image, topk_latents_by_activation, delta_for_scale
from qa_utils import mask_energy, monotonicity

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to SAE run dir (models/sae_final/...)")
    ap.add_argument("--pairs", default="data/pairs.csv")
    ap.add_argument("--k", type=int, default=3, help="Top-k latents per image")
    ap.add_argument("--energy_gate", type=float, default=0.05)
    ap.add_argument("--mono_gate", type=float, default=0.80)
    ap.add_argument("--labeled_only", action="store_true", help="Only count latents that have a mined label")
    ap.add_argument("--labels_json", default="models/latent_labels.json")
    ap.add_argument("--out_csv", default="models/gate_eval.csv")
    ap.add_argument("--out_images_csv", default="models/gate_images.csv")
    ap.add_argument("--out_pass_list", default="models/gate_pass_images.txt")
    args = ap.parse_args()

    # load images (ALL, in order)
    df = pd.read_csv(args.pairs)[["image","report"]]
    paths = df["image"].tolist()

    labels_map = {}
    lj = Path(args.labels_json)
    if lj.exists():
        labels_map = json.loads(lj.read_text())

    w = SAEWrapper(args.run)

    per_latent_rows = []
    per_image_rows = []

    passed_images = []
    seen_images = set()

    for img_path in paths:
        try:
            x = load_image(img_path, size=w.cfg["img_size"])
        except Exception as e:
            print(f"[skip] load failed for {img_path}: {e}")
            continue

        # top-k latents
        _, z = w.reconstruct(x)
        idx, vals = topk_latents_by_activation(z, k=args.k)

        items = []
        for j, v in zip(idx, vals):
            has_label = (str(int(j)) in labels_map)
            if args.labeled_only and not has_label:
                continue
            items.append((int(j), float(v), labels_map.get(str(int(j)), {}).get("label", "")))

        any_pass = False

        # compute QA for each kept latent
        alpha_grid = [0.0, 1.25, 1.5]      # ablate, +25%, +50% scale
        step_grid  = [0.0, 0.25, 0.50]     # for correlation axis

        for (j, zval, label) in items:
            energies = [mask_energy(delta_for_scale(w, x, j, a)) for a in alpha_grid]
            E0 = float(energies[0])
            mono = float(monotonicity(step_grid, energies))
            passed = int((E0 >= args.energy_gate) and (mono >= args.mono_gate))
            any_pass = any_pass or bool(passed)

            per_latent_rows.append({
                "image": img_path,
                "latent": j,
                "label": label,
                "z": zval,
                "E0_mean": E0,
                "mono_corr": mono,
                "passed": passed
            })

        # image-level summary
        if img_path not in seen_images:
            seen_images.add(img_path)
            # count only among evaluated items (after labeled_only filter)
            passed_count = sum(r["passed"] for r in per_latent_rows if r["image"] == img_path)
            per_image_rows.append({
                "image": img_path,
                "topk_evaluated": len(items),
                "num_passed": int(passed_count),
                "image_passes": int(any_pass)
            })
            if any_pass:
                passed_images.append(img_path)

    # write outputs
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_latent_rows).to_csv(args.out_csv, index=False)

    Path(args.out_images_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_image_rows).to_csv(args.out_images_csv, index=False)

    Path(args.out_pass_list).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_pass_list).write_text("\n".join(passed_images))

    # console summary
    total_imgs = len(per_image_rows)
    imgs_with_any = sum(r["image_passes"] for r in per_image_rows)
    total_latents = len(per_latent_rows)
    passed_latents = sum(r["passed"] for r in per_latent_rows)
    print(f"Run: {args.run}")
    print(f"Images evaluated     : {total_imgs}")
    print(f"Images with ≥1 pass  : {imgs_with_any}/{total_imgs} ({100*imgs_with_any/max(1,total_imgs):.1f}%)")
    print(f"Latents evaluated    : {total_latents}")
    print(f"Latents passing gates: {passed_latents} ({100*passed_latents/max(1,total_latents):.1f}%)")
    print(f"Wrote per-latent CSV : {args.out_csv}")
    print(f"Wrote per-image CSV  : {args.out_images_csv}")
    print(f"Wrote pass list      : {args.out_pass_list}")

if __name__ == "__main__":
    main()