python app/scout_strict.py \
  --run models/sae_final/m384_lam0.0005_lr0.001_bs8_e12_seed42__ft1 \
  --k 3 --N 3 --require_label --save_cache \
  --mono_gate 0.85 --emin_gate 0.05 --area_min 0.01 --area_max 0.12 \
  --stab_iou_min 0.50 --q 0.10
