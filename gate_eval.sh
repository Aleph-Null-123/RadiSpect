python app/eval_gates.py \
  --run models/sae_final/m384_lam0.0005_lr0.001_bs8_e12_seed42__ft1 \
  --k 3 --energy_gate 0.05 --mono_gate 0.80 \
  --labeled_only \
  --out_csv models/gate_eval.csv \
  --out_images_csv models/gate_images.csv \
  --out_pass_list models/gate_pass_images.txt