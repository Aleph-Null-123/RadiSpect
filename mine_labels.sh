python app/label_mining.py --run models/sae_final/m384_lam0.0005_lr0.001_bs8_e12_seed42__ft1 \
  --topK 60 --min_support 5 --min_enrichment 0.02 --min_pos_ratio 0.55 \
  --out models/latent_labels.json
