python app/finetune_moredata.py \
  --from_run models/sae_final/m384_lam0.0005_lr0.001_bs8_e12_seed42 \
  --train_glob "data/images_normalized/*.png" \
  --epochs 8 --batch_size 8 --lr 1e-3 \
  --save_to models/sae_final/m384_lam0.0005_lr0.001_bs8_e12_seed42__ft1
