python app/train_conv_sae.py \
  --train-size 1000 --val-size 200 \
  --epochs 12 --min-epochs 4 --patience 2 \
  --batch-size 8 \
  --latents 256 384 \
  --lambdas 3e-4 5e-4 8e-4 \
  --lrs 1e-3 \
  --save-epoch best_mse \
  --runs-dir "models/sae_final"
  
