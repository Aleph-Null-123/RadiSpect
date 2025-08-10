python app/train_conv_sae.py \
  --train-size 1000 --val-size 200 \
  --epochs 12 --min-epochs 4 --patience 2 \
  --batch-size 8 \
  --latents 256 \
  --lambdas 0 \
  --lrs 1e-3 \
  --runs-dir "models/sae_baseline_run"