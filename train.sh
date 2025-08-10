python app/train_conv_sae.py \
  --train-size 1000 --val-size 200 \
  --epochs 12 --min-epochs 4 --patience 2 \
  --batch-size 8 \
  --latents 128 256 \
  --lambdas 1e-3 2e-3 \
  --lrs 1e-3 \
  --save-epoch best_mse
