export CUDA_VISIBLE_DEVICES=1

python exp_plas.py \
  --data-path /data/fno \
  --ntrain 900 \
  --ntest 80 \
  --ntotal 987 \
  --in_dim 1 \
  --out_dim 4 \
  --h 101 \
  --w 31 \
  --h-down 1 \
  --w-down 1 \
  --T-in 20 \
  --batch-size 5 \
  --learning-rate 0.0005 \
  --model FNO_3D \
  --d-model 32 \
  --num-basis 12 \
  --num-token 4 \
  --patch-size 7,2,2 \
  --padding 12,1,11 \
  --model-save-path ./checkpoints/plas \
  --model-save-name fno.pt