export CUDA_VISIBLE_DEVICES=6

python exp_darcy.py \
  --data-path /home/wuhaixu/ \
  --ntrain 1000 \
  --ntest 200 \
  --ntotal 1200 \
  --in_dim 1 \
  --out_dim 1 \
  --h 421 \
  --w 421 \
  --h-down 5 \
  --w-down 5 \
  --batch-size 20 \
  --learning-rate 0.001 \
  --model LSM_2D \
  --d-model 64 \
  --num-basis 12 \
  --num-token 4 \
  --patch-size 3,3 \
  --padding 11,11 \
  --model-save-path ./checkpoints/darcy \
  --model-save-name lsm.pt