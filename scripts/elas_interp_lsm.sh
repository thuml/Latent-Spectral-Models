export CUDA_VISIBLE_DEVICES=5

python exp_elas_interp.py \
  --data-path /home/wuhaixu/elasticity \
  --ntrain 1000 \
  --ntest 200 \
  --ntotal 2000 \
  --in_dim 1 \
  --out_dim 1 \
  --h 41 \
  --w 41 \
  --h-down 1 \
  --w-down 1 \
  --batch-size 20 \
  --learning-rate 0.001 \
  --model LSM_2D \
  --d-model 32 \
  --num-basis 12 \
  --num-token 4 \
  --patch-size 3,3 \
  --padding 7,7 \
  --model-save-path ./checkpoints/elas_interp \
  --model-save-name lsm.pt