export CUDA_VISIBLE_DEVICES=0

python exp_airfoils.py \
  --data-path /home/wuhaixu/airfoil \
  --ntrain 1000 \
  --ntest 200 \
  --ntotal 1200 \
  --in_dim 2 \
  --out_dim 1 \
  --h 221 \
  --w 51 \
  --h-down 1 \
  --w-down 1 \
  --batch-size 20 \
  --learning-rate 0.001 \
  --model FNO_2D \
  --d-model 32 \
  --num-basis 12 \
  --num-token 4 \
  --patch-size 14,4 \
  --padding 13,3 \
  --model-save-path ./checkpoints/airfoil \
  --model-save-name fno.pt