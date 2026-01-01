export CUDA_VISIBLE_DEVICES=2

python canny_batch.py \
 --input_dir ./data/BSDS300/images/test \
 --output_dir ./output_canny \
 --t1 50 --t2 150 --recursive
