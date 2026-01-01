export CUDA_VISIBLE_DEVICES=3

python quant_demo.py --input_dir ./data/BSDS300/images/test \
 --pre_weight ./weight/bsds.pt \
 --out_dir ./output_quant \
 --bs 8 \
 --input_quant \
 --weight_quant