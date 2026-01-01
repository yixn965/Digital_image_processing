export CUDA_VISIBLE_DEVICES=5

# python sample_cond_ldm.py \
#  --cfg ./configs/BSDS_sample.yaml

# 自然
python demo.py --input_dir ./data/BSDS300/images/test \
 --pre_weight ./weight/bsds.pt \
 --out_dir ./output_2 \
 --bs 8

# # 室内
# python demo.py --input_dir ./data/BSDS300/images/test \
#  --pre_weight ./weight/nyud.pt \
#  --out_dir ./output_2 \
#  --bs 8

# # 户外
# python demo.py --input_dir ./data/BSDS300/images/test \
#  --pre_weight ./weight/biped.pt \
#  --out_dir ./output_3 \
#  --bs 8