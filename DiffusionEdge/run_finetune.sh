export CUDA_VISIBLE_DEVICES=3,5

# 微调保存相应权重
python finetune_logvar.py \
  --cfg configs/default.yaml \
  --hed_root ./data/HED-BSDS \
  --epochs 5 \
  --batch_size 8 \
  --lr 1e-4 \
  --save_path ./weight/autoencoder_logvar_finetuned.pt

# 运行微调后demo
python demo.py --input_dir ./data/BSDS300/images/test \
 --pre_weight ./weight/bsds.pt \
 --out_dir ./output_3 \
 --bs 8
