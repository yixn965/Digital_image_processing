# finetune_logvar.py
# 用 HED-BSDS 边缘 GT 微调 AutoencoderKL 的 logvar 输出头

import os
import yaml
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from tqdm.auto import tqdm
from fvcore.common.config import CfgNode

from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL 

import os, sys, io
import time
from datetime import datetime

class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass
        return len(s)
    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def setup_stdout_stderr_tee(save_dir, filename=f"log_{current_datetime}.txt"):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, filename)
    f = open(log_path, "w", buffering=1, encoding="utf-8")  # line-buffered
    sys.stdout = Tee(sys.stdout, f)
    sys.stderr = Tee(sys.stderr, f)
    print(f"[logger] capturing stdout/stderr to {log_path}")
    return f


# ----------------- 和 train_cond_ldm / demo 一样的配置读取 -----------------

def load_conf(config_file, conf={}):
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf


def parse_args():
    parser = argparse.ArgumentParser(description="finetune logvar head of VAE")
    parser.add_argument(
        "--cfg", type=str, required=True,
        help="experiment configure yaml (同 train_cond_ldm / demo)"
    )
    parser.add_argument(
        "--hed_root", type=str, required=True,
        help="HED-BSDS 根目录，比如 data/HED-BSDS"
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="finetune 轮数，先用小一点试通流程"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="训练 batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="只训练 logvar 头的学习率"
    )
    parser.add_argument(
        "--save_path", type=str, default="logvar_ckpts/autoencoder_logvar_finetuned.pth",
        help="保存微调后 AutoencoderKL state_dict 的路径"
    )
    args = parser.parse_args()
    args.cfg = load_conf(args.cfg)
    return args

# ----------------- HED-BSDS 边缘 GT Dataset -----------------

class HEDBSDSEdgeDataset(Dataset):
    """
    只用 HED-BSDS 的 GT 边缘图训练 VAE：
    - 读取 hed_root/train_pair.lst
    - 每行: <img_rel> <gt_rel>
    - 我们只用第二列 gt_rel
    """
    def __init__(self, root, image_size):
        self.root = root
        self.image_size = image_size

        pair_list = os.path.join(root, "train_pair.lst")
        assert os.path.isfile(pair_list), f"{pair_list} 不存在，请确认 HED-BSDS 路径是否正确"

        self.gt_paths = []
        with open(pair_list, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_rel, gt_rel = line.split()
                self.gt_paths.append(gt_rel)

        # 统一 resize 到和模型一致的 image_size
        # image_size 一般是 [H, W]，从 cfg.model.image_size 里拿
        h, w = image_size
        self.transform = T.Compose([
            T.Resize((h, w), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),          # [0,1], 形状 [1, H, W]
        ])

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, idx):
        gt_rel = self.gt_paths[idx]
        gt_path = os.path.join(self.root, gt_rel)
        edge = Image.open(gt_path).convert("L")  # 灰度图
        edge = self.transform(edge)             # [1, H, W]

        # 这里可以确保是 0/1 边缘（如果 HED-BSDS 是 0/255 的话）
        # edge = (edge > 0.5).float()
        return edge


# ----------------- 主训练流程：只训练 decoder.conv_logvar -----------------

def main():
    args = parse_args()
    cfg = CfgNode(args.cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import torchvision.utils as tv

    vis_dir = "./output_finetune_vis"
    os.makedirs(vis_dir, exist_ok=True)

    log_file = setup_stdout_stderr_tee(vis_dir)

    # 1. 构建 AutoencoderKL，并加载原来的 first_stage 权重
    model_cfg = cfg.model
    first_stage_cfg = model_cfg.first_stage
    first_stage_model = AutoencoderKL(
        ddconfig=first_stage_cfg.ddconfig,
        lossconfig=first_stage_cfg.lossconfig,
        embed_dim=first_stage_cfg.embed_dim,
        ckpt_path=first_stage_cfg.ckpt_path,   # 原始 VAE ckpt
    ).to(device)

    print("==> AutoencoderKL loaded from:", first_stage_cfg.ckpt_path)

    # 2. 冻结所有参数，只训练 decoder.conv_logvar
    for name, p in first_stage_model.named_parameters():
        p.requires_grad = False

    trainable_params = []
    for name, p in first_stage_model.named_parameters():
        if "decoder.conv_logvar" in name:
            p.requires_grad = True
            trainable_params.append(p)
            print("[trainable]", name)

    assert len(trainable_params) > 0, "没有找到 decoder.conv_logvar 参数，请确认已经在 Decoder 里加了这个分支"

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    # 3. HED-BSDS 边缘 GT DataLoader
    image_size = model_cfg.image_size  # 比如 [320, 320]
    train_set = HEDBSDSEdgeDataset(root=args.hed_root, image_size=image_size)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # 4. 训练循环：edge_gt -> encode -> z -> decode -> (mu, logvar)
    first_stage_model.train()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=120)
        for edge_gt in pbar:
            edge_gt = edge_gt.to(device)  # [B,1,H,W]

            # encode：只用边缘图本身作为 VAE 输入
            posterior = first_stage_model.encode(edge_gt)
            # 用 mode 而不是 sample，减少噪声
            if hasattr(posterior, "mode"):
                z = posterior.mode()
            else:
                z = posterior.sample()

            # decode：你已经把 decode 改成返回 (dec, x_logvar)
            x_mu, x_logvar = first_stage_model.decode(z)

            # 防止 logvar 爆炸
            x_logvar = x_logvar.clamp(-3.0, 3.0)

            # BCE(logits, target)：这里假设 x_mu 是 logits，edge_gt∈[0,1]
            bce = F.binary_cross_entropy_with_logits(
                x_mu, edge_gt, reduction="none"
            )

            # heteroscedastic loss: exp(-logvar) * BCE + logvar
            loss_het = (torch.exp(-x_logvar) * bce + x_logvar).mean()

            optimizer.zero_grad()
            loss_het.backward()
            optimizer.step()

            # ====== 可视化部分 ======
            if global_step % 500 == 0:   # 每 500 step 存一次图
                with torch.no_grad():
                    # 取 batch 里的第一张
                    gt_edge = edge_gt[0:1]              # [1,1,H,W]
                    pred_prob = torch.sigmoid(x_mu[0:1])  # [1,1,H,W]

                    sigma = torch.exp(0.5 * x_logvar[0:1])  # [1,1,H,W]
                    # 归一化到 [0,1] 方便看
                    sigma_norm = (sigma - sigma.min()) / (sigma.max() - sigma.min() + 1e-8)

                    # 分别保存三张
                    tv.save_image(gt_edge,    f"{vis_dir}/step{global_step:06d}_gt.png")
                    tv.save_image(pred_prob,  f"{vis_dir}/step{global_step:06d}_pred.png")
                    tv.save_image(sigma_norm, f"{vis_dir}/step{global_step:06d}_uncert.png")
            # ====== 可视化结束 ======

            global_step += 1
            pbar.set_postfix({"loss_het": f"{loss_het.item():.4f}"})

        print(f"[Epoch {epoch}] loss_het = {loss_het.item():.4f}")

    # 5. 保存微调后的 AutoencoderKL 权重
    torch.save(first_stage_model.state_dict(), args.save_path)
    print("==> finetuned AutoencoderKL (with logvar head) saved to:", args.save_path)


if __name__ == "__main__":
    main()
