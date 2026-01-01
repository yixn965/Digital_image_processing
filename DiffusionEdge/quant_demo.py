import numpy as np
import yaml
import argparse
import math
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from denoising_diffusion_pytorch.utils import *
import torchvision as tv
from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
# from denoising_diffusion_pytorch.transmodel import TransModel
from denoising_diffusion_pytorch.uncond_unet import Unet
from denoising_diffusion_pytorch.data import *
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from fvcore.common.config import CfgNode
from scipy import integrate
import matplotlib.cm as cm
import numpy as np
from quant import *
from quant.build_model import build_model
from utils.logger_utils import logger1, logger2, logger3, logger4, logger5, outpath
from quant.quant_model_smooth import quant_model_smooth, set_quant_state

def load_conf(config_file, conf={}):
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf
def parse_args():
    parser = argparse.ArgumentParser(description="demo configure")
    parser.add_argument("--cfg", help="experiment configure file name", type=str, default="./configs/default.yaml")
    parser.add_argument("--input_dir", help='input directory', type=str, required=True)
    parser.add_argument("--pre_weight", help='path of pretrained weight', type=str, required=True)
    parser.add_argument("--sampling_timesteps", help='sampling timesteps', type=int, default=1)
    parser.add_argument("--out_dir", help='output directory', type=str, required=True)
    parser.add_argument("--bs", help='batch_size for inference', type=int, default=8)
    parser.add_argument('--input_quant', action="store_true",help="Enable or disable input quantization")
    parser.add_argument('--weight_quant', action="store_true", help="Enable or disable weight quantization")
    parser.add_argument("--include_layers", type=str, default="", help="Comma-separated list of layers to include.")
    parser.add_argument("--exclude_layers", type=str, default="", help="Comma-separated list of layers to exclude.")
    args = parser.parse_args()
    args.cfg = load_conf(args.cfg)
    return args


def main(args):
    cfg = CfgNode(args.cfg)
    torch.manual_seed(42)
    np.random.seed(42)
    # random.seed(seed)
    # logger = create_logger(root_dir=cfg['out_path'])
    # writer = SummaryWriter(cfg['out_path'])
    model_cfg = cfg.model
    first_stage_cfg = model_cfg.first_stage

    # 1. 构建好原始的 VAE
    first_stage_model = AutoencoderKL(
        ddconfig=first_stage_cfg.ddconfig,
        lossconfig=first_stage_cfg.lossconfig,
        embed_dim=first_stage_cfg.embed_dim,
        ckpt_path=first_stage_cfg.ckpt_path,  # 原始 VAE ckpt
    )

    # 2. 加载你微调出来的 VAE state_dict
    ft_sd = torch.load("./weight/autoencoder_logvar_finetuned.pth", map_location="cpu")

    # 3. 只把特定几层拷过去
    with torch.no_grad():
        base_sd = first_stage_model.state_dict()
        for k, v in ft_sd.items():
            # 只更新 decoder.conv_logvar 相关层
            if k.startswith("decoder.conv_logvar"):
                print("update key:", k)
                base_sd[k] = v
        first_stage_model.load_state_dict(base_sd, strict=False)

    if model_cfg.model_name == 'cond_unet':
        from denoising_diffusion_pytorch.mask_cond_unet import Unet
        unet_cfg = model_cfg.unet
        unet = Unet(dim=unet_cfg.dim,
                    channels=unet_cfg.channels,
                    dim_mults=unet_cfg.dim_mults,
                    learned_variance=unet_cfg.get('learned_variance', False),
                    out_mul=unet_cfg.out_mul,
                    cond_in_dim=unet_cfg.cond_in_dim,
                    cond_dim=unet_cfg.cond_dim,
                    cond_dim_mults=unet_cfg.cond_dim_mults,
                    window_sizes1=unet_cfg.window_sizes1,
                    window_sizes2=unet_cfg.window_sizes2,
                    fourier_scale=unet_cfg.fourier_scale,
                    cfg=unet_cfg,
                    )
    else:
        raise NotImplementedError
    if model_cfg.model_type == 'const_sde':
        from denoising_diffusion_pytorch.ddm_const_sde import LatentDiffusion
    else:
        raise NotImplementedError(f'{model_cfg.model_type} is not surportted !')
    ldm = LatentDiffusion(
        model=unet,
        auto_encoder=first_stage_model,
        train_sample=model_cfg.train_sample,
        image_size=model_cfg.image_size,
        timesteps=model_cfg.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        loss_type=model_cfg.loss_type,
        objective=model_cfg.objective,
        scale_factor=model_cfg.scale_factor,
        scale_by_std=model_cfg.scale_by_std,
        scale_by_softsign=model_cfg.scale_by_softsign,
        default_scale=model_cfg.get('default_scale', False),
        input_keys=model_cfg.input_keys,
        ckpt_path=model_cfg.ckpt_path,
        ignore_keys=model_cfg.ignore_keys,
        only_model=model_cfg.only_model,
        start_dist=model_cfg.start_dist,
        perceptual_weight=model_cfg.perceptual_weight,
        use_l1=model_cfg.get('use_l1', True),
        cfg=model_cfg,
    )
    # ldm.init_from_ckpt(cfg.sampler.ckpt_path, use_ema=cfg.sampler.get('use_ema', True))
    ckpt = torch.load(args.pre_weight, map_location="cpu")  # 或 cfg.sampler.ckpt_path

    if cfg.sampler.use_ema:
        sd = ckpt["ema"]
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("ema_model."):
                new_k = k[len("ema_model."):]
                new_sd[new_k] = v
        sd = new_sd
        msg = ldm.load_state_dict(sd, strict=False)
        print("==> EMA load_state_dict missing:", msg.missing_keys)
        print("==> EMA load_state_dict unexpected:", msg.unexpected_keys)
    else:
        msg = ldm.load_state_dict(ckpt["model"], strict=False)
        print("==> load_state_dict missing:", msg.missing_keys)
        print("==> load_state_dict unexpected:", msg.unexpected_keys)

    if "scale_factor" in ckpt["model"]:
        ldm.scale_factor = ckpt["model"]["scale_factor"]

    ldm = wo(ldm, args)
    wq_params = {'n_bits': 8, 'channel_wise': False}
    aq_params = {'n_bits': 8, 'channel_wise': False}
    q_model = quant_model_smooth(ldm, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_model.eval()
    logger4.info("Quantized model created and moved to device.")
    logger4.info("model structure:\n" + str(q_model))  

    input_quant = args.input_quant
    weight_quant = args.weight_quant

    include_layers = args.include_layers.split(",") if args.include_layers else None
    exclude_layers = args.exclude_layers.split(",") if args.exclude_layers else None
    set_quant_state(
        q_model,
        input_quant=input_quant,
        weight_quant=weight_quant,
        include_layers=include_layers,
        exclude_layers=exclude_layers,
    )

    logger4.info(f"Setting quantization state: input_quant={input_quant}, weight_quant={weight_quant}")
    # logger4.info(f"Include layers: {include_layers}")
    # logger4.info(f"Exclude layers: {exclude_layers}")
    logger4.info("model structure:\n" + str(q_model))

    # linear_layer_names = []

    # for name, module in ldm.named_modules():
    #     if isinstance(module, (nn.Linear,nn.Conv2d, nn.Conv1d)):
    #         linear_layer_names.append(name)

    # # 指定保存路径
    # save_path = "./linear_layers.txt"

    # # 保存到文件
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # with open(save_path, "w") as f:
    #     for name in linear_layer_names:
    #         f.write(name + "\n")

    # print(f"✅ 已保存 {len(linear_layer_names)} 个 Linear 层名称到 {save_path}")


    data_cfg = cfg.data
    if data_cfg['name'] == 'edge':
        dataset = EdgeDatasetTest(
            data_root=args.input_dir,
            image_size=model_cfg.image_size,
        )
        # dataset = torch.utils.data.ConcatDataset([dataset] * 5)
    else:
        raise NotImplementedError
    dl = DataLoader(dataset, batch_size=cfg.sampler.batch_size, shuffle=False, pin_memory=True,
                    num_workers=data_cfg.get('num_workers', 2))
    # for slide sampling, we only support batch size = 1
    sampler_cfg = cfg.sampler
    sampler_cfg.save_folder = args.out_dir
    sampler_cfg.ckpt_path = args.pre_weight
    sampler_cfg.batch_size = args.bs
    sampler = Sampler(
        ldm, dl, batch_size=sampler_cfg.batch_size,
        sample_num=sampler_cfg.sample_num,
        results_folder=sampler_cfg.save_folder, cfg=cfg,
    )
    sampler.calibrate(calib_n=16)
    sampler.sample()


class Sampler(object):
    def __init__(
            self,
            model,
            data_loader,
            sample_num=1000,
            batch_size=16,
            results_folder='./results',
            rk45=False,
            cfg={},
    ):
        super().__init__()
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision='no',
            kwargs_handlers=[ddp_handler],
        )
        self.model = model
        self.sample_num = sample_num
        self.rk45 = rk45

        self.batch_size = batch_size
        self.batch_num = math.ceil(sample_num // batch_size)

        self.image_size = model.image_size
        self.cfg = cfg

        dl = self.accelerator.prepare(data_loader)
        self.dl = dl
        self.results_folder = Path(results_folder)
        if self.accelerator.is_main_process:
            self.results_folder.mkdir(exist_ok=True, parents=True)

        self.model = self.accelerator.prepare(self.model)
        # data = torch.load(cfg.sampler.ckpt_path, map_location=lambda storage, loc: storage)

    def calibrate(self, calib_n=16):
        """
        用 edge 数据集做校准，跑前 calib_n 张图（或若干 batch），
        让量化的 forward 收集统计量。
        """
        accelerator = self.accelerator
        device = accelerator.device

        self.model.eval()
        num_seen = 0          # 已经用来校准的样本数
        total_needed = calib_n

        with torch.no_grad():
            for idx, batch in enumerate(self.dl):
                # 超过 calib_n 就停
                bs = batch['cond'].shape[0]
                if num_seen >= total_needed:
                    break

                # 把 batch 搬到 device
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)

                cond = batch['cond']
                raw_w = batch["raw_size"][0].item()
                raw_h = batch["raw_size"][1].item()
                mask = batch['ori_mask'] if 'ori_mask' in batch else None

                if self.cfg.sampler.sample_type == 'whole':
                    _ = self.whole_sample(
                        cond,
                        raw_size=(raw_h, raw_w),
                        mask=mask,
                        calib5=True
                    )
                elif self.cfg.sampler.sample_type == 'slide':
                    _pred, _logvar = self.slide_sample(
                        cond,
                        crop_size=self.cfg.sampler.get('crop_size', [320, 320]),
                        stride=self.cfg.sampler.stride,
                        mask=mask,
                        bs=self.batch_size,
                        calib5=True
                    )
                else:
                    raise NotImplementedError

                num_seen += bs

        accelerator.print(f"[Calibration] Done with {num_seen} samples.")

    def sample(self):
        accelerator = self.accelerator
        device = accelerator.device
        batch_num = self.batch_num
        with torch.no_grad():
            self.model.eval()
            psnr = 0.
            num = 0
            for idx, batch in tqdm(enumerate(self.dl)):
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key].to(device)
                # image = batch["image"]
                cond = batch['cond']
                raw_w = batch["raw_size"][0].item()      # default batch size = 1
                raw_h = batch["raw_size"][1].item()
                img_name = batch["img_name"][0]

                mask = batch['ori_mask'] if 'ori_mask' in batch else None
                bs = cond.shape[0]
                if self.cfg.sampler.sample_type == 'whole':
                    batch_pred = self.whole_sample(cond, raw_size=(raw_h, raw_w), mask=mask)
                elif self.cfg.sampler.sample_type == 'slide':
                    batch_pred, batch_logvar = self.slide_sample(cond, crop_size=self.cfg.sampler.get('crop_size', [320, 320]),
                                                   stride=self.cfg.sampler.stride, mask=mask, bs=self.batch_size)
                else:
                    raise NotImplementedError
                # for j, (img, c) in enumerate(zip(batch_pred, cond)):
                #     file_name = self.results_folder / img_name
                #     tv.utils.save_image(img, str(file_name)[:-4] + ".png")
                for j, (logit, logvar, c) in enumerate(zip(batch_pred, batch_logvar, cond)):
                    # logit: (1, H, W) 边缘 logits
                    # logvar: (1, H, W) log σ^2

                    # 1) 边缘概率图
                    edge_prob = logit

                    # 2) 不确定度图：先从 logvar -> σ，再归一化到 [0,1]
                    sigma = torch.exp(0.5 * logvar)          # 标准差 σ
                    sigma = sigma.detach()
                    confidence = torch.exp(-sigma).clamp(0.0, 1.0)  # 置信度

                    sigma_th = 1.0
                    p_th = 0.3  # or 0.3/0.7 你可以自己调
                    edge_mask = (edge_prob > p_th).float()          # 细边缘 mask

                    conf_on_edge = confidence * edge_mask
                    uncertainty = (1.0 - confidence) * edge_mask  # [B,1,H,W], 越大越不确
                    file_name = self.results_folder / img_name 

                    unc_map = uncertainty[0].detach().cpu().numpy()   # (H, W), 0~1 之间

                    # 如果想增强对比度，可以只对边缘处做归一化：
                    # mask 里为 1 的才参与 min/max
                    edge_mask_np = edge_mask[0].detach().cpu().numpy().astype(bool)
                    if edge_mask_np.any():
                        v_min = unc_map[edge_mask_np].min()
                        v_max = unc_map[edge_mask_np].max()
                        # 防止除 0
                        if v_max > v_min:
                            unc_norm = np.zeros_like(unc_map)
                            unc_norm[edge_mask_np] = (unc_map[edge_mask_np] - v_min) / (v_max - v_min + 1e-8)
                        else:
                            unc_norm = unc_map  # 全部一样就随便
                    else:
                        unc_norm = unc_map

                    # 用 colormap 映射到 RGB
                    cmap = cm.get_cmap('coolwarm')                   # 低值蓝，高值红
                    color = cmap(unc_norm)[..., :3]                  # (H, W, 3)

                    # 转成 tensor：[3, H, W]
                    color_tensor = torch.from_numpy(color).permute(2, 0, 1).float()

                    # 关键：在 RGB 上再乘一次 edge_mask，让背景变成真正的黑色
                    edge_mask_3 = edge_mask[0].detach().cpu().repeat(3, 1, 1)        # (3, H, W)
                    color_tensor = color_tensor * edge_mask_3        # 非边缘位置 → 0

                    tv.utils.save_image(color_tensor, str(file_name)[:-4] + "_uncert_heatmap.png")
                    # 保存边缘概率图
                    tv.utils.save_image(edge_prob, str(file_name)[:-4] + "_edge.png")
                    # 保存不确定度热力图（灰度图）
                    tv.utils.save_image(conf_on_edge, str(file_name)[:-4] + "_uncert.png")

        accelerator.print('sampling complete')

    # ----------------------------------waiting revision------------------------------------
    def slide_sample(self, inputs, crop_size, stride, mask=None, bs=8,calib5=False, **kwargs):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = 1
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        # 整图尺寸的累积张量（注意：用 inputs.new_zeros，自动对齐 device/dtype）
        preds        = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        logvar_preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat    = inputs.new_zeros((batch_size, 1,         h_img, w_img))

        crop_imgs = []
        x1s, x2s, y1s, y2s = [], [], [], []

        # 1) 先生成所有 patch 及对应坐标
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                crop_imgs.append(crop_img)
                x1s.append(x1)
                x2s.append(x2)
                y1s.append(y1)
                y2s.append(y2)

        crop_imgs = torch.cat(crop_imgs, dim=0)   # (num_windows * batch_size, C, h_crop, w_crop)

        crop_seg_logits_list = []
        crop_logvar_list     = []
        num_windows = crop_imgs.shape[0]
        length = math.ceil(num_windows / bs)

        # 2) 分批送进 model.sample，拿到每个 patch 的 seg + logvar
        for i in range(length):
            if i == length - 1:
                crop_imgs_temp = crop_imgs[bs * i:num_windows, ...]
            else:
                crop_imgs_temp = crop_imgs[bs * i:bs * (i + 1), ...]

            if isinstance(self.model, nn.parallel.DistributedDataParallel):
                crop_seg_logits, crop_logvar = self.model.module.sample(
                    batch_size=crop_imgs_temp.shape[0],
                    cond=crop_imgs_temp,
                    mask=mask,
                    calib5=calib5
                )
            elif isinstance(self.model, nn.Module):
                crop_seg_logits, crop_logvar = self.model.sample(
                    batch_size=crop_imgs_temp.shape[0],
                    cond=crop_imgs_temp,
                    mask=mask,
                    calib5=calib5
                )
            else:
                raise NotImplementedError

            crop_seg_logits_list.append(crop_seg_logits)  # (bs_i, 1, h_crop, w_crop)
            crop_logvar_list.append(crop_logvar)          # (bs_i, 1, h_crop, w_crop)

        crop_seg_logits = torch.cat(crop_seg_logits_list, dim=0)  # (N, 1, h_crop, w_crop)
        crop_logvar     = torch.cat(crop_logvar_list,     dim=0)  # (N, 1, h_crop, w_crop)

        # 3) 一次循环，同时把 seg 和 logvar 贴回整图并计数
        for seg_patch, logvar_patch, x1, x2, y1, y2 in zip(
            crop_seg_logits, crop_logvar, x1s, x2s, y1s, y2s
        ):
            # seg 部分：pad到整图大小再累加
            preds += F.pad(
                seg_patch,     # (1, 1, h_crop, w_crop)
                (int(x1), int(w_img - x2), int(y1), int(h_img - y2))
            )
            # logvar 部分：同样 pad + 累加
            logvar_preds += F.pad(
                logvar_patch,
                (int(x1), int(w_img - x2), int(y1), int(h_img - y2))
            )

            # 计数矩阵：对应区域 +1
            count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0

        # 4) 对重叠区域取平均，得到整图的 logits 和 logvar
        seg_logits  = preds / count_mat          # (batch_size, 1, h_img, w_img)
        full_logvar = logvar_preds / count_mat   # (batch_size, 1, h_img, w_img)

        return seg_logits, full_logvar

    def whole_sample(self, inputs, raw_size, mask=None):

        inputs = F.interpolate(inputs, size=(416, 416), mode='bilinear', align_corners=True)

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            seg_logits = self.model.module.sample(batch_size=inputs.shape[0], cond=inputs, mask=mask)
        elif isinstance(self.model, nn.Module):
            seg_logits = self.model.sample(batch_size=inputs.shape[0], cond=inputs, mask=mask)
        seg_logits = F.interpolate(seg_logits, size=raw_size, mode='bilinear', align_corners=True)
        return seg_logits



if __name__ == "__main__":
    args = parse_args()
    main(args)