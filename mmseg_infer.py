#!/usr/bin/env python3
# =============================================================
# MMSegmentation 通用推理脚本
# 支持对任意 mmsegmentation 训练模型的推理验证测试
# 使用方式: python mmseg_infer.py --config <config.py> --checkpoint <.pth> --img <image_or_dir> [--out <output_dir>]
# =============================================================

import os
import os
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import argparse
import glob
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmseg.apis import MMSegInferencer

# 默认调色板（RGB，0-255）
DEFAULT_PALETTE = {
    'cityscapes': [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                    [0, 80, 100], [0, 0, 230], [119, 11, 32]],
    'ade20k': [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 153],
               [230, 2, 153], [230, 230, 230], [4, 250, 7], [224, 5, 255],
               [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51],
               [255, 6, 82], [143, 255, 167], [204, 255, 35], [0, 102, 200],
               [8, 230, 220], [220, 220, 220], [140, 140, 136]],
    'pv': [[255, 255, 255], [0, 0, 255]],   # 背景白 + 光伏蓝
    'building': [[255, 255, 255], [0, 0, 255]],  # 背景白 + 建筑蓝
}

# ============================================================
# 固化参数（通用分割模型）
# ============================================================
HARDCODED_CLASSES = ['background', 'object']
HARDCODED_PALETTE = [[255, 255, 255], [0, 0, 255]]  # RGB: 背景白、目标蓝


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSegmentation 通用推理脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单张图片推理
  python mmseg_infer.py --config config.py --checkpoint best.pth --img photo.jpg

  # 文件夹批量推理
  python mmseg_infer.py --config config.py --checkpoint best.pth --img ./test_images/ --out results/

  # 指定输出目录和调色板
  python mmseg_infer.py --config config.py --checkpoint best.pth --img ./test/ --out ./output/ --palette pv

  # 指定推理尺寸和设备
  python mmseg_infer.py --config config.py --checkpoint best.pth --img ./test/ \\
         --resize 512 512 --device cuda:0 --batch-size 4

  # 自定义调色板
  python mmseg_infer.py --config config.py --checkpoint best.pth --img ./test/ \\
         --palette-list 255 255 255 0 0 255 128 128 128
"""
    )
    parser.add_argument('--config', '-c', required=True,
                        help='模型配置文件 (.py)')
    parser.add_argument('--checkpoint', '-w', required=True,
                        help='模型权重文件 (.pth)')
    parser.add_argument('--img', '-i', required=True,
                        help='输入图片路径或图片目录')
    parser.add_argument('--out', '-o', default='./mmseg_output',
                        help='输出目录 (默认: ./mmseg_output)')
    parser.add_argument('--palette', '-p', default=None,
                        help='调色板预设: cityscapes, ade20k, pv 等')
    parser.add_argument('--palette-list', nargs='+', type=int, default=None,
                        help='自定义调色板: RGB RGB RGB ... (3的倍数)')
    parser.add_argument('--resize', nargs=2, type=int, default=None,
                        help='推理图片尺寸: 宽 高')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                        help='批次大小 (默认: 1)')
    parser.add_argument('--device', '-d', default='cuda:0',
                        help='推理设备 (默认: cuda:0)')
    parser.add_argument('--save-vis', action='store_true',
                        help='保存可视化结果图')
    parser.add_argument('--save-mask', action='store_true',
                        help='保存分割掩码图')
    parser.add_argument('--opacity', type=float, default=0.6,
                        help='可视化叠加透明度 (0.0-1.0, 默认: 0.6)')
    parser.add_argument('--classes', nargs='+', default=None,
                        help='类别名称列表，如: background person car')
    parser.add_argument('--max-images', type=int, default=None,
                        help='最多推理图片数量 (默认: 无限制)')
    parser.add_argument('--show', action='store_true',
                        help='显示结果图 (需要GUI环境)')
    parser.add_argument('--save-merged', action='store_true',
                        help='保存原图+分割结果叠加图')
    return parser.parse_args()


def build_palette(palette_arg, palette_list, classes):
    """构建调色板"""
    if palette_list:
        # 用户通过 --palette-list 提供
        if len(palette_list) % 3 != 0:
            raise ValueError('--palette-list 必须为3的倍数（每像素RGB）')
        return [palette_list[i:i+3] for i in range(0, len(palette_list), 3)]

    if palette_arg and palette_arg in DEFAULT_PALETTE:
        return DEFAULT_PALETTE[palette_arg]

    # 尝试从config中获取调色板
    return None


def clear_output_dir(out_dir):
    """清空输出目录（如果存在）"""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        print(f'已清空输出目录: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)


def collect_images(img_path):
    """收集图片列表（自动去重）"""
    if os.path.isfile(img_path):
        return [img_path]

    # 是目录
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(img_path, pat)))
        # Windows 下 .tif/.tiff 实际是同一类型，跳过大写匹配避免重复
        if not pat.endswith(('.tif', '.tiff')):
            files.extend(glob.glob(os.path.join(img_path, pat.upper())))
    # 按绝对路径去重（Windows 下 .tif/.tiff 视为同一文件）
    seen = set()
    unique = []
    for f in files:
        ab = os.path.abspath(f)
        if ab not in seen:
            seen.add(ab)
            unique.append(f)
    return sorted(unique)


def main():
    args = parse_args()

    # ========== 0. 基础检查 ==========
    if not os.path.exists(args.config):
        raise FileNotFoundError(f'配置文件不存在: {args.config}')
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'权重文件不存在: {args.checkpoint}')

    device = args.device if torch.cuda.is_available() else 'cpu'
    if device != 'cpu':
        if not torch.cuda.is_available():
            print('警告: CUDA不可用，降级到CPU')
            device = 'cpu'

    print('=' * 60)
    print('MMSegmentation 通用推理脚本')
    print('=' * 60)
    print(f'  配置: {args.config}')
    print(f'  权重: {args.checkpoint}')
    print(f'  输入: {args.img}')
    print(f'  输出: {args.out}')
    print(f'  设备: {device}')
    print(f'  批次: {args.batch_size}')
    if args.resize:
        print(f'  尺寸: {args.resize[0]}x{args.resize[1]}')
    print('=' * 60)

    # ========== 1. 收集图片 ==========
    image_list = collect_images(args.img)
    if not image_list:
        raise FileNotFoundError(f'未找到任何图片: {args.img}')
    if args.max_images:
        image_list = image_list[:args.max_images]

    print(f'找到 {len(image_list)} 张图片')

    # ========== 2. 构建调色板（使用固化参数）============
    palette = HARDCODED_PALETTE
    classes = HARDCODED_CLASSES
    print(f'  固化类别: {classes}')
    print(f'  固化调色板: {palette}')

    # ========== 3. 加载配置并修改test_pipeline ==========
    cfg = Config.fromfile(args.config)

    resize_scale = tuple(args.resize) if args.resize else None

    # 在 LoadImageFromFile 之后插入 Resize（如果指定了resize）
    if resize_scale:
        load_img_idx = None
        for idx, transform in enumerate(cfg.test_pipeline):
            if transform.get('type') == 'LoadImageFromFile':
                load_img_idx = idx
                break
        if load_img_idx is not None:
            # 检查是否已有Resize，有则跳过
            has_resize = any(t.get('type') == 'Resize' for t in cfg.test_pipeline)
            if not has_resize:
                cfg.test_pipeline.insert(
                    load_img_idx + 1,
                    dict(type='Resize', scale=resize_scale, keep_ratio=False)
                )
                print(f'已插入Resize: scale={resize_scale}')

    # ========== 4. 初始化推理器 ==========
    infer_kwargs = dict(
        model=cfg,
        weights=args.checkpoint,
        device=device,
        classes=classes,
        palette=palette,
    )

    inferencer = MMSegInferencer(**infer_kwargs)
    print('推理器初始化完成')

    # ========== 5. 清空并创建输出目录 ==========
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(os.path.join(args.out, 'vis'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'mask'), exist_ok=True)

    # ========== 6. 分批次推理 ==========
    total = len(image_list)
    autocast_ctx = torch.cuda.amp.autocast() if 'cuda' in device else torch.nullcontext()

    for batch_start in range(0, total, args.batch_size):
        batch_imgs = image_list[batch_start:batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        batch_total = (total + args.batch_size - 1) // args.batch_size
        print(f'处理第 {batch_num}/{batch_total} 批 ({len(batch_imgs)} 张)...')

        with autocast_ctx:
            # 完全依赖 inferencer 内部写入：vis -> {out}/vis/，mask -> {out}/mask/
            inferencer(
                batch_imgs,
                batch_size=args.batch_size,
                out_dir=args.out,
                show=False,
                wait_time=0,
                img_out_dir='vis',
                pred_out_dir='mask',
                return_vis=False,
                return_datasamples=False,
                with_labels=False
            )

        # 清理GPU缓存
        if 'cuda' in device:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # ========== 7. 完成 ==========
    print()
    print('=' * 60)
    print('推理完成!')
    print(f'  图片总数: {total}')
    print(f'  输出目录: {args.out}')
    print(f'  可视化图: {os.path.join(args.out, "vis")}')
    print(f'  掩码图:   {os.path.join(args.out, "mask")}')
    print('=' * 60)


if __name__ == '__main__':
    main()
