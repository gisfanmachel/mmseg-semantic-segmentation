# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import cv2
import numpy as np
from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--with-labels',
        action='store_true',
        default=False,
        help='Whether to display the class labels.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test a single image
    result = inference_model(model, args.img)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        title=args.title,
        opacity=args.opacity,
        with_labels=args.with_labels,
        draw_gt=False,
        show=False if args.out_file is not None else True,
        out_file=args.out_file)
    #
    # # 语义分割预测
    # pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
    #
    # palette_dict={"1":(0, 0, 0), "2":(0, 255, 0), "3":(255, 0, 0), "4":(0, 0, 255), "5":(255, 255, 0), "6":(255, 0, 255), "7":(0, 255, 255), "8":(255, 255, 255), "9":(128, 0, 0), "10":(0, 128, 0), "11":(128, 128,225)}
    # # 将预测的整数ID，映射为对应类别的颜色
    # pred_mask_bgr = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
    # for idx in palette_dict.keys():
    #     pred_mask_bgr[np.where(pred_mask == idx)] = palette_dict[idx]
    # pred_mask_bgr = pred_mask_bgr.astype('uint8')
    #
    # # 将语义分割预测图和原图叠加显示
    # pred_viz = cv2.addWeighted(args.img, args.opacity, pred_mask_bgr, 1 - args.opacity, 0)




if __name__ == '__main__':
    main()
