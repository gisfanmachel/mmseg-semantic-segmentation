# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmseg.apis import MMSegInferencer


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('model', help='Config file')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='', help='Path to save result file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='Whether to display the drawn image.')
    parser.add_argument(
        '--dataset-name',
        default='cityscapes',
        help='Color palette used for segmentation map')
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
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    inferencer = MMSegInferencer(
        args.model,
        args.checkpoint,
        dataset_name=args.dataset_name,
        device=args.device)

    # test a single imagel
    inferencer(
        args.img,
        show=args.show,
        out_dir=args.out_dir,
        opacity=args.opacity,
        with_labels=args.with_labels)

    # # 输入一个图片 list
    # >> > images = [image1, image2, ...]  # image1 可以是文件路径或 np.ndarray
    # >> > inferencer(images, show=True, wait_time=0.5)  # wait_time 是延迟时间，0 表示无限
    #
    # # 或输入图像目录
    # >> > images = $IMAGESDIR
    # >> > inferencer(images, show=True, wait_time=0.5)
    #
    # # 保存可视化渲染彩色分割图和预测结果
    # # out_dir 是保存输出结果的目录，img_out_dir 和 pred_out_dir 为 out_dir 的子目录
    # # 以保存可视化渲染彩色分割图和预测结果
    # >> > inferencer(images, out_dir='outputs', img_out_dir='vis', pred_out_dir='pred')

    # result = inferencer('demo/demo.png')
    # # 结果是一个包含 'visualization' 和 'predictions' 两个 key 的 `dict`
    # # 'visualization' 包含彩色分割图
    # print(result['visualization'].shape)
    # # (512, 683, 3)
    #
    # # 'predictions' 包含带有标签索引的分割掩膜
    # print(result['predictions'].shape)
    # # (512, 683)
    #
    # result = inferencer('demo/demo.png', return_datasamples=True)
    # print(type(result))
    # # <class 'mmseg.structures.seg_data_sample.SegDataSample'>
    #
    # # 输入一个图片 list
    # results = inferencer(images)
    # # 输出为列表
    # print(type(results['visualization']), results['visualization'][0].shape)
    # # <class 'list'> (512, 683, 3)
    # print(type(results['predictions']), results['predictions'][0].shape)
    # # <class 'list'> (512, 683)
    #
    # results = inferencer(images, return_datasamples=True)
    # # <class 'list'>
    # print(type(results[0]))
    # # <class 'mmseg.structures.seg_data_sample.SegDataSample'>


if __name__ == '__main__':
    # main()
    img = 'demo.png'
    config_file = '../configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
    checkpoint_file = '../pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
    inferencer = MMSegInferencer(
        config_file,
        checkpoint_file,
        device="cuda:0")

    # test a single imagel
    result=inferencer(
        img)
    print(result)