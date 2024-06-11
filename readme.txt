在mmseg/datasets 增加__init__.py,ZihaoDataset.py(定义类别和配色方案)

pip install -v -e .   这种安装，会引用最新的代码

训练：
 python tools/train.py data/mmseg-semseg/scene/scene_mmseg_v1-semseg-pspnet_mask.py --work-dir work_dirs/scene_mmseg_v1-semseg-pspnet_mask2

 python tools/train.py demo/pspnet_dump.py --work-dir work_dirs/scene_mmseg_v1-semseg-pspnet_mask



python tools/train.py data/mmseg-semseg/boat/boat_mmseg_v1-semseg-pspnet_mask.py --work-dir work_dirs/boat_mmseg_v1-semseg-pspnet_mask


python tools/train.py Zihao-Configs/ZihaoDataset_PSPNet_20230818.py --work-dir work_dirs/ZihaoDataset_PSPNet_20230818


python tools/train.py Zihao-Configs/LandcoverDataset_PSPNet_20240611.py --work-dir work_dirs/LandcoverDataset_PSPNet_20240611

推理