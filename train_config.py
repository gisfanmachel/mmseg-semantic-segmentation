"""
#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
@Project :mmsegmentation_main
@File    :train_cmd.py
@IDE     :PyCharm
@Author  :chenxw
@Date    :2024/6/3 11:29
@Descr:
"""

from mmengine import Config

config_file = "configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py"
config_file = "demo/pspnet_dump.py"
config_file = "work_dirs/scene_mmseg_v1-semseg-pspnet_mask/pspnet_dump.py"
config_file = "data/mmseg-semseg/scene/scene_mmseg_v1-semseg-pspnet_mask.py"
cfg = Config.fromfile(config_file)
print(f'Config:\n{cfg.pretty_text}')
cfg.dump('pspnet_dump2.py')
