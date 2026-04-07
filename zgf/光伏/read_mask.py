import cv2
import numpy as np

mask_path = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\光伏\save_ceshi\mask\00000003_pred.png'
mask_path=r"d:\data\PV08_350252_1189866.png"

# 先以二进制方式读取文件，再用 cv2.imdecode 解码
with open(mask_path, 'rb') as f:
    img_data = f.read()
mask = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_UNCHANGED)

print("图像形状:", mask.shape)
print("数据类型:", mask.dtype)
print("唯一像素值:", np.unique(mask))
print("像素值统计:", np.bincount(mask.flatten()))