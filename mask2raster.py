import glob
import os
import numpy as np
from PIL import Image
import cv2
import rasterio
from rasterio.plot import show
from osgeo import gdal

MASKS_DIR = os.path.join(os.getcwd(),"masks")
OUTPUT_DIR =os.path.join(os.getcwd(),"masks2")
mask_paths = glob.glob(os.path.join(MASKS_DIR, "*.tif"))
palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34],
           [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51]]

for i, mask_path in enumerate(mask_paths):
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]

    # seg_map = np.loadtxt(osp.join(data_root, ann_dir, file)).astype(np.uint8)
    seg_map=cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), -1)
    seg_img = Image.fromarray(seg_map).convert('P')
    seg_img.putpalette(np.array(palette, dtype=np.uint8))
    out_mask_path = os.path.join(OUTPUT_DIR, "{}.png".format(mask_filename))
    seg_img.save(out_mask_path)

    # # 打开单波段TIFF文件，报错utf-8
    # with rasterio.open(mask_path) as src:
    #     # 读取波段数据
    #     band = src.read()

    #     # 将像素值为1的像素修改为129, 127, 38
    #     # 创建一个与原波段相同大小的数组，用于存放修改后的三波段数据
    #     new_bands = np.zeros((3, band.shape[0], band.shape[1]), dtype=band.dtype)

    #     # 遍历单波段数据，根据条件修改像素值
    #     for i in range(band.shape[0]):
    #         for j in range(band.shape[1]):
    #             # 背景（0）,建筑(1)，林地(2)，水(3)，道路(4)
    #             if band[i, j] == 0:
    #                 new_bands[0, i, j] = 128  # 第一个波段的像素值设置为128
    #                 new_bands[1, i, j] = 128  # 第二个波段的像素值设置为128
    #                 new_bands[2, i, j] = 128  # 第三个波段的像素值设置为128
    #             elif band[i, j] == 1:
    #                 new_bands[0, i, j] = 129  # 第一个波段的像素值设置为129
    #                 new_bands[1, i, j] = 127  # 第二个波段的像素值设置为127
    #                 new_bands[2, i, j] = 38  # 第三个波段的像素值设置为38
    #             elif band[i, j] == 2:
    #                 new_bands[0, i, j] = 120
    #                 new_bands[1, i, j] = 69
    #                 new_bands[2, i, j] = 125
    #             elif band[i, j] == 3:
    #                 new_bands[0, i, j] = 53
    #                 new_bands[1, i, j] = 125
    #                 new_bands[2, i, j] = 34
    #             elif band[i, j] == 4:
    #                 new_bands[0, i, j] = 0
    #                 new_bands[1, i, j] = 11
    #                 new_bands[2, i, j] = 123
    #             else:
    #                 new_bands[:, i, j] = band[i, j]  # 其他像素值保持不变

    # # 更新元数据以反映新的三波段结构
    # meta = src.meta.copy()
    # meta.update(count=3)  # 更新波段数为3

    # out_mask_path = os.path.join(OUTPUT_DIR, "{}.tif".format(mask_filename))
    # # 写入新的三波段 TIFF 文件
    # with rasterio.open(out_mask_path, 'w', **meta) as dst:
    #     dst.write(new_bands, indexes=1)  # 写入数据，indexes=1 表示写入第一个波段
    #     dst.write(new_bands, indexes=2)  # 写入数据，indexes=2 表示写入第二个波段
    #     dst.write(new_bands, indexes=3)  # 写入数据，indexes=3 表示写入第三个波段



    # # 打开单波段TIFF文件
    # src_ds = gdal.Open(mask_path, gdal.GA_ReadOnly)
    # if src_ds is None:
    #     raise Exception(f"Unable to open {mask_path}")
    #
    # # 读取单波段数据
    # src_band = src_ds.GetRasterBand(1)
    # band_data = src_band.ReadAsArray()
    # out_mask_path = os.path.join(OUTPUT_DIR, "{}.tif".format(mask_filename))
    # # 创建输出数据集，设置为三波段
    # driver = gdal.GetDriverByName('GTiff')
    # out_ds = driver.Create(out_mask_path, src_ds.RasterXSize, src_ds.RasterYSize, 3, gdal.GDT_Byte)
    #
    # # 根据单波段的像素值来设置三波段的像素值
    # band_data_3d = np.stack((band_data,) * 3, axis=-1)  # 创建三维数组，用于存储结果
    # band_data_3d[band_data == 1] = [255, 34, 55]  # 将像素值为1的像素设置为(255, 34, 55)
    # band_data_3d[band_data == 2] = [134, 35, 23]  # 将像素值为2的像素设置为(134, 35, 23)
    #
    # # 写入数据到每个波段
    # for i in range(1, 4):
    #     out_band = out_ds.GetRasterBand(i)
    #     out_band.WriteArray(band_data_3d[..., i-1])  # 写入对应波段的数据
    #
    # # 设置地理空间信息（如果需要）
    # out_ds.SetGeoTransform(src_ds.GetGeoTransform())
    # out_ds.SetProjection(src_ds.GetProjection())
    #
    #
    #
    #
    # # # 根据单波段的像素值来设置三波段的像素值
    # # for i in range(1, 4):
    # #     out_band = out_ds.GetRasterBand(i)
    # #     if i == 1:
    # #         # 红色通道：像素值为1的变为255，像素值为2的变为134
    # #         out_band.WriteArray(np.where(band_data == 1, 255, np.where(band_data == 2, 134, 0)))
    # #     elif i == 2:
    # #         # 绿色通道：像素值为1的变为34，像素值为2的变为35
    # #         out_band.WriteArray(np.where(band_data == 1, 34, np.where(band_data == 2, 35, 0)))
    # #     elif i == 3:
    # #         # 蓝色通道：像素值为1的变为55，像素值为2的变为23
    # #         out_band.WriteArray(np.where(band_data == 1, 55, np.where(band_data == 2, 23, 0)))
    #
    # # 清理
    # src_ds = None
    # out_ds = None


