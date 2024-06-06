#!/usr/bin/env python3

import glob
import os

import cv2

current_number = 1

def get_next_seven_digit_number():
    global current_number
    next_number = current_number
    current_number += 1
    return f"{next_number:07}"





# ...每次调用都会增加1，并格式化为7位数字字符串
def write_list_to_file(file_path, list_data):
    with open(file_path, "w") as file:
        for temp in list_data:
            file.write(temp + "\n")


IMGS_DIR = "./images"
MASKS_DIR = "./masks2"
OUTPUT_NAME = "output5"
OUTPUT_IAMGES_DIR = "./{}/images".format(OUTPUT_NAME)
OUTPUT_LABELS_DIR = "./{}/labels".format(OUTPUT_NAME)
OUTPUT_SPLITS_DIR = "./{}/splits".format(OUTPUT_NAME)

TARGET_SIZE_X = 2048
TARGET_SIZE_Y = 1024

img_paths = glob.glob(os.path.join(IMGS_DIR, "*.tif"))
mask_paths = glob.glob(os.path.join(MASKS_DIR, "*.png"))

img_paths.sort()
mask_paths.sort()

os.makedirs(OUTPUT_IAMGES_DIR)
os.makedirs(OUTPUT_LABELS_DIR)
os.makedirs(OUTPUT_SPLITS_DIR)
# 按照指定大小将大图裁切成小图
for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
    img_filename = os.path.splitext(os.path.basename(img_path))[0]
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    assert img_filename == mask_filename and img.shape[:2] == mask.shape[:2]

    k = 0
    for y in range(0, img.shape[0], TARGET_SIZE_Y):
        for x in range(0, img.shape[1], TARGET_SIZE_X):
            img_tile = img[y:y + TARGET_SIZE_Y, x:x + TARGET_SIZE_X]
            mask_tile = mask[y:y + TARGET_SIZE_Y, x:x + TARGET_SIZE_X]

            if img_tile.shape[0] == TARGET_SIZE_Y and img_tile.shape[1] == TARGET_SIZE_X:
                filename=get_next_seven_digit_number()
                out_img_path = os.path.join(OUTPUT_IAMGES_DIR, "{}.jpg".format(filename))
                cv2.imwrite(out_img_path, img_tile)

                out_mask_path = os.path.join(OUTPUT_LABELS_DIR, "{}.png".format(filename))
                cv2.imwrite(out_mask_path, mask_tile)

            k += 1

    print("Processed {} {}/{}".format(img_filename, i + 1, len(img_paths)))

# 检查分割的图片大小是否正确
for filename in os.listdir(OUTPUT_IAMGES_DIR):
    img_path = os.path.join(OUTPUT_IAMGES_DIR, filename)
    img = cv2.imread(img_path)
    assert img.shape[:2] == (TARGET_SIZE_Y, TARGET_SIZE_X)
print("All images are correct size.")

# 获取文件夹内所有文件名
filenames = os.listdir(OUTPUT_IAMGES_DIR)
# 计算拆分比例
split_point = int(len(filenames) * 0.8)
# 按比例拆分文件名列表
train_filenames = filenames[:split_point]
val_filenames = filenames[split_point:]
write_list_to_file(os.path.join("./{}/splits/train.txt".format(OUTPUT_NAME)),
                   [train_filename.split(".")[0] for train_filename in train_filenames])
write_list_to_file(os.path.join("./{}/splits/val.txt".format(OUTPUT_NAME)),
                   [val_filename.split(".")[0] for val_filename in val_filenames])
