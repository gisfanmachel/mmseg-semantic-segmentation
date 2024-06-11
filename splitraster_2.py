#!/usr/bin/env python3

import glob
import os
import random
import shutil

import cv2

current_number = 1


# 将原始tif和标注单波段tif分割小图样本
# train和val分开存储
# 分割后的图片为jpg,标注为png
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


def split_files_by_radio(IAMGE_DIR):
    train_radio = 0.8
    valid_radio = 0.2
    test_radio = 0.0
    img_list = os.listdir(IAMGE_DIR)
    train_img_list = []
    valid_img_list = []
    test_img_list = []
    train_num = int(len(img_list) * train_radio)
    valid_num = int(len(img_list) * valid_radio)
    test_num = int(len(img_list) * test_radio)
    train_img_list += img_list[:train_num]
    valid_img_list += img_list[train_num:train_num + valid_num]
    test_img_list += img_list[train_num + valid_num:train_num + valid_num + test_num]

    # 打乱数据
    random.shuffle(train_img_list)
    random.shuffle(valid_img_list)
    random.shuffle(test_img_list)
    return train_img_list, valid_img_list, test_img_list


def move_data(src_dir, img_list, dest_dir):
    index = 0
    for img_file in img_list:
        if os.path.isfile(os.path.join(src_dir, img_file)):
            index += 1
            print("{}/{}将{}移动到{}".format(index, len(img_list), img_file, dest_dir))
            shutil.move(os.path.join(src_dir, img_file), dest_dir)


BASE_DIR = r"F:\BaiduNetdiskDownload\landcover.ai.v1"
IMGS_DIR = os.path.join(BASE_DIR, "images")
MASKS_DIR = os.path.join(BASE_DIR, "masks")
OUTPUT_DIR = os.path.join(BASE_DIR, "train_data")
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_IAMGES_DIR = os.path.join(OUTPUT_DIR, "img_dir")
OUTPUT_LABELS_DIR = os.path.join(OUTPUT_DIR, "ann_dir")
OUTPUT_TRAIN_IAMGES_DIR = os.path.join(OUTPUT_IAMGES_DIR, "train")
OUTPUT_VAL_IAMGES_DIR = os.path.join(OUTPUT_IAMGES_DIR, "val")
OUTPUT_TEST_IAMGES_DIR = os.path.join(OUTPUT_IAMGES_DIR, "test")
OUTPUT_TRAIN_LABELS_DIR = os.path.join(OUTPUT_LABELS_DIR, "train")
OUTPUT_VAL_LABELS_DIR = os.path.join(OUTPUT_LABELS_DIR, "val")
OUTPUT_TEST_LABELS_DIR = os.path.join(OUTPUT_LABELS_DIR, "test")
TARGET_SIZE_X = 800
TARGET_SIZE_Y = 800

img_paths = glob.glob(os.path.join(IMGS_DIR, "*.tif"))
mask_paths = glob.glob(os.path.join(MASKS_DIR, "*.tif"))

img_paths.sort()
mask_paths.sort()

os.makedirs(OUTPUT_IAMGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_TRAIN_IAMGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_VAL_IAMGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_TEST_IAMGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_TRAIN_LABELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_VAL_LABELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_TEST_LABELS_DIR, exist_ok=True)

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
                filename = get_next_seven_digit_number()
                out_img_path = os.path.join(OUTPUT_IAMGES_DIR, "{}.jpg".format(filename))
                cv2.imwrite(out_img_path, img_tile)

                out_mask_path = os.path.join(OUTPUT_LABELS_DIR, "{}.png".format(filename))
                cv2.imwrite(out_mask_path, mask_tile)

            k += 1

    print("Processed {} {}/{}".format(img_filename, i + 1, len(img_paths)))

# 检查分割的图片大小是否正确
for filename in os.listdir(OUTPUT_IAMGES_DIR):
    img_path = os.path.join(OUTPUT_IAMGES_DIR, filename)
    if os.path.isfile(img_path):
        img = cv2.imread(img_path)
        assert img.shape[:2] == (TARGET_SIZE_Y, TARGET_SIZE_X)
print("All images are correct size.")

train_img_list, valid_img_list, test_img_list = split_files_by_radio(OUTPUT_IAMGES_DIR)
train_label_list, valid_label_list, test_label_list = split_files_by_radio(OUTPUT_LABELS_DIR)

move_data(OUTPUT_IAMGES_DIR, train_img_list, OUTPUT_TRAIN_IAMGES_DIR)
move_data(OUTPUT_IAMGES_DIR, valid_img_list, OUTPUT_VAL_IAMGES_DIR)
move_data(OUTPUT_IAMGES_DIR, test_img_list, OUTPUT_TEST_IAMGES_DIR)

move_data(OUTPUT_LABELS_DIR, train_label_list, OUTPUT_TRAIN_LABELS_DIR)
move_data(OUTPUT_LABELS_DIR, valid_label_list, OUTPUT_VAL_LABELS_DIR)
move_data(OUTPUT_LABELS_DIR, test_label_list, OUTPUT_TEST_LABELS_DIR)
