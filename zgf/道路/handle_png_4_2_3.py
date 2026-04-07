from PIL import Image

# 打开 RGBA 图片
img = Image.open(r"D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\道路\ceshitupian\TW2021_4326_XIAO_MORE2.png")
# 转换为 RGB 模式（丢弃 Alpha）
rgb_img = img.convert("RGB")
# 保存为新的图片（可以是 PNG 或 JPG）
rgb_img.save(r"D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\道路\ceshitupian\TW2021_4326_XIAO_MORE2-2.png")