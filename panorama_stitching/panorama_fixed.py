import cv2
import numpy as np
import os
import sys

print("全景图像拼接 - 兼容性版本")
print("="*50)

# 1. 检查OpenCV版本并选择可用算法
try:
    # 尝试创建SIFT检测器（部分环境需要opencv-contrib-python）
    detector = cv2.SIFT_create()
    print("✅ 使用 SIFT 特征检测器")
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
except:
    # 如果失败，使用ORB（通常都可用）
    detector = cv2.ORB_create(nfeatures=1000)
    print("✅ 使用 ORB 特征检测器")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 2. 加载图片
image_dir = "images"
images = []
files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

if len(files) < 2:
    print("错误：需要至少2张图片。")
    sys.exit()

for f in files:
    img = cv2.imread(os.path.join(image_dir, f))
    if img is not None:
        # 缩小图片以加快处理（保持宽高比）
        h, w = img.shape[:2]
        if w > 800:
            scale = 800 / w
            new_w = 800
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        images.append(img)
        print(f"已加载: {f} ({img.shape[1]}x{img.shape[0]})")

# 3. 方法一：优先尝试OpenCV内置拼接器（最稳定）
print("\n尝试使用OpenCV内置拼接器...")
stitcher = cv2.Stitcher_create()
status, panorama = stitcher.stitch(images)

if status == cv2.Stitcher_OK:
    output_path = "panorama_result_fixed.jpg"
    cv2.imwrite(output_path, panorama)
    print(f"✅ 内置拼接器成功！结果保存为: {output_path}")
    cv2.imshow("全景图 (内置拼接器)", panorama)
else:
    print(f"内置拼接器失败(错误码:{status})，尝试特征点匹配方法...")
    # 方法二：使用特征点匹配（自定义实现，更可控但复杂）
    # 这里可以调用你 panorama_stitching.py 里的核心函数
    # 例如: from panorama_stitching import stitch_multiple_images
    # result = stitch_multiple_images(images)
    print("提示：可在此处调用 panorama_stitching.py 中的算法")

cv2.waitKey(0)
cv2.destroyAllWindows()