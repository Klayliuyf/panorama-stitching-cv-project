import cv2
import numpy as np
import os

def main():
    print("=== 全景拼接简化版（使用OpenCV内置函数）===")
    
    # 1. 加载`images`文件夹里的所有图片
    image_dir = "images"
    images = []
    files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for f in sorted(files):
        img_path = os.path.join(image_dir, f)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            print(f"加载: {f}")
    
    if len(images) < 2:
        print("错误：需要至少2张图片。")
        return
    
    # 2. 使用OpenCV内置拼接器
    stitcher = cv2.Stitcher_create()
    status, panorama = stitcher.stitch(images)
    
    # 3. 输出结果
    if status == cv2.Stitcher_OK:
        output_path = "panorama_result.jpg"
        cv2.imwrite(output_path, panorama)
        print(f"✅ 拼接成功！结果已保存为: {output_path}")
        print(f"图片尺寸: {panorama.shape[1]} x {panorama.shape[0]}")
        
        # 尝试显示图片（如果环境支持）
        try:
            cv2.imshow("全景图", panorama)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            pass
    else:
        print(f"❌ 拼接失败，错误码: {status}")

if __name__ == "__main__":
    main()