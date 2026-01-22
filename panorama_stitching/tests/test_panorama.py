import cv2
import numpy as np
import os

def generate_test_images():
    """生成测试图像（如果没有真实图片）"""
    print("生成测试图像...")
    
    # 创建images目录
    os.makedirs("images", exist_ok=True)
    
    # 生成3张有重叠区域的测试图像
    for i in range(3):
        # 创建基础图像
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # 添加不同颜色的矩形作为特征
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
        # 每张图像稍微偏移
        offset_x = i * 200  # 重叠200像素
        
        # 绘制一些特征
        for j in range(3):
            x1 = offset_x + 50 + j * 100
            y1 = 100 + j * 80
            x2 = x1 + 80
            y2 = y1 + 80
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[j], -1)
        
        # 添加文字
        cv2.putText(img, f"Image {i+1}", (offset_x + 50, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 保存
        cv2.imwrite(f"images/test{i+1}.jpg", img)
        print(f"生成 images/test{i+1}.jpg")
    
    print("测试图像生成完成！")

if __name__ == "__main__":
    generate_test_images()