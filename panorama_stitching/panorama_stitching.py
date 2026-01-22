
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class PanoramaStitcher:
    def __init__(self):
        """初始化拼接器"""
        # 使用SIFT算法检测特征点
        self.sift = cv2.SIFT_create()
        # 使用FLANN匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
    def load_images(self, image_paths):
        """加载并预处理图像"""
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"无法加载图像: {path}")
            images.append(img)
        return images
    
    def find_keypoints(self, image):
        """检测图像的特征点和描述符"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """匹配两个图像的特征点"""
        matches = self.flann.knnMatch(desc1, desc2, k=2)
        
        # 应用Lowe's ratio test筛选好的匹配
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        return good_matches
    
    def find_homography(self, kp1, kp2, matches):
        """计算单应性矩阵"""
        if len(matches) < 4:
            raise ValueError("匹配点太少，无法计算单应性矩阵")
        
        # 提取匹配点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # 使用RANSAC算法计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # 统计内点数量
        matches_mask = mask.ravel().tolist()
        inliers = matches_mask.count(1)
        inlier_ratio = inliers / len(matches)
        
        print(f"总匹配数: {len(matches)}, 内点数: {inliers}, 内点比例: {inlier_ratio:.2%}")
        
        return H, matches_mask
    
    def warp_images(self, img1, img2, H):
        """将图像2变换到图像1的坐标系"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 获取变换后的图像角点
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        warped_corners2 = cv2.perspectiveTransform(corners2, H)
        
        # 所有角点合并
        all_corners = np.concatenate((corners1, warped_corners2), axis=0)
        
        # 计算拼接后画布的大小
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        # 计算平移变换
        translation = np.array([[1, 0, -x_min],
                                [0, 1, -y_min],
                                [0, 0, 1]])
        
        # 应用平移
        result_width = x_max - x_min
        result_height = y_max - y_min
        result = cv2.warpPerspective(img2, translation @ H, (result_width, result_height))
        
        # 将第一张图像放到正确位置
        result[-y_min:h1-y_min, -x_min:w1-x_min] = img1
        
        return result
    
    def simple_blend(self, img1, img2):
        """简单的图像融合（重叠区域平均）"""
        # 创建掩码
        mask1 = (img1 > 0).astype(np.uint8)
        mask2 = (img2 > 0).astype(np.uint8)
        
        # 找到重叠区域
        overlap = mask1 * mask2
        
        # 重叠区域取平均值
        blended = img1.copy()
        blended[overlap == 1] = (img1[overlap == 1] // 2 + img2[overlap == 1] // 2)
        blended[mask2 == 1] = img2[mask2 == 1]
        
        return blended
    
    def stitch_two_images(self, img1, img2, show_matches=False):
        """拼接两张图像"""
        print("正在检测特征点...")
        kp1, desc1 = self.find_keypoints(img1)
        kp2, desc2 = self.find_keypoints(img2)
        
        print("正在匹配特征点...")
        matches = self.match_features(desc1, desc2)
        
        if show_matches:
            self.draw_matches(img1, kp1, img2, kp2, matches[:50])
        
        print("正在计算单应性矩阵...")
        H, mask = self.find_homography(kp1, kp2, matches)
        
        print("正在进行图像变换...")
        result = self.warp_images(img1, img2, H)
        
        return result
    
    def stitch_multiple_images(self, image_paths):
        """拼接多张图像"""
        print(f"开始拼接 {len(image_paths)} 张图像...")
        
        # 加载所有图像
        images = self.load_images(image_paths)
        
        # 从中间图像开始拼接（效果更好）
        center_idx = len(images) // 2
        
        # 向左拼接
        left_result = images[center_idx].copy()
        for i in range(center_idx - 1, -1, -1):
            print(f"拼接第 {i+1} 张图像到左侧...")
            left_result = self.stitch_two_images(images[i], left_result)
        
        # 向右拼接
        result = left_result.copy()
        for i in range(center_idx + 1, len(images)):
            print(f"拼接第 {i+1} 张图像到右侧...")
            result = self.stitch_two_images(result, images[i])
        
        return result
    
    def draw_matches(self, img1, kp1, img2, kp2, matches):
        """绘制匹配的特征点"""
        draw_params = dict(matchColor=(0, 255, 0),
                          singlePointColor=(255, 0, 0),
                          flags=cv2.DrawMatchesFlags_DEFAULT)
        
        matches_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
        
        # 显示图像
        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
        plt.title("特征点匹配结果")
        plt.axis('off')
        plt.show()
    
    def save_result(self, result, output_path):
        """保存拼接结果"""
        cv2.imwrite(output_path, result)
        print(f"拼接结果已保存到: {output_path}")

def main():
    """主函数"""
    print("=== 全景图像拼接程序 ===")
    
    # 1. 设置图像路径
    image_dir = "images"
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    
    if len(image_paths) < 2:
        print("错误：至少需要2张图像")
        return
    
    print(f"找到 {len(image_paths)} 张图像: {image_files}")
    
    # 2. 创建拼接器
    stitcher = PanoramaStitcher()
    
    # 3. 执行拼接
    try:
        result = stitcher.stitch_multiple_images(image_paths)
        
        # 4. 显示结果
        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("拼接结果")
        plt.axis('off')
        plt.show()
        
        # 5. 保存结果
        output_path = "panorama_result.jpg"
        stitcher.save_result(result, output_path)
        
        print("✅ 拼接完成！")
        
    except Exception as e:
        print(f"❌ 拼接失败: {str(e)}")
        print("请检查：")
        print("1. 图片是否有足够重叠区域（20-30%）")
        print("2. 图片是否在同一水平线拍摄")
        print("3. 图片亮度是否一致")

def quick_test():
    """快速测试（使用OpenCV内置stitcher）"""
    print("=== 快速测试模式（使用OpenCV内置拼接器）===")
    
    # 创建拼接器
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    
    # 加载图像
    images = []
    for i in range(1, 4):  # 假设有3张图片
        img_path = f"images/test{i}.jpg"
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    
    if len(images) < 2:
        print("需要至少2张测试图片")
        return
    
    # 拼接
    status, panorama = stitcher.stitch(images)
    
    if status == cv2.Stitcher_OK:
        cv2.imwrite("panorama_opencv.jpg", panorama)
        print("✅ 快速测试成功！")
        
        # 显示结果
        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
        plt.title("OpenCV拼接结果")
        plt.axis('off')
        plt.show()
    else:
        print(f"❌ 拼接失败，错误码: {status}")

if __name__ == "__main__":
    # 可以根据需要选择运行模式
    print("请选择运行模式：")
    print("1. 完整实现（推荐）")
    print("2. 快速测试（使用OpenCV内置函数）")
    
    choice = input("输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        quick_test()
    else:
        print("无效选择，运行完整实现")
        main()