import sys
import subprocess
import pkg_resources

def main():
    print("=== 环境检查 ===")
    
    # 检查Python版本
    print(f"Python版本: {sys.version_info.major}.{sys.version_info.minor}")
    
    # 检查关键包
    required_packages = ['opencv-python', 'numpy', 'matplotlib']
    print("\n检查依赖包:")
    for pkg in required_packages:
        try:
            version = pkg_resources.get_distribution(pkg).version
            print(f"  ✅ {pkg:20s} 版本: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"  ❌ {pkg:20s} 未安装")
    
    # 测试OpenCV功能
    try:
        import cv2
        print(f"\nOpenCV版本: {cv2.__version__}")
        
        # 测试SIFT
        try:
            sift = cv2.SIFT_create()
            print("✅ SIFT功能: 正常")
        except:
            print("⚠️  SIFT功能: 不可用 (可能需要opencv-contrib-python)")
            
        # 测试Stitcher
        try:
            stitcher = cv2.Stitcher_create()
            print("✅ 拼接器功能: 正常")
        except:
            print("⚠️  拼接器功能: 不可用")
            
    except ImportError:
        print("❌ 无法导入OpenCV")

if __name__ == "__main__":
    main()