import os
import sys

def check_project():
    print("提交前最终检查清单")
    print("="*60)
    
    score = 0
    total = 11
    
    # 1. 检查根目录核心文件
    print("\n1. 核心代码文件:")
    core_files = [
        ('panorama_stitching.py', '主算法实现'),
        ('panorama_simple.py', '简化实现'),
        ('requirements.txt', '依赖列表'),
        ('README.md', '项目说明'),
    ]
    for f, desc in core_files:
        if os.path.exists(f):
            print(f"   ✅ {f:25} - {desc}")
            score += 1
        else:
            print(f"   ❌ {f:25} - {desc} [缺失]")
    
    # 2. 检查必要目录
    print("\n2. 必要目录:")
    needed_dirs = ['images/', 'tests/']
    for d in needed_dirs:
        if os.path.exists(d) and os.path.isdir(d):
            print(f"   ✅ {d:25} - 存在")
            score += 1
        else:
            print(f"   ❌ {d:25} - 缺失")
    
    # 3. 检查images中是否有测试图
    print("\n3. 测试资源:")
    if os.path.exists('images'):
        img_files = [f for f in os.listdir('images') if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if len(img_files) >= 2:
            print(f"   ✅ images/ 目录有 {len(img_files)} 张测试图片")
            score += 1
        else:
            print(f"   ❌ images/ 目录图片不足（至少需要2张）")
    
    # 4. 检查报告
    print("\n4. 课程报告:")
    report_found = False
    for f in os.listdir('.'):
        if f.lower().endswith('.pdf') and ('报告' in f or 'report' in f.lower()):
            print(f"   ✅ 找到PDF报告: {f}")
            score += 2  # 报告分值更高
            report_found = True
            break
    if not report_found:
        print("   ❌ 未找到PDF格式的最终报告 (Final_Report.pdf)")
    
    # 总结
    print("\n" + "="*60)
    print(f"检查完成: {score}/{total} 项通过")
    print(f"完成度: {score/total*100:.1f}%")
    
    if score >= total - 2:
        print("🎉 项目完整性良好，可以提交！")
        return True
    else:
        print("⚠️  请根据上方提示补全缺失项。")
        return False

if __name__ == "__main__":
    check_project()