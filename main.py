#!/usr/bin/env python3
"""
计算机视觉中期作业 - 主程序入口
显著特征提取、匹配与图像拼接

作者: Zhang Zhang
日期: 2026年4月27日
"""

import sys
import os
import argparse

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_all_modules():
    """测试所有模块功能"""
    print("计算机视觉中期作业 - 模块测试")
    print("=" * 60)

    try:
        # 测试工具模块
        from utils import load_image, get_image_pairs, create_results_dir
        print("✓ 工具模块导入成功")

        # 测试特征检测模块
        from feature_detection import harris_corner_detection, sift_feature_detection
        print("✓ 特征检测模块导入成功")

        # 测试特征匹配模块
        from feature_matching import ratio_test_matching
        print("✓ 特征匹配模块导入成功")

        # 测试图像拼接模块
        from image_stitching import compute_homography, stitch_images
        print("✓ 图像拼接模块导入成功")

        # 测试UI模块
        try:
            from ui import ComputerVisionApp
            print("✓ UI模块导入成功")
        except ImportError as e:
            print(f"⚠ UI模块导入警告: {e}")

        print("\n所有核心模块导入成功！")
        return True

    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        return False

def run_full_pipeline():
    """运行完整的图像拼接流程"""
    print("\n运行完整图像拼接流程...")
    print("-" * 60)

    try:
        from utils import load_image, get_image_pairs, create_results_dir, save_image
        from feature_detection import sift_feature_detection
        from feature_matching import ratio_test_matching
        from image_stitching import compute_homography, stitch_images

        # 获取图像对
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        pairs = get_image_pairs(data_dir)

        if not pairs:
            print("错误: 未找到图像对！")
            return False

        print(f"找到 {len(pairs)} 对图像")

        # 处理第一对图像
        left_path, right_path = pairs[0]
        print(f"处理图像对: {os.path.basename(left_path)} - {os.path.basename(right_path)}")

        # 加载图像
        image1 = load_image(left_path)
        image2 = load_image(right_path)

        if image1 is None or image2 is None:
            print("错误: 无法加载图像！")
            return False

        print(f"图像1尺寸: {image1.shape}")
        print(f"图像2尺寸: {image2.shape}")

        # 特征检测
        print("\n1. 特征检测...")
        keypoints1, descriptors1 = sift_feature_detection(image1)
        keypoints2, descriptors2 = sift_feature_detection(image2)
        print(f"  图像1检测到 {len(keypoints1)} 个特征点")
        print(f"  图像2检测到 {len(keypoints2)} 个特征点")

        # 特征匹配
        print("\n2. 特征匹配...")
        matches = ratio_test_matching(descriptors1, descriptors2)
        print(f"  找到 {len(matches)} 个匹配对")

        if len(matches) < 4:
            print("错误: 匹配点太少，无法进行图像拼接！")
            return False

        # 图像拼接
        print("\n3. 图像拼接...")
        H = compute_homography(keypoints1, keypoints2, matches)
        if H is None:
            print("错误: 无法计算单应性矩阵！")
            return False

        stitched_image = stitch_images(image1, image2, H)
        print(f"  拼接图像尺寸: {stitched_image.shape}")

        # 保存结果
        print("\n4. 保存结果...")
        results_dir = create_results_dir()
        output_path = os.path.join(results_dir, "stitched_result.png")

        if save_image(stitched_image, output_path):
            print(f"  结果已保存: {output_path}")
        else:
            print("  保存结果失败！")

        print("\n✓ 完整流程执行成功！")
        return True

    except Exception as e:
        print(f"✗ 流程执行失败: {e}")
        return False

def run_gui():
    """运行图形用户界面"""
    print("\n启动图形用户界面...")

    try:
        import tkinter as tk
        from ui import ComputerVisionApp

        root = tk.Tk()
        app = ComputerVisionApp(root)
        root.mainloop()

        return True
    except Exception as e:
        print(f"✗ GUI启动失败: {e}")
        return False

def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='计算机视觉中期作业 - 特征提取与图像拼接')
    parser.add_argument('--test', action='store_true', help='测试所有模块')
    parser.add_argument('--pipeline', action='store_true', help='运行完整流程')
    parser.add_argument('--gui', action='store_true', help='启动图形界面')

    args = parser.parse_args()

    print("计算机视觉中期作业 - 特征提取、匹配与图像拼接")
    print("=" * 60)

    # 检查依赖库
    try:
        import cv2
        import numpy as np
        import matplotlib
        print("✓ 所有依赖库已正确安装")
        print(f"  - OpenCV版本: {cv2.__version__}")
        print(f"  - NumPy版本: {np.__version__}")
        print(f"  - Matplotlib版本: {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ 依赖库导入失败: {e}")
        sys.exit(1)

    # 检查数据目录
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if os.path.exists(data_dir):
        print(f"✓ 数据目录存在: {data_dir}")
        # 列出数据文件
        image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  找到 {len(image_files)} 个图像文件: {', '.join(image_files)}")
    else:
        print(f"✗ 数据目录不存在: {data_dir}")

    # 根据参数执行相应操作
    if args.test:
        test_all_modules()
    elif args.pipeline:
        run_full_pipeline()
    elif args.gui:
        run_gui()
    else:
        # 默认行为：测试模块
        if test_all_modules():
            print("\n使用说明:")
            print("  python main.py --test    测试所有模块")
            print("  python main.py --pipeline 运行完整流程")
            print("  python main.py --gui     启动图形界面")
            print("\n建议先运行测试确保所有模块正常工作。")

if __name__ == "__main__":
    main()