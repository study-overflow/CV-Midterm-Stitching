#!/usr/bin/env python3
"""
工具函数模块
包含图像处理、文件操作等辅助函数
"""

import cv2
import numpy as np
import os
from typing import Tuple, List, Optional

def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    加载图像文件

    Args:
        image_path: 图像文件路径

    Returns:
        numpy数组表示的图像，如果加载失败返回None
    """
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法加载图像: {image_path}")
        return None

    return image

def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    将BGR图像转换为RGB格式

    Args:
        image: BGR格式的图像

    Returns:
        RGB格式的图像
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def resize_image(image: np.ndarray, max_size: int = 800) -> np.ndarray:
    """
    调整图像大小，保持宽高比

    Args:
        image: 输入图像
        max_size: 最大尺寸（宽或高）

    Returns:
        调整大小后的图像
    """
    h, w = image.shape[:2]

    if max(h, w) <= max_size:
        return image

    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)

    return cv2.resize(image, (new_w, new_h))

def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    保存图像文件

    Args:
        image: 要保存的图像
        output_path: 输出路径

    Returns:
        保存成功返回True，失败返回False
    """
    try:
        cv2.imwrite(output_path, image)
        print(f"图像已保存: {output_path}")
        return True
    except Exception as e:
        print(f"保存图像失败: {e}")
        return False

def get_image_pairs(data_dir: str) -> List[Tuple[str, str]]:
    """
    从数据目录获取图像对

    Args:
        data_dir: 数据目录路径

    Returns:
        图像对列表，每个元素为(left_image_path, right_image_path)
    """
    image_pairs = []

    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return image_pairs

    # 查找所有图像文件
    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    # 简单的配对逻辑：按文件名排序后相邻配对
    for i in range(0, len(image_files) - 1, 2):
        left_path = os.path.join(data_dir, image_files[i])
        right_path = os.path.join(data_dir, image_files[i + 1])
        image_pairs.append((left_path, right_path))

    return image_pairs

def create_results_dir() -> str:
    """
    创建结果目录

    Returns:
        结果目录路径
    """
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def draw_keypoints(image: np.ndarray, keypoints: List[cv2.KeyPoint], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    在图像上绘制特征点

    Args:
        image: 输入图像
        keypoints: 特征点列表
        color: 绘制颜色 (BGR格式)

    Returns:
        绘制了特征点的图像
    """
    result = image.copy()

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        size = int(kp.size)

        # 绘制特征点
        cv2.circle(result, (x, y), 3, color, -1)

        # 绘制方向（如果有）
        if kp.angle != -1:
            angle_rad = np.radians(kp.angle)
            end_x = int(x + size * np.cos(angle_rad))
            end_y = int(y + size * np.sin(angle_rad))
            cv2.line(result, (x, y), (end_x, end_y), color, 1)

    return result

if __name__ == "__main__":
    # 测试工具函数
    print("工具函数模块测试...")

    # 测试图像对获取
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    pairs = get_image_pairs(data_dir)
    print(f"找到 {len(pairs)} 对图像")

    for i, (left, right) in enumerate(pairs):
        print(f"Pair {i+1}: {os.path.basename(left)} - {os.path.basename(right)}")