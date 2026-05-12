#!/usr/bin/env python3
"""
特征检测模块
实现Harris角点检测和SIFT特征点检测
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict

def harris_corner_detection(image: np.ndarray, block_size: int = 2, ksize: int = 3, k: float = 0.04) -> np.ndarray:
    """
    Harris角点检测

    Args:
        image: 输入图像（灰度图）
        block_size: 邻域大小
        ksize: Sobel算子孔径大小
        k: Harris检测器自由参数

    Returns:
        角点响应图
    """
    # 转换为灰度图（如果输入是彩色图）
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 转换为float32
    gray = np.float32(gray)

    # Harris角点检测
    dst = cv2.cornerHarris(gray, block_size, ksize, k)

    return dst

def get_harris_keypoints(harris_response: np.ndarray, threshold: float = 0.01) -> List[Tuple[int, int]]:
    """
    从Harris响应图中提取关键点坐标

    Args:
        harris_response: Harris角点响应图
        threshold: 阈值（相对于最大响应的比例）

    Returns:
        关键点坐标列表 [(x, y), ...]
    """
    # 直接在原始响应上做阈值（避免normalize类型问题）
    max_response = harris_response.max()
    if max_response <= 0:
        return []

    threshold_value = max_response * threshold

    # 向量化查找超过阈值的角点位置
    ys, xs = np.where(harris_response > threshold_value)

    # 非极大值抑制：使用膨胀操作找局部极大值
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(harris_response, kernel)
    local_max = (harris_response == dilated)

    # 同时满足阈值和局部极大值
    mask = (harris_response > threshold_value) & local_max
    ys, xs = np.where(mask)

    keypoints = list(zip(xs.tolist(), ys.tolist()))
    return keypoints

def sift_feature_detection(image: np.ndarray, nfeatures: int = 0, nOctaveLayers: int = 3,
                          contrastThreshold: float = 0.04, edgeThreshold: float = 10,
                          sigma: float = 1.6) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    SIFT特征点检测

    Args:
        image: 输入图像
        nfeatures: 保留的最佳特征数量（0表示无限制）
        nOctaveLayers: 每个octave中的层数
        contrastThreshold: 对比度阈值
        edgeThreshold: 边缘阈值
        sigma: 高斯模糊参数

    Returns:
        (关键点列表, 描述子矩阵)
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 创建SIFT检测器
    sift = cv2.SIFT_create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers,
                           contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold,
                           sigma=sigma)

    # 检测特征点和计算描述子
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors

def compare_detection_methods(image: np.ndarray) -> Dict[str, int]:
    """
    比较Harris和SIFT特征检测方法

    Args:
        image: 输入图像

    Returns:
        包含两种方法检测到的特征点数量的字典
    """
    results = {}

    # Harris角点检测
    harris_response = harris_corner_detection(image)
    harris_keypoints = get_harris_keypoints(harris_response)
    results['harris'] = len(harris_keypoints)

    # SIFT特征检测
    sift_keypoints, _ = sift_feature_detection(image)
    results['sift'] = len(sift_keypoints)

    return results

def visualize_detection_results(image: np.ndarray, harris_keypoints: List[Tuple[int, int]],
                               sift_keypoints: List[cv2.KeyPoint]) -> np.ndarray:
    """
    可视化特征检测结果

    Args:
        image: 原始图像
        harris_keypoints: Harris角点坐标
        sift_keypoints: SIFT关键点

    Returns:
        可视化结果图像
    """
    # 复制原始图像
    result = image.copy()

    # 绘制Harris角点（红色）
    for x, y in harris_keypoints:
        cv2.circle(result, (x, y), 3, (0, 0, 255), -1)  # 红色

    # 绘制SIFT特征点（绿色）
    for kp in sift_keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(result, (x, y), 3, (0, 255, 0), -1)  # 绿色

        # 绘制方向（如果有）
        if kp.angle != -1:
            angle_rad = np.radians(kp.angle)
            end_x = int(x + kp.size * np.cos(angle_rad))
            end_y = int(y + kp.size * np.sin(angle_rad))
            cv2.line(result, (x, y), (end_x, end_y), (0, 255, 0), 1)

    return result

if __name__ == "__main__":
    # 测试特征检测模块
    print("特征检测模块测试...")

    # 测试图像加载
    from utils import load_image

    # 加载测试图像
    test_image_path = "../data/left.png"
    image = load_image(test_image_path)

    if image is not None:
        print(f"图像尺寸: {image.shape}")

        # Harris角点检测
        harris_response = harris_corner_detection(image)
        harris_keypoints = get_harris_keypoints(harris_response)
        print(f"Harris检测到 {len(harris_keypoints)} 个角点")

        # SIFT特征检测
        sift_keypoints, descriptors = sift_feature_detection(image)
        print(f"SIFT检测到 {len(sift_keypoints)} 个特征点")

        # 比较两种方法
        comparison = compare_detection_methods(image)
        print(f"特征点数量比较: Harris={comparison['harris']}, SIFT={comparison['sift']}")

    else:
        print("无法加载测试图像")