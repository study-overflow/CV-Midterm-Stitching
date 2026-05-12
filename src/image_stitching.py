#!/usr/bin/env python3
"""
图像拼接模块
实现单应性矩阵计算和图像拼接
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict

def compute_homography(keypoints1: List[cv2.KeyPoint], keypoints2: List[cv2.KeyPoint],
                       matches: List[Tuple[int, int, float]], method: str = 'RANSAC',
                       ransac_reproj_threshold: float = 5.0) -> Optional[np.ndarray]:
    """
    计算单应性矩阵

    Args:
        keypoints1: 左图像的关键点（参考图像）
        keypoints2: 右图像的关键点（变换图像）
        matches: 匹配对列表
        method: 计算方法 ('RANSAC', 'LMEDS', 'RHO')
        ransac_reproj_threshold: RANSAC重投影阈值

    Returns:
        从右图像到左图像的单应性矩阵，如果计算失败返回None
    """
    if len(matches) < 4:
        print("错误: 需要至少4个匹配点来计算单应性矩阵")
        return None

    # 提取匹配点的坐标
    # 注意：我们需要从右图像到左图像的变换，所以源点是右图像，目标点是左图像
    src_pts = []  # 右图像的关键点
    dst_pts = []  # 左图像的关键点

    for match in matches:
        idx1, idx2, _ = match
        src_pts.append(keypoints2[idx2].pt)  # 右图像的关键点
        dst_pts.append(keypoints1[idx1].pt)  # 左图像的关键点

    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

    # 计算单应性矩阵：从右图像(src)到左图像(dst)
    if method == 'RANSAC':
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold)
    elif method == 'LMEDS':
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
    elif method == 'RHO':
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RHO)
    else:
        H, mask = cv2.findHomography(src_pts, dst_pts, 0)

    # 统计内点数量
    if mask is not None:
        inliers = np.sum(mask)
        print(f"单应性矩阵计算成功，内点数量: {inliers}/{len(matches)}")
    else:
        print("单应性矩阵计算失败")
        return None

    return H

def warp_image(image: np.ndarray, H: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """
    应用单应性矩阵变换图像

    Args:
        image: 输入图像
        H: 单应性矩阵
        output_size: 输出图像尺寸 (width, height)

    Returns:
        变换后的图像
    """
    return cv2.warpPerspective(image, H, output_size)

def stitch_images(image1: np.ndarray, image2: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    拼接两幅图像

    按照标准流程：
    1. 使用单应性矩阵H对右图（image2）进行透视变换
    2. 输出画布宽度 = 左图宽度 + 右图宽度
    3. 将左图内容复制到画布上

    Args:
        image1: 左图像（参考图像）
        image2: 右图像（变换图像）
        H: 从image2到image1的单应性矩阵

    Returns:
        拼接后的全景图像
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # 输出画布：宽度 = 左图宽 + 右图宽，高度取较大值
    output_width = w1 + w2
    output_height = max(h1, h2)

    # 对右图进行透视变换
    warped_image2 = cv2.warpPerspective(image2, H, (output_width, output_height))

    # 创建结果画布，先放置变换后的右图
    result = warped_image2.copy()

    # 将左图复制到画布左侧
    result[0:h1, 0:w1] = image1

    return result

def blend_images(image1: np.ndarray, image2: np.ndarray, blend_width: int = 100) -> np.ndarray:
    """
    融合两幅图像的重叠区域

    Args:
        image1: 第一幅图像
        image2: 第二幅图像
        blend_width: 融合区域宽度

    Returns:
        融合后的图像
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # 创建结果图像
    result = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    result[:h1, :w1] = image1

    # 找到重叠区域
    overlap_start = max(0, w1 - blend_width)
    overlap_end = min(w1, w1 + blend_width)

    # 线性融合重叠区域
    for x in range(overlap_start, overlap_end):
        alpha = (x - overlap_start) / (overlap_end - overlap_start)
        if x < w1 and x - w1 + w2 >= 0:
            result[:, x] = (1 - alpha) * image1[:, x] + alpha * image2[:, x - w1]

    # 复制非重叠区域
    result[:h2, w1:] = image2[:, blend_width:]

    return result

def evaluate_stitching_quality(stitched_image: np.ndarray, original_images: List[np.ndarray]) -> Dict[str, float]:
    """
    评估拼接质量

    Args:
        stitched_image: 拼接后的图像
        original_images: 原始图像列表

    Returns:
        质量评估指标
    """
    metrics = {}

    # 计算拼接图像的大小
    h, w = stitched_image.shape[:2]
    metrics['stitched_size'] = w * h

    # 计算原始图像的总大小
    original_size = sum(img.shape[0] * img.shape[1] for img in original_images)
    metrics['original_size'] = original_size

    # 计算重叠区域比例
    overlap_ratio = 1 - (metrics['stitched_size'] / original_size)
    metrics['overlap_ratio'] = overlap_ratio

    # 计算图像质量指标（简单的梯度方差）
    gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    metrics['sharpness'] = np.var(gradient_magnitude)

    return metrics

def test_image_stitching_pipeline(image1_path: str, image2_path: str) -> Dict[str, any]:
    """
    测试完整的图像拼接流程

    Args:
        image1_path: 第一幅图像路径
        image2_path: 第二幅图像路径

    Returns:
        包含所有中间结果和最终结果的字典
    """
    from utils import load_image
    from feature_detection import sift_feature_detection
    from feature_matching import ratio_test_matching

    results = {}

    # 加载图像
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    if image1 is None or image2 is None:
        print("错误: 无法加载图像")
        return results

    results['image1'] = image1
    results['image2'] = image2

    # 特征检测
    keypoints1, descriptors1 = sift_feature_detection(image1)
    keypoints2, descriptors2 = sift_feature_detection(image2)
    results['keypoints1'] = keypoints1
    results['keypoints2'] = keypoints2

    # 特征匹配
    matches = ratio_test_matching(descriptors1, descriptors2)
    results['matches'] = matches

    # 计算单应性矩阵
    H = compute_homography(keypoints1, keypoints2, matches)
    if H is None:
        print("错误: 无法计算单应性矩阵")
        return results

    results['homography'] = H

    # 图像拼接
    stitched_image = stitch_images(image1, image2, H)
    results['stitched_image'] = stitched_image

    # 质量评估
    quality = evaluate_stitching_quality(stitched_image, [image1, image2])
    results['quality_metrics'] = quality

    return results

if __name__ == "__main__":
    # 测试图像拼接模块
    print("图像拼接模块测试...")

    # 测试拼接流程
    image1_path = "../data/left.png"
    image2_path = "../data/right.png"

    results = test_image_stitching_pipeline(image1_path, image2_path)

    if results:
        print(f"图像1尺寸: {results['image1'].shape}")
        print(f"图像2尺寸: {results['image2'].shape}")
        print(f"检测到特征点: {len(results['keypoints1'])} 和 {len(results['keypoints2'])}")
        print(f"找到匹配对: {len(results['matches'])}")
        print(f"拼接图像尺寸: {results['stitched_image'].shape}")
        print(f"质量指标: {results['quality_metrics']}")

        # 保存结果
        from utils import save_image
        save_image(results['stitched_image'], "../results/stitched_result.png")
    else:
        print("拼接测试失败")