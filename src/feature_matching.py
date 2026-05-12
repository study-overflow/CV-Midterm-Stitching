#!/usr/bin/env python3
"""
特征匹配模块
实现特征描述子和匹配算法
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

def compute_sift_descriptors(keypoints: List[cv2.KeyPoint], image: np.ndarray) -> np.ndarray:
    """
    计算SIFT描述子

    Args:
        keypoints: 关键点列表
        image: 输入图像（灰度）

    Returns:
        描述子矩阵
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 创建SIFT描述子计算器
    sift = cv2.SIFT_create()

    # 计算描述子
    _, descriptors = sift.compute(gray, keypoints)

    return descriptors

def compute_orb_descriptors(keypoints: List[cv2.KeyPoint], image: np.ndarray) -> np.ndarray:
    """
    计算ORB描述子（低成本描述子）

    Args:
        keypoints: 关键点列表
        image: 输入图像（灰度）

    Returns:
        描述子矩阵
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 创建ORB描述子计算器
    orb = cv2.ORB_create()

    # 计算描述子
    _, descriptors = orb.compute(gray, keypoints)

    return descriptors

def ssd_distance(desc1: np.ndarray, desc2: np.ndarray) -> float:
    """
    计算平方差和（SSD）距离

    Args:
        desc1: 第一个描述子
        desc2: 第二个描述子

    Returns:
        SSD距离
    """
    return np.sum((desc1 - desc2) ** 2)


def _compute_ssd_distance_matrix(descriptors1: np.ndarray, descriptors2: np.ndarray) -> np.ndarray:
    """
    使用numpy向量化计算两组描述子之间的SSD距离矩阵

    利用 ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b 展开，避免逐对循环

    Args:
        descriptors1: 第一幅图像的描述子 (N, D)
        descriptors2: 第二幅图像的描述子 (M, D)

    Returns:
        距离矩阵 (N, M)，dist[i][j] = SSD(desc1[i], desc2[j])
    """
    d1 = descriptors1.astype(np.float64)
    d2 = descriptors2.astype(np.float64)
    # ||a||^2 各行的平方和
    sq1 = np.sum(d1 ** 2, axis=1, keepdims=True)  # (N, 1)
    sq2 = np.sum(d2 ** 2, axis=1, keepdims=True)  # (M, 1)
    # SSD = ||a||^2 + ||b||^2 - 2*a·b
    dist = sq1 + sq2.T - 2.0 * d1 @ d2.T
    # 数值误差可能导致极小负值
    np.maximum(dist, 0, out=dist)
    return dist


def brute_force_matching(descriptors1: np.ndarray, descriptors2: np.ndarray,
                        distance_threshold: float = 0.7) -> List[Tuple[int, int, float]]:
    """
    暴力匹配器（使用SSD距离 + 比值检验）

    Args:
        descriptors1: 第一幅图像的描述子
        descriptors2: 第二幅图像的描述子
        distance_threshold: 距离阈值（比率检验）

    Returns:
        匹配对列表，每个元素为(index1, index2, distance)
    """
    if descriptors1 is None or descriptors2 is None:
        return []
    if len(descriptors2) < 2:
        return []

    # 向量化计算距离矩阵
    dist_matrix = _compute_ssd_distance_matrix(descriptors1, descriptors2)

    # 对每一行找最近和次近
    # partition比full sort快: 只保证前2个是最小的
    idx2 = np.argpartition(dist_matrix, 2, axis=1)[:, :2]
    rows = np.arange(len(descriptors1))

    d_first = dist_matrix[rows, idx2[:, 0]]
    d_second = dist_matrix[rows, idx2[:, 1]]

    # 确保 first <= second
    swap = d_first > d_second
    idx2[swap, 0], idx2[swap, 1] = idx2[swap, 1], idx2[swap, 0]
    d_first, d_second = np.minimum(d_first, d_second), np.maximum(d_first, d_second)

    # 比值检验
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = d_first / d_second

    mask = ratio < distance_threshold
    matches = []
    for i in np.where(mask)[0]:
        matches.append((int(i), int(idx2[i, 0]), float(d_first[i])))

    return matches


def ratio_test_matching(descriptors1: np.ndarray, descriptors2: np.ndarray,
                       ratio_threshold: float = 0.7) -> List[Tuple[int, int, float]]:
    """
    比值检验匹配器（向量化实现）

    Args:
        descriptors1: 第一幅图像的描述子
        descriptors2: 第二幅图像的描述子
        ratio_threshold: 比值阈值

    Returns:
        匹配对列表，每个元素为(index1, index2, distance)
    """
    if descriptors1 is None or descriptors2 is None:
        return []
    if len(descriptors2) < 2:
        return []

    # 向量化计算距离矩阵
    dist_matrix = _compute_ssd_distance_matrix(descriptors1, descriptors2)

    # 对每一行找最近和次近
    idx2 = np.argpartition(dist_matrix, 2, axis=1)[:, :2]
    rows = np.arange(len(descriptors1))

    d_first = dist_matrix[rows, idx2[:, 0]]
    d_second = dist_matrix[rows, idx2[:, 1]]

    # 确保 first <= second
    swap = d_first > d_second
    idx2[swap, 0], idx2[swap, 1] = idx2[swap, 1], idx2[swap, 0]
    d_first, d_second = np.minimum(d_first, d_second), np.maximum(d_first, d_second)

    # 比值检验
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = d_first / d_second

    mask = ratio < ratio_threshold
    matches = []
    for i in np.where(mask)[0]:
        matches.append((int(i), int(idx2[i, 0]), float(d_first[i])))

    return matches


def ssd_matching(descriptors1: np.ndarray, descriptors2: np.ndarray,
                threshold: float = None) -> List[Tuple[int, int, float]]:
    """
    纯SSD最近邻匹配（不做比值检验），用于与ratio test做对比实验

    Args:
        descriptors1: 第一幅图像的描述子
        descriptors2: 第二幅图像的描述子
        threshold: 绝对距离阈值（None则不过滤）

    Returns:
        匹配对列表，每个元素为(index1, index2, distance)
    """
    if descriptors1 is None or descriptors2 is None:
        return []
    if len(descriptors2) < 1:
        return []

    dist_matrix = _compute_ssd_distance_matrix(descriptors1, descriptors2)

    best_idx = np.argmin(dist_matrix, axis=1)
    best_dist = dist_matrix[np.arange(len(descriptors1)), best_idx]

    matches = []
    for i in range(len(descriptors1)):
        d = float(best_dist[i])
        if threshold is None or d < threshold:
            matches.append((i, int(best_idx[i]), d))

    return matches

def compute_hamming_distance_matrix(descriptors1: np.ndarray, descriptors2: np.ndarray) -> np.ndarray:
    """
    计算二进制描述子（如ORB）的Hamming距离矩阵

    使用查表法（lookup table）向量化计算popcount，避免逐行循环

    Args:
        descriptors1: 第一幅图像的二进制描述子 (N, D) uint8
        descriptors2: 第二幅图像的二进制描述子 (M, D) uint8

    Returns:
        距离矩阵 (N, M)
    """
    # 构建0-255的popcount查找表
    lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

    n = len(descriptors1)
    m = len(descriptors2)
    dist = np.zeros((n, m), dtype=np.int32)

    # 逐字节XOR后查表求popcount，利用numpy广播
    # descriptors1: (N, D), descriptors2: (M, D)
    # 一次性XOR: (N, 1, D) ^ (1, M, D) -> (N, M, D) 可能内存太大
    # 分块处理以平衡速度和内存
    block_size = 512
    for i in range(0, n, block_size):
        i_end = min(i + block_size, n)
        # (block, 1, D) XOR (1, M, D) -> (block, M, D)
        xor = np.bitwise_xor(
            descriptors1[i:i_end, np.newaxis, :],
            descriptors2[np.newaxis, :, :]
        )
        # 查表popcount并沿D轴求和 -> (block, M)
        dist[i:i_end] = lut[xor].sum(axis=2)

    return dist


def orb_ratio_test_matching(descriptors1: np.ndarray, descriptors2: np.ndarray,
                            ratio_threshold: float = 0.75) -> List[Tuple[int, int, float]]:
    """
    ORB二进制描述子的比值检验匹配（使用Hamming距离）

    Args:
        descriptors1: 第一幅图像的ORB描述子
        descriptors2: 第二幅图像的ORB描述子
        ratio_threshold: 比值阈值

    Returns:
        匹配对列表
    """
    if descriptors1 is None or descriptors2 is None:
        return []
    if len(descriptors2) < 2:
        return []

    dist_matrix = compute_hamming_distance_matrix(descriptors1, descriptors2)

    idx2 = np.argpartition(dist_matrix, 2, axis=1)[:, :2]
    rows = np.arange(len(descriptors1))

    d_first = dist_matrix[rows, idx2[:, 0]].astype(np.float64)
    d_second = dist_matrix[rows, idx2[:, 1]].astype(np.float64)

    swap = d_first > d_second
    idx2[swap, 0], idx2[swap, 1] = idx2[swap, 1], idx2[swap, 0]
    d_first, d_second = np.minimum(d_first, d_second), np.maximum(d_first, d_second)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = d_first / d_second

    mask = ratio < ratio_threshold
    matches = []
    for i in np.where(mask)[0]:
        matches.append((int(i), int(idx2[i, 0]), float(d_first[i])))

    return matches

def visualize_matches(image1: np.ndarray, image2: np.ndarray,
                     keypoints1: List[cv2.KeyPoint], keypoints2: List[cv2.KeyPoint],
                     matches: List[Tuple[int, int, float]],
                     max_matches: int = 50) -> np.ndarray:
    """
    可视化匹配结果

    Args:
        image1: 第一幅图像
        image2: 第二幅图像
        keypoints1: 第一幅图像的关键点
        keypoints2: 第二幅图像的关键点
        matches: 匹配对列表
        max_matches: 最大显示匹配数

    Returns:
        可视化匹配结果的图像
    """
    # 水平拼接两幅图像
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    max_height = max(h1, h2)
    result = np.zeros((max_height, w1 + w2, 3), dtype=np.uint8)

    result[:h1, :w1] = image1
    result[:h2, w1:w1+w2] = image2

    # 随机选择部分匹配进行可视化
    if len(matches) > max_matches:
        import random
        matches = random.sample(matches, max_matches)

    # 绘制匹配线
    for match in matches:
        idx1, idx2, _ = match

        kp1 = keypoints1[idx1]
        kp2 = keypoints2[idx2]

        pt1 = (int(kp1.pt[0]), int(kp1.pt[1]))
        pt2 = (int(kp2.pt[0] + w1), int(kp2.pt[1]))

        # 随机颜色
        color = tuple(np.random.randint(0, 255, 3).tolist())

        cv2.line(result, pt1, pt2, color, 1)
        cv2.circle(result, pt1, 3, color, -1)
        cv2.circle(result, pt2, 3, color, -1)

    return result

if __name__ == "__main__":
    import time
    # 测试特征匹配模块
    print("特征匹配模块测试...")

    from utils import load_image
    from feature_detection import sift_feature_detection

    image1 = load_image("../data/left.png")
    image2 = load_image("../data/right.png")

    if image1 is not None and image2 is not None:
        keypoints1, descriptors1 = sift_feature_detection(image1)
        keypoints2, descriptors2 = sift_feature_detection(image2)
        print(f"图像1: {len(keypoints1)} 个特征点")
        print(f"图像2: {len(keypoints2)} 个特征点")

        # SIFT + Ratio Test
        t0 = time.time()
        matches_rt = ratio_test_matching(descriptors1, descriptors2)
        t_rt = time.time() - t0
        print(f"SIFT Ratio Test: {len(matches_rt)} 个匹配, 耗时 {t_rt:.3f}s")

        # SIFT + 纯SSD
        t0 = time.time()
        matches_ssd = ssd_matching(descriptors1, descriptors2)
        t_ssd = time.time() - t0
        print(f"SIFT 纯SSD:      {len(matches_ssd)} 个匹配, 耗时 {t_ssd:.3f}s")

        # ORB
        orb_desc1 = compute_orb_descriptors(keypoints1, image1)
        orb_desc2 = compute_orb_descriptors(keypoints2, image2)
        if orb_desc1 is not None and orb_desc2 is not None:
            t0 = time.time()
            matches_orb = orb_ratio_test_matching(orb_desc1, orb_desc2)
            t_orb = time.time() - t0
            print(f"ORB Ratio Test:  {len(matches_orb)} 个匹配, 耗时 {t_orb:.3f}s")
    else:
        print("无法加载测试图像")