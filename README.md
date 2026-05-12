# 计算机视觉中期作业 - 特征提取、匹配与图像拼接

学号：2023010916　姓名：张章

GitHub仓库：https://github.com/study-overflow/CV-Midterm-Stitching

## 环境要求

- Python 3.8+
- 操作系统：Windows / Linux 均可

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行方式

### 图形界面（推荐）

```bash
python main.py --gui
```

界面提供以下按钮：
- **加载预设图像**：从 `data/` 目录选择已有的图像对
- **自选图像对**：通过文件对话框选取任意两张图片
- **特征检测**：执行 Harris 角点检测和 SIFT 特征检测
- **特征匹配**：使用 Ratio Test 进行特征匹配
- **图像拼接**：计算单应性矩阵并生成全景图
- **保存结果**：将拼接结果保存至 `results/` 目录

### 命令行

```bash
# 测试所有模块是否正常
python main.py --test

# 运行完整拼接流程（使用 data/ 中的第一对图像）
python main.py --pipeline
```

## 项目结构

```
├── main.py              # 主程序入口
├── requirements.txt     # 依赖库
├── src/
│   ├── feature_detection.py   # Harris + SIFT 特征检测
│   ├── feature_matching.py    # SSD / Ratio Test 匹配，SIFT + ORB 描述子
│   ├── image_stitching.py     # 单应性矩阵计算与图像拼接
│   ├── utils.py               # 图像读写等工具函数
│   └── ui.py                  # Tkinter 图形界面
├── data/                # 图像数据集（9对 + 实拍测试）
├── results/             # 实验结果与可视化
└── report/
    ├── main.tex         # 报告 LaTeX 源文件
    └── main.pdf         # 报告 PDF
```
