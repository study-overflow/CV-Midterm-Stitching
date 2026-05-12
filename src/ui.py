#!/usr/bin/env python3
"""
用户界面模块
实现图形用户界面，提供功能按钮和结果可视化
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

from utils import load_image, resize_image, save_image, get_image_pairs, create_results_dir
from feature_detection import harris_corner_detection, get_harris_keypoints, sift_feature_detection, visualize_detection_results
from feature_matching import ratio_test_matching, visualize_matches
from image_stitching import compute_homography, stitch_images

class ComputerVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("计算机视觉中期作业 - 特征提取与图像拼接")
        self.root.geometry("1200x800")

        # 初始化变量
        self.image1 = None
        self.image2 = None
        self.keypoints1 = None
        self.keypoints2 = None
        self.descriptors1 = None
        self.descriptors2 = None
        self.matches = None
        self.homography = None
        self.stitched_image = None

        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="5")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # 按钮
        ttk.Button(control_frame, text="加载预设图像", command=self.load_image_pair).grid(row=0, column=0, padx=5, pady=2)
        ttk.Button(control_frame, text="自选图像对", command=self.load_custom_pair).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(control_frame, text="特征检测", command=self.detect_features).grid(row=0, column=2, padx=5, pady=2)
        ttk.Button(control_frame, text="特征匹配", command=self.match_features).grid(row=0, column=3, padx=5, pady=2)
        ttk.Button(control_frame, text="图像拼接", command=self.stitch_images).grid(row=0, column=4, padx=5, pady=2)
        ttk.Button(control_frame, text="保存结果", command=self.save_results).grid(row=0, column=5, padx=5, pady=2)
        ttk.Button(control_frame, text="重置", command=self.reset).grid(row=0, column=6, padx=5, pady=2)

        # 状态显示
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(control_frame, textvariable=self.status_var).grid(row=0, column=7, padx=10, pady=2)

        # 图像显示区域
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 左图像显示
        left_frame = ttk.LabelFrame(display_frame, text="左图像")
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.left_canvas = tk.Canvas(left_frame, width=400, height=300, bg='white')
        self.left_canvas.pack(padx=5, pady=5)

        # 右图像显示
        right_frame = ttk.LabelFrame(display_frame, text="右图像")
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.right_canvas = tk.Canvas(right_frame, width=400, height=300, bg='white')
        self.right_canvas.pack(padx=5, pady=5)

        # 结果显示区域
        result_frame = ttk.LabelFrame(main_frame, text="结果")
        result_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.result_canvas = tk.Canvas(result_frame, width=800, height=300, bg='white')
        self.result_canvas.pack(padx=5, pady=5)

        # 配置网格权重
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(0, weight=1)

    def load_image_pair(self):
        """从数据目录加载预设图像对"""
        self.status_var.set("正在加载图像...")
        self.root.update()

        try:
            # 从数据目录获取图像对
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
            pairs = get_image_pairs(data_dir)

            if not pairs:
                messagebox.showerror("错误", "未找到图像对！请确保数据目录包含图像文件。")
                self.status_var.set("错误: 未找到图像")
                return

            # 如果有多对，让用户选择
            if len(pairs) > 1:
                pair_names = [f"{os.path.basename(l)} - {os.path.basename(r)}" for l, r in pairs]
                # 创建选择对话框
                select_win = tk.Toplevel(self.root)
                select_win.title("选择图像对")
                select_win.geometry("400x300")
                select_win.transient(self.root)
                select_win.grab_set()

                tk.Label(select_win, text="请选择要加载的图像对：", font=("", 12)).pack(pady=10)
                listbox = tk.Listbox(select_win, font=("", 10))
                for name in pair_names:
                    listbox.insert(tk.END, name)
                listbox.selection_set(0)
                listbox.pack(fill=tk.BOTH, expand=True, padx=10)

                selected_idx = [0]
                def on_confirm():
                    sel = listbox.curselection()
                    if sel:
                        selected_idx[0] = sel[0]
                    select_win.destroy()

                ttk.Button(select_win, text="确定", command=on_confirm).pack(pady=10)
                self.root.wait_window(select_win)

                left_path, right_path = pairs[selected_idx[0]]
            else:
                left_path, right_path = pairs[0]

            self._load_and_display(left_path, right_path)

        except Exception as e:
            messagebox.showerror("错误", f"加载图像时出错: {e}")
            self.status_var.set("错误: 加载失败")

    def load_custom_pair(self):
        """让用户自选两张图像文件"""
        self.status_var.set("请选择左图...")
        self.root.update()

        filetypes = [("图像文件", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("所有文件", "*.*")]

        left_path = filedialog.askopenfilename(title="选择左图（参考图像）", filetypes=filetypes)
        if not left_path:
            self.status_var.set("已取消")
            return

        right_path = filedialog.askopenfilename(title="选择右图（待变换图像）", filetypes=filetypes)
        if not right_path:
            self.status_var.set("已取消")
            return

        self._load_and_display(left_path, right_path)

    def _load_and_display(self, left_path, right_path):
        """加载并显示一对图像（内部方法）"""
        self.image1 = load_image(left_path)
        self.image2 = load_image(right_path)

        if self.image1 is None or self.image2 is None:
            messagebox.showerror("错误", "无法加载图像文件！")
            self.status_var.set("错误: 无法加载图像")
            return

        # 重置中间状态
        self.keypoints1 = None
        self.keypoints2 = None
        self.descriptors1 = None
        self.descriptors2 = None
        self.matches = None
        self.homography = None
        self.stitched_image = None
        self.result_canvas.delete("all")

        # 调整图像大小用于显示
        self.display_image1 = resize_image(self.image1, 400)
        self.display_image2 = resize_image(self.image2, 400)

        # 显示图像
        self.display_image_on_canvas(self.display_image1, self.left_canvas)
        self.display_image_on_canvas(self.display_image2, self.right_canvas)

        self.status_var.set(f"已加载: {os.path.basename(left_path)} - {os.path.basename(right_path)}")

    def detect_features(self):
        """特征检测"""
        if self.image1 is None or self.image2 is None:
            messagebox.showwarning("警告", "请先加载图像！")
            return

        self.status_var.set("正在检测特征...")
        self.root.update()

        try:
            # Harris角点检测
            harris_response1 = harris_corner_detection(self.image1)
            harris_response2 = harris_corner_detection(self.image2)

            harris_keypoints1 = get_harris_keypoints(harris_response1)
            harris_keypoints2 = get_harris_keypoints(harris_response2)

            # SIFT特征检测
            self.keypoints1, self.descriptors1 = sift_feature_detection(self.image1)
            self.keypoints2, self.descriptors2 = sift_feature_detection(self.image2)

            # 可视化结果
            result_image1 = visualize_detection_results(self.image1, harris_keypoints1, self.keypoints1)
            result_image2 = visualize_detection_results(self.image2, harris_keypoints2, self.keypoints2)

            # 调整大小用于显示
            result_image1 = resize_image(result_image1, 400)
            result_image2 = resize_image(result_image2, 400)

            # 显示结果
            self.display_image_on_canvas(result_image1, self.left_canvas)
            self.display_image_on_canvas(result_image2, self.right_canvas)

            self.status_var.set(f"特征检测完成: Harris={len(harris_keypoints1)}/{len(harris_keypoints2)}, SIFT={len(self.keypoints1)}/{len(self.keypoints2)}")

        except Exception as e:
            messagebox.showerror("错误", f"特征检测时出错: {e}")
            self.status_var.set("错误: 特征检测失败")

    def match_features(self):
        """特征匹配"""
        if self.keypoints1 is None or self.keypoints2 is None:
            messagebox.showwarning("警告", "请先进行特征检测！")
            return

        self.status_var.set("正在匹配特征...")
        self.root.update()

        try:
            # 特征匹配
            self.matches = ratio_test_matching(self.descriptors1, self.descriptors2)

            # 可视化匹配结果
            match_image = visualize_matches(self.image1, self.image2,
                                          self.keypoints1, self.keypoints2,
                                          self.matches)

            # 调整大小用于显示
            match_image = resize_image(match_image, 800)

            # 显示匹配结果
            self.display_image_on_canvas(match_image, self.result_canvas)

            self.status_var.set(f"特征匹配完成: 找到 {len(self.matches)} 个匹配对")

        except Exception as e:
            messagebox.showerror("错误", f"特征匹配时出错: {e}")
            self.status_var.set("错误: 特征匹配失败")

    def stitch_images(self):
        """图像拼接"""
        if self.matches is None:
            messagebox.showwarning("警告", "请先进行特征匹配！")
            return

        self.status_var.set("正在拼接图像...")
        self.root.update()

        try:
            # 计算单应性矩阵
            self.homography = compute_homography(self.keypoints1, self.keypoints2, self.matches)

            if self.homography is None:
                messagebox.showerror("错误", "无法计算单应性矩阵！")
                self.status_var.set("错误: 单应性矩阵计算失败")
                return

            # 图像拼接
            self.stitched_image = stitch_images(self.image1, self.image2, self.homography)

            # 显示拼接结果
            stitched_display = resize_image(self.stitched_image, 800)
            self.display_image_on_canvas(stitched_display, self.result_canvas)

            self.status_var.set("图像拼接完成")

        except Exception as e:
            messagebox.showerror("错误", f"图像拼接时出错: {e}")
            self.status_var.set("错误: 图像拼接失败")

    def save_results(self):
        """保存结果"""
        if self.stitched_image is None:
            messagebox.showwarning("警告", "请先完成图像拼接！")
            return

        try:
            # 创建结果目录
            results_dir = create_results_dir()

            # 保存拼接结果
            output_path = os.path.join(results_dir, "stitched_result.png")
            if save_image(self.stitched_image, output_path):
                messagebox.showinfo("成功", f"结果已保存到: {output_path}")
                self.status_var.set("结果已保存")
            else:
                messagebox.showerror("错误", "保存结果失败！")

        except Exception as e:
            messagebox.showerror("错误", f"保存结果时出错: {e}")

    def reset(self):
        """重置所有状态"""
        self.image1 = None
        self.image2 = None
        self.keypoints1 = None
        self.keypoints2 = None
        self.descriptors1 = None
        self.descriptors2 = None
        self.matches = None
        self.homography = None
        self.stitched_image = None

        # 清空画布
        self.left_canvas.delete("all")
        self.right_canvas.delete("all")
        self.result_canvas.delete("all")

        self.status_var.set("已重置")

    def display_image_on_canvas(self, image: np.ndarray, canvas: tk.Canvas):
        """在画布上显示图像"""
        # 转换图像格式
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 转换为PIL图像
        pil_image = Image.fromarray(image_rgb)

        # 调整图像大小以适应画布
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            pil_image.thumbnail((canvas_width - 10, canvas_height - 10), Image.Resampling.LANCZOS)

        # 转换为Tkinter图像
        tk_image = ImageTk.PhotoImage(pil_image)

        # 清空画布并显示图像
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=tk_image)

        # 保存引用防止垃圾回收
        canvas.tk_image = tk_image

def main():
    """主函数"""
    root = tk.Tk()
    app = ComputerVisionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()