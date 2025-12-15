import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps, ImageTk
import numpy as np
import os
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =================配置区域=================
# 这里的参数必须和 train.py 保持完全一致
MODEL_PATH = os.path.join(BASE_DIR, "models", "mnist_cnn.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# 画布配置
CANVAS_WIDTH = 280
CANVAS_HEIGHT = 280
# 画笔粗细：这个参数很关键，决定了缩放后是否和MNIST的笔触相似
# 280 -> 28 缩小10倍，所以画笔 20-25 左右对应 MNIST 的 2-2.5 像素宽
BRUSH_SIZE = 22 

# =================模型定义=================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =================GUI 主程序=================
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别 v2.0 (深度优化版)")
        
        # 加载模型
        self.load_model()
        
        # 布局容器
        main_frame = tk.Frame(root)
        main_frame.pack(padx=10, pady=10)
        
        # --- 左侧：绘图区 ---
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT)
        
        self.canvas = tk.Canvas(left_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg='black', cursor="cross")
        self.canvas.pack()
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.drawing)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)
        
        # 按钮区
        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(btn_frame, text="清空 (Clear)", command=self.clear_canvas, bg="#ffcccc").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(btn_frame, text="预测 (Predict)", command=self.predict, bg="#ccffcc").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # --- 右侧：信息与调试区 ---
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, padx=20, fill=tk.Y)
        
        tk.Label(right_frame, text="模型看到的输入:", font=("Arial", 10)).pack(anchor="w")
        
        # 调试预览窗口 (显示 28x28 的放大版)
        self.debug_label = tk.Label(right_frame, bg="gray", width=140, height=140)
        self.debug_label.pack(pady=5)
        
        tk.Label(right_frame, text="预测结果:", font=("Arial", 12, "bold")).pack(anchor="w", pady=(10, 0))
        self.result_label = tk.Label(right_frame, text="-", font=("Arial", 48, "bold"), fg="blue")
        self.result_label.pack()
        
        self.probs_label = tk.Label(right_frame, text="", justify=tk.LEFT, font=("Consolas", 9))
        self.probs_label.pack(anchor="w")

        # --- 核心：内存绘图对象 (Shadow Buffer) ---
        # 我们不在屏幕截图，而是直接在内存里的 PIL Image 上画
        # 背景全黑(0)，笔刷全白(255)，与 MNIST 原始格式一致（之后会做预处理）
        self.image = Image.new("L", (CANVAS_WIDTH, CANVAS_HEIGHT), 0)
        self.draw_engine = ImageDraw.Draw(self.image)
        
        self.last_x = None
        self.last_y = None

    def load_model(self):
        try:
            self.model = Net().to(DEVICE)
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}")
            
            # 兼容 CPU 加载 GPU 训练的模型
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"✅ 模型加载成功，运行在: {DEVICE}")
            print(f"📦 模型架构:\n{self.model}")
        except Exception as e:
            messagebox.showerror("模型错误", str(e))
            self.root.destroy()

    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def drawing(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            # 1. 在屏幕 GUI 上画（为了让你看见）
            self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                    width=BRUSH_SIZE, fill='white', capstyle=tk.ROUND, smooth=True)
            # 2. 在内存 PIL Image 上画（为了给模型看）
            self.draw_engine.line([self.last_x, self.last_y, x, y], fill=255, width=BRUSH_SIZE)
        
        self.last_x = x
        self.last_y = y

    def end_draw(self, event):
        # 每次画完一笔，自动触发一次简单预览或预测（可选，这里暂不自动预测）
        pass

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_engine.rectangle((0, 0, CANVAS_WIDTH, CANVAS_HEIGHT), fill=0)
        self.result_label.config(text="-")
        self.probs_label.config(text="")
        self.debug_label.config(image='', bg="gray")

    def preprocess_image(self):
        """
        核心预处理管道：
        1. 裁剪黑边
        2. 等比例缩放以适应 20x20 的盒子
        3. 粘贴到 28x28 的中心
        4. 标准化
        """
        # 获取图像数据
        img = self.image.copy()
        
        # 1. 裁剪内容区域 (Bounding Box)
        bbox = img.getbbox()
        if bbox is None:
            print("⚠️ 画布为空，请先画数字")
            return None # 空白画布
        
        print(f"\n📍 调试信息:")
        print(f"   1️⃣ 裁剪区域 bbox: {bbox}")
            
        img_cropped = img.crop(bbox)
        
        # 2. 计算缩放比例，保持长宽比，放入 20x20 的盒子中
        # MNIST 数字通常占据 28x28 中的中心 20x20 区域
        target_size = 20
        w, h = img_cropped.size
        ratio = min(target_size / w, target_size / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        print(f"   2️⃣ 原始数字大小: {w}x{h}, 缩放比例: {ratio:.2f}, 新大小: {new_w}x{new_h}")
        
        # 使用高质量重采样
        if hasattr(Image, "Resampling"):
            resample_mode = Image.Resampling.LANCZOS
        else:
            resample_mode = Image.ANTIALIAS
            
        img_resized = img_cropped.resize((new_w, new_h), resample_mode)
        
        # 3. 创建 28x28 黑色底图并居中粘贴
        final_img = Image.new("L", (28, 28), 0)
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        final_img.paste(img_resized, (paste_x, paste_y))
        
        # --- 更新调试预览图 ---
        # 放大显示给用户看模型到底看到了什么
        preview_img = final_img.resize((140, 140), resample_mode)
        self.tk_preview = ImageTk.PhotoImage(preview_img)
        self.debug_label.config(image=self.tk_preview)
        
        # 4. 转 Tensor 并标准化
        img_np = np.array(final_img, dtype=np.float32) / 255.0
        tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # 标准化 (x - mean) / std
        tensor = (tensor - MNIST_MEAN) / MNIST_STD
        
        print(f"   3️⃣ 最终张量形状: {tensor.shape}, 值范围: [{tensor.min():.2f}, {tensor.max():.2f}]")
        
        return tensor

    def predict(self):
        tensor = self.preprocess_image()
        if tensor is None:
            self.result_label.config(text="?")
            return
            
        with torch.no_grad():
            output = self.model(tensor)
            # 加上 Softmax 获取概率
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            
            print(f"   🎯 预测结果: {pred_idx}, 置信度: {confidence*100:.1f}%")
            
            # 更新 UI
            self.result_label.config(text=str(pred_idx))
            
            # 显示 Top 3 概率
            top3_indices = probs.argsort()[::-1][:3]
            info_text = "置信度:\n"
            for idx in top3_indices:
                info_text += f"{idx}: {probs[idx]*100:.1f}%\n"
            self.probs_label.config(text=info_text)

if __name__ == "__main__":
    root = tk.Tk()
    # 绑定回车键预测
    root.bind('<Return>', lambda event: app.predict())
    app = DigitRecognizerApp(root)
    root.mainloop()