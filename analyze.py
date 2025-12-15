"""
模型分析工具 - 可视化CNN各层的输出和模型性能
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# ==================== 设置中文字体 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 配置
MODEL_PATH = "models/mnist_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "analysis_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== 模型定义（必须和train.py一致） ====================
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

# ==================== 特殊模型：用于提取中间层输出 ====================
class NetWithHooks(nn.Module):
    """能看到中间层输出的模型"""
    def __init__(self, base_model):
        super().__init__()
        self.conv1 = base_model.conv1
        self.conv2 = base_model.conv2
        self.pool = base_model.pool
        self.fc1 = base_model.fc1
        self.fc2 = base_model.fc2
        
        # 存储中间输出
        self.features = {}
    
    def forward(self, x):
        # Conv1
        x = F.relu(self.conv1(x))
        self.features['conv1'] = x.detach()
        
        # Conv2 + Pool
        x = F.relu(self.conv2(x))
        self.features['conv2'] = x.detach()
        x = self.pool(x)
        self.features['pool'] = x.detach()
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC1
        x = F.relu(self.fc1(x))
        self.features['fc1'] = x.detach()
        
        # FC2
        x = self.fc2(x)
        return x


def load_model():
    """加载模型"""
    model = Net().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def visualize_conv_filters(model, save_path="conv_filters.png"):
    """
    可视化第一层卷积核
    每个卷积核都是一个3×3的小图像，学到了什么特征
    """
    print("\n🎨 可视化Conv1的32个卷积核...")
    
    weights = model.conv1.weight.data.cpu().numpy()  # shape: (32, 1, 3, 3)
    
    # 标准化到0-1
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    
    fig, axes = plt.subplots(4, 8, figsize=(12, 8))
    fig.suptitle('Conv1层的32个卷积核（3×3滤波器）', fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        ax.imshow(weights[idx, 0], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Filter {idx+1}', fontsize=8)
    
    full_path = os.path.join(RESULTS_DIR, save_path)
    plt.tight_layout()
    plt.savefig(full_path, dpi=100, bbox_inches='tight')
    print(f"✅ 已保存: {full_path}")
    plt.close()


def visualize_feature_maps(model_with_hooks, test_image, digit_label, save_path="feature_maps.png"):
    """
    可视化一张图像在各层的特征图
    看模型是怎么处理输入的
    """
    print(f"\n🔍 可视化数字'{digit_label}'在各层的特征图...")
    
    with torch.no_grad():
        output = model_with_hooks(test_image)
    
    features = model_with_hooks.features
    
    # ========== 原始图像 ==========
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.suptitle(f'CNN各层特征提取过程（输入数字：{digit_label}）', fontsize=16, fontweight='bold')
    
    # 第1行：原始图像 + Conv1的4个特征图
    original = test_image.cpu().numpy()[0, 0]
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('原始图像\n(28×28)', fontweight='bold')
    axes[0, 0].axis('off')
    
    conv1_feat = features['conv1'].cpu().numpy()[0]  # shape: (32, 28, 28)
    for i in range(1, 4):
        ax = axes[0, i]
        ax.imshow(conv1_feat[i*8], cmap='hot')
        ax.set_title(f'Conv1滤波器{i*8+1}\n(28×28)', fontsize=9)
        ax.axis('off')
    
    # 第2行：Conv2的4个特征图
    conv2_feat = features['conv2'].cpu().numpy()[0]  # shape: (64, 28, 28)
    for i in range(4):
        ax = axes[1, i]
        ax.imshow(conv2_feat[i*16], cmap='hot')
        ax.set_title(f'Conv2滤波器{i*16+1}\n(28×28)', fontsize=9)
        ax.axis('off')
    
    # 第3行：Pool后的特征图
    pool_feat = features['pool'].cpu().numpy()[0]  # shape: (64, 14, 14)
    for i in range(4):
        ax = axes[2, i]
        ax.imshow(pool_feat[i*16], cmap='hot')
        ax.set_title(f'MaxPool后\n(14×14)', fontsize=9)
        ax.axis('off')
    
    full_path = os.path.join(RESULTS_DIR, save_path)
    plt.tight_layout()
    plt.savefig(full_path, dpi=100, bbox_inches='tight')
    print(f"✅ 已保存: {full_path}")
    plt.close()


def evaluate_and_confusion_matrix(model, test_loader, save_path="confusion_matrix.png"):
    """
    评估模型，生成混淆矩阵
    看哪些数字最容易被识别错
    """
    print("\n📊 计算混淆矩阵（评估模型性能）...")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算整体准确率
    accuracy = (all_preds == all_labels).mean()
    print(f"整体准确率: {accuracy*100:.2f}%")
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘制
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
                xticklabels=range(10), yticklabels=range(10))
    ax.set_xlabel('预测结果', fontsize=12, fontweight='bold')
    ax.set_ylabel('真实标签', fontsize=12, fontweight='bold')
    ax.set_title(f'混淆矩阵（整体准确率：{accuracy*100:.2f}%）', fontsize=14, fontweight='bold')
    
    full_path = os.path.join(RESULTS_DIR, save_path)
    plt.tight_layout()
    plt.savefig(full_path, dpi=100, bbox_inches='tight')
    print(f"✅ 已保存: {full_path}")
    plt.close()
    
    # 按类别打印精度
    print("\n📈 各数字的识别精度:")
    for digit in range(10):
        correct = cm[digit, digit]
        total = cm[digit].sum()
        acc = correct / total if total > 0 else 0
        print(f"  数字{digit}: {acc*100:.1f}% ({correct}/{total})")


def main():
    # 加载模型和数据
    print("加载模型...")
    model = load_model()
    model_with_hooks = NetWithHooks(model).to(DEVICE)
    
    print("加载MNIST测试数据...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # ==================== 分析 ====================
    
    # 1️⃣ 可视化卷积核
    visualize_conv_filters(model)
    
    # 2️⃣ 选几个数字，看它们怎么被处理的
    print("\n🎬 选取不同数字，可视化特征提取过程...")
    for digit_label in [0, 3, 5, 8]:
        # 找一个标签为digit_label的图像
        for images, labels in test_loader:
            mask = labels == digit_label
            if mask.any():
                test_image = images[mask][0:1].to(DEVICE)
                save_name = f"feature_maps_digit{digit_label}.png"
                visualize_feature_maps(model_with_hooks, test_image, digit_label, save_name)
                break
    
    # 3️⃣ 混淆矩阵
    evaluate_and_confusion_matrix(model, test_loader)
    
    print("\n" + "="*50)
    print("✅ 分析完成！已生成以下文件：")
    print("  1. conv_filters.png - Conv1的32个卷积核")
    print("  2. feature_maps_digit*.png - 各数字的特征提取过程")
    print("  3. confusion_matrix.png - 混淆矩阵")
    print("="*50)


if __name__ == "__main__":
    main()
