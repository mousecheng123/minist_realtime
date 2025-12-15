# train.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 超参
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
MODEL_PATH = "models/mnist_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 调试输出
print(f"🎯 运行设备: {DEVICE}")
print(f"📊 超参: batch_size={BATCH_SIZE}, epochs={EPOCHS}, lr={LR}")

os.makedirs("models", exist_ok=True)

# 模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)               # /2
        self.fc1 = nn.Linear(12544, 128)       # after pooling twice or once
        # Above we will actually apply pool once to reduce to 14x14
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))  # -> (64,14,14)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 数据（注意：训练用 Normalize 使用 MNIST 的 mean/std）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=1000, shuffle=False, num_workers=0)

model = Net().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# 训练
for epoch in range(1, EPOCHS+1):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (batch_idx+1) % 100 == 0:
            pbar.set_postfix(loss=running_loss / (batch_idx+1))

    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            pred = outputs.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    acc = 100.0 * correct / total
    print(f"Epoch {epoch} - Test accuracy: {acc:.2f}%")

# 保存权重（CPU/GPU 通用）
torch.save(model.state_dict(), MODEL_PATH)
print("Saved model to", MODEL_PATH)
