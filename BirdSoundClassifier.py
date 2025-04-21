import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np

# 参数设置
data_dir = 'training_data'
save_model_path = 'audio_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_rate = 22050
duration = 5  # 每个音频统一裁剪为5秒
n_mfcc = 40

# 自定义数据集
class BirdSoundDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.labels = []
        self.label_map = {name: idx for idx, name in enumerate(sorted(os.listdir(root_dir)))}
        self.root_dir = root_dir
        for label in self.label_map:
            folder = os.path.join(root_dir, label)
            for file in os.listdir(folder):
                if file.endswith('.wav'):
                    self.data.append(os.path.join(folder, file))
                    self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]
        y, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
        if len(y) < sample_rate * duration:
            y = np.pad(y, (0, sample_rate * duration - len(y)))  # 补零
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc = mfcc[:, :216]  # 固定时间步长
        mfcc = torch.tensor(mfcc).float().unsqueeze(0)  # [1, 40, 216]
        return mfcc, label

# 模型定义
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 10 * 54, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 加载数据
dataset = BirdSoundDataset(data_dir)
num_classes = len(set(dataset.labels))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 初始化模型
model = SimpleCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 开始训练
print("🎙️ 开始训练语音模型...")
for epoch in range(20):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels)
        total += labels.size(0)
    acc = correct.double() / total
    print(f"Epoch [{epoch+1}/20], Loss: {running_loss/total:.4f}, Accuracy: {acc:.4f}")

# 保存模型
torch.save(model.state_dict(), save_model_path)
print(f"✅ 语音模型保存为 {save_model_path}")
