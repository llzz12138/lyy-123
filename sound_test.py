import torch
import torch.nn as nn
import librosa
import numpy as np
import pyaudio
import keyboard
import time

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🐦 类别设置
num_classes = 3
label_map = {0: 'crow', 1: 'magpie', 2: 'sparrow'}

# 🎵 模型结构（和训练时保持一致）
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

# 📦 加载模型
model = SimpleCNN(num_classes).to(device)
model.load_state_dict(torch.load('audio_model.pth', map_location=device))
model.eval()

# 音频参数
sample_rate = 22050
duration = 5
n_mfcc = 40
threshold = 0.5  # 最低置信度

# 实时录音设置
p = pyaudio.PyAudio()
chunk = 1024
channels = 1
format = pyaudio.paInt16
rate = sample_rate

def listen_and_recognize():
    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("🎧 识别模式开启，按 'q' 键退出")
    while True:
        if keyboard.is_pressed('q'):
            print("❌ 退出识别模式")
            break

        print("🔴 正在录音...")
        audio_data = []
        for _ in range(0, int(sample_rate / chunk * duration)):
            data = stream.read(chunk)
            audio_data.append(np.frombuffer(data, dtype=np.int16))
        audio_data = np.hstack(audio_data)

        # 🎼 特征提取
        mfcc = librosa.feature.mfcc(y=audio_data.astype(float), sr=sample_rate, n_mfcc=n_mfcc)
        if mfcc.shape[1] < 216:
            pad_width = 216 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :216]

        mfcc_tensor = torch.tensor(mfcc).float().unsqueeze(0).unsqueeze(0).to(device)

        # 🔍 模型预测
        with torch.no_grad():
            output = model(mfcc_tensor)
            prob = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(prob, 1)

        # 📢 输出结果
        if confidence.item() > threshold:
            bird_name = label_map[predicted_class.item()]
            print(f"✅ 检测结果: {bird_name}（置信度 {confidence.item():.2f}）")
        else:
            print("⚠️ 置信度低，无法判断鸟类")

        time.sleep(1)  # 间隔一秒再次识别

    stream.stop_stream()
    stream.close()

# 🕹️ 主程序：按键启动识别
def start_recognition():
    print("📢 按 's' 开始识别，按 'q' 退出程序")
    while True:
        if keyboard.is_pressed('s'):
            listen_and_recognize()
        if keyboard.is_pressed('q'):
            print("🚪 退出程序")
            break

start_recognition()
