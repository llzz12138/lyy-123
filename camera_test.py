import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import ttk
from threading import Thread
from playsound import playsound
import os

# ========== 配置 ==========
model_path = 'bird_model.pth'
class_names = ['crow', 'magpie', 'sparrow', 'not_bird']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 音频文件路径 ==========
bird_sounds = {
    'crow': 'sounds/owl.mp3',
    'magpie': 'sounds/snake.mp3',
    'sparrow': 'sounds/eagle.mp3'
}

# ========== 图像预处理 ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ========== 加载模型 ==========
model = models.squeezenet1_1(weights=None)  # 避免使用预训练权重
model.classifier[1] = nn.Conv2d(512, len(class_names), kernel_size=(1, 1))
model.num_classes = len(class_names)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

print("✅ 模型已加载，等待启动摄像头识别...")

# ========== 识别线程函数 ==========
def start_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取摄像头画面")
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            label = class_names[pred.item()]

        # 在帧上显示识别结果
        cv2.putText(frame, f"Detected: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Real-time Bird Detection", frame)

        # 播放驱逐音效（仅当启用音频 且是目标鸟类）
        if audio_enabled.get() and label in bird_sounds:
            Thread(target=playsound, args=(bird_sounds[label],), daemon=True).start()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ========== 图形界面 ==========
root = tk.Tk()
root.title("多模态害鸟识别系统")

# 布局
frm = ttk.Frame(root, padding=20)
frm.grid()

# 音频启用按钮
audio_enabled = tk.BooleanVar(master=root, value=True)
chk_audio = ttk.Checkbutton(frm, text="启用语音驱逐", variable=audio_enabled)
chk_audio.grid(row=0, column=0, sticky="w", pady=10)

# 启动识别按钮
btn_start = ttk.Button(frm, text="启动摄像头识别", command=lambda: Thread(target=start_detection, daemon=True).start())
btn_start.grid(row=1, column=0, pady=10)

# 退出按钮
btn_exit = ttk.Button(frm, text="退出", command=root.destroy)
btn_exit.grid(row=2, column=0, pady=10)

root.mainloop()
