import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import tkinter as tk
from tkinter import filedialog

# 设置路径
save_model_path = ('bird_model.pth')

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载模型并修改最后一层
model = models.squeezenet1_1(pretrained=False)
model.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1,1), stride=(1,1))  # 4类输出
model.load_state_dict(torch.load(save_model_path, map_location=torch.device('cpu')))
model = model.to(device)
model.eval()  # 设置为评估模式

# 创建 tkinter 窗口并让用户选择文件
root = tk.Tk()
root.withdraw()  # 隐藏主窗口

# 打开文件选择对话框
test_image_path = filedialog.askopenfilename(
    title="选择一张图片",
    filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
)

# 检查图片文件是否存在
if not os.path.isfile(test_image_path):
    print(f"错误: 图片文件 {test_image_path} 未找到!")
else:
    # 加载并预处理图片
    image = Image.open(test_image_path)
    image = transform(image).unsqueeze(0).to(device)

    # 进行预测
    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)

    # 显示预测结果
    class_names = ['crow', 'magpie', 'sparrow', 'not_bird']  # 根据你的数据集调整类别
    print(f"预测类别: {class_names[predicted_class.item()]}")
