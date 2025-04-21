from BirdSoundClassifier import BirdSoundClassifier

# 设置训练数据路径（确保路径正确）
train_folder = 'training_data'  # 你的训练数据文件夹路径

# 创建并训练模型
clf = BirdSoundClassifier()
clf.train_classifier(train_folder)

# 保存模型
clf.save_model('bird_model.pkl')  # 训练后直接保存模型
print("Model has been saved as 'bird_model.pkl'.")
