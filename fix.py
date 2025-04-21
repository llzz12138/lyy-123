import os
import ffmpeg

# 定义文件夹路径
data_dir = 'training_data'

# 函数：重新编码音频文件为标准的wav格式
def reencode_audio(input_path, output_path):
    try:
        ffmpeg.input(input_path).output(output_path, acodec='pcm_s16le', ar='22050').run()
        print(f"文件已重新编码：{input_path} -> {output_path}")
    except ffmpeg.Error as e:
        print(f"重新编码失败：{input_path} -> {output_path}")
        print(e)

# 遍历文件夹并重新编码所有 .wav 文件
def reencode_all_wav_files(data_dir):
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(root, f"reencoded_{file}")
                # 调用重新编码函数
                reencode_audio(input_path, output_path)

# 执行重新编码操作
reencode_all_wav_files(data_dir)
