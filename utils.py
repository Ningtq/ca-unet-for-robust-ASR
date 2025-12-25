import os
import shutil

# 获取目录下所有文件和文件夹的名称列表
dir_path = "/s4home/ntq523/CA-UNet/results/whisper-large-v2/baselines/OA/chime4-simu"
files = os.listdir(dir_path)

# 获取完整路径
for item in files:
    item_new = item.replace('_', '-').replace('-eval', '').replace('mossfromergan', 'mossformergan')
    shutil.move(f"{dir_path}/{item}", f"{dir_path}/{item_new}")
