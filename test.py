import os
import shutil
import cv2
import numpy as np
import torch
from torchvision import transforms
from unet import UNet
from datasets import custom_test_dataset
import config as cfg
import time
res_dir = cfg.res_dir

# 删除已存在的结果目录
if os.path.exists(res_dir):
    shutil.rmtree(res_dir)
os.mkdir(res_dir)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'device: {device}')

transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

test_dir = cfg.test_dir
test_dataset = custom_test_dataset(test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.test_bs, shuffle=False)

print(f'\nlen(test_dataset): {len(test_dataset)}')
print(f'len(test_loader): {len(test_loader)}  @bs={cfg.test_bs}')

# 加载预训练模型
model = UNet(n_classes=1, padding=True).to(device)
ckpt_path = os.path.join(cfg.models_dir, cfg.ckpt)
ckpt = torch.load(ckpt_path)
model_state_dict = ckpt['model_state_dict']
model.load_state_dict(model_state_dict)
model.to(device)
start_time = time.time()  # 记录开始时间
print('\nDenoising noisy images...')


model.eval()
with torch.no_grad():
    sample_counter = 0  # 全局样本计数器
    for batch_idx, noisy_imgs in enumerate(test_loader):
        # rrr=cv2.imread(test_dir+'/1_100.jpg')
        # print(noisy_imgs.shape)
        print(f'batch: {str(batch_idx + 1).zfill(len(str(len(test_loader))))}/{len(test_loader)}', end='\r')
        noisy_imgs = noisy_imgs.to(device)
        # print(noisy_imgs.shape)
        out,yL1, y_HL1, y_LH1, y_HH1 = model(noisy_imgs)


        # 逐个处理每个样本的输出
        for i in range(len(out)):
            # 提取单通道输出并转换格式
            denoised = out[i][0].cpu().numpy()
            denoised = np.clip(denoised, 0.0, 1.0) * 255.0
            denoised = np.uint8(denoised)
            # print(denoised.shape)
            # 生成唯一文件名
            filename = f'denoised_{sample_counter:03d}.png'
            cv2.imwrite(os.path.join(res_dir, filename), denoised)
            sample_counter += 1
end_time = time.time()
total_time = end_time - start_time


print(f'\n\nResults saved in "{res_dir}" directory')
print('\nFin.')
print(f'\nTotal runtime: {total_time:.2f} seconds')