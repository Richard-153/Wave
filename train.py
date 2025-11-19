import sys, os, time, glob, time, pdb, cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import kornia.losses as losses
from pytorch_wavelets import DWTForward
from torch.optim.lr_scheduler import CosineAnnealingLR
# L1_loss = nn.L1Loss()
# ssim_loss = losses.SSIMLoss(window_size=11, reduction='mean')
# a = 1
# b = 0
# c = 0.1
plt.switch_backend('agg') # for servers not supporting display
# import neccesary libraries for defining the optimizers
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms

from unet import UNet
from datasets import DAE_dataset
import config as cfg

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('device: ', device)

script_time = time.time()

def q(text = ''):
    print('> {}'.format(text))
    sys.exit()

data_dir = cfg.data_dir
train_image_dir = cfg.train_dir
# train_mask_dir =cfg.train_dir

    
models_dir = cfg.models_dir
if not os.path.exists(models_dir):
    os.mkdir(models_dir)

losses_dir = cfg.losses_dir
if not os.path.exists(losses_dir):
    os.mkdir(losses_dir)

def count_parameters(model):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters/1e6 # in terms of millions

def plot_losses(running_train_loss, running_val_loss, train_epoch_loss, val_epoch_loss, epoch):
    fig = plt.figure(figsize=(16,16))
    fig.suptitle('loss trends', fontsize=20)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.title.set_text('epoch train loss VS #epochs')
    ax1.set_xlabel('#epochs')
    ax1.set_ylabel('epoch train loss')
    ax1.plot(train_epoch_loss)
    
    ax2.title.set_text('epoch val loss VS #epochs')
    ax2.set_xlabel('#epochs')
    ax2.set_ylabel('epoch val loss')
    ax2.plot(val_epoch_loss)
 
    ax3.title.set_text('batch train loss VS #batches')
    ax3.set_xlabel('#batches')
    ax3.set_ylabel('batch train loss')
    ax3.plot(running_train_loss)

    ax4.title.set_text('batch val loss VS #batches')
    ax4.set_xlabel('#batches')
    ax4.set_ylabel('batch val loss')
    ax4.plot(running_val_loss)
    
    plt.savefig(os.path.join(losses_dir,'losses_{}.png'.format(str(epoch + 1).zfill(2))))

transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
Train_dir = os.path.join(data_dir, train_image_dir)
# print(Train_dir)
# Train_mask_dir = os.path.join(data_dir, train_mask_dir)
# print(Train_mask_dir)
Val_dir = os.path.join(cfg.data_dir,cfg.val_dir)
# print(Val_dir)
# Val_mask_dir = os.path.join(cfg.data_dir,cfg.val_dir)
# print(Val_mask_dir)
train_dataset       = DAE_dataset(data_dir=Train_dir, transform = transform)
val_dataset         = DAE_dataset(data_dir=Val_dir, transform = transform)

print('\nlen(train_dataset) : ', len(train_dataset))
print('len(val_dataset)   : ', len(val_dataset))

batch_size = cfg.batch_size

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = not True)

print('\nlen(train_loader): {}  @bs={}'.format(len(train_loader), batch_size))
print('len(val_loader)  : {}  @bs={}'.format(len(val_loader), batch_size))

# defining the model
model = UNet(n_classes = 1, padding = True).to(device) # try decreasing the depth value if there is a memory error

resume = cfg.resume

if not resume:
    print('\nfrom scratch')
    train_epoch_loss = []
    val_epoch_loss = []
    running_train_loss = []
    running_val_loss = []
    epochs_till_now = 0
else:
    ckpt_path = os.path.join(models_dir, cfg.ckpt)
    ckpt = torch.load(ckpt_path)
    print(f'\nckpt loaded: {ckpt_path}')
    model_state_dict = ckpt['model_state_dict']
    model.load_state_dict(model_state_dict)
    model.to(device)
    losses = ckpt['losses']
    running_train_loss = losses['running_train_loss']
    running_val_loss = losses['running_val_loss']
    train_epoch_loss = losses['train_epoch_loss']
    val_epoch_loss = losses['val_epoch_loss']
    epochs_till_now = ckpt['epochs_till_now']

lr = 3 * cfg.lr
# lr = 0.0000296230
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
loss_fn = nn.MSELoss()
log_interval = cfg.log_interval
epochs = cfg.epochs
# 计算总训练步数
total_train_batches = epochs * len(train_loader)
scheduler = CosineAnnealingLR(optimizer, T_max=total_train_batches, eta_min=1e-6)

###
print('\nmodel has {} M parameters'.format(count_parameters(model)))
print(f'\nloss_fn        : {loss_fn}')
print(f'lr             : {lr}')
print(f'epochs_till_now: {epochs_till_now}')
print(f'epochs from now: {epochs}')
###


wt = DWTForward(J=1, mode='zero', wave='haar').to(device)
for epoch in range(epochs_till_now, epochs_till_now + epochs):
    print(f'\n===== EPOCH {epoch + 1}/{epochs_till_now + epochs} =====')
    print('\nTRAINING...')
    epoch_train_start_time = time.time()
    batch_start_time = time.time()
    model.train()
    for batch_idx, (input_images, target_images) in enumerate(train_loader):  # input_images是拼接后的图像
        input_images = input_images.to(device)
        # print(input_images)
        target_images = target_images.to(device)
        # print(target_images.shape)
        # optimizer.zero_grad()
        out,yL1, y_HL1, y_LH1, y_HH1 = model(input_images)

        #target
        yL2, yH2 = wt(target_images)
        yL2 = yL2.to(device)
        y_HL2 = yH2[0][:, :, 0, ::].to(device)
        y_LH2 = yH2[0][:, :, 1, ::].to(device)
        y_HH2 = yH2[0][:, :, 2, ::].to(device)

        loss = loss_fn(out, target_images) + loss_fn(yL1, yL2) + loss_fn(y_HL1, y_HL2) + loss_fn(y_LH1, y_LH2) + loss_fn(y_HH1, y_HH2)
        # loss = loss_fn(out, target_images)
        running_train_loss.append(loss.item())
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if (batch_idx + 1)%log_interval == 0:
            batch_time = time.time() - batch_start_time
            m,s = divmod(batch_time, 60)
            print('train loss @batch_idx {}/{}: {} in {} mins {} secs (per batch)'.format(str(batch_idx+1).zfill(len(str(len(train_loader)))), len(train_loader), loss.item(), int(m), round(s, 2)))
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Current LR: {current_lr:.10f}')  # 查看小数点后更多位
    train_epoch_loss.append(np.array(running_train_loss).mean())

    epoch_train_time = time.time() - epoch_train_start_time
    m,s = divmod(epoch_train_time, 60)
    h,m = divmod(m, 60)
    print('\nepoch train time: {} hrs {} mins {} secs'.format(int(h), int(m), int(s)))

    print('\nVALIDATION...')
    epoch_val_start_time = time.time()
    model.eval()
    with torch.no_grad():
        for batch_idx, (noisy_imgs, imgs) in enumerate(val_loader):

            imgs = imgs.to(device)
            # print(imgs.shape)
            noisy_imgs = noisy_imgs.to(device)
            # print(noisy_imgs.shape)

            out, yL1, y_HL1, y_LH1, y_HH1 = model(noisy_imgs)

            # target
            # yL2, yH2 = wt(imgs)
            # yL2 = yL2.to(device)
            # y_HL2 = yH2[0][:, :, 0, ::].to(device)
            # y_LH2 = yH2[0][:, :, 1, ::].to(device)
            # y_HH2 = yH2[0][:, :, 2, ::].to(device)

            loss = loss_fn(out, imgs)
            # loss = loss_fn(out, imgs) + loss_fn(yL1, yL2) + loss_fn(y_HL1, y_HL2) + loss_fn(y_LH1, y_LH2) + loss_fn(
            #     y_HH1, y_HH2)
            running_val_loss.append(loss.item())

            if (batch_idx + 1)%log_interval == 0:
                print('val loss   @batch_idx {}/{}: {}'.format(str(batch_idx+1).zfill(len(str(len(val_loader)))), len(val_loader), loss.item()))

    val_epoch_loss.append(np.array(running_val_loss).mean())

    epoch_val_time = time.time() - epoch_val_start_time
    m,s = divmod(epoch_val_time, 60)
    h,m = divmod(m, 60)
    print('\nepoch val   time: {} hrs {} mins {} secs'.format(int(h), int(m), int(s)))

    plot_losses(running_train_loss, running_val_loss, train_epoch_loss, val_epoch_loss,  epoch)   

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'losses': {'running_train_loss': running_train_loss, 
                           'running_val_loss': running_val_loss, 
                           'train_epoch_loss': train_epoch_loss, 
                           'val_epoch_loss': val_epoch_loss}, 
                'epochs_till_now': epoch+1}, 
                os.path.join(models_dir, 'model{}.pth'.format(str(epoch + 1).zfill(2))))
    # 在训练代码末尾添加（或在每个epoch保存模型时同时保存权重）
    # torch.save(train_dataset.weights, os.path.join(models_dir, 'trained_weights.pth'))
total_script_time = time.time() - script_time
m, s = divmod(total_script_time, 60)
h, m = divmod(m, 60)
print(f'\ntotal time taken for running this script: {int(h)} hrs {int(m)} mins {int(s)} secs')
  
print('\nFin.')
