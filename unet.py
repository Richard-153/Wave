import torch
from torch import nn
import torch.nn.functional as F
from models.FCSTN import FCSTN
from pytorch_wavelets import DWTForward


def dilate(tensor, kernel, iterations=1):
    """
    PyTorch张量膨胀操作
    输入：
        tensor : 二值掩膜张量 (支持2D/3D/4D)
        kernel : 结构元素 (2D)
        iterations: 膨胀次数
    输出：
        dilated : 膨胀后的张量 (保持原始维度)
    """
    # 维度统一处理
    original_ndim = tensor.ndim
    tensor = tensor.float()  # 确保为浮点型
    
    # 将输入统一为4D: [batch, channel, H, W]
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)               # [1,C,H,W]
    
    # 核处理（必须为2D）
    assert kernel.ndim == 2, "Kernel必须是2D"
    kernel = kernel.to(tensor.device)
    
    # 迭代膨胀
    for _ in range(iterations):
        # 反射填充（四边各填充2像素）
        padded = torch.nn.functional.pad(
            tensor, 
            pad=(2, 2, 2, 2),  # 左右上下填充
            mode='reflect'
        )
        
        # 用分组卷积实现膨胀（最高效的方式）
        groups = padded.size(1)  # 通道数
        unfolded = torch.nn.functional.conv2d(
            padded,
            kernel.view(1, 1, *kernel.shape).repeat(groups, 1, 1, 1),
            stride=1,
            padding=0,
            groups=groups
        )
        
        # 阈值处理（模拟二值膨胀）
        tensor = (unfolded > 0).float()
    
    # 恢复原始维度
    if original_ndim == 2:
        return tensor.squeeze().squeeze().to(torch.uint8)
    elif original_ndim == 3:
        return tensor.squeeze(0).to(torch.uint8)
    return tensor.to(torch.uint8)

class MASK(nn.Module):
    def __init__(self, in_ch, learnable_threshold=True):
        super(MASK, self).__init__()
        self.dwt = DWTForward(J=1, wave='haar')
    def forward(self, x):
        B, C, H, W = x.shape

        # 小波变换
        yL, yH = self.dwt(x)
        hl = yH[0][:, :, 0, :, :]
        lh = yH[0][:, :, 1, :, :]
        hh = yH[0][:, :, 2, :, :]

        # 高频能量计算
        hf_energy = torch.sqrt(lh ** 2 + hl ** 2 + hh ** 2 + 1e-6)
        mask_threshold = 0.3
        hf_energy_normalized = (hf_energy - torch.min(hf_energy)) / (torch.max(hf_energy) - torch.min(hf_energy))
        hf_mask = (hf_energy_normalized > mask_threshold).type(torch.uint8) * 255
        hf_mask_resized = torch.nn.functional.interpolate(hf_mask.float(), size=(H, W), mode='bilinear',align_corners=False).to(torch.uint8)
        # 1. 阈值处理 (THRESH_BINARY等效)
        binary = torch.where(hf_mask_resized > 127, 
                    torch.tensor(255, device=hf_mask_resized.device),
                    torch.tensor(0, device=hf_mask_resized.device))
        # 2. 创建结构元素 (MORPH_RECT等效)
        kernel = torch.ones((5, 5), 
                    dtype=torch.float32, 
                    device=hf_mask_resized.device)  # 确保设备一致
        #print('111111',binary.shape)
        dilated = dilate(binary, kernel)
        # print('1111',dilated.shape)
        #print('22222',dilated.shape)
        return dilated
#xiaorong
# class MASK(nn.Module):
#     def __init__(self, in_ch, learnable_threshold=True):
#         super(MASK, self).__init__()
#         #self.dwt = DWTForward(J=1, wave='haar')
#     def forward(self, x):
#         dilated = x
#         #print('22222',dilated.shape)
#         return dilated

class WT(nn.Module):
    def __init__(self, in_ch):
        super(WT, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]

        return yL,y_HL,y_LH,y_HH

# class downWT1(nn.Module):
#     def __init__(self, in_ch):
#         super(downWT1, self).__init__()
#
#
#     def forward(self, x):
#         # print(x.shape)
#         x =x
#
#         return x
class downWT1(nn.Module):
    def __init__(self, in_ch):
        super(downWT1, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv = nn.Conv2d(in_ch*4, in_ch*64, 1, 1)

    def forward(self, x):
        # print(x.shape)
        y = x
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1) # 4 4 256 320
        # print(x.shape)
        # x = self.conv(x)
        # print(x.shape)
        x = F.interpolate(x, size=(y.shape[2], y.shape[3]), mode='bilinear', align_corners=False) #4 4 512 640
        # print(x.shape)
        x = self.conv(x) # 4 1 512 640
        # print(x.shape)

        return x
#xiaorong2
# class downWT(nn.Module):
#     def __init__(self, in_ch):
#         super(downWT, self).__init__()
#         self.wt = DWTForward(J=1, mode='zero', wave='haar')
#         self.conv = nn.Conv2d(in_ch*4, in_ch*2, 1, 1)
#
#     def forward(self, x):
#         # print(x.shape)
#         x =x
#
#         return x
class downWT(nn.Module):
    def __init__(self, in_ch):
        super(downWT, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv = nn.Conv2d(in_ch*4, in_ch*2, 1, 1)

    def forward(self, x):
        # print(x.shape)
        y = x
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1) # 4 4 256 320
        # print(x.shape)
        # x = self.conv(x)
        # print(x.shape)
        x = F.interpolate(x, size=(y.shape[2], y.shape[3]), mode='bilinear', align_corners=False) #4 4 512 640
        # print(x.shape)
        x = self.conv(x) # 4 1 512 640
        # print(x.shape)

        return x

class upWT(nn.Module):
    def __init__(self, in_ch):
        super(upWT, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv = nn.Conv2d(in_ch*4, in_ch//2, 1, 1)

    def forward(self, x):
        # print(x.shape)
        y = x
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1) # 4 4 256 320
        # print(x.shape)
        # x = self.conv(x)
        # print(x.shape)
        x = F.interpolate(x, size=(y.shape[2]*2, y.shape[3]*2), mode='bilinear', align_corners=False) #4 4 512 640
        # print(x.shape)
        x = self.conv(x) # 4 1 512 640
        # print(x.shape)

        return x
#xiaorong1
# class upWT(nn.Module):
#     def __init__(self, in_ch):
#         super(upWT, self).__init__()
#
#
#     def forward(self, x):
#         # print(x.shape)
#         x = x
#
#         return x

class UNet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            wf=6,
            padding=False,
            batch_norm=False,
            up_mode='upconv'
    ):
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding

        # ==================== 下采样路径 ====================
        # 下采样第一级 (572x572 -> 286x286)
        self.down1 = nn.Sequential(
            UNetConvBlock(in_channels, 2 ** (wf + 0), padding, batch_norm),
            FCSTN(channel=2 ** (wf + 0))
        )

        # 下采样第二级 (286x286 -> 143x143)
        self.down2 = nn.Sequential(
            UNetConvBlock(2 ** (wf + 0), 2 ** (wf + 1), padding, batch_norm),
            FCSTN(channel=2 ** (wf + 1))
        )

        # 下采样第三级 (143x143 -> 71x71)
        self.down3 = nn.Sequential(
            UNetConvBlock(2 ** (wf + 1), 2 ** (wf + 2), padding, batch_norm),
            FCSTN(channel=2 ** (wf + 2))
        )

        # 下采样第四级 (71x71 -> 保持不变)
        self.down4 = nn.Sequential(
            UNetConvBlock(2 ** (wf + 2), 2 ** (wf + 3), padding, batch_norm),
            FCSTN(channel=2 ** (wf + 3))
        )

        # ==================== 上采样路径 ====================
        # 上采样第三级 (71x71 -> 142x142)
        self.up3 = UNetUpBlock(2 ** (wf + 3), 2 ** (wf + 2), up_mode, padding, batch_norm)

        # 上采样第二级 (142x142 -> 284x284)
        self.up2 = UNetUpBlock(2 ** (wf + 2), 2 ** (wf + 1), up_mode, padding, batch_norm)

        # 上采样第一级 (284x284 -> 568x568)
        self.up1 = UNetUpBlock(2 ** (wf + 1), 2 ** (wf + 0), up_mode, padding, batch_norm)

        # 输出层
        self.last = nn.Conv2d(2 ** (wf + 0), n_classes, kernel_size=1)

        # up小波第3级
        self.uwt3 = upWT(in_ch=128)

        # up小波第2级
        self.uwt2 = upWT(in_ch=256)

        # up小波第1级
        self.uwt1 = upWT(in_ch=512)

        # down小波第四级
        self.dwt4 = downWT(in_ch=256)

        # down小波第三级
        self.dwt3 = downWT(in_ch=128)

        # down小波第二级
        self.dwt2 = downWT(in_ch=64)

        # down小波第一级
        self.dwt1 = downWT1(in_ch=1)

        # mask第一级
        self.mask1 = MASK(in_ch=3)

        # mask第二级
        self.mask2 = MASK(in_ch=64)

        # mask第三级
        self.mask3 = MASK(in_ch=128)

        # mask第四级
        self.mask4 = MASK(in_ch=256)

        # 通道数调整1
        self.mask_fusion1 = nn.Conv2d(64 + 1, 64, kernel_size=1)
        # 通道数调整2
        self.mask_fusion2 = nn.Conv2d(128 + 64, 128, kernel_size=1)
        # 通道数调整3
        self.mask_fusion3 = nn.Conv2d(256 + 128, 256, kernel_size=1)
        # 通道数调整4
        self.mask_fusion4 = nn.Conv2d(512 + 256, 512, kernel_size=1)
        # 通道数调整5
        self.mask_fusion5 = nn.Conv2d(512, 256, kernel_size=1)
        # 通道数调整6
        self.mask_fusion6 = nn.Conv2d(256, 128, kernel_size=1)
        # 通道数调整7
        self.mask_fusion7 = nn.Conv2d(128, 64, kernel_size=1)


        self.fcstn = FCSTN(channel=64)

        self.WT = WT(in_ch=64)

    def forward(self, x):
        # ========== 下采样过程 ==========
        # x # 4 1 512 640
        # 第一级
        x1 = self.down1(x)  # [4,64,512,640] 此处x1是conv+fcstn 64
        downwt1 = self.dwt1(x) #4 64 512 640
        # print(downwt1.shape)
        x1_1 = downwt1 + x1 #4 64 512 640
        mask1 = self.mask1(x) #4 1 512 640
        x1_2 = torch.cat([x1_1, mask1], dim=1)  # [4, 65, 512, 640]
        #print('333',x1_2.shape)
        x1_3 = self.mask_fusion1(x1_2)  # [4, 64, 512, 640]
        x1_pool = F.max_pool2d(x1_3, 2)  #4 64 256 320

        # 第二级
        x2 = self.down2(x1_pool)# [4 128,256,320]
        downwt2 = self.dwt2(x1_pool) #4 64 256 320
        x2_1 = downwt2 + x2 # 4 128 256 320
        mask2 = self.mask2(x1_pool) # 4 1 256 320
        #print('444',mask2.shape)
        x2_2 = torch.cat([x2_1,mask2],dim=1) # 4 129 256 320
        #print('555',x2_2.shape)
        x2_3 = self.mask_fusion2(x2_2) # 4 128 256 320
        x2_pool = F.max_pool2d(x2_3, 2)  # [4 128,128,160]

        # 第三级
        x3 = self.down3(x2_pool)  #4 256 128 160
        downwt3 = self.dwt3(x2_pool) #4 256 128 160
        x3_1 = downwt3 + x3 # 4 256 128 160
        mask3 = self.mask3(x2_pool) # 4 1 128 160
        x3_2 = torch.cat([x3_1,mask3],dim=1) #4 257 128 160
        #print('111',x3_2.shape) #384
        x3_3 = self.mask_fusion3(x3_2) # 4 256 128 160
        #print('222',x3_3.shape) # 256
        x3_pool = F.max_pool2d(x3_3, 2)  # 4 256 64 80

        # 第四级（无池化）
        x4 = self.down4(x3_pool)  # 4 512 64 80
        downwt4 = self.dwt4(x3_pool) # 4 512 64 80
        x4_1 = downwt4 + x4 # 4 512 64 80
        mask4 = self.mask4(x3_pool) # 4 1 64 80
        x4_2 = torch.cat([x4_1,mask4],dim=1) # 4 513 64 80
        #print('333',x4_2.shape) # 768
        x4_3 = self.mask_fusion4(x4_2) # 4 512 64 80

        # ========== 上采样过程 ==========
        # 上采样第三级
        u3 = self.up3(x4_3, x3)  # 4 256 128 160
        upwt1 = self.uwt1(x4_3) # 4 256 128 160
        u3_1 = u3 + upwt1 #4 256 128 160
        #print('00',u3_1)
        upmask4 = self.mask4(x3_3) #4 1 128 160
        #print('0',upmask4.shape)#256
        # u3_1 = F.interpolate(u3_1, size=(135, 240), mode='bilinear', align_corners=False)
        u3_2 = torch.cat([upmask4,u3_1],dim=1) #4 257 128 160
        #print('444',u3_2.shape) # 512 
        u3_3 = self.mask_fusion5(u3_2) # 4 256 128 160

        # 上采样第二级
        u2 = self.up2(u3_3, x2)  # 4 128 256 320
        upwt2 = self.uwt2(u3_3) # 4 128 256 320
        u2_1 = u2 + upwt2 # 4 128 256 320
        upmask3 = self.mask3(x2_3) #4 1 236 320
        u2_2 = torch.cat([upmask3,u2_1],dim=1) # 4 129 236 320
        #print('000',u2_2.shape)
        u2_3 = self.mask_fusion6(u2_2) # 4 128 236 320

        # 上采样第一级
        u1 = self.up1(u2_3, x1)  # 4 64 512 640
        upwt3 = self.uwt3(u2_3) #4 64 512 640
        u1_1 = u1 + upwt3 # 4 64 512 640
        upmask2 = self.mask2(x1_3) # 4 1 512 640
        u1_2 = torch.cat([upmask2,u1_1],dim=1) # 4 65 512 640
        u1_3 = self.mask_fusion7(u1_2) # 4 64 512 640

        # 最终输出
        output = self.fcstn(u1_3)
        output = self.last(output)  # 4 1 512 640

        yL, y_HL, y_LH, y_HH = self.WT(output)

        return output, yL, y_HL, y_LH, y_HH



class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        block.append(nn.Dropout2d(p=0.15)) # edited
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        # print(x.shape)
        # print(self.block)
        out = self.block(x)
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )
        self.att = FCSTN(channel=out_size * 2)  # 拼接后通道数是两倍
        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.att(out)
        out = self.conv_block(out)

        return out
