
from torch import nn


class FCSTN(nn.Module):

    # 定义一个全连接注意力（FCAttention）模块，接受通道数 channel 和其他参数
    def __init__(self, channel, b=1, gamma=2):  # channel = 64
        super(FCSTN, self).__init__()

    def forward(self, input):
        out = input
        # 将输入特征图与计算得到的权重相乘，得到增强后的输出特征图。
        return out
