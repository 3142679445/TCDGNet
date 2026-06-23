# import torchsummary
# import torchvision
# from torch import nn
#
# from testnet.day1222 import testnet11
# from testnet.testmodule import xyASPP4, xyASPP5
#
#
# # from model.DenseNetUNet import DenseUNet
# # from model.day1215 import Net3
#
# # model = Net3(n_channels=3, n_classes=1, kernel_size=9, number=16, imgsize=512, att='FEM')
# class DeformConv(nn.Module):
#
#     def __init__(self, in_channels, out_channel,groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
#         super(DeformConv, self).__init__()
#
#         self.offset_net = nn.Conv2d(in_channels=in_channels,
#                                     out_channels=2 * kernel_size[0] * kernel_size[1],
#                                     kernel_size=kernel_size,
#                                     padding=padding,
#                                     stride=stride,
#                                     dilation=dilation,
#                                     bias=True)
#
#         self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
#                                                         out_channels=out_channel,
#                                                         kernel_size=kernel_size,
#                                                         padding=padding,
#                                                         groups=groups,
#                                                         stride=stride,
#                                                         dilation=dilation,
#                                                         bias=False)
#
#     def forward(self, x):
#         offsets = self.offset_net(x)
#         out = self.deform_conv(x, offsets)
#         return out
# model=testnet11(128)
# torchsummary.summary(model,input_size=(3,512,512),device='cpu')
# total = sum([param.nelement() for param in model.parameters()])
# # 精确地计算：1MB=1024KB=1048576字节
# print('Number of parameter: % .4fM' % (total / 1e6))
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

from net.APFnet import PMC_Net
from net.AttentionUNet import AttentionUNet
from net.CTCNet import CTCNet
from net.DeepLabV3Plus import DeepLab
from net.UNet import UNet
from net.UNetPlusPlus import UNetPlusPlus
from net.deepcrack import DeepCrack

supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
}
# model=CTCNet()
# model=DeepCrack()
# model=UNet()
# model = UNetPlusPlus()
# model = DeepLab(num_classes=1, backbone='resnet')
model = PMC_Net()
# model = AttentionUNet()
model.cuda(1).eval()
shape=(3, 512, 512)
input = (torch.randn((1, *shape), device=next(model.parameters()).device))
print(len(input))
for i in input:
    print(i.shape)
params = parameter_count(model)[""]
Gflops, unsupported = flop_count(model=model, inputs=input, supported_ops=supported_ops)

del model, input
# return sum(Gflops.values()) * 1e9
# return f"params {params} GFLOPs {sum(Gflops.values())}"
print("---"*20)
print(params*1e-6)
print("---"*20)
print(sum(Gflops.values()))
print(Gflops.values())