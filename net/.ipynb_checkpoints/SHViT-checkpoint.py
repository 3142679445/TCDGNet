import torch
import itertools

from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite
from torch import nn

from GatedBottConv import CAGU
from WTconv import TDWT
from attention import HybridSEBlock


class GroupNorm(torch.nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0)
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x



class DWA(torch.nn.Module):
    def __init__(self, dim, r=0.5, qk_dim=16,dilations=(1,2)):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = int(dim*r)
        self.WTdim=(dim-self.pdim)//2
        self.inproj=Conv2d_BN(dim-self.pdim,self.WTdim,ks=1)
        self.WTconv0=nn.Sequential(
            TDWT(self.WTdim,self.WTdim,kernel_size=3,dilation=dilations[0]),
            nn.BatchNorm2d(self.WTdim),
            nn.ReLU()
        )
        self.WTconv1 = nn.Sequential(
            TDWT(self.WTdim, self.WTdim, kernel_size=3, dilation=dilations[1]),
            nn.BatchNorm2d(self.WTdim),
            nn.ReLU()
        )
        self.se=HybridSEBlock('avg',self.WTdim*2,2)
        self.conv=nn.Sequential(
            nn.Conv2d(self.WTdim*2,self.WTdim*2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(self.WTdim*2),
        )

        self.pre_norm = GroupNorm(self.pdim)
        self.qkv = Conv2d_BN(self.pdim, qk_dim * 2 + self.pdim)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            dim, dim, bn_weight_init=0))

    def forward(self, x):
        B, C, H, W = x.shape
        #x1 = r*C = pdim          x2 = dim-pdim = (1-r)*C
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)

        x2=self.inproj(x2)
        x2_0=self.WTconv0(x2)
        x2_1=self.WTconv1(x2_0)
        x2=self.se(torch.cat([x2_0,x2_1],dim=1))
        x2=self.conv(x2)
        x = self.proj(torch.cat([x1, x2], dim=1))
        return x

class DGWA(torch.nn.Module):
    def __init__(self, indim,outdim,r=0.5, qk_dim=16):
        super().__init__()
        self.conv = Residual(Conv2d_BN(indim, outdim, 3, 1, 1, groups=1, bn_weight_init=0))
        self.mixer = Residual(DWA(outdim, r,qk_dim))
        self.ffn = Residual(FFN(outdim, int(outdim * 2)))
        self.gate = CAGU(outdim)
    def forward(self, x):
        y = self.conv(x)
        y = self.gate(x,y) + x
        y = self.gate(self.mixer(y),y) + y
        return self.ffn(y)