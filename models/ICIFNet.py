import torch
import torch.nn as nn
from .resnet import resnet18
from .pvtv2 import pvt_v2_b1
import torch.nn.functional as F
import numpy as np
import math
from torch import nn, einsum
from einops import rearrange


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 

class ConvRelPosEnc(nn.Module):
    """ Convolutional relative position encoding. """
    def __init__(self, Ch, h, window):
        """
        Initialization.
            Ch: Channels per head.
            h: Number of heads.
            window: Window size(s) in convolutional relative positional encoding. It can have two forms:
                    1. An integer of window size, which assigns all attention heads with the same window size in ConvRelPosEnc.
                    2. A dict mapping window size to #attention head splits (e.g. {window size 1: #attention head split 1, window size 2: #attention head split 2})
                       It will apply different window size to the attention head splits.
        """
        super().__init__()

        if isinstance(window, int):
            window = {window: h}                                                         # Set the same window size for all attention heads.
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()            
        
        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1                                                                 # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2         # Determine padding size. Ref: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
            cur_conv = nn.Conv2d(cur_head_split*Ch, cur_head_split*Ch,
                kernel_size=(cur_window, cur_window), 
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),                          
                groups=cur_head_split*Ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x*Ch for x in self.head_splits]

    def forward(self, q, v, size):
        B, h, N, Ch = q.shape
        H, W = size
        assert N == H * W
        # print(q.shape,v.shape)
        # Convolutional relative position encoding.
        # q_img = q                                                             # Shape: [B, h, H*W, Ch].
        # v_img = v                                                             # Shape: [B, h, H*W, Ch].
        # print(q.shape,v.shape)
        v_img = rearrange(v, 'B h (H W) Ch -> B (h Ch) H W', H=H, W=W)               # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)                      # Split according to channels.
        conv_v_img_list = [conv(x) for conv, x in zip(self.conv_list, v_img_list)]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        conv_v_img = rearrange(conv_v_img, 'B (h Ch) H W -> B h (H W) Ch', h=h)          # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].

        EV_hat_img = q* conv_v_img
        # print(EV_hat_img.shape)
        zero = torch.zeros((B, h, 0, Ch), dtype=q.dtype, layout=q.layout, device=q.device)
        EV_hat = torch.cat((zero, EV_hat_img), dim=2)                                # Shape: [B, h, N, Ch].
        # print(EV_hat.shape)
        return EV_hat

class FactorAtt_ConvRelPosEnc(nn.Module):
    """ Factorized attention with convolutional relative position encoding class. """
    def __init__(self, dim, num_heads=8, qkv_bias=False,  proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)                                       # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = ConvRelPosEnc(Ch=dim // num_heads, h=num_heads, window={3:2, 5:3, 7:3})

    def forward(self, q,k,v, size):
        B, N, C = size[0],size[1],size[2]

        # # Generate Q, K, V.
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # Shape: [3, B, h, N, Ch].
        # q, k, v = qkv[0], qkv[1], qkv[2]                                                 # Shape: [B, h, N, Ch].

        # Factorized attention.
        k_softmax = k.softmax(dim=2)                                                     # Softmax on dim N.
        k_softmax_T_dot_v = einsum('b h n k, b h n v -> b h k v', k_softmax, v)          # Shape: [B, h, Ch, Ch].
        factor_att        = einsum('b h n k, b h k v -> b h n v', q, k_softmax_T_dot_v)  # Shape: [B, h, N, Ch].

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=[size[3],size[4]])                                                # Shape: [B, h, N, Ch].

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C)                                           # Shape: [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C].

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x   

class MultiHeadDense(nn.Module):
    def __init__(self, d, bias=False):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))
        if bias:
            raise NotImplementedError()
            self.bias = Parameter(torch.Tensor(d, d))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x:[b, h*w, d]
        b, wh, d = x.size()
        x = torch.bmm(x, self.weight.repeat(b, 1, 1))
        # x = F.linear(x, self.weight, self.bias)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def positional_encoding_2d(self, d_model, height, width):
        """
        reference: wzlxjtu/PositionalEncoding2D

        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        try:
            pe = pe.to(torch.device("cuda:0"))
        except RuntimeError:
            pass
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe

    def forward(self, x):
        raise NotImplementedError()

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000
                         **(torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x,
                             device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y,
                             device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()),
                          dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2),
                          device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)

class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)        
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)
        

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

class MultiHeadCrossAttention(MultiHeadAttention):
    def __init__(self, channelY, channelS, ch_out, drop_rate=0.2,qkv_bias=False):
        super(MultiHeadCrossAttention, self).__init__()
        self.Sconv = nn.Sequential(
            nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.Yconv = nn.Sequential(
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
            
        self.query = MultiHeadDense(channelS, bias=False)
        self.key = MultiHeadDense(channelS, bias=False)
        self.value = MultiHeadDense(channelS, bias=False)

        self.softmax = nn.Softmax(dim=1)
        self.Spe = PositionalEncodingPermute2D(channelS)
        self.Ype = PositionalEncodingPermute2D(channelY)

        self.qkv = nn.Linear(channelS, channelS * 3, bias=qkv_bias)
        self.num_heads = 8
        head_dim = channelS// 8
        self.scale = head_dim ** -0.5
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(channelS,self.num_heads,qkv_bias=qkv_bias,  proj_drop=drop_rate)
        self.residual = Residual(channelS*2, ch_out)
        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, Y, S):
        Sb, Sc, Sh, Sw = S.size()
        Yb, Yc, Yh, Yw = Y.size()
        
        Spe = self.Spe(S)
        S = S + Spe
        S1 = self.Sconv(S)
        S1=S1.reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)

        Ype = self.Ype(Y)
        Y = Y + Ype
        Y1 = self.Yconv(Y).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)

        B, N, C = Y1.shape
        size=[B, N, C,Sh, Sw ]

        qkv_l= self.qkv(Y1)
        qkv_l=qkv_l.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # Shape: [3, B, h, N, Ch].
        q_l, k_l, v_l = qkv_l[0], qkv_l[1], qkv_l[2] 

        qkv_g = self.qkv(S1)
        qkv_g=qkv_g.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # Shape: [3, B, h, N, Ch].
        q_g, k_g, v_g = qkv_g[0], qkv_g[1], qkv_g[2] 

        cur1 = self.factoratt_crpe(q_g, k_l, v_l, size).permute(0, 2, 1).reshape(Yb, Sc, Yh, Yw) 
        cur2 = self.factoratt_crpe(q_l, k_g, v_g, size).permute(0, 2, 1).reshape(Yb, Sc, Yh, Yw)

        fuse = self.residual(torch.cat([cur1,cur2], 1))
        if self.drop_rate > 0:
            return self.dropout(fuse),cur1,cur2
        else:
            return fuse,cur1,cur2

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        # print("++",x.size()[1],self.inp_dim,x.size()[1],self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class decode(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, out_channel,norm_layer=nn.BatchNorm2d):
        super(decode, self).__init__()
        self.conv_d1 = nn.Conv2d(in_channel_down, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(in_channel_left, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channel*2, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(out_channel)

    def forward(self, left, down):
        down_mask = self.conv_d1(down)
        left_mask = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')
            z1 = F.relu(left_mask * down_, inplace=True)
        else:
            z1 = F.relu(left_mask * down, inplace=True)

        if down_mask.size()[2:] != left.size()[2:]:
            down_mask = F.interpolate(down_mask, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_mask * left, inplace=True)

        out = torch.cat((z1, z2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)

def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
        
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1+in_ch2, out_ch)

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2):

        x1 = self.up(x1)
        # input is CHW
        
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        if self.attn_block is not None:
            x2 = self.attn_block(x1, x2)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
                )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x)+self.identity(x))

class ICIFNet(nn.Module):
    def __init__(self,num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False):
        super(ICIFNet, self).__init__()

        self.show_Feature_Maps = False
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.backbone = pvt_v2_b1()  # [64, 128, 320, 512]
        path = './pretrained/pvt_v2_b1.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.final_x = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )

        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )

        self.drop = nn.Dropout2d(drop_rate)

        self.cross2 = MultiHeadCrossAttention(256, 320,ch_out=256, drop_rate=drop_rate/2,qkv_bias=True)
        self.cross3 = MultiHeadCrossAttention(128, 128,ch_out=128, drop_rate=drop_rate/2,qkv_bias=True)
        self.cross4 = MultiHeadCrossAttention(64, 64,ch_out=64, drop_rate=drop_rate/2,qkv_bias=True)

        self.cross2_img2 = MultiHeadCrossAttention(256, 320,ch_out=256, drop_rate=drop_rate/2,qkv_bias=True)
        self.cross3_img2 = MultiHeadCrossAttention(128, 128,ch_out=128, drop_rate=drop_rate/2,qkv_bias=True)
        self.cross4_img2 = MultiHeadCrossAttention(64, 64,ch_out=64, drop_rate=drop_rate/2,qkv_bias=True)

        self.up2 = Up(256, 128, 128, attn=True)
        self.up3 = Up(128, 64, 64, attn=True)

        self.up2_img2 = Up(256, 128, 128, attn=True)
        self.up3_img2 = Up(128, 64, 64, attn=True)
        
        # low-level & high-level
        self.Translayer2_g = BasicConv2d(320,128, 1)
        self.fam43_1 = decode(128,128,128)
        self.Translayer3_g = BasicConv2d(128,64, 1)
        self.fam32_1 = decode(64,64,64)

        self.Translayer2_l = BasicConv2d(320,128, 1)
        self.fam43_2 = decode(128,128,128)
        self.Translayer3_l = BasicConv2d(128,64, 1)
        self.fam32_2 = decode(64,64,64)

        self.Translayer2_g_img2 = BasicConv2d(320,128, 1)
        self.fam43_1_img2 = decode(128,128,128)
        self.Translayer3_g_img2 = BasicConv2d(128,64, 1)
        self.fam32_1_img2 = decode(64,64,64)

        self.Translayer2_l_img2 = BasicConv2d(320,128, 1)
        self.fam43_2_img2 = decode(128,128,128)
        self.Translayer3_l_img2 = BasicConv2d(128,64, 1)
        self.fam32_2_img2 = decode(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        pvt = self.backbone(imgs1)
        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        # c1 = self.drop(c1)
        c2 = self.resnet.layer2(c1)
        # c2 = self.drop(c2)
        c3 = self.resnet.layer3(c2)
        # c3 = self.drop(c3)

        pvt_img2 = self.backbone(imgs2)
        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        # c1_img2 = self.drop(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        # c2_img2 = self.drop(c2_img2)
        c3_img2 = self.resnet.layer3(c2_img2)
        # c3_img2 = self.drop(c3_img2)

        cross_2, curg_2, curl_2 = self.cross2(c3, pvt[2]) # 128 320 320
        cross_3, curg_3, curl_3 = self.cross3(c2, pvt[1]) # 64 128 128
        cross_4, curg_4, curl_4 = self.cross4(c1, pvt[0]) # 32 64 64

        x_up_2 = self.up2(cross_2, cross_3)
        x_up_3 = self.up3(x_up_2, cross_4)

        out3_g = self.fam43_1(curg_3, self.Translayer2_g(curg_2))
        out2_g = self.fam32_1(curg_4, self.Translayer3_g(out3_g))

        out3_l = self.fam43_2(curl_3, self.Translayer2_l(curl_2))
        out2_l = self.fam32_2(curl_4, self.Translayer3_l(out3_l))

        cross_2_img2,curg_2_img2,curl_2_img2=self.cross2_img2(c3_img2,pvt_img2[2])
        cross_3_img2,curg_3_img2,curl_3_img2=self.cross3_img2(c2_img2,pvt_img2[1])
        cross_4_img2,curg_4_img2,curl_4_img2=self.cross4_img2(c1_img2,pvt_img2[0])

        x_up_2_img2 = self.up2_img2(cross_2_img2,cross_3_img2)
        x_up_3_img2 = self.up3_img2(x_up_2_img2,cross_4_img2)                                 #decoder rdio the most

        out3_g_img2 = self.fam43_1_img2(curg_3_img2, self.Translayer2_g_img2(curg_2_img2))
        out2_g_img2 = self.fam32_1_img2(curg_4_img2, self.Translayer3_g_img2(out3_g_img2))

        out3_l_img2 = self.fam43_2_img2(curl_3_img2, self.Translayer2_l_img2(curl_2_img2))
        out2_l_img2 = self.fam32_2_img2(curl_4_img2, self.Translayer3_l_img2(out3_l_img2))

        final2 = self.upsamplex4(torch.abs(out2_g-out2_g_img2))
        final1 = self.upsamplex4(torch.abs(out2_l-out2_l_img2))
        finalx = self.upsamplex4(torch.abs(x_up_3-x_up_3_img2))

        map_x = self.final_2(final2)
        map_1 = self.final_1(final1)
        map_2 = self.final_x(finalx)

        return map_x, map_1, map_2

    def init_weights(self):
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)

        self.Translayer2_g.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_g.apply(init_weights)
        self.fam32_1.apply(init_weights)

        self.Translayer2_l.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_l.apply(init_weights)
        self.fam32_2.apply(init_weights)
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)
        self.cross4.apply(init_weights)
        self.up2.apply(init_weights)
        self.up3.apply(init_weights)

        self.Translayer2_g_img2.apply(init_weights)
        self.fam43_1_img2.apply(init_weights)
        self.Translayer3_g_img2.apply(init_weights)
        self.fam32_1_img2.apply(init_weights)

        self.Translayer2_l_img2.apply(init_weights)
        self.fam43_2_img2.apply(init_weights)
        self.Translayer3_l_img2.apply(init_weights)
        self.fam32_2_img2.apply(init_weights)
        self.cross2_img2.apply(init_weights)
        self.cross3_img2.apply(init_weights)
        self.cross4_img2.apply(init_weights)
        self.up2_img2.apply(init_weights)
        self.up3_img2.apply(init_weights)