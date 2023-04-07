import os
import sys
from models.deform_conv import *
# import re
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
import copy
from functools import partial, reduce
import numpy as np
import itertools
import math
from collections import OrderedDict
import sys

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ReflectionPad2d(paddings)(images)
    return images
def pixelUnshuffle(x, r=1):
    b, c, h, w = x.size()
    out_chl = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    x = x.view(b, c, out_h, r, out_w, r)
    out = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_chl, out_h, out_w)

    return out
def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()
    
    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, nf, kernel_size=3, stride=1, padding=1, dilation=1, act='relu'):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))

        return out + x
class ContrastiveBaesdFusion(nn.Module):
    def __init__(self, in_channels):
        super(ContrastiveBaesdFusion, self).__init__()
        self.up_sample_1 = nn.Sequential(*[nn.ConvTranspose2d(in_channels,in_channels,6,stride=2,padding=2),nn.PReLU()])
        self.up_sample_2 = nn.Sequential(*[nn.ConvTranspose2d(in_channels,in_channels,6,stride=2,padding=2),nn.PReLU()])
        self.down_sample = nn.Sequential(*[nn.Conv2d(in_channels,in_channels,6,stride=2,padding=2),nn.PReLU()])
        self.encoder_1=ResidualBlock(in_channels,kernel_size=3,act='relu')
        self.encoder_2=ResidualBlock(in_channels,kernel_size=3,act='relu')
        self.encoder_3=ResidualBlock(in_channels,kernel_size=3,act='relu')
        self.conv_cat = nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self,lr,lr_up,warp_ref):
        res_up=self.encoder_1(lr_up-warp_ref)
        enhance_warp_ref=warp_ref+res_up
        
        enhance_down=self.down_sample(enhance_warp_ref)
        res_down=self.encoder_2(lr-enhance_down)
        res_lr=self.encoder_3(enhance_down-lr)
        res_down_up=self.up_sample_1(res_down)
        
        out_warp=res_down_up+enhance_warp_ref
        out_lr=self.up_sample_2(lr+res_lr)
        out=self.act(self.conv_cat(torch.cat([out_warp, out_lr ], dim=1)))           
        return out
    
class LRTExtract(nn.Module):
    def __init__(self,filter_num=64):
        super(LRTExtract,self).__init__()    
        #LRTExtract
        self.conv_hgap   =   nn.Conv2d(1,1,stride=[1,1],kernel_size=[1,1])
        self.conv_wgap   =   nn.Conv2d(1,1,stride=[1,1],kernel_size=[1,1])
        self.conv_cgap   =   nn.Conv2d(filter_num,filter_num,stride=[1,1],kernel_size=[1,1])

    def forward(self,input):
        gap_Height  =   torch.mean(torch.mean(input,dim=1,keepdim=True),dim=3,keepdim=True) #N,1,H,1
        gap_Weight  =   torch.mean(torch.mean(input,dim=1,keepdim=True),dim=2,keepdim=True) #N,1,1,W
        gap_Channel =   torch.mean(torch.mean(input,dim=2,keepdim=True),dim=3,keepdim=True) #N,C,1,1
        
        convHeight_GAP  =   F.sigmoid(self.conv_hgap(gap_Height))
        convWeight_GAP  =   F.sigmoid(self.conv_wgap(gap_Weight))
        convChannel_GAP  =   F.sigmoid(self.conv_cgap(gap_Channel))
        
        vecConHeight_GAP    =   convHeight_GAP.view([convHeight_GAP.size()[0],1,convHeight_GAP.size()[2]]).permute(0,2,1)  #N,H,1
        vecConWeight_GAP    =   convWeight_GAP.view([convWeight_GAP.size()[0],1,convWeight_GAP.size()[3]])  #N,1,W
        vecConChannel_GAP   =   convChannel_GAP.view([convChannel_GAP.size()[0],convChannel_GAP.size()[1],1]).permute(0,2,1)   #N,1,c
        
        matHWmulT=torch.matmul(vecConHeight_GAP,vecConWeight_GAP )#N,H,W
        
        matHWmulT_size=matHWmulT.size()

        vecHWmulT   =   matHWmulT.view([matHWmulT_size[0],matHWmulT_size[1]*matHWmulT_size[2],1])   #N,H*W,1
        matHWCmulT  =   torch.matmul(vecHWmulT,vecConChannel_GAP)    #N,H*W,C
        recon=matHWCmulT.view(input.size()[0],input.size()[1],input.size()[2],input.size()[3])
        return recon

class LowRankTensor(nn.Module):
    def __init__(self,in_channel,filter_num=64,rank=4):
        super(LowRankTensor,self).__init__()    
        block=functools.partial(LRTExtract,filter_num=filter_num)
        self.rank=rank
        self.n_layers=2*self.rank
        layers=[]
        for _ in range(self.n_layers):
            layers.append(block())
        self.lrtelayer=nn.Sequential(*layers)
        #encoding
        self.conv2=nn.Conv2d(filter_num,filter_num,stride=[1,1],kernel_size=[3,3])
        self.enc=nn.Conv2d(filter_num,filter_num,stride=[1,1],kernel_size=[3,3])
        self.con_cat=nn.Conv2d(filter_num*self.rank,filter_num,stride=[1,1],kernel_size=[3,3])
        self.dec=nn.Conv2d(filter_num,in_channel,stride=[1,1],kernel_size=[3,3])
        self.relu=nn.ReLU(inplace=True)
    def encoding(self,input):
        temp_input=self.same_padding(input,[3,3],[1,1],[1,1])
        temp_input=self.relu(self.conv2(temp_input))
        out=self.same_padding(temp_input,[3,3],[1,1],[1,1])
        out=self.enc(out)
        return out
    def same_padding(self,input, ksizes, strides, rates):
        assert len(input.size()) == 4
        _, _, rows, cols = input.size()
        out_rows = (rows + strides[0] - 1) // strides[0]
        out_cols = (cols + strides[1] - 1) // strides[1]
        effective_k_row = (ksizes[0] - 1) * rates[0] + 1
        effective_k_col = (ksizes[1] - 1) * rates[1] + 1
        padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
        padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
        # Pad the input
        padding_top = int(padding_rows / 2.)
        padding_left = int(padding_cols / 2.)
        padding_bottom = padding_rows - padding_top
        padding_right = padding_cols - padding_left
        paddings = (padding_left, padding_right, padding_top, padding_bottom)
        out = torch.nn.ZeroPad2d(paddings)(input)
        return out
    def ResBlock(self,input,index):
        xup     =   self.lrtelayer[index](input)
        xres    =   input   -   xup
        xdn     =   self.lrtelayer[index+1](xres)
        xdn     =   xdn +   xres
        return xup, xdn
    
    def LRTE(self,input):
        xup,xdn=self.ResBlock(input,0)
        temp_xup=xdn
        output=xup
        for i in range(2,self.n_layers,2):
            temp_xup,temp_xdn=self.ResBlock(temp_xup,i)
            xup=xup+temp_xup
            output=torch.cat([output,xup],1)
            temp_xup=temp_xdn
        return output
    
    def forward(self,input):
        enc=self.encoding(input)
        low_rank_map_cat=self.LRTE(enc)
        low_rank_map_cat=self.same_padding(low_rank_map_cat,[3,3],[1,1],[1,1])
        low_rank_map=self.con_cat(low_rank_map_cat)
        feature_lowrank=torch.mul(enc,low_rank_map)
        minxed_feature=feature_lowrank+input
        return minxed_feature   
    
class ImageRegistration(nn.Module):
    """Pre-activation version of the BasicBlock + Conditional instance normalization"""
    expansion = 1

    def __init__(self, in_channel):
        super(ImageRegistration, self).__init__()
        self.norm = nn.InstanceNorm2d(in_channel, affine=False)
        self.conv_shared = nn.Sequential(nn.Conv2d(in_channel*2 , in_channel, 3, 1, 1, bias=True),nn.ReLU(inplace=True))
        self.conv_gamma = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True)
        self.conv_beta = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True)

        
        self.conv_gamma.weight.data.zero_()
        self.conv_beta.weight.data.zero_()
        self.conv_gamma.bias.data.zero_()
        self.conv_beta.bias.data.zero_()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, sr,ref):
        style = self.conv_shared(torch.cat([sr, ref], dim=1))
        gamma = self.conv_gamma(style)
        beta = self.conv_beta(style)
        b, c, h, w = sr.size()
        sr = sr.view(b, c, h * w)
        sr_mean = torch.mean(sr, dim=-1, keepdim=True).unsqueeze(3)
        sr_std = torch.std(sr, dim=-1, keepdim=True).unsqueeze(3)
        out = self.norm(ref)

        out = (sr_std + gamma) * out + (beta+sr_mean)
        out = F.leaky_relu(out, negative_slope=0.2)
        out = self.conv1(out)
        out = (sr_std + gamma) * out + (beta+sr_mean)
        out = self.conv2(F.leaky_relu(out, negative_slope=0.2))
    
        out += ref

        return out
#编码器
class Encoder(nn.Module):
    def __init__(self, in_chl, nf, n_blks=[1, 1, 1], act='relu'):
        super(Encoder, self).__init__()

        block = functools.partial(ResidualBlock, nf=nf)

        self.conv_L1 = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)
        self.blk_L1 = make_layer(block, n_layers=n_blks[0])

        self.conv_L2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.blk_L2 = make_layer(block, n_layers=n_blks[1])

        self.conv_L3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.blk_L3 = make_layer(block, n_layers=n_blks[2])

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        fea_L1 = self.blk_L1(self.act(self.conv_L1(x)))
        fea_L2 = self.blk_L2(self.act(self.conv_L2(fea_L1)))
        fea_L3 = self.blk_L3(self.act(self.conv_L3(fea_L2)))
        return [fea_L1, fea_L2, fea_L3]


#DRAM   可替换为cross non-local里那种

class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center
class ACM(nn.Module):
    def __init__(self, low_in_channels, high_in_channels, key_channels, value_channels, out_channels=None, scale=1, norm_type=None,psp_size=(1,3,6,8)):
        super(ACM, self).__init__()
        self.scale = scale
        self.in_channels = low_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = high_in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key_img = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_key_lang = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_query_img = nn.Sequential(
            nn.Conv2d(in_channels=high_in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_query_lang = nn.Sequential(
            nn.Conv2d(in_channels=high_in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_value_img = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.f_value_lang = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W_img = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
        self.W_lang = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                               kernel_size=1, stride=1, padding=0)

        self.psp = PSPModule(psp_size)
        self.relu = nn.PReLU()
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.conv = nn.Conv2d(self.in_channels*2, self.out_channels,
                              kernel_size=3, stride=1, padding=1, bias=False)
        # nn.init.constant_(self.W.weight, 0)
        # nn.init.constant_(self.W.bias, 0)

    def forward(self, img_feats, lang_feats):
        batch_size, h, w = img_feats.size(0), img_feats.size(2), img_feats.size(3)

        value_img = self.psp(self.f_value_img(img_feats)).permute(0, 2, 1)
        value_lang = self.psp(self.f_value_lang(lang_feats)).permute(0, 2, 1)

        query_img = self.f_query_img(img_feats).view(batch_size, self.key_channels, -1).permute(0, 2, 1)
        query_lang = self.f_query_lang(lang_feats).view(batch_size, self.key_channels, -1).permute(0, 2, 1)

        key_img = self.psp(self.f_key_img(img_feats))
        key_lang = self.psp(self.f_key_lang(lang_feats))

        sim_map_img = torch.matmul(query_img, key_img)
        sim_map_img = (self.key_channels ** -.5) * sim_map_img

        sim_map_lang = torch.matmul(query_lang, key_lang)
        sim_map_lang = (self.key_channels ** -.5) * sim_map_lang

        sim_map = F.softmax(sim_map_img+sim_map_lang, dim=-1)

        context_img = torch.matmul(sim_map, value_img)
        context_img = context_img.permute(0, 2, 1).contiguous()
        context_img = context_img.view(batch_size, self.value_channels, *img_feats.size()[2:])
        context_img = self.W_img(context_img)

        context_lang = torch.matmul(sim_map, value_lang)
        context_lang = context_lang.permute(0, 2, 1).contiguous()
        context_lang = context_lang.view(batch_size, self.value_channels, *img_feats.size()[2:])
        context_lang = self.W_lang(context_lang)
        
        out = torch.cat([context_img, context_lang], dim=1)
        out =lang_feats.mul(self.relu((self.conv(out))))
        return out
class Contrastive_offset_generator(nn.Module):
    def __init__(self,nf):
        super(Contrastive_offset_generator, self).__init__()
        block = functools.partial(ResidualBlock, nf=nf)
        self.blk_sr = make_layer(block, n_layers=2)
        self.blk_ref=make_layer(block, n_layers=2)
        self.contrastive=ACM(low_in_channels=nf, high_in_channels=nf, key_channels=nf//16,
                                    value_channels=nf//16, out_channels=nf)
        self.blk_out=make_layer(block, n_layers=2)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
    def forward(self,sr,ref):
        sr_feature=self.blk_sr(sr)
        ref_feature=self.blk_ref(ref)
        contrastive_ref=self.contrastive(sr_feature,ref_feature)
        out=self.relu(self.blk_out(contrastive_ref))
        return out
class Decoder(nn.Module):
    def __init__(self, nf, out_chl, n_blks=[1, 1, 1, 1, 1, 1]):
        super(Decoder, self).__init__()

        block = functools.partial(ResidualBlock, nf=nf)

        self.conv_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.blk_L3 = make_layer(block, n_layers=n_blks[0])

        self.conv_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.blk_L2 = make_layer(block, n_layers=n_blks[1])

        self.conv_L1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.blk_L1 = make_layer(block, n_layers=n_blks[2])

        self.merge_warp_x1 = nn.Conv2d(nf * 3, nf, 3, 1, 1, bias=True)
        self.blk_x1 = make_layer(block, n_blks[3])

        self.fusion_x2 = ContrastiveBaesdFusion(nf)
        self.blk_x2 = make_layer(block, n_blks[4])

        self.fusion_x4 = ContrastiveBaesdFusion(nf)
        self.blk_x4 = make_layer(functools.partial(ResidualBlock, nf=64), n_blks[5])

        self.conv_out = nn.Conv2d(64, out_chl, 3, 1, 1, bias=True)

        self.act = nn.ReLU(inplace=True)
        self.registration_x1=ImageRegistration(nf)
        self.registration_x2=ImageRegistration(nf)
        self.registration_x4=ImageRegistration(nf)
    
        self.contras_x1=ACM(low_in_channels=nf, high_in_channels=nf, key_channels=nf//2,
                                    value_channels=nf//2, out_channels=nf)
        self.contras_x2=ACM(low_in_channels=nf, high_in_channels=nf, key_channels=nf//2,
                                    value_channels=nf//2, out_channels=nf)
        self.contras_x4=ACM(low_in_channels=nf, high_in_channels=nf, key_channels=nf//2,
                                    value_channels=nf//2, out_channels=nf)
        self.LRTE_x2=LowRankTensor(nf)
        self.LRTE_x4=LowRankTensor(nf)


    def forward(self, lr_l, warp_ref_l):
        fea_L3 = self.act(self.conv_L3(lr_l[2]))
        fea_L3 = self.blk_L3(fea_L3)

        fea_L3_up = F.interpolate(fea_L3, scale_factor=2, mode='bilinear', align_corners=False)

        fea_L2 = self.act(self.conv_L2(torch.cat([fea_L3_up, lr_l[1]], dim=1)))

        fea_L2 = self.blk_L2(fea_L2)
        fea_L2_up = F.interpolate(fea_L2, scale_factor=2, mode='bilinear', align_corners=False)
        fea_L1 = self.act(self.conv_L1(torch.cat([fea_L2_up, lr_l[0]], dim=1)))
        fea_L1 = self.blk_L1(fea_L1)
                
        warp_ref_x1=self.registration_x1(fea_L1,warp_ref_l[2])
        attention_x1=self.contras_x1(fea_L1,warp_ref_x1)
        fea_x1 = self.act(self.merge_warp_x1(torch.cat([fea_L1, warp_ref_x1,attention_x1], dim=1)))
        
                
        fea_x1 = self.blk_x1(fea_x1)
        fea_x1_up = F.interpolate(fea_x1, scale_factor=2, mode='bilinear', align_corners=False)
        
   
        warp_ref_x2=self.registration_x2(fea_x1_up,warp_ref_l[1])
        
        attention_x2=self.contras_x2(fea_x1_up,warp_ref_x2)
        
        fea_x2=self.fusion_x2(fea_x1,warp_ref_x2,attention_x2)
        fea_x2 = self.blk_x2(fea_x2)
        fea_x2=self.LRTE_x2(fea_x2)
        fea_x2_up = F.interpolate(fea_x2, scale_factor=2, mode='bilinear', align_corners=False)

        
        warp_ref_x4=self.registration_x4(fea_x2_up,warp_ref_l[0])
        attention_x4=self.contras_x4(fea_x2_up,warp_ref_x4)
        
        fea_x4=self.fusion_x4(fea_x2,warp_ref_x4,attention_x4)
       
        fea_x4 = self.blk_x4(fea_x4)
        fea_x4=self.LRTE_x4(fea_x4)
        out = self.conv_out(fea_x4)
        return out

class Contrastive_offset_generator(nn.Module):
    def __init__(self,nf):
        super(Contrastive_offset_generator, self).__init__()
        block = functools.partial(ResidualBlock, nf=nf)
        self.blk_sr = make_layer(block, n_layers=2)
        self.blk_ref=make_layer(block, n_layers=2)
        self.contrastive=ACM(low_in_channels=nf, high_in_channels=nf, key_channels=nf//16,
                                    value_channels=nf//16, out_channels=nf)
        self.blk_out=make_layer(block, n_layers=2)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
    def forward(self,sr,ref):
        sr_feature=self.blk_sr(sr)
        ref_feature=self.blk_ref(ref)
        contrastive_ref=self.contrastive(sr_feature,ref_feature)
        out=self.relu(self.blk_out(contrastive_ref))
        return out
class DeformableConvBlock(nn.Module):
    def __init__(self,nf,scale):
        super(DeformableConvBlock, self).__init__()
        self.offset_generator = Contrastive_offset_generator(nf)
        self.deformconv = DeformConv2d(inc=nf, outc=nf, kernel_size=scale,stride=scale,
                                       padding=1)
    def forward(self, lr, ref):
        offset = self.offset_generator(lr, ref)
        output = self.deformconv(x=ref, offset=offset)
        return output



class FeatureMatching(nn.Module):
    def __init__(self, ksize=3, stride=1):
        super(FeatureMatching, self).__init__()
        self.ksize = ksize
        self.stride = stride  
    def forward(self, query, key):
        #input query and key, return matching
        shape_query = query.shape
        query = extract_image_patches(query, ksizes=[self.ksize, self.ksize], strides=[self.stride,self.stride], rates=[1, 1], padding='same') 
        w = extract_image_patches(key, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride], rates=[1, 1], padding='same')
        w = w.permute(0, 2, 1)   
        w = F.normalize(w, dim=2) # [N, Hr*Wr, C*k*k]
        query  = F.normalize(query, dim=1) # [N, C*k*k, H*W]
        y = torch.bmm(w, query) #[N, Hr*Wr, H*W]
        relavance_maps, hard_indices = torch.max(y, dim=1) #[N, H*W]   
        relavance_maps = relavance_maps.view(shape_query[0], 1, shape_query[2], shape_query[3])      
        return relavance_maps,  hard_indices
    
class CADN(nn.Module):
    def __init__(self, args):
        super(CADN, self).__init__()
        in_chl = args.input_nc
        nf = args.nf
        n_blks = [4, 4, 4]
        n_blks_dec = [2, 2, 2, 12, 8, 4]

        self.scale = args.sr_scale
        self.num_nbr = args.num_nbr
        self.psize = 3
        self.lr_block_size = 8
        self.ref_down_block_size = 1.5
        self.dilations = [1, 2, 3]
        self.rank=args.low_rank_tensor
        self.enc = Encoder(in_chl=in_chl, nf=nf, n_blks=n_blks)
        self.decoder = Decoder(nf, in_chl, n_blks=n_blks_dec)
        self.criterion = nn.L1Loss(reduction='mean')
        self.conv=nn.Conv2d(nf,3,stride=[1,1],kernel_size=[1,1])
        self.feature_match = FeatureMatching(ksize=3,stride=1)
        self.deform_x1=DeformableConvBlock(nf,4)
        self.deform_x2=DeformableConvBlock(nf,4)
        self.deform_x4=DeformableConvBlock(nf,4)
        self.weight_init(scale=0.1)

    def weight_init(self, scale=0.1):
        for name, m in self.named_modules():
            classname = m.__class__.__name__
            if classname == 'DCN':
                continue
            elif classname == 'Conv2d' or classname == 'ConvTranspose2d':
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.5 * math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('BatchNorm') != -1:
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif classname.find('Linear') != -1:
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data = torch.ones(m.bias.data.size())

        for name, m in self.named_modules():
            classname = m.__class__.__name__
            if classname == 'ResidualBlock':
                m.conv1.weight.data *= scale
                m.conv2.weight.data *= scale
            if classname == 'ImageRegistration':
                # initialization
                m.conv_gamma.weight.data.zero_()
                m.conv_beta.weight.data.zero_()
    def warp(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N,  ]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.clone().view(views).expand(expanse)
        return torch.gather(input, dim, index)
    def transfer(self,lr,ref,ksize,stride,index):
        shape_out = list(lr.size())   # b*c*h*w
        unfolded_ref = extract_image_patches(ref, ksizes=[ksize,ksize],  strides=[stride,stride], rates=[1, 1], padding='same') # [N, C*k*k, L]
        warpped_ref = self.warp(unfolded_ref, 2, index)
        warpped_features = F.fold(warpped_ref, output_size=(shape_out[2]*4, shape_out[3]*4), kernel_size=(ksize, ksize), padding=0, stride=stride) 
        return warpped_features    
    def forward(self, lr, ref, ref_down, gt=None,coarse = False):
        _, _, h, w = lr.size()

        lrsr = F.interpolate(lr, scale_factor=self.scale, mode='bicubic')
        #fea_lr_l [0]:9, 64, 40, 40;[1]:9, 64, 20, 20;[2]:9, 64, 10, 10
        #fea_ref_l [0]:9, 64, 160, 160;[1]:9, 64, 80, 80;[2]:9, 64, 40, 40 
        fea_lr_l = self.enc(lr)
        fea_reflr_l = self.enc(ref_down)
        fea_ref_l = self.enc(ref)
        #计算置信度图和序列图C和P 
        confidence_map,  index_map = self.feature_match(fea_lr_l[0],  fea_reflr_l[0])
        #coarse warp
        
        warp_ref_l_x1 = self.transfer(fea_lr_l[2],fea_ref_l[2], 1,1,index_map)    
        warp_ref_l_x2 = self.transfer(fea_lr_l[1],fea_ref_l[1], 2,2,index_map)    
        warp_ref_l_x4 = self.transfer(fea_lr_l[0],fea_ref_l[0], 4,4,index_map) 
        fealr_x2=F.interpolate(fea_lr_l[0], scale_factor=2, mode='bicubic')   
        fealr_x4=F.interpolate(fealr_x2, scale_factor=2, mode='bicubic') 
        warp_ref_l_x1=self.deform_x1(fea_lr_l[0],warp_ref_l_x1)
        warp_ref_l_x2=self.deform_x2(fealr_x2,warp_ref_l_x2)
        warp_ref_l_x4=self.deform_x2(fealr_x4,warp_ref_l_x4)
        
        warp_ref_l = [warp_ref_l_x4, warp_ref_l_x2, warp_ref_l_x1]
        


        out = self.decoder(fea_lr_l, warp_ref_l)
        out = out + lrsr
        
        if gt is not None:
            L1_loss = self.criterion(out, gt)
            loss_dict = OrderedDict(L1=L1_loss)
            return loss_dict
        else:
            return out




if __name__ == "__main__":
    pass