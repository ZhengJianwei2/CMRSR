import torch.nn as nn
import torch
from torchvision.models import vgg as vgg
import torch.nn.functional as F
class FeatureMatching(nn.Module):
    def __init__(self, ksize=3,  scale=2, stride=1):
        super(FeatureMatching, self).__init__()
        self.ksize = ksize
        self.stride = stride  
        self.scale = scale
        match0 = [nn.Conv2d(in_channels=128, out_channels=16, kernel_size=1, stride=1, bias=True),nn.LeakyReLU(0.2, inplace=True)] 
        match0=nn.Sequential(*match0)

        state_dict = torch.load('/home/qwe/oneflow/v2-single/MASA-SR-main/vgg19-dcbb9e9d.pth')
        vgg19 = vgg.vgg19(pretrained=False)
        vgg19.load_state_dict(state_dict)
        vgg_pretrained_features = vgg19.features
        
        self.feature_extract = torch.nn.Sequential()
        
        for x in range(10):
            self.feature_extract.add_module(str(x), vgg_pretrained_features[x])
            
        self.feature_extract.add_module('map', match0)
        
   
        for param in self.feature_extract.parameters():
            param.requires_grad = True
            

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 , 0.224, 0.225 )
        self.sub_mean = MeanShift(1, vgg_mean, vgg_std) 
        self.avgpool = nn.AvgPool2d((self.scale,self.scale),(self.scale,self.scale))            


    def forward(self, query, key):
        #input query and key, return matching
    
        query = self.sub_mean(query)
        query  = F.interpolate(query, scale_factor=self.scale, mode='bicubic',align_corners=True)
        # there is a pooling operation in self.feature_extract
        query = self.feature_extract(query)
        shape_query = query.shape
        query = extract_image_patches(query, ksizes=[self.ksize, self.ksize], strides=[self.stride,self.stride], rates=[1, 1], padding='same') 
      

        key = self.avgpool(key)
        key = self.sub_mean(key)
        key  = F.interpolate(key, scale_factor=self.scale, mode='bicubic',align_corners=True)
        # there is a pooling operation in self.feature_extract
        key = self.feature_extract(key)
        shape_key = key.shape
        w = extract_image_patches(key, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride], rates=[1, 1], padding='same')
    

        w = w.permute(0, 2, 1)   
        w = F.normalize(w, dim=2) # [N, Hr*Wr, C*k*k]
        query  = F.normalize(query, dim=1) # [N, C*k*k, H*W]
        y = torch.bmm(w, query) #[N, Hr*Wr, H*W]
        relavance_maps, hard_indices = torch.max(y, dim=1) #[N, H*W]   
        relavance_maps = relavance_maps.view(shape_query[0], 1, shape_query[2], shape_query[3])      

        return relavance_maps,  hard_indices



class CoarseWarp(nn.Module):
    def __init__(self,  ksize=3, stride=1):
        super(CoarseWarp, self).__init__()
        self.ksize = ksize
        self.stride = stride

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

    def forward(self, lr,ref, index_map ):
        # value there can be features or image in ref view

        # b*c*h*w
        shape_out = list(lr.size())   # b*c*h*w
 
        unfolded_ref = extract_image_patches(ref, ksizes=[self.ksize, self.ksize],  strides=[self.stride,self.stride], rates=[1, 1], padding='same') # [N, C*k*k, L]
        warpped_ref = self.warp(unfolded_ref, 2, index_map)
        warpped_features = F.fold(warpped_ref, output_size=(shape_out[2]*4, shape_out[3]*4), kernel_size=(self.ksize, self.ksize), padding=0, stride=self.stride) 
        return warpped_features      


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)

        self.weight.requires_grad = False
        self.bias.requires_grad = False   
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