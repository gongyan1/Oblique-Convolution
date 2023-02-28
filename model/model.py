import numpy as np
import torch
from .cc_attention import CrissCrossAttention
#from .backbone import resnet
from .resnet_dcn import resnet18,resnet34
from .oblique_conv import oblique_conv_res
from .Affinetransformation import Affinetrans,Affinetrans2
import math
from inplace_abn import InPlaceABN, InPlaceABNSync
from torch.nn import functional as F
class conv_bn_relu(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size,
            stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class parsingNet(torch.nn.Module):
    def __init__(self, size=(288, 800), pretrained=True, backbone='34', cls_dim=(37, 10, 4), use_aux=False):
        super(parsingNet, self).__init__()

        self.size = size
        self.w = size[0]
        self.h = size[1]
        self.cls_dim = cls_dim # (num_gridding, num_cls_per_lane, num_of_lanes)
        # num_cls_per_lane is the number of row anchors
        self.use_aux = use_aux
        self.total_dim = np.prod(cls_dim)
    
        # input : nchw,
        # output: (w+1) * sample_rows * 4
        #self.model = resnet(backbone, pretrained=pretrained)
        self.model1 = oblique_conv_res(1.80759513,'18',pretrained=True)
        self.model2 = oblique_conv_res(0.6381599, '18', pretrained=True)
        self.model3 = oblique_conv_res(39.31144689, '18', pretrained=True)
        self.conv2_ = conv_bn_relu(192, 128, 1)
        self.conv3_ = conv_bn_relu(384, 256, 1)
        self.convfea_ = conv_bn_relu(768, 512, 1)
        if self.use_aux:
            self.aux_header2 = torch.nn.Sequential(
                conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_header3 = torch.nn.Sequential(
                conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_header4 = torch.nn.Sequential(
                conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_combine = torch.nn.Sequential(
                conv_bn_relu(384, 256, 3,padding=2,dilation=2),
                conv_bn_relu(256, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=4,dilation=4),
                torch.nn.Conv2d(128, cls_dim[-1] + 1,1)
                # output : n, num_of_lanes+1, h, w
            )
            initialize_weights(self.aux_header2,self.aux_header3,self.aux_header4,self.aux_combine)
    
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim),
        )
    
        self.pool = torch.nn.Conv2d(512,8,1) if backbone in ['34','18'] else torch.nn.Conv2d(2048,8,1)
        # 1/32,2048 channel
        # 288,800 -> 9,40,2048
        # (w+1) * sample_rows * 4
        # 37 * 10 * 4
        initialize_weights(self.cls)
    
    def forward(self, x):
        # n c h w - > n 2048 sh sw
        # -> n 2048
        #x2,x3,fea = self.model(x)
        x2_1, x3_1, fea_1 = self.model1(x)
        #print(x2_1.shape)
        x2_2, x3_2, fea_2 = self.model2(x)
        x2_3, x3_3, fea_3 = self.model3(x)
        x2 = self.conv2_(torch.cat([x2_1, x2_2, x2_3],dim=1))
        x3 = self.conv3_(torch.cat([x3_1, x3_2, x3_3], dim=1))
        fea = self.convfea_(torch.cat([fea_1, fea_2, fea_3], dim=1))
    
        #print(x2_1.shape,x2_2.shape,x2_3.shape)
        #print(x3_1.shape, x3_2.shape, x3_3.shape)
        #print(fea_1.shape, fea_2.shape, fea_3.shape)
        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3 = torch.nn.functional.interpolate(x3,scale_factor = 2,mode='bilinear')
            x4 = self.aux_header4(fea)
            x4 = torch.nn.functional.interpolate(x4,scale_factor = 4,mode='bilinear')
            aux_seg = torch.cat([x2,x3,x4],dim=1)
            aux_seg = self.aux_combine(aux_seg)
        else:
            aux_seg = None
    
        fea = self.pool(fea).view(-1, 1800)
    
        group_cls = self.cls(fea).view(-1, *self.cls_dim)
    
        if self.use_aux:
            return group_cls, aux_seg
    
        return group_cls

class parsingNet1(torch.nn.Module):
    def __init__(self, size=(288, 800), backbone='18', cls_dim=(37, 10, 4),k=[1.80759513,0.6381599,39.31144689], use_aux=False):
        super(parsingNet1, self).__init__()
        self.k=k
        self.size = size
        self.w = size[0]
        self.h = size[1]
        self.cls_dim = cls_dim # (num_gridding, num_cls_per_lane, num_of_lanes)
        # num_cls_per_lane is the number of row anchors
        self.use_aux = use_aux
        self.total_dim = np.prod(cls_dim)

        # input : nchw,
        # output: (w+1) * sample_rows * 4
        #self.model = resnet(backbone, pretrained=pretrained)
        self.model = resnet34(pretrained=False)
        self.conv2_ = conv_bn_relu(192, 128, 1)
        self.conv3_ = conv_bn_relu(384, 256, 1)
        self.convfea_ = conv_bn_relu(768, 512, 1)
        if self.use_aux:
            self.aux_header2 = torch.nn.Sequential(
                conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_header3 = torch.nn.Sequential(
                conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_header4 = torch.nn.Sequential(
                conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_combine = torch.nn.Sequential(
                conv_bn_relu(384, 256, 3,padding=2,dilation=2),
                conv_bn_relu(256, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=4,dilation=4),
                torch.nn.Conv2d(128, cls_dim[-1] + 1,1)
                # output : n, num_of_lanes+1, h, w
            )
            initialize_weights(self.aux_header2,self.aux_header3,self.aux_header4,self.aux_combine)
    
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim),
        )
    
        self.pool = torch.nn.Conv2d(512,8,1) if backbone in ['34','18'] else torch.nn.Conv2d(2048,8,1)
        # 1/32,2048 channel
        # 288,800 -> 9,40,2048
        # (w+1) * sample_rows * 4
        # 37 * 10 * 4
        initialize_weights(self.cls)
    
    def forward(self, x):
        # n c h w - > n 2048 sh sw
        # -> n 2048
        #x2,x3,fea = self.model(x)
    
        x2_list=[]
        x3_list=[]
        fea_list=[]
        padding_csize = (math.sqrt(x.shape[2] ** 2 + x.shape[3] ** 2)) // 1
        top_pad = np.int((padding_csize - x.shape[0]) // 2 )
        left_pad = np.int((padding_csize - x.shape[1]) // 2 )
        x = F.pad(x, [left_pad, left_pad, top_pad, top_pad])
        for i in range(len(self.k)):
            angle=math.atan(self.k[i])
            theta1 = torch.tensor([
                [math.cos(angle), math.sin(-angle), 0],
                [math.sin(angle), math.cos(angle), 0]
            ], dtype=torch.float).unsqueeze(0)
            theta1 = theta1.expand(x.shape[0], 2, 3).cuda()
            theta1T = torch.tensor([
                [math.cos(-angle), math.sin(angle), 0],
                [math.sin(-angle), math.cos(-angle), 0]
            ], dtype=torch.float).unsqueeze(0)
            theta1T = theta1T.expand(x.shape[0], 2, 3).cuda()
            theta2 = torch.tensor([
                [math.cos(-angle), math.sin(angle), 0],
                [math.sin(-angle), math.cos(-angle), 0]
            ], dtype=torch.float).unsqueeze(0)
            theta2 = theta2.expand(x.shape[0], 2, 3).cuda()
            theta2T = torch.tensor([
                [math.cos(angle), math.sin(-angle), 0],
                [math.sin(angle), math.cos(angle), 0]
            ], dtype=torch.float).unsqueeze(0)
            theta2T = theta2T.expand(x.shape[0], 2, 3).cuda()
            # 正转
            T = F.affine_grid(theta1, x.shape)
            x01 = F.grid_sample(x, T)
            x2_1, x3_1, fea_1 = self.model(x01)
    
            #
            T2_1 = F.affine_grid(theta1T, x2_1.shape)
            x2_1 = F.grid_sample(x2_1, T2_1)
            x2_1 = x2_1[:, :, left_pad // (2 ** 3):-1+(-left_pad // (2 ** 3)), top_pad // (2 ** 3):-top_pad // (2 ** 3)]
            T3_1 = F.affine_grid(theta1T, x3_1.shape)
            x3_1 = F.grid_sample(x3_1, T3_1)
            x3_1 = x3_1[:, :, left_pad // (2 ** 4):-left_pad // (2 ** 4), top_pad // (2 ** 4):-top_pad // (2 ** 4)]
            Tfea_1 = F.affine_grid(theta1T, fea_1.shape)
            fea_1 = F.grid_sample(fea_1, Tfea_1)
            fea_1 = fea_1[:, :, left_pad // (2 ** 5):-left_pad // (2 ** 5), top_pad // (2 ** 5):-top_pad // (2 ** 5)]
            x2_list.append(x2_1)
            x3_list.append(x3_1)
            fea_list.append(fea_1)
    
            # 反转
            T = F.affine_grid(theta2, x.shape)
            x02 = F.grid_sample(x, T)
            x2_2, x3_2, fea_2 = self.model(x02)
            T2_2 = F.affine_grid(theta2T, x2_2.shape)
            x2_2 = F.grid_sample(x2_2, T2_2)
            x2_2 = x2_2[:, :, left_pad // (2 ** 3):-1+(-left_pad // (2 ** 3)), top_pad // (2 ** 3):-top_pad // (2 ** 3)]
            T3_2 = F.affine_grid(theta2T, x3_2.shape)
            x3_2 = F.grid_sample(x3_2, T3_2)
            x3_2 = x3_2[:, :, left_pad // (2 ** 4):-left_pad // (2 ** 4), top_pad // (2 ** 4):-top_pad // (2 ** 4)]
            Tfea_2 = F.affine_grid(theta2T, fea_2.shape)
            fea_2 = F.grid_sample(fea_2, Tfea_2)
            fea_2 = fea_2[:, :, left_pad // (2 ** 5):-left_pad // (2 ** 5), top_pad // (2 ** 5):-top_pad // (2 ** 5)]
    
            x2_list.append(x2_2)
            x3_list.append(x3_2)
            fea_list.append(fea_2)
    
        x2 = self.conv2_(torch.cat(x2_list,dim=1))
        x3 = self.conv3_(torch.cat(x3_list, dim=1))
        fea = self.convfea_(torch.cat(fea_list, dim=1))
    
        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3 = torch.nn.functional.interpolate(x3,scale_factor = 2,mode='bilinear')
            x4 = self.aux_header4(fea)
            x4 = torch.nn.functional.interpolate(x4,scale_factor = 4,mode='bilinear')
            aux_seg = torch.cat([x2,x3,x4],dim=1)
            aux_seg = self.aux_combine(aux_seg)
        else:
            aux_seg = None
    
        fea = self.pool(fea).view(-1, 1800)
        group_cls = self.cls(fea).view(-1, *self.cls_dim)
        if self.use_aux:
            return group_cls, aux_seg
        return group_cls

def initialize_weights(*models):
    for model in models:
        real_init_weights(model)
def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)


class parsingNet2(torch.nn.Module):
    def __init__(self, size=(288, 800), pretrained=True, backbone='34', cls_dim=(37, 10, 4), use_aux=False):
        super(parsingNet2, self).__init__()
        self.size = size
        self.w = size[0]
        self.h = size[1]
        self.cls_dim = cls_dim # (num_gridding, num_cls_per_lane, num_of_lanes)
        # num_cls_per_lane is the number of row anchors
        self.use_aux = use_aux
        self.total_dim = np.prod(cls_dim)
        # input : nchw,
        # output: (w+1) * sample_rows * 4
        #self.model = resnet(backbone, pretrained=pretrained)
        self.model = Affinetrans('self',[1.26126253,782.22222222,408.88888889,92.35312561],3,[128,256,512])#[1.80759513,0.6381599,39.31144689]
        if self.use_aux:
            self.aux_header2 = torch.nn.Sequential(
                conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18','self'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_header3 = torch.nn.Sequential(
                conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18','self'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_header4 = torch.nn.Sequential(
                conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18','self'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_combine = torch.nn.Sequential(
                conv_bn_relu(384, 256, 3,padding=2,dilation=2),
                conv_bn_relu(256, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=4,dilation=4),
                torch.nn.Conv2d(128, cls_dim[-1] + 1,1)
                # output : n, num_of_lanes+1, h, w
            )
            initialize_weights(self.aux_header2,self.aux_header3,self.aux_header4,self.aux_combine)
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim),
        )
        self.pool = torch.nn.Conv2d(512,8,1) if backbone in ['34','18','self'] else torch.nn.Conv2d(2048,8,1)
        # 1/32,2048 channel
        # 288,800 -> 9,40,2048
        # (w+1) * sample_rows * 4
        # 37 * 10 * 4
        initialize_weights(self.cls)
    def forward(self, x):
        # n c h w - > n 2048 sh sw
        # -> n 2048
        x2,x3,fea = self.model(x)
        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3 = torch.nn.functional.interpolate(x3,scale_factor = 2,mode='bilinear')
            x4 = self.aux_header4(fea)
            x4 = torch.nn.functional.interpolate(x4,scale_factor = 4,mode='bilinear')
            aux_seg = torch.cat([x2,x3,x4],dim=1)
            aux_seg = self.aux_combine(aux_seg)
        else:
            aux_seg = None
        fea = self.pool(fea).view(-1, 1800)
        #print(fea.shape)
        group_cls = self.cls(fea).view(-1, *self.cls_dim)
        if self.use_aux:
            return group_cls, aux_seg
        return group_cls
class RCCAModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = torch.nn.Sequential(torch.nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   InPlaceABNSync(inter_channels))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = torch.nn.Sequential(torch.nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   InPlaceABNSync(inter_channels))

        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_channels),
            torch.nn.Dropout2d(0.1),
            torch.nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
    
    def forward(self, x, recurrence=1):
        output = self.conva(x)
        #print('output',output.shape)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)
    
        output = self.bottleneck(torch.cat([x, output], 1))
        return output

class parsingNet3(torch.nn.Module):
    def __init__(self, size=(288, 800), pretrained=True, backbone='34', cls_dim=(37, 10, 4), use_aux=False):
        super(parsingNet3, self).__init__()


       # self.head = RCCAModule(512, 128, 512)
    
        self.size = size
        self.w = size[0]
        #print('w',self.w) #288
        self.h = size[1]
       # print('h',self.h)  #800
        self.cls_dim = cls_dim # (num_gridding, num_cls_per_lane, num_of_lanes)
        # num_cls_per_lane is the number of row anchors
       # print('cls_dim',self.cls_dim) #101,56,4
        self.use_aux = use_aux
        self.total_dim = np.prod(cls_dim)
       # print('total_dim',self.total_dim) #22624
        # input : nchw,
        # output: (w+1) * sample_rows * 4
        #self.model = resnet(backbone, pretrained=pretrained)#res34_
        self.model = Affinetrans2('res34_',[0.9362591854472577, 3.21191038651291, 1.318984580691007, 0.41028590825592043],3,[128,256,512])#[1.80759513,0.6381599,39.31144689]
        if self.use_aux:
            self.aux_header2 = torch.nn.Sequential(
                conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18','self'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_header3 = torch.nn.Sequential(
                conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18','self'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_header4 = torch.nn.Sequential(
                conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18','self'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_combine = torch.nn.Sequential(
                conv_bn_relu(384, 256, 3,padding=2,dilation=2),
                conv_bn_relu(256, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=4,dilation=4),
                torch.nn.Conv2d(128, cls_dim[-1] + 1,1)
                # output : n, num_of_lanes+1, h, w
            )
            initialize_weights(self.aux_header2,self.aux_header3,self.aux_header4,self.aux_combine)
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim),
        )
        self.pool = torch.nn.Conv2d(512,8,1) if backbone in ['34','18','self'] else torch.nn.Conv2d(2048,8,1) # 加RCCA是128，否则是512
        # 1/32,2048 channel
        # 288,800 -> 9,40,2048
        # (w+1) * sample_rows * 4
        # 37 * 10 * 4
        initialize_weights(self.cls)
    def forward(self, x):
        # n c h w - > n 2048 sh sw
        # -> n 2048
        x2,x3,fea = self.model(x)
        # print('x2',x2.shape)#([2, 128, 36, 100])
        # print('x3',x3.shape) #([2, 256, 18, 50])
        # print('fea',fea.shape)# ([2, 512, 9, 25])
        if self.use_aux:
            x2 = self.aux_header2(x2)
            #print('x2u',x2.shape)# ([2, 128, 36, 100])
            x3 = self.aux_header3(x3)
           # print('x3u1',x3.shape) #([2, 128, 18, 50])
            x3 = torch.nn.functional.interpolate(x3,scale_factor = 2,mode='bilinear')
           # print('x3u2',x3.shape) #([2, 128, 36, 100])
            x4 = self.aux_header4(fea)
           # print('x4u',x4.shape)#([2, 128, 9, 25])
            x4 = torch.nn.functional.interpolate(x4,scale_factor = 4,mode='bilinear')
           # print('x4u2',x4.shape) #([2, 128, 36, 100])
            aux_seg = torch.cat([x2,x3,x4],dim=1)
          #  print('aux_seg',aux_seg.shape)#([2, 384, 36, 100])
            aux_seg = self.aux_combine(aux_seg)
           # print('aux_seg1',aux_seg.shape) #([2, 5, 36, 100])
        else:
            aux_seg = None
        #print('fea',fea.shape)
       # fea = self.head(fea, 2)
        #print('feax',fea.shape)
        fea = self.pool(fea).view(-1, 1800)
      #  print('fea1',fea.shape) #([2, 1800])
        #print(fea.shape)
        group_cls = self.cls(fea).view(-1, *self.cls_dim)
        #print('group_cls',group_cls.shape)
        #print(group_cls[0])
      #  print('group_cls',group_cls.shape)
        if self.use_aux:
            return group_cls, aux_seg
        return group_cls

'''
pars=parsingNet3(use_aux=True).cuda()
input=torch.randn([8,3,288,800]).cuda()
output=pars(input)


288,800
torch.Size([4, 128, 36, 100]) torch.Size([4, 128, 36, 100]) torch.Size([4, 128, 36, 100])
torch.Size([4, 256, 18, 50]) torch.Size([4, 256, 18, 50]) torch.Size([4, 256, 18, 50])
torch.Size([4, 512, 9, 25]) torch.Size([4, 512, 9, 25]) torch.Size([4, 512, 9, 25])
'''