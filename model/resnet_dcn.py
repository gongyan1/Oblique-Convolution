import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from mmcv.ops import DeformConv2dPack as DCN # 这里多加了个mmcv

from .eca_module import eca_layer

import math
__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}
def conv3x3(in_planes, out_planes, stride=1,groups=1):
    "3x3 convolution with padding"
    if stride!=1:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,groups=groups)
    else:
        return nn.Conv2d(in_planes, out_planes,  kernel_size=3, stride=1, padding=1,bias=False,groups=groups)

    

        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes,stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.eca = eca_layer(planes, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class BasicBlock_(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_, self).__init__()
        self.conv1 = conv3x3(inplanes, planes,stride,groups=4)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,groups=4)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        # self.eca = eca_layer(planes, 3)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.eca(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        
        self.eca = eca_layer(planes * 4, k_size)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DCN_out(nn.Module):
    def __init__(self,planes,kernel_size):
        super(DCN_out, self).__init__()
        self.kernel_size=kernel_size
        self.conv = DCN(planes,planes,kernel_size=[kernel_size,1])
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        #self.conv_ = nn.Conv2d(planes,planes,)
    def forward(self,x):
        #x=torch.cat([x,x[:,:,:-1,:]],dim=2)
        x=F.pad(x,[0,0,self.kernel_size//2,self.kernel_size//2])
        x=self.conv(x)
        return self.relu(self.bn1(x))

class CBRP(nn.Module):
    def __init__(self):
        super(CBRP, self).__init__()
        self.cbr = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    def forward(self,x):
        return self.cbr(x)
class CBRP_(nn.Module):
    def __init__(self):
        super(CBRP_, self).__init__()
        self.cbr = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    def forward(self,x):
        return self.cbr(x)
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, img_channel=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        '''
        self.cbr1 = nn.Sequential(nn.Conv2d(img_channel, 64, kernel_size=7, stride=2, padding=3),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))'''
        self.layer1 = self._make_layer(block, 64, layers[0])
        #self.layer1_out = DCN_out(16,9)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        #self.layer2_out = DCN_out(32,7)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer3_out = DCN_out(64,5)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.layer4_out = DCN_out(128,3)
        
        
        self.conv1 = nn.Sequential(DCN(64,64,kernel_size=(217,1)),nn.BatchNorm2d(64),nn.ReLU())
        self.conv2 = nn.Sequential(DCN(128,128,kernel_size=(109,1)),nn.BatchNorm2d(128),nn.ReLU())
        self.conv3 = nn.Sequential(DCN(256,256,kernel_size=(55,1)),nn.BatchNorm2d(256),nn.ReLU())
        self.conv4 = nn.Sequential(DCN(512,512,kernel_size=(27,1)),nn.BatchNorm2d(512),nn.ReLU())
        
        self.c_pool1 = nn.Sequential(nn.MaxPool2d(1),nn.Conv2d(64,64,kernel_size=1),nn.BatchNorm2d(64),nn.ReLU())
        self.c_pool2 = nn.Sequential(nn.MaxPool2d(1),nn.Conv2d(128,128,kernel_size=1),nn.BatchNorm2d(128),nn.ReLU())
        self.c_pool3 = nn.Sequential(nn.MaxPool2d(1),nn.Conv2d(256,256,kernel_size=1),nn.BatchNorm2d(256),nn.ReLU())
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,padding=0,bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #x = self.cbr1(x)
        f1 = self.layer1(x)
#         print(f1.size)
        f1=F.pad(f1,[0,0,108,108])
        f1=self.conv1(f1)
        f1 = self.c_pool1(f1)
        #print(f1.size())
        print("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
        f2 = self.layer2(f1)
        f2=F.pad(f2,[0,0,54,54])
        f2=self.conv2(f2)
        f2 = self.c_pool2(f2)
        
        
        f3 = self.layer3(f2)
        f3=F.pad(f3,[0,0,27,27])
        f3=self.conv3(f3)
        f3 = self.c_pool3(f3)
        f4 = self.layer4(f3)
        
#         f4=F.pad(f4,[0,0,13,13])
#         f4=self.conv4(f4)
        
        return f2, f3, f4
    
    
    
    
    
    
class ResNet_(nn.Module):
    def __init__(self, block, layers, num_classes=10, img_channel=3):
        self.inplanes = 64
        super(ResNet_, self).__init__()
        '''
        self.cbr1 = nn.Sequential(nn.Conv2d(img_channel, 64, kernel_size=7, stride=2, padding=3),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))'''
        self.layer1 = self._make_layer(block, 64, layers[0])
        #self.layer1_out = DCN_out(16,9)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        #self.layer2_out = DCN_out(32,7)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer3_out = DCN_out(64,5)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.layer4_out = DCN_out(128,3)
        
        self.conv1 = nn.Sequential(DCN(1,64,kernel_size=(217,3)),nn.BatchNorm2d(64),nn.ReLU())
        self.conv2 = nn.Sequential(DCN(1,128,kernel_size=(109,3)),nn.BatchNorm2d(128),nn.ReLU())
        self.conv3 = nn.Sequential(DCN(1,256,kernel_size=(55,3)),nn.BatchNorm2d(256),nn.ReLU())
        self.conv4 = nn.Sequential(DCN(1,512,kernel_size=(27,3)),nn.BatchNorm2d(512),nn.ReLU())
        self.drop=nn.Dropout(p=0.1)
#         self.c_pool1 = nn.Sequential(nn.MaxPool2d(1),nn.Conv2d(64,64,kernel_size=1),nn.BatchNorm2d(64),nn.ReLU())
#         self.c_pool2 = nn.Sequential(nn.MaxPool2d(1),nn.Conv2d(128,128,kernel_size=1),nn.BatchNorm2d(128),nn.ReLU())
#         self.c_pool3 = nn.Sequential(nn.MaxPool2d(1),nn.Conv2d(256,256,kernel_size=1),nn.BatchNorm2d(256),nn.ReLU())
        
#         self.avg_pool_1 = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(64, 64, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.sigmoid_1 = nn.Sigmoid()
        
#         self.avg_pool_2 = nn.AdaptiveAvgPool2d(1)
#         self.fc2 = nn.Conv2d(128, 128, 1, bias=False)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.sigmoid_2 = nn.Sigmoid()
        
#         self.avg_pool_3 = nn.AdaptiveAvgPool2d(1)
#         self.fc3 = nn.Conv2d(256, 256, 1, bias=False)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.sigmoid_3 = nn.Sigmoid()
        
#         self.avg_pool_4 = nn.AdaptiveAvgPool2d(1)
#         self.fc4 = nn.Conv2d(512, 512, 1, bias=False)
#         self.bn4 = nn.BatchNorm2d(512)
#         self.sigmoid_4 = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,padding=0,bias=False,groups=4),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #x = self.cbr1(x)
        f1 = self.layer1(x)
        print('f11',f1.size())
        # print('f11',f1.shape)



        f1_1, _ = torch.max(f1, dim=1, keepdim=True)
        f1_1 = F.pad(f1_1, [1, 1, 108, 108])
        f1_1 = self.conv1(f1_1)
        f1 = f1 * f1_1
        print('f12',f1.shape)
    
        # f1_1, _ = torch.max(f1, dim=1, keepdim=True)
        # b=torch.cat((f1_1[:,:,108:,:],f1_1),dim=2)
        # b=torch.cat((b,f1_1[:,:,:108,:]),dim=2)
        # b=F.pad(b, [1, 1, 0, 0])
        # f1_1 = self.conv1(b)
        # f1 = f1 * f1_1
     
        f2 = self.layer2(f1)
        print('f21',f2.shape)
        
        # f2_1, _ = torch.max(f2, dim=1, keepdim=True)
        # b=torch.cat((f2_1[:,:,54:,:],f2_1),dim=2)
        # b=torch.cat((b,f2_1[:,:,:54,:]),dim=2)
        # b=F.pad(b, [1, 1, 0, 0])
       
        f2_1, _ = torch.max(f2, dim=1, keepdim=True)
        f2_1 = F.pad(f2_1, [1, 1, 54, 54])
        f2_1 = self.conv2(f2_1)
        f2 = f2 * f2_1
        # print('f22',f2.shape)
    
        f3 = self.layer3(f2)
        # print('f31',f3.shape)
        f3_1, _ = torch.max(f3, dim=1, keepdim=True)
        f3_1 = F.pad(f3_1, [1, 1, 27, 27])
        f3_1 = self.conv3(f3_1)
        f3 = f3 * f3_1
        # print('f32',f3.shape)
        # f3_1, _ = torch.max(f3, dim=1, keepdim=True)
        # b=torch.cat((f3_1[:,:,27:,:],f3_1),dim=2)
        # b=torch.cat((b,f3_1[:,:,:27,:]),dim=2)
        # b=F.pad(b, [1, 1, 0, 0])
        # f3_1 = self.conv3(b)
        # f3 = f3 * f3_1
      
        f4 = self.layer4(f3)
        # print('f41',f4.shape)
        f4_1, _ = torch.max(f4, dim=1, keepdim=True)
        f4_1 = F.pad(f4_1, [1, 1, 13, 13])
        f4_1 = self.conv4(f4_1)
        f4 = f4 * f4_1
        # print('f42',f4.shape)
        # f4_1, _ = torch.max(f4, dim=1, keepdim=True)
        # b=torch.cat((f4_1[:,:,14:,:],f4_1),dim=2)
        # b=torch.cat((b,f4_1[:,:,:13,:]),dim=2)
        # b=F.pad(b, [1, 1, 0, 0])
        # f4_1 = self.conv4(b)
      
        # f4 = f4 * f4_1
    
        return f2, f3, f4
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]), strict=False)
    return model
def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]), strict=False)
    return model
def resnet34_(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet_(BasicBlock_, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]), strict=False)
    return model
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]), strict=False)
    return model


def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]), strict=False)
    return model
def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model
'''
ccc=CBRP().cuda()
a=resnet34(pretrained=False).cuda()
b=torch.randn([1,3,288,800]).cuda()
b=a(ccc(b))'''