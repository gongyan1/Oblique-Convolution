import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from .resnet_dcn import resnet18,resnet34,resnet50,CBRP,resnet34_,CBRP_
from .densenet import densenet
from .try_conv import model_self,model_self1
from torch.autograd import Variable
'''
d=5
size=[288,800]
size_=[size[0]/2**5,size[1]/2**5]
diameter = math.sqrt(size_[0]**2+size_[1]**2)
print(diameter)
t=math.ceil((diameter-size_[0])/2)
l=math.ceil((diameter-size_[1])/2)
print(t,l)'''

class Affinetrans(nn.Module):
    def __init__(self,model,k_list,down,channel_list):
        super(Affinetrans, self).__init__()
        self.k_list=k_list
        self.CBRP=CBRP()
        if model=='res34':
            self.model = resnet34()
        elif model=='self':
            self.model = model_self()
        self.down=down#下采样倍数
        self.c1 = nn.Sequential(nn.Conv2d(channel_list[0] * len(k_list), channel_list[0], 1, 1),nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(channel_list[1] * len(k_list), channel_list[1], 1, 1),nn.ReLU())
        self.c3 = nn.Sequential(nn.Conv2d(channel_list[2] * len(k_list), channel_list[2], 1, 1),nn.ReLU())
    def affine(self,angle,feature):
        b,c,h,w = feature.shape
        diameter = math.sqrt((h/(2**self.down))**2+(w/(2**self.down))**2)
        top_pad = math.ceil((diameter-(h/(2**self.down)))/2)
        left_pad = math.ceil((diameter-(w/(2**self.down)))/2)
        feature=F.pad(feature, [left_pad*(2**self.down), left_pad*(2**self.down), top_pad*(2**self.down), top_pad*(2**self.down)])
        #开始仿射变换
        #旋转theta
        theta = torch.tensor([
            [math.cos(angle), math.sin(-angle), 0],
            [math.sin(angle), math.cos(angle), 0]
        ], dtype=torch.float).unsqueeze(0)
        theta = theta.expand(b, 2, 3).cuda()
        #转回theta
        theta_ = torch.tensor([
            [math.cos(-angle), math.sin(angle), 0],
            [math.sin(-angle), math.cos(-angle), 0]
        ], dtype=torch.float).unsqueeze(0)
        theta_ = theta_.expand(b, 2, 3).cuda()
        #进行仿射变换
        T = F.affine_grid(theta, feature.shape)
        feature = F.grid_sample(feature,T)
        #模型提取特征
        f2,f3,f4 = self.model(feature)
        o_list = [f2,f3,f4]
        #转回并去除pad
        output_list=[]
        down_list=sorted(range(len(o_list)),reverse=True)
        for i in range(len(o_list)):
            T = F.affine_grid(theta_, o_list[i].shape)
            output = F.grid_sample(o_list[i],T)
            output = output[:,:,top_pad*(2**down_list[i]):-1*top_pad*(2**down_list[i]),left_pad*(2**down_list[i]):-1*left_pad*(2**down_list[i])]
            output_list.append(output)
        return output_list
    def affine_once(self,k,feature):
        angle = math.atan(k)
        feature1_list = self.affine(angle,feature)
        feature2_list = self.affine(-1*angle,feature)
        return [torch.cat([feature1_list[i],feature2_list[i]],dim=1) for i in range(len(feature1_list))]
    def forward(self,feature):
        feature=self.CBRP(feature)
        f_list=[]
        for k in self.k_list:
            f_list.append(self.affine_once(k,feature))
        out_list=[]
        for m in range(len(f_list[0])):
            out_l=[]
            for n in range(len(f_list)):
                out_l.append(f_list[n][m])
                #print(f_list[n][m].shape)
            out_list.append(torch.cat(out_l,dim=1))
        #print(self.c1(out_list[0]).shape,self.c2(out_list[1]).shape,self.c3(out_list[2]).shape)
        return self.c1(out_list[0]),self.c2(out_list[1]),self.c3(out_list[2])
''''''
class Affinetrans2(nn.Module):
    def __init__(self,model,k_list,down,channel_list):
        super(Affinetrans2, self).__init__()
        self.k_list=k_list
        self.CBRP=CBRP_()
        if model=='res34':
            self.model = resnet34()
        elif model=='res34_':
            self.model = resnet34_()
        elif model=='self':
            self.model = model_self1()
        elif model=='densenet':
            self.model = densenet()
        self.down=down#下采样倍数
    def affine(self,angle,feature):
        b,c,h,w = feature.shape
        diameter = math.sqrt((h/(2**self.down))**2+(w/(2**self.down))**2)
        top_pad = math.ceil((diameter-(h/(2**self.down)))/2)
        left_pad = math.ceil((diameter-(w/(2**self.down)))/2)
        #feature=F.pad(feature, [left_pad*(2**self.down), left_pad*(2**self.down), top_pad*(2**self.down), top_pad*(2**self.down)])
        #开始仿射变换
        #旋转theta
        theta = torch.tensor([
            [math.cos(angle), math.sin(-angle), 0],
            [math.sin(angle), math.cos(angle), 0]
        ], dtype=torch.float).unsqueeze(0)
        theta = theta.expand(b, 2, 3).cuda()
        #转回theta
        theta_ = torch.tensor([
            [math.cos(-angle), math.sin(angle), 0],
            [math.sin(-angle), math.cos(-angle), 0]
        ], dtype=torch.float).unsqueeze(0)
        theta_ = theta_.expand(b, 2, 3).cuda()
        #进行仿射变换
        return [theta,theta_]#output_list
    def affine_once(self,k,feature):
        angle = math.atan(k)
        theta1,theta1_ = self.affine(angle, feature)
        theta2,theta2_ = self.affine(-1 * angle, feature)
        '''     
        feature1_list = self.affine(angle,feature)
        feature2_list = self.affine(-1*angle,feature)'''
        return theta1,theta2,theta1_,theta2_
    def forward(self,feature):
        theta_list=[]
        theta_list_=[]
        feature=self.CBRP(feature)
        for k in self.k_list:
            theta1, theta2, theta1_, theta2_=self.affine_once(k,feature)
            theta_list.append(theta1)
            theta_list.append(theta2)
            theta_list_.append(theta1_)
            theta_list_.append(theta2_)
        b,c,h,w = feature.shape
        diameter = math.sqrt((h/(2**self.down))**2+(w/(2**self.down))**2)
        top_pad = math.ceil((diameter-(h/(2**self.down)))/2)
        left_pad = math.ceil((diameter-(w/(2**self.down)))/2)
        feature=F.pad(feature, [left_pad*(2**self.down), left_pad*(2**self.down), top_pad*(2**self.down), top_pad*(2**self.down)])
        T_list=[]
        for i in theta_list:
            T_list.append(F.affine_grid(i, feature.shape))
        T_total=torch.cat(T_list,dim=1)
        feature=F.grid_sample(feature,T_total)#.view(feature.shape[0],feature.shape[1]*8,feature.shape[2],feature.shape[3])
        cost = Variable(torch.FloatTensor(feature.size()[0], feature.size()[1], feature.size()[2]//8,feature.size()[3]).zero_()).cuda()
        for i in range(8):
            cost[:,feature.shape[1]//8*i:feature.shape[1]//8*(i+1),:,:]=feature[:,feature.shape[1]//8*i:feature.shape[1]//8*(i+1),feature.shape[2]//8*i:feature.shape[2]//8*(i+1),:]
        f2,f3,f4=self.model(cost)
        T2_list = []
        T3_list = []
        T4_list = []
        for i in range(8):
            b2,c2,w2,h2 = f2[:,i*f2.shape[1]//8:(i+1)*f2.shape[1]//8,:,:].shape
            b3,c3,w3,h3 = f3[:,i*f3.shape[1]//8:(i+1)*f3.shape[1]//8,:,:].shape
            b4,c4,w4,h4 = f4[:,i*f4.shape[1]//8:(i+1)*f4.shape[1]//8,:,:].shape
            T2_ = F.affine_grid(theta_list_[i],[b2,c2//8,w2,h2])
            T3_ = F.affine_grid(theta_list_[i],[b3,c3//8,w3,h3])
            T4_ = F.affine_grid(theta_list_[i],[b4,c4//8,w4,h4])
            T2_list.append(T2_)
            T3_list.append(T3_)
            T4_list.append(T4_)
        T2_= torch.cat(T2_list,dim=1)
        T3_ = torch.cat(T3_list,dim=1)
        T4_ = torch.cat(T4_list, dim=1)
        #print(F.grid_sample(f2, T2_).shape,f3.shape,f4.shape)
        f2 = F.grid_sample(f2, T2_)#[:,:,top_pad*(2**2):-1*top_pad*(2**2),left_pad*(2**2):-1*left_pad*(2**2)]
        f3 = F.grid_sample(f3, T3_)#[:,:,top_pad*(2**1):-1*top_pad*(2**1),left_pad*(2**1):-1*left_pad*(2**1)]
        f4 = F.grid_sample(f4, T4_)#[:,:,top_pad*(2**0):-1*top_pad*(2**0),left_pad*(2**0):-1*left_pad*(2**0)]
        cost2 = Variable(torch.FloatTensor(f2.size()[0], f2.size()[1], f2.size()[2] // 8,
                                          f2.size()[3]).zero_()).cuda()
        cost3 = Variable(torch.FloatTensor(f3.size()[0], f3.size()[1], f3.size()[2] // 8,
                                          f3.size()[3]).zero_()).cuda()
        cost4 = Variable(torch.FloatTensor(f4.size()[0], f4.size()[1], f4.size()[2] // 8,
                                          f4.size()[3]).zero_()).cuda()
        for i in range(8):
            cost2[:,f2.shape[1]//8*i:f2.shape[1]//8*(i+1),:,:]=f2[:,f2.shape[1]//8*i:f2.shape[1]//8*(i+1),f2.shape[2]//8*i:f2.shape[2]//8*(i+1),:]
            cost3[:, f3.shape[1] // 8 * i:f3.shape[1] // 8 * (i + 1), :, :] = f3[:,f3.shape[1] // 8 * i:f3.shape[1] // 8 * (i + 1),f3.shape[2] // 8 * i:f3.shape[2] // 8 * (i + 1), :]
            cost4[:, f4.shape[1] // 8 * i:f4.shape[1] // 8 * (i + 1), :, :] = f4[:,f4.shape[1] // 8 * i:f4.shape[1] // 8 * (i + 1),f4.shape[2] // 8 * i:f4.shape[2] // 8 * (i + 1), :]
        #print(cost2[:,:,top_pad*(2**2):-1*top_pad*(2**2),left_pad*(2**2):-1*left_pad*(2**2)].shape,cost3[:,:,top_pad*(2**1):-1*top_pad*(2**1),left_pad*(2**1):-1*left_pad*(2**1)].shape,cost4[:,:,top_pad*(2**0):-1*top_pad*(2**0),left_pad*(2**0):-1*left_pad*(2**0)].shape)
        #print(self.c1(out_list[0]).shape,self.c2(out_list[1]).shape,self.c3(out_list[2]).shape)
        #print(cost2[:,:,top_pad*(2**2):-1*top_pad*(2**2),left_pad*(2**2):-1*left_pad*(2**2)].shape,cost3[:,:,top_pad*(2**1):-1*top_pad*(2**1),left_pad*(2**1):-1*left_pad*(2**1)].shape,cost4[:,:,top_pad*(2**0):-1*top_pad*(2**0),left_pad*(2**0):-1*left_pad*(2**0)].shape)
        return cost2[:,:,top_pad*(2**2):-1*top_pad*(2**2),left_pad*(2**2):-1*left_pad*(2**2)],cost3[:,:,top_pad*(2**1):-1*top_pad*(2**1),left_pad*(2**1):-1*left_pad*(2**1)],cost4[:,:,top_pad*(2**0):-1*top_pad*(2**0),left_pad*(2**0):-1*left_pad*(2**0)]
#5.840723037719727---1.7261924743652344
#107.14200925827026---18
'''input=torch.randn([1,3,288,800]).cuda()
o1=torch.randn([1, 128, 36, 100]).cuda()
o2=torch.randn([1, 256, 18, 50]).cuda()
o3=torch.randn([1, 512, 9, 25]).cuda()
mmm=Affinetrans('self',[1.26126253,782.22222222,408.88888889,92.35312561],3,[128,256,512]).cuda()
out=mmm(input)
import time
T=0
for i in range(100):
    a=time.time()
    out=mmm(input)
    ((out[0]-o1).mean()+(out[1]-o2).mean()+(out[2]-o3).mean()).backward()
    b=time.time()
    T+=(b-a)
print(T)
'''