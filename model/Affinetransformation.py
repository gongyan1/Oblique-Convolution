import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from .resnet_dcn import resnet18,resnet34,resnet50,CBRP,resnet34_,CBRP_
from .densenet import densenet
from .try_conv import model_self,model_self1
from torch.autograd import Variable


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
        self.down=down
    def affine(self,angle,feature):
        b,c,h,w = feature.shape
        diameter = math.sqrt((h/(2**self.down))**2+(w/(2**self.down))**2)
        top_pad = math.ceil((diameter-(h/(2**self.down)))/2)
        left_pad = math.ceil((diameter-(w/(2**self.down)))/2)
       
        theta = torch.tensor([
            [math.cos(angle), math.sin(-angle), 0],
            [math.sin(angle), math.cos(angle), 0]
        ], dtype=torch.float).unsqueeze(0)
        theta = theta.expand(b, 2, 3).cuda()
      
        theta_ = torch.tensor([
            [math.cos(-angle), math.sin(angle), 0],
            [math.sin(-angle), math.cos(-angle), 0]
        ], dtype=torch.float).unsqueeze(0)
        theta_ = theta_.expand(b, 2, 3).cuda()
      
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
        return cost2[:,:,top_pad*(2**2):-1*top_pad*(2**2),left_pad*(2**2):-1*left_pad*(2**2)],cost3[:,:,top_pad*(2**1):-1*top_pad*(2**1),left_pad*(2**1):-1*left_pad*(2**1)],cost4[:,:,top_pad*(2**0):-1*top_pad*(2**0),left_pad*(2**0):-1*left_pad*(2**0)]
