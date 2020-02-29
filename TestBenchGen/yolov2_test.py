# --------------------------------------------------------
# Pytorch Yolov2
# Licensed under The MIT License [see LICENSE for details]
# Written by Jingru Tan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import config as cfg
from darknet import Darknet19
import alexnet, vgg
from darknet import conv_bn_leaky
from loss import build_target, yolo_loss

archs={
        'alex': alexnet.alexnet,
        'vgg11': vgg.vgg11_bn,
        'vgg16': vgg.vgg16_bn,
        }
outfmaps={
        'alex': 256,
        'vgg11': 512,
        'vgg16': 512,
        }

class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.data.size()
        ws = self.stride
        hs = self.stride
        x = x.view(B, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4).contiguous()
        x = x.view(B, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, int(H / hs), int(W / ws)).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, int(H / hs), int(W / ws))
        return x


class Yolov2(nn.Module):

    num_classes = 10
    num_anchors = 5

    def __init__(self, classes=None, pretrained=False, arch='alex'):
        super(Yolov2, self).__init__()
        if classes:
            self.num_classes = len(classes)

        self.trunk = archs[arch](pretrained=pretrained)

        # detection layers
        self.conv3 = nn.Sequential(conv_bn_leaky(outfmaps[arch], outfmaps[arch], kernel_size=3, return_module=True),
                                   conv_bn_leaky(outfmaps[arch], outfmaps[arch]//2, kernel_size=3, return_module=True))


        self.conv4 = nn.Sequential(conv_bn_leaky(outfmaps[arch]//2, outfmaps[arch]//2, kernel_size=3, return_module=True),
                                   nn.Conv2d(outfmaps[arch]//2, (5 + self.num_classes) * self.num_anchors, kernel_size=1))

        #self.reorg = ReorgLayer()

    def forward(self, x, gt_boxes=None, gt_classes=None, num_boxes=None, training=False):
        """
        x: Variable
        gt_boxes, gt_classes, num_boxes: Tensor
        """

        part_module = self.trunk.features
        # ------------------------------------------------
        # Printout model
        # ------------------------------------------------
        print(part_module)

        # --------------------------------------------
        # Profiling
        # --------------------------------------------
        with torch.autograd.profiler.profile(use_cuda=False) as prof:
            tmp = x
            tmp = self.trunk(tmp)
            tmp = self.conv3(tmp)
            out = self.conv4(tmp)
        print(prof)
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))

        # ------------------------------------------------
        # Access parameters, 
        #  and generate weight and bias for the 1st layer
        # ------------------------------------------------
        for n, p in part_module.named_parameters():
            print("layer", n)
            print("shape", p.shape)

            if n == '0.weight':
                param = p.detach().numpy()
                print('weight',param)
                fname = 'weight_l0.txt'
                print('Weight Paramter File -> %s' % fname)
                np.savetxt(fname, param.reshape(-1,), fmt="%.4f", delimiter="")

            if n == '0.bias':
                param = p.detach().numpy()
                print('bias',param)
                fname = 'bias_l0.txt'
                print('Bias Paramter File -> %s' % fname)
                np.savetxt(fname, param.reshape(-1,), fmt="%.4f", delimiter="")

        # ------------------------------------------------
        # Test(Inference) step-by-step manner
        #  and generate testbenches
        # ------------------------------------------------
        y = x
        for layer_idx, module in enumerate(part_module):
            print("layer idx",layer_idx)

            # Printout layer information (to be uncommented)
            #print("module",module)
            #print(type(module))
            #print(module.__dict__)
            #print(module.__dict__['_modules'])

            # To access a specified layer (just example)
            #if isinstance( module, torch.nn.modules.conv.Conv2d):

            # Generate Testbench
            if layer_idx == 0:
                print("conv2d in the 1st layer!!")

                bench_input = y
                #bench_input = bench_input.detach().numpy()
                bench_input = bench_input.numpy()
                print(bench_input.shape)
                print("bench_input",bench_input)
                bench_input = bench_input.reshape(-1,)

                #print(bench_input)
                fname = 'testbench_input.txt'
                print('Test Bench Input File -> %s' % fname)
                np.savetxt(fname, bench_input, fmt="%.5f", delimiter=",")

                print('input.dimension',y.shape)
                y = module(y) # (1,3,416,416)
                print('output.dimension',y.shape)

                print('y', y.shape) # (1,64,102,102)

                bench_output = y
                bench_output = bench_output.detach().numpy()
                bench_output = bench_output.reshape(-1,)

                #print(bench_output)
                fname = 'testbench_output.txt'
                print('Test Bench Output File -> %s' % fname)
                np.savetxt(fname, bench_output, fmt="%.5f", delimiter=",")

            else:
                y = module(y)

            '''
            if ii == 0:
                #print('x',x.shape)
                tmp = x.numpy()
                tmp = module(tmp)
                y = torch.from_numpy(tmp)
                #print(y.shape)
            elif ii == 1:
                y = module(y)
                #print(y.shape)
            elif ii == 2:
                tmp = y.numpy()
                tmp = channel_shift( tmp, 512, 7)
                y = torch.from_numpy(tmp)
                #print(y.shape)
            elif ii == 3:
                #print(y.shape)
                y = module(y)
            '''

        x = y

        # inference latter layers
        x = self.conv3(x)
        out = self.conv4(x)

        if cfg.debug:
            print('check output', out.view(-1)[0:10])

        # --------------------------------------------
        # Post processing
        # --------------------------------------------
        # out -- tensor of shape (B, num_anchors * (5 + num_classes), H, W)
        bsize, _, h, w = out.size()

        # 5 + num_class tensor represents (t_x, t_y, t_h, t_w, t_c) and (class1_score, class2_score, ...)
        # reorganize the output tensor to shape (B, H * W * num_anchors, 5 + num_classes)
        out = out.permute(0, 2, 3, 1).contiguous().view(bsize, h * w * self.num_anchors, 5 + self.num_classes)

        # activate the output tensor
        # `sigmoid` for t_x, t_y, t_c; `exp` for t_h, t_w;
        # `softmax` for (class1_score, class2_score, ...)

        xy_pred = torch.sigmoid(out[:, :, 0:2])
        conf_pred = torch.sigmoid(out[:, :, 4:5])
        hw_pred = torch.exp(out[:, :, 2:4])
        class_score = out[:, :, 5:]
        class_pred = F.softmax(class_score, dim=-1)
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)

        if training:
            output_variable = (delta_pred, conf_pred, class_score)
            output_data = [v.data for v in output_variable]
            gt_data = (gt_boxes, gt_classes, num_boxes)
            target_data = build_target(output_data, gt_data, h, w)

            target_variable = [Variable(v) for v in target_data]
            box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)

            return box_loss, iou_loss, class_loss

        return delta_pred, conf_pred, class_pred

if __name__ == '__main__':
    model = Yolov2(arch='vgg11')
    im = np.random.randn(1, 3, 256, 256)
    im_variable = Variable(torch.from_numpy(im)).float()
    out = model(im_variable)
    delta_pred, conf_pred, class_pred = out
    print('delta_pred size:', delta_pred.size())
    print('conf_pred size:', conf_pred.size())
    print('class_pred size:', class_pred.size())



