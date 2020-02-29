import os
import argparse
import time
import torch
from torch.autograd import Variable
from PIL import Image
from test import prepare_im_data
from yolov2_test import Yolov2
from yolo_eval import yolo_eval
from util.visualize import draw_detection_boxes
import matplotlib.pyplot as plt
from util.network import WeightLoader

import numpy as np

float_dtype='float' # for xilinx
float_np_dtype=np.float16
modifier=''

def save_header(file_name,string_data):
#    print('\tINFO: Convert -> %s' % file_name)
    f = open(file_name, 'a')
    f.write(string_data)
    f.close()

def arr2header(arr, dtype='float', width=160):
#    if dtype=='float': formatter={'float':lambda x: "%23.16e" % x}
    if dtype=='float': formatter={'float':lambda x: "%2.5e" % x}
    else:              formatter={'int':lambda x: "%d" % x}
    np.set_printoptions(threshold=np.inf)
    print(arr.shape)
    if arr.ndim == 4:
        d1, d2, d3, d4 = arr.shape
        if d3 == 1 and d4 ==1:
            print("!!pointwise weight!!")
            dummy = np.zeros((d1,d2))
            arr = arr.transpose(2,3,0,1)
            dummy = arr[0,0]
            arr = dummy
            print(" -> new", arr.shape)
#    exit()
#    return np.array2string(arr, threshold=np.nan, separator=',', formatter=formatter, max_line_width=width).replace(']', '}').replace('[', '{')+';\n'
#    return np.array2string(arr, threshold=np.inf, separator=',', formatter=formatter, max_line_width=width).replace(']', '}').replace('[', '{')+';\n'
    return np.array2string(arr, separator=',', formatter=formatter, max_line_width=width).replace(']', '}').replace('[', '{')+';\n'

def parse_args():

    parser = argparse.ArgumentParser('Yolo v2')
    parser.add_argument('--output_dir', dest='output_dir',
                        default='output', type=str)
    parser.add_argument('--arch', default='alex', type=str)
    parser.add_argument('--model_name', dest='model_name',
                        default='yolov2_epoch_160', type=str)
    parser.add_argument('--cuda', dest='use_cuda',
                        default=False, type=bool)

    args = parser.parse_args()
    return args


def demo():
    args = parse_args()
    print('call with args: {}'.format(args))

    # input images
    images_dir = 'images'
    images_names = ['trainval1.jpg', 'trainval2.jpg', 'test1.jpg', 'test2.jpg']


    classes = ("car", "bus", "truck", "svehicle", "pedestrian", "motorbike", "bicycle", "train", "signal", "signs")

    model = Yolov2(arch=args.arch)
    #weight_loader = WeightLoader()
    #weight_loader.load(model, 'yolo-voc.weights')
    #print('loaded')

    model_path = os.path.join(args.output_dir, args.model_name + '.pth')
    print('loading model from {}'.format(model_path))
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    if args.use_cuda:
        model.cuda()

    model.eval()
    print('model loaded')
    print(model)

    ## generate weight
    idx = 0
    for ii, module in enumerate(model.trunk.features):
        #print("ii",ii)
        #print("module",module)
        #print(type(module))
        #print(module.__dict__)

        if isinstance( module, torch.nn.modules.conv.Conv2d):
            print("conv2d layer_%d" % idx)
            #print("weight",module.__dict__['_parameters']['weight'])
            weight = module.__dict__['_parameters']['weight']
            weight = weight.detach().numpy() # nn.tensor -> numpy
            #print(weight)
            #print(weight.shape)
            
            if ii == 0:
                header_w = float_dtype+' w_%s[%d][%d][%d][%d]=\n' % ((str(ii),)+(weight.shape)) + arr2header(weight)
                save_header('./weight_l0.h', header_w)

            #print("bias",module.__dict__['_parameters']['bias'])
            bias = module.__dict__['_parameters']['bias']
            bias = bias.detach().numpy() # nn.tensor -> numpy
            #print(bias)

            if ii == 0:
                coef = bias.reshape(-1,).astype(float_np_dtype)
                header = float_dtype+(' b_%s[%d]=\n' % ((str(ii)),len(coef))) + arr2header(coef)
                save_header('./bias_l0.h', header)

            idx += 1
    exit()


    for image_name in images_names:
        image_path = os.path.join(images_dir, image_name)
        img = Image.open(image_path)
        im_data, im_info = prepare_im_data(img)

        if args.use_cuda:
            im_data_variable = Variable(im_data).cuda()
        else:
            im_data_variable = Variable(im_data)

        tic = time.time()

        yolo_output = model(im_data_variable)
        yolo_output = [item[0].data for item in yolo_output]
        detections = yolo_eval(yolo_output, im_info, conf_threshold=0.2, nms_threshold=0.4)
        ##print(detections)

        toc = time.time()
        cost_time = toc - tic
        print('im detect, cost time {:4f}, FPS: {}'.format(
            toc-tic, int(1 / cost_time)))

        det_boxes = detections[:, :5].cpu().numpy()
        det_classes = detections[:, -1].long().cpu().numpy()
        im2show = draw_detection_boxes(img, det_boxes, det_classes, class_names=classes)
        plt.figure()
        plt.imshow(im2show)
        #plt.show()

        save_image_path = os.path.join(images_dir, image_name + "_detect.jpg")
        print("save -> " + save_image_path)
        plt.savefig(save_image_path)

if __name__ == '__main__':
    demo()
