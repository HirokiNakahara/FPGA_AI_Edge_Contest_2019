import os
import argparse
import time
import torch
from torch.autograd import Variable
from PIL import Image
from test import prepare_im_data
from yolov2 import Yolov2
from yolo_eval import yolo_eval
from util.visualize import draw_detection_boxes
import matplotlib.pyplot as plt
from util.network import WeightLoader


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
        print(detections)

        toc = time.time()
        cost_time = toc - tic
        print('im detect, cost time {:4f}, FPS: {}'.format(
            toc-tic, int(1 / cost_time)))

        det_boxes = detections[:, :5].cpu().numpy()
        det_classes = detections[:, -1].long().cpu().numpy()
        im2show = draw_detection_boxes(img, det_boxes, det_classes, class_names=classes)
        plt.figure()
        plt.imshow(im2show)
        plt.show()

if __name__ == '__main__':
    demo()
