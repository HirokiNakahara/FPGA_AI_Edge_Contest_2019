from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import numpy as np
import argparse
import time
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.factory import get_imdb
from dataset.roidb import RoiDataset, detection_collate
from yolov2 import Yolov2
from torch import optim
from torch.optim.lr_scheduler import StepLR
from util.network import adjust_learning_rate
from tensorboardX import SummaryWriter
from config import config as cfg


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Yolo v2')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--arch', default='alex', type=str)
    parser.add_argument('--max_epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=100, type=int) # 160 -> 100
    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=1, type=int)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of workers to load training data',
                        default=8, type=int)
    parser.add_argument('--output_dir', dest='output_dir',
                        default='output', type=str)
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        default=False, type=bool)
    parser.add_argument('--display_interval', dest='display_interval',
                        default=100, type=int) # 10 -> 100
    parser.add_argument('--mGPUs', dest='mGPUs',
                        default=False, type=bool)
    parser.add_argument('--save_interval', dest='save_interval',
                        default=20, type=int)
    parser.add_argument('--cuda', dest='use_cuda',
                        default=False, type=bool)
    parser.add_argument('--resume', dest='resume',
                        default=False, type=bool)
    parser.add_argument('--checkpoint_epoch', dest='checkpoint_epoch',
                        default=100, type=int) # 160 -> 100
    parser.add_argument('--exp_name', dest='exp_name',
                        default='default', type=str)

    args = parser.parse_args()
    return args

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train():

    # define the hyper parameters first
    args = parse_args()
    args.steplr_epoch = cfg.steplr_epoch
    args.steplr_factor = cfg.steplr_factor
    args.weight_decay = cfg.weight_decay
    args.momentum = cfg.momentum

    print('Called with args:')
    print(args)

    lr = args.lr

    # initial tensorboardX writer
    if args.use_tfboard:
        if args.exp_name == 'default':
            writer = SummaryWriter()
        else:
            writer = SummaryWriter('runs/' + args.exp_name)

    args.imdb_name = 'trainval'
    args.imdbval_name = 'trainval'

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load dataset
    print('loading dataset....')
    train_dataset = RoiDataset(get_imdb(args.imdb_name))

    print('dataset loaded.')

    print('training rois number: {}'.format(len(train_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  collate_fn=detection_collate, drop_last=True)

    # initialize the model
    print('initialize the model')
    tic = time.time()
    model = Yolov2(pretrained=True, arch=args.arch)
    toc = time.time()
    print('model loaded: cost time {:.2f}s'.format(toc-tic))

    # initialize the optimizer
    optimizer = optim.SGD( [
        {"params": model.trunk.parameters(), "lr": args.lr*cfg.former_lr_decay},
        {"params": model.conv3.parameters()},
        {"params": model.conv4.parameters()} ],
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.steplr_epoch, gamma=args.steplr_factor)

    if args.resume:
        print('resume training enable')
        resume_checkpoint_name = 'yolov2_epoch_{}.pth'.format(args.checkpoint_epoch)
        resume_checkpoint_path = os.path.join(output_dir, resume_checkpoint_name)
        print('resume from {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        args.start_epoch = checkpoint['epoch'] + 1
        lr = checkpoint['lr']
        print('learning rate is {}'.format(lr))
        adjust_learning_rate(optimizer, lr)

    if args.use_cuda:
        model.cuda()

    if args.mGPUs:
        model = nn.DataParallel(model)

    # set the model mode to train because we have some layer whose behaviors are different when in training and testing.
    # such as Batch Normalization Layer.
    model.train()

    iters_per_epoch = int(len(train_dataset) / args.batch_size)

    # start training
    for epoch in range(args.start_epoch, args.max_epochs+1):
        loss_temp = 0
        tic = time.time()
        train_data_iter = iter(train_dataloader)

        scheduler.step()
        lr=get_lr(optimizer)

        if cfg.multi_scale and epoch in cfg.epoch_scale:
            cfg.scale_range = cfg.epoch_scale[epoch]
            print('change scale range to {}'.format(cfg.scale_range))

        for step in range(iters_per_epoch):

            if cfg.multi_scale and (step + 1) % cfg.scale_step == 0:
                scale_index = np.random.randint(*cfg.scale_range)
                cfg.input_size = cfg.input_sizes[scale_index]
                print('change input size {}'.format(cfg.input_size))

            im_data, boxes, gt_classes, num_obj = next(train_data_iter)
            if args.use_cuda:
                im_data = im_data.cuda()
                boxes = boxes.cuda()
                gt_classes = gt_classes.cuda()
                num_obj = num_obj.cuda()

            im_data_variable = Variable(im_data)

            box_loss, iou_loss, class_loss = model(im_data_variable, boxes, gt_classes, num_obj, training=True)

            loss = box_loss.mean()+ iou_loss.mean() \
                   + class_loss.mean()

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            loss_temp += loss.item()

            if (step + 1) % args.display_interval == 0:
                toc = time.time()
                loss_temp /= args.display_interval

                iou_loss_v = iou_loss.mean().item()
                box_loss_v = box_loss.mean().item()
                class_loss_v = class_loss.mean().item()

                print("[epoch %2d][step %4d/%4d] loss: %.4f, lr: %.2e, iou_loss: %.4f, box_loss: %.4f, cls_loss: %.4f" \
                      % (epoch, step+1, iters_per_epoch, loss_temp, lr, iou_loss_v, box_loss_v, class_loss_v))

                if args.use_tfboard:

                    n_iter = (epoch - 1) * iters_per_epoch + step + 1
                    writer.add_scalar('lr', lr, n_iter)
                    writer.add_scalar('losses/loss', loss_temp, n_iter)
                    writer.add_scalar('losses/iou_loss', iou_loss_v, n_iter)
                    writer.add_scalar('losses/box_loss', box_loss_v, n_iter)
                    writer.add_scalar('losses/cls_loss', class_loss_v, n_iter)

                loss_temp = 0
                tic = time.time()

        if epoch % args.save_interval == 0:
            save_name = os.path.join(output_dir, 'yolov2_epoch_{}.pth'.format(epoch))
            torch.save({
                'model': model.module.state_dict() if args.mGPUs else model.state_dict(),
                'epoch': epoch,
                'lr': lr
                }, save_name)

if __name__ == '__main__':
    train()

