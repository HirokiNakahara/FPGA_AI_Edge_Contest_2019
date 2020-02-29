# AI Edge Contest: TestBench Generator
(Also, profile analysis of a pre-trained model)

## OS: 
Ubuntu 18.04 LTS (or Ubuntu 16.04 LTS, but not tested...)

Note: I strongly recommended to run this repo at Host PC (not Ultra96 board!!!)

## Requrements:
torch 1.4.0
torchvision 0.5.0
opencv
matplotlib

## 1. Download Pre-trained model (or use your own trained model) and store to (your model path)
https://drive.google.com/file/d/1xUznZs1YPh4XBPri9zcpfk2WTSlDhcVK/view?usp=sharing

(this repo directory)$ mkdir yolov2_alex_1

(this repo directory)$ cp (Download directory)/yolov2_epoch_40.pth ./yolov2_alex_1/yolov2_epoch_40.pth

## 2. make a symbolik link to a traning dataset directory

(this repo directory)$ ln -s (path to your training dataset directory) data

## 3. execute script as follows:

(this repo directory)$ $ python gen_testbench.py --arch alex --dataset test --bs 1 --output_dir yolov2_alex_1 --model_name yolov2_epoch_40
