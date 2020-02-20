## Yolov2 Pytorch Training Script
## Prerequisites
- python 3.6.x (or more)
- torch 1.4.0
- torchvision 0.5.0
- tensorboardX
- opencv3
- pillow

We confiremed at Ubuntu 18.04LTS, 16.04 LTS, and Google Colaboratory (20, Feb., 2020).

## Tutorial

See, Qiita (available, soon).

## Preparation

First clone the code

    git clone https://github.com/tztztztztz/yolov2.pytorch.git
    
Install dependencies

	pip install -r requirements.txt

Then create some folder

    mkdir output 
    mkdir data
    
### Train the model

    python train.py --cuda true
         
## Testing 
 
    python test.py --cuda true

 
 

















