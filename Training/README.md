## Yolov2 Pytorch Training Script
## Prerequisites
- python 3.5.x (or more)
- pytorch 0.4.1 (or more)
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

 
 

















