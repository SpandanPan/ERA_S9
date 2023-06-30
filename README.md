# Image Classification on CFAR10 using Pytorch

### Data Description
##### 1. Training DataSize : 50000 images
##### 2. Test DataSize: 10000 images
##### 3. Image Size: 3*32*32

### Data Augmentations Used
#### Library: albumentations
##### Transformations
##### 1. ShifScaleRotate
##### 2. Horizontal Flip
##### 3. Colour Jitter
##### 4. Cutout


# Model Structure

### Convolution Block1
#### Layer 1 - Normal CNN, BN, Relu
#### Layer 2-  Dilated Conv(Dilation=2),BN,Relu
#### TB - 1*1 Conv, CNN with stride 2

### Convolution Block2
#### Layer 1 - Depthwise CNN,BN,Relu, Pointwise Conv (1*1)
#### Layer 2 - Dilated Conv(Dilation=2),BN,Relu
#### TB - 1*1 Conv, Dilated Conv(Dilation=3)

### Convolution Block3
#### Layer 1 - Normal Conv,BN,Relu
#### Layer 2 - Dilated Conv
#### TB - Strided conv,1*1

### Convolution Block4
#### Layer 1 - Normal Conv,BN,Relu
#### Layer 2 - Dilated Conv,BN, RELU
#### Layer 3 - Normal Conv,BN,Relu

### Output Layer
#### GAP
#### FC

### Optimizers
#### SGD
#### Step Scheduler (Size=15)
#### Epoch 50

### Model Summary

 Conv2d-1           [-1, 64, 30, 30]           1,728
              ReLU-2           [-1, 64, 30, 30]               0
       BatchNorm2d-3           [-1, 64, 30, 30]             128
           Dropout-4           [-1, 64, 30, 30]               0
            Conv2d-5           [-1, 64, 30, 30]          36,864
              ReLU-6           [-1, 64, 30, 30]               0
       BatchNorm2d-7           [-1, 64, 30, 30]             128
           Dropout-8           [-1, 64, 30, 30]               0
            Conv2d-9           [-1, 30, 30, 30]           1,920
           Conv2d-10           [-1, 30, 14, 14]           8,100
             ReLU-11           [-1, 30, 14, 14]               0
           Conv2d-12           [-1, 30, 14, 14]             270
             ReLU-13           [-1, 30, 14, 14]               0
      BatchNorm2d-14           [-1, 30, 14, 14]              60
          Dropout-15           [-1, 30, 14, 14]               0
           Conv2d-16           [-1, 64, 14, 14]           1,920
           Conv2d-17           [-1, 64, 14, 14]          36,864
             ReLU-18           [-1, 64, 14, 14]               0
      BatchNorm2d-19           [-1, 64, 14, 14]             128
          Dropout-20           [-1, 64, 14, 14]               0
           Conv2d-21           [-1, 30, 14, 14]           1,920
           Conv2d-22             [-1, 30, 8, 8]           8,100
             ReLU-23             [-1, 30, 8, 8]               0
           Conv2d-24             [-1, 64, 8, 8]          17,280
             ReLU-25             [-1, 64, 8, 8]               0
      BatchNorm2d-26             [-1, 64, 8, 8]             128
          Dropout-27             [-1, 64, 8, 8]               0
           Conv2d-28             [-1, 64, 8, 8]          36,864
             ReLU-29             [-1, 64, 8, 8]               0
      BatchNorm2d-30             [-1, 64, 8, 8]             128
          Dropout-31             [-1, 64, 8, 8]               0
           Conv2d-32             [-1, 30, 8, 8]           1,920
           Conv2d-33             [-1, 30, 4, 4]           8,100
      BatchNorm2d-34             [-1, 30, 4, 4]              60
             ReLU-35             [-1, 30, 4, 4]               0
           Conv2d-36             [-1, 35, 4, 4]           9,450
             ReLU-37             [-1, 35, 4, 4]               0
      BatchNorm2d-38             [-1, 35, 4, 4]              70
          Dropout-39             [-1, 35, 4, 4]               0
           Conv2d-40             [-1, 35, 4, 4]          11,025
             ReLU-41             [-1, 35, 4, 4]               0
      BatchNorm2d-42             [-1, 35, 4, 4]              70
          Dropout-43             [-1, 35, 4, 4]               0
           Conv2d-44             [-1, 44, 4, 4]          13,860
             ReLU-45             [-1, 44, 4, 4]               0
      BatchNorm2d-46             [-1, 44, 4, 4]              88
AdaptiveAvgPool2d-47             [-1, 44, 1, 1]               0
           Linear-48                   [-1, 10]             440
================================================================
Total params: 197,613
Trainable params: 197,613
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 4.87
Params size (MB): 0.75
Estimated Total Size (MB): 5.64
----------------------------------------------------------------
