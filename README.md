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

