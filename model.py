import torch
import torch.nn as nn

dropout_value = 0.1
class Model_CFAR(nn.Module):
    def __init__(self):
        super(Model_CFAR, self).__init__()

        # *******************************Conv Block 1*************************
        # Block1-Layer1
        self.convblk1_ly1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value))

        # Block1-Layer2
        self.convblk1_ly2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=2,  bias=False,dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value))

        # Block1 - TB
        self.convblk1_tb = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=30, kernel_size=(1, 1), padding=0, bias=False),
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(3, 3), padding=0,stride=2, bias=False),
            nn.ReLU())
        # ****************************Conv Block 2****************************
        # Block 2 - Layer1 - Depthwise
        self.convblk2_ly1 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(3, 3), padding=1, bias=False,groups=30),
            nn.ReLU(),
            nn.BatchNorm2d(30),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=30, out_channels=64, kernel_size=(1, 1), padding=0, bias=False))

         #Block 2- Layer2
        self.convblk2_ly2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=2, bias=False,dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value))

        # Block 2 - TB - Using Dilated Conv Layer in place of strided conv
        self.convblk2_tb = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=30, kernel_size=(1, 1), padding=0, bias=False),
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(3, 3), padding=0,stride=1,dilation=3, bias=False),
            nn.ReLU())
        # ****************************Conv Block 3****************************
        # Block 3 - Layer 1

        self.convblk3_ly1 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value))

        #Block 3 - Layer 2
        self.convblk3_ly2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=2, bias=False,dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value))

        # Block 3- TB
        self.convblk3_tb = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=30, kernel_size=(1, 1), padding=0, bias=False),
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(3, 3), padding=1,stride=2, dilation=1, bias=False),
            nn.BatchNorm2d(30),
            nn.ReLU())
        # ****************************Conv Block 4****************************
        # Block 4 - Layer 1

        self.convblk4_ly1 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=35, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(35),
            nn.Dropout(dropout_value))

        #Block 4 - Layer 2
        self.convblk4_ly2 = nn.Sequential(
            nn.Conv2d(in_channels=35, out_channels=35, kernel_size=(3, 3), padding=2, bias=False,dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(35),
            nn.Dropout(dropout_value))

        # Block 4- Layer 3
        self.convblk4_ly3 = nn.Sequential(
            nn.Conv2d(in_channels=35, out_channels=44, kernel_size=(3, 3), padding=1,stride=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(44))
        #****************************Output Block*************************
        # GAP Layer
        self.gap = nn.AdaptiveAvgPool2d(1)

       # FC Layer
        self.fc = nn.Linear(44, 10,bias=False)

    def forward(self, x):
        x = self.convblk1_ly1(x)
        x = x+self.convblk1_ly2(x)
        x = self.convblk1_tb(x)
        x = self.convblk2_ly1(x)
        x = x+self.convblk2_ly2(x)
        x = self.convblk2_tb(x)
        x = self.convblk3_ly1(x)
        x = x+self.convblk3_ly2(x)
        x = self.convblk3_tb(x)
        x = self.convblk4_ly1(x)
        x = x+self.convblk4_ly2(x)
        x = self.convblk4_ly3(x)
        x = self.gap(x)
        x = x.view(-1, 1*1*44)
        x = self.fc(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
