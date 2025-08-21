# ============================================================================
# File: model.py
# Date: 2025-03-11
# Author: TA
# Description: Model architecture.
# ============================================================================

import torch
import torch.nn as nn
import torchvision.models as models

class MyNet(nn.Module): 
    def __init__(self):
        super(MyNet, self).__init__()
        
        ################################################################
        # TODO:                                                        #
        # Define your CNN model architecture. Note that the first      #
        # input channel is 3, and the output dimension is 10 (class).  #
        ################################################################
        def conv_block(in_channels, out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels)
            )
        
        def shortcut(in_channels, out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16

        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8

        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4

        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4 -> 2x2

        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        # )
        self.cnn = nn.Sequential(
            conv_block(3, 32),          # 0: 3 -> 32, 32x32
            nn.MaxPool2d(2, 2),         # 1: 32x32 -> 16x16
            conv_block(32, 64),         # 2: 32 -> 64, 16x16
            nn.MaxPool2d(2, 2),         # 3: 16x16 -> 8x8
            conv_block(64, 128),        # 4: 64 -> 128, 8x8
            nn.MaxPool2d(2, 2),         # 5: 8x8 -> 4x4
            conv_block(128, 256),       # 6: 128 -> 256, 4x4
            conv_block(256, 384),       # 7: 256 -> 384, 4x4
        )

        # 獨立的殘差 shortcut
        self.shortcut1 = shortcut(64, 128)  
        self.shortcut2 = shortcut(128, 256) 
        self.shortcut3 = shortcut(256, 384)  


        self.fc = nn.Sequential(
            nn.Linear(384 * 4 * 4, 1024), #512 4 4 1024
            nn.BatchNorm1d(1024), # 1024
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 512),    # 1024 512 
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(512, 10)
            # nn.BatchNorm1d(512),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.2),          
            # nn.Linear(512, 10),
        )


    def forward(self, x):

        ##########################################
        # TODO:                                  #
        # Define the forward path of your model. #
        ##########################################

        # x = self.cnn(x)
        x = self.cnn[0](x)  # 3 -> 32
        x = self.cnn[1](x)  # Pool: 32x32 -> 16x16

        # Block 2
        x = self.cnn[2](x)  # 32 -> 64
        x = self.cnn[3](x)  # Pool: 16x16 -> 8x8

        # Block 3 with residual
        residual = self.shortcut1(x)  # 64 -> 128
        x = self.cnn[4](x)           # 64 -> 128
        x = x + residual
        x = torch.relu(x)
        x = self.cnn[5](x)           # Pool: 8x8 -> 4x4

        # Block 4 with residual
        residual = self.shortcut2(x)  # 128 -> 256
        x = self.cnn[6](x)           # 128 -> 256
        x = x + residual
        x = torch.relu(x)

        # Block 5 with residual
        residual = self.shortcut3(x)  # 256 -> 512
        x = self.cnn[7](x)           # 256 -> 512
        x = x + residual
        x = torch.relu(x)

        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)

        return x
    
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        ############################################
        # NOTE:                                    #
        # Pretrain weights on ResNet18 is allowed. #
        ############################################

        # (batch_size, 3, 32, 32)
        # try to load the pretrained weights
        # self.resnet = models.resnet18(weights=None)  # Python3.8 w/ torch 2.2.1
        self.resnet = models.resnet18(pretrained=True)  # Python3.6 w/ torch 1.10.1
        # (batch_size, 512)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        # (batch_size, 10)

        #######################################################################
        # TODO (optional):                                                     #
        # Some ideas to improve accuracy if you can't pass the strong         #
        # baseline:                                                           #
        #   1. reduce the kernel size, stride of the first convolution layer. # 
        #   2. remove the first maxpool layer (i.e. replace with Identity())  #
        # You can run model.py for resnet18's detail structure                #
        #######################################################################

        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.resnet.maxpool = nn.Identity()

        ############################## TODO End ###############################

    def forward(self, x):
        return self.resnet(x)
    
if __name__ == '__main__':
    model = ResNet18()
    print(model)

    # # print architecture and number of parameters
    # from torchsummary import summary
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # MyNet
    # print("MyNet Architecture and Parameters:")
    # mynet = MyNet().to(device)
    # summary(mynet, input_size=(3, 32, 32))

    # # ResNet18
    # print("\nResNet18 Architecture and Parameters:")
    # resnet = ResNet18().to(device)
    # summary(resnet, input_size=(3, 32, 32))