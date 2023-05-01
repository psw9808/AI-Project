from __future__ import unicode_literals
import dataset
import feature
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.optim as optim
from torchsummary import summary
import time
import pandas as pd
import pickle
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt


class MAESTRA(nn.Module):
    def __init__(self):
        super(MAESTRA, self).__init__()

        # 네트워크에서 반복적으로 사용되는 Conv + BatchNorm + ReLu를 합쳐서 하나의 함수로 정의
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)  # *으로 list unpacking

            return cbr

        # Contracting path
        # 1x168x216 => 64x168x216
        self.down1 = nn.Sequential(
            CBR2d(1, 64),
            CBR2d(64, 64)
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.down2 = nn.Sequential(
            CBR2d(64, 128),
            CBR2d(128, 128)
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.down3 = nn.Sequential(
            CBR2d(128, 256),
            CBR2d(256, 256)
        )

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.down4 = nn.Sequential(
            CBR2d(256, 512),
            CBR2d(512, 512),
            # nn.Dropout(p=0.5)
        )

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.bottleNeck = nn.Sequential(
            CBR2d(512, 1024),
            CBR2d(1024, 1024)
        )

        self.unpool4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512,
                                          kernel_size=2, stride=2)

        self.up4 = nn.Sequential(
            CBR2d(1024, 512),
            CBR2d(512, 512)
        )

        self.unpool3 = nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                          kernel_size=2, stride=2)

        self.up3 = nn.Sequential(
            CBR2d(512, 256),
            CBR2d(256, 256)
        )

        self.unpool2 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                          kernel_size=2, stride=2)

        self.up2 = nn.Sequential(
            CBR2d(256, 128),
            CBR2d(128, 128)
        )

        self.unpool1 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                          kernel_size=2, stride=2)

        self.up1 = nn.Sequential(
            CBR2d(128, 64),
            CBR2d(64, 64)
        )

        self.fc = nn.Conv2d(64, 1, kernel_size=1, stride=1)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),  # 160x208->160x208
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->80x104

            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->40x52

            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->20x26

            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # ->10x13
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=33280, out_features=16640),
            nn.ReLU(),
            nn.Linear(in_features=16640, out_features=1)
        )

    # __init__ 함수에서 선언한 layer들 연결해서 data propa flow 만들기
    def forward(self, x):

        layer1 = self.down1(x)

        out = self.pool1(layer1)

        layer2 = self.down2(out)

        out = self.pool2(layer2)

        layer3 = self.down3(out)

        out = self.pool3(layer3)

        layer4 = self.down4(out)

        out = self.pool4(layer4)

        bottle_neck = self.bottleNeck(out)

        unpool4 = self.unpool4(bottle_neck)
        cat4 = torch.cat((transforms.CenterCrop((unpool4.shape[2], unpool4.shape[3]))
                          (layer4), unpool4), dim=1)

        dec_layer4 = self.up4(cat4)

        unpool3 = self.unpool3(dec_layer4)
        cat3 = torch.cat((transforms.CenterCrop((unpool3.shape[2], unpool3.shape[3]))
                          (layer3), unpool3), dim=1)

        dec_layer3 = self.up3(cat3)

        unpool2 = self.unpool2(dec_layer3)
        cat2 = torch.cat((transforms.CenterCrop((unpool2.shape[2], unpool2.shape[3]))
                          (layer2), unpool2), dim=1)

        dec_layer2 = self.up2(cat2)

        unpool1 = self.unpool1(dec_layer2)
        cat1 = torch.cat((transforms.CenterCrop((unpool1.shape[2], unpool1.shape[3]))
                          (layer1), unpool1), dim=1)
        # cat1 = torch.cat((unpool1, layer1), dim=1)

        dec_layer1 = self.up1(cat1)

        out = self.fc(dec_layer1)
        out = self.features(out)
        out = torch.flatten(out, 1)
        out = self.fc_layer(out)

        return out


class MAESTRA2_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class MAESTRA2_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
        

class MAESTRA2(nn.Module):
    def __init__(self, num_classes, input_channels=1, deep_supervision=False):
        super().__init__()

        num_filter = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # DownSampling
        self.conv0_0 = MAESTRA2_block(input_channels, num_filter[0], num_filter[0])
        self.conv1_0 = MAESTRA2_block(num_filter[0], num_filter[1], num_filter[1])
        self.conv2_0 = MAESTRA2_block(num_filter[1], num_filter[2], num_filter[2])
        self.conv3_0 = MAESTRA2_block(num_filter[2], num_filter[3], num_filter[3])
        self.conv4_0 = MAESTRA2_block(num_filter[3], num_filter[4], num_filter[4])

        # Upsampling & Dense skip
        # N to 1 skip
        self.conv0_1 = MAESTRA2_block(num_filter[0] + num_filter[1], num_filter[0], num_filter[0])
        self.conv1_1 = MAESTRA2_block(num_filter[1] + num_filter[2], num_filter[1], num_filter[1])
        self.conv2_1 = MAESTRA2_block(num_filter[2] + num_filter[3], num_filter[2], num_filter[2])
        self.conv3_1 = MAESTRA2_block(num_filter[3] + num_filter[4], num_filter[3], num_filter[3])

        # N to 2 skip
        self.conv0_2 = MAESTRA2_block(num_filter[0] * 2 + num_filter[1], num_filter[0], num_filter[0])
        self.conv1_2 = MAESTRA2_block(num_filter[1] * 2 + num_filter[2], num_filter[1], num_filter[1])
        self.conv2_2 = MAESTRA2_block(num_filter[2] * 2 + num_filter[3], num_filter[2], num_filter[2])

        # N to 3 skip
        self.conv0_3 = MAESTRA2_block(num_filter[0] * 3 + num_filter[1], num_filter[0], num_filter[0])
        self.conv1_3 = MAESTRA2_block(num_filter[1] * 3 + num_filter[2], num_filter[1], num_filter[1])

        # N to 4 skip
        self.conv0_4 = MAESTRA2_block(num_filter[0] * 4 + num_filter[1], num_filter[0], num_filter[0])

        if self.deep_supervision:
            self.output1 = nn.Conv2d(num_filter[0], num_classes, kernel_size=1)
            self.output2 = nn.Conv2d(num_filter[0], num_classes, kernel_size=1)
            self.output3 = nn.Conv2d(num_filter[0], num_classes, kernel_size=1)
            self.output4 = nn.Conv2d(num_filter[0], num_classes, kernel_size=1)

        else:
            self.output = nn.Conv2d(num_filter[0], num_classes, kernel_size=1)
        """
        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # torch.nn.init.kaiming_uniform_(m, a=0, mode='fan_in', nonlinearity='leaky_relu')
                torch.nn.init.uniform_(m.weight.data)
                # init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                # torch.nn.init.kaiming_uniform_(m, a=0, mode='fan_in', nonlinearity='leaky_relu')
                torch.nn.init.uniform_(m.weight.data)
                # init_weights(m, init_type='kaiming')
        """
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),  # 160x208->160x208
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->80x104

            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->40x52

            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->20x26

            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # ->10x13
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=33280, out_features=16640),
            nn.ReLU(),
            nn.Linear(in_features=16640, out_features=1)
        )

    def forward(self, x):

        x0_0 = self.conv0_0(x)  # 32x168x216
        x1_0 = self.conv1_0(self.pool(x0_0))  # 64x84x108
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))  # 96x168x216->32x168x216

        x2_0 = self.conv2_0(self.pool(x1_0))  # 128x42x54
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))  # 192x84x108->64x84x108
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))  # 320x168x216->32x168x216

        x3_0 = self.conv3_0(self.pool(x2_0))  # 256x21x27
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))  # 384x42x54->128x42x54
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))  # 640x84x108->64x84x108
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))  # 1088x168x216->32x168x216

        x4_0 = self.conv4_0(self.pool(x3_0))  # 512x10x13
        xup = self.up(x4_0)
        x3_1 = self.conv3_1(torch.cat([transforms.CenterCrop((xup.shape[2], xup.shape[3]))(x3_0),
                                       xup], dim=1))  # 768x20x26->256x20x26
        xup = self.up(x3_1)
        x2_2 = self.conv2_2(torch.cat([transforms.CenterCrop((xup.shape[2], xup.shape[3]))(x2_0),
                                       transforms.CenterCrop((xup.shape[2], xup.shape[3]))(x2_1),
                                       xup], dim=1))  # 1280x40x52->128x40x52
        xup = self.up(x2_2)
        x1_3 = self.conv1_3(torch.cat([transforms.CenterCrop((xup.shape[2], xup.shape[3]))(x1_0),
                                       transforms.CenterCrop((xup.shape[2], xup.shape[3]))(x1_1),
                                       transforms.CenterCrop((xup.shape[2], xup.shape[3]))(x1_2),
                                       xup], dim=1))  # 2176x80x104->64x80x104
        xup = self.up(x1_3)
        x0_4 = self.conv0_4(torch.cat([transforms.CenterCrop((xup.shape[2], xup.shape[3]))(x0_0),
                                       transforms.CenterCrop((xup.shape[2], xup.shape[3]))(x0_1),
                                       transforms.CenterCrop((xup.shape[2], xup.shape[3]))(x0_2),
                                       transforms.CenterCrop((xup.shape[2], xup.shape[3]))(x0_3),
                                       xup], dim=1))  # 3712x160x208->32x160x208

        if self.deep_supervision:
            output1 = self.output1(transforms.CenterCrop((x0_4.shape[2], x0_4.shape[3]))(x0_1))
            output2 = self.output2(transforms.CenterCrop((x0_4.shape[2], x0_4.shape[3]))(x0_2))
            output3 = self.output3(transforms.CenterCrop((x0_4.shape[2], x0_4.shape[3]))(x0_3))
            output4 = self.output4(transforms.CenterCrop((x0_4.shape[2], x0_4.shape[3]))(x0_4))
            output = (output1 + output2 + output3 + output4) / 4

        else:
            output = self.output(x0_4)  # 1x160x208

        output = self.features(output)
        output = torch.flatten(output, 1)
        output = self.fc_layer(output)

        return output


def conv_block_1(in_dim,out_dim, activation,stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=1, stride=stride),
        nn.BatchNorm2d(out_dim),
        activation
    )
    return model


def conv_block_3(in_dim,out_dim, activation, stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_dim),
        activation
    )
    return model


class BottleNeck(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, activation, down=False):
        super(BottleNeck, self).__init__()
        self.down = down

        # 특성지도의 크기가 감소하는 경우
        if self.down:
            self.layer = nn.Sequential(
                conv_block_1(in_dim, mid_dim, activation, stride=2),
                conv_block_3(mid_dim, mid_dim, activation, stride=1),
                conv_block_1(mid_dim, out_dim, activation, stride=1)
            )

            # 특성지도 크기 + 채널을 맞춰주는 부분
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2)

        # 특성지도의 크기가 그대로인 경우
        else:
            self.layer = nn.Sequential(
                conv_block_1(in_dim, mid_dim, activation, stride=1),
                conv_block_3(mid_dim, mid_dim, activation, stride=1),
                conv_block_1(mid_dim, out_dim, activation, stride=1)
            )

        # 채널을 맞춰주는 부분
        self.dim_equalizer = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        if self.down:
            downsample = self.downsample(x)
            out = self.layer(x)
            out = out + downsample
        else:
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            out = out + x
        return out

# 이 아래부턴 Resnet 테스트용
# 50-layer
class ResNet(nn.Module):

    def __init__(self, base_dim, num_classes=1, batch_size=32):
        super(ResNet, self).__init__()
        self.batch_size = batch_size
        self.activation = nn.ReLU()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, base_dim, 7, 2, 3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer_2 = nn.Sequential(
            BottleNeck(base_dim, base_dim, base_dim * 4, self.activation),
            BottleNeck(base_dim * 4, base_dim, base_dim * 4, self.activation),
            BottleNeck(base_dim * 4, base_dim, base_dim * 4, self.activation, down=True)
        )
        self.layer_3 = nn.Sequential(
            BottleNeck(base_dim * 4, base_dim * 2, base_dim * 8, self.activation),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.activation),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.activation),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.activation, down=True)
        )
        self.layer_4 = nn.Sequential(
            BottleNeck(base_dim * 8, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation, down=True)
        )
        self.layer_5 = nn.Sequential(
            BottleNeck(base_dim * 16, base_dim * 8, base_dim * 32, self.activation),
            BottleNeck(base_dim * 32, base_dim * 8, base_dim * 32, self.activation),
            BottleNeck(base_dim * 32, base_dim * 8, base_dim * 32, self.activation)
        )
        self.avgpool = nn.AvgPool2d(1, 1)
        self.fc_layer = nn.Sequential(
            nn.Linear(base_dim * 32 * 6 * 7, num_classes)
        )

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.avgpool(out)
        # out = torch.flatten(out, 1)
        out = out.view(out.shape[0], -1)
        out = self.fc_layer(out)

        return out
        
        
class MAESTRA3(nn.Module):
    def __init__(self, num_classes, base_dim, batch_size=32, input_channels=1, deep_supervision=False):
        super().__init__()

        num_filter = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # DownSampling
        self.conv0_0 = MAESTRA2_block(input_channels, num_filter[0], num_filter[0])
        self.conv1_0 = MAESTRA2_block(num_filter[0], num_filter[1], num_filter[1])
        self.conv2_0 = MAESTRA2_block(num_filter[1], num_filter[2], num_filter[2])
        self.conv3_0 = MAESTRA2_block(num_filter[2], num_filter[3], num_filter[3])
        self.conv4_0 = MAESTRA2_block(num_filter[3], num_filter[4], num_filter[4])

        # Upsampling & Dense skip
        # N to 1 skip
        self.conv0_1 = MAESTRA2_block(num_filter[0] + num_filter[1], num_filter[0], num_filter[0])
        self.conv1_1 = MAESTRA2_block(num_filter[1] + num_filter[2], num_filter[1], num_filter[1])
        self.conv2_1 = MAESTRA2_block(num_filter[2] + num_filter[3], num_filter[2], num_filter[2])
        self.conv3_1 = MAESTRA2_block(num_filter[3] + num_filter[4], num_filter[3], num_filter[3])

        # N to 2 skip
        self.conv0_2 = MAESTRA2_block(num_filter[0] * 2 + num_filter[1], num_filter[0], num_filter[0])
        self.conv1_2 = MAESTRA2_block(num_filter[1] * 2 + num_filter[2], num_filter[1], num_filter[1])
        self.conv2_2 = MAESTRA2_block(num_filter[2] * 2 + num_filter[3], num_filter[2], num_filter[2])

        # N to 3 skip
        self.conv0_3 = MAESTRA2_block(num_filter[0] * 3 + num_filter[1], num_filter[0], num_filter[0])
        self.conv1_3 = MAESTRA2_block(num_filter[1] * 3 + num_filter[2], num_filter[1], num_filter[1])

        # N to 4 skip
        self.conv0_4 = MAESTRA2_block(num_filter[0] * 4 + num_filter[1], num_filter[0], num_filter[0])

        if self.deep_supervision:
            self.output1 = nn.Conv2d(num_filter[0], num_classes, kernel_size=1)
            self.output2 = nn.Conv2d(num_filter[0], num_classes, kernel_size=1)
            self.output3 = nn.Conv2d(num_filter[0], num_classes, kernel_size=1)
            self.output4 = nn.Conv2d(num_filter[0], num_classes, kernel_size=1)

        else:
            self.output = nn.Conv2d(num_filter[0], num_classes, kernel_size=1)
        """
        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # torch.nn.init.kaiming_uniform_(m, a=0, mode='fan_in', nonlinearity='leaky_relu')
                torch.nn.init.uniform_(m.weight.data)
                # init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                # torch.nn.init.kaiming_uniform_(m, a=0, mode='fan_in', nonlinearity='leaky_relu')
                torch.nn.init.uniform_(m.weight.data)
                # init_weights(m, init_type='kaiming')
        """
        
        self.batch_size = batch_size
        self.activation = nn.ReLU()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, base_dim, 7, 2, 3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer_2 = nn.Sequential(
            BottleNeck(base_dim, base_dim, base_dim * 4, self.activation),
            BottleNeck(base_dim * 4, base_dim, base_dim * 4, self.activation),
            BottleNeck(base_dim * 4, base_dim, base_dim * 4, self.activation, down=True)
        )
        self.layer_3 = nn.Sequential(
            BottleNeck(base_dim * 4, base_dim * 2, base_dim * 8, self.activation),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.activation),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.activation),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.activation, down=True)
        )
        self.layer_4 = nn.Sequential(
            BottleNeck(base_dim * 8, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation, down=True)
        )
        self.layer_5 = nn.Sequential(
            BottleNeck(base_dim * 16, base_dim * 8, base_dim * 32, self.activation),
            BottleNeck(base_dim * 32, base_dim * 8, base_dim * 32, self.activation),
            BottleNeck(base_dim * 32, base_dim * 8, base_dim * 32, self.activation)
        )
        self.avgpool = nn.AvgPool2d(1, 1)
        self.fc_layer = nn.Sequential(
            nn.Linear(base_dim * 32 * 5 * 7, num_classes)
        )

    def forward(self, x):

        x0_0 = self.conv0_0(x)  # 32x168x216
        x1_0 = self.conv1_0(self.pool(x0_0))  # 64x84x108
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))  # 96x168x216->32x168x216

        x2_0 = self.conv2_0(self.pool(x1_0))  # 128x42x54
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))  # 192x84x108->64x84x108
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))  # 320x168x216->32x168x216

        x3_0 = self.conv3_0(self.pool(x2_0))  # 256x21x27
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))  # 384x42x54->128x42x54
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))  # 640x84x108->64x84x108
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))  # 1088x168x216->32x168x216

        x4_0 = self.conv4_0(self.pool(x3_0))  # 512x10x13
        xup = self.up(x4_0)
        x3_1 = self.conv3_1(torch.cat([transforms.CenterCrop((xup.shape[2], xup.shape[3]))(x3_0),
                                       xup], dim=1))  # 768x20x26->256x20x26
        xup = self.up(x3_1)
        x2_2 = self.conv2_2(torch.cat([transforms.CenterCrop((xup.shape[2], xup.shape[3]))(x2_0),
                                       transforms.CenterCrop((xup.shape[2], xup.shape[3]))(x2_1),
                                       xup], dim=1))  # 1280x40x52->128x40x52
        xup = self.up(x2_2)
        x1_3 = self.conv1_3(torch.cat([transforms.CenterCrop((xup.shape[2], xup.shape[3]))(x1_0),
                                       transforms.CenterCrop((xup.shape[2], xup.shape[3]))(x1_1),
                                       transforms.CenterCrop((xup.shape[2], xup.shape[3]))(x1_2),
                                       xup], dim=1))  # 2176x80x104->64x80x104
        xup = self.up(x1_3)
        x0_4 = self.conv0_4(torch.cat([transforms.CenterCrop((xup.shape[2], xup.shape[3]))(x0_0),
                                       transforms.CenterCrop((xup.shape[2], xup.shape[3]))(x0_1),
                                       transforms.CenterCrop((xup.shape[2], xup.shape[3]))(x0_2),
                                       transforms.CenterCrop((xup.shape[2], xup.shape[3]))(x0_3),
                                       xup], dim=1))  # 3712x160x208->32x160x208

        if self.deep_supervision:
            output1 = self.output1(transforms.CenterCrop((x0_4.shape[2], x0_4.shape[3]))(x0_1))
            output2 = self.output2(transforms.CenterCrop((x0_4.shape[2], x0_4.shape[3]))(x0_2))
            output3 = self.output3(transforms.CenterCrop((x0_4.shape[2], x0_4.shape[3]))(x0_3))
            output4 = self.output4(transforms.CenterCrop((x0_4.shape[2], x0_4.shape[3]))(x0_4))
            output = (output1 + output2 + output3 + output4) / 4

        else:
            output = self.output(x0_4)  # 1x160x208

        output = self.layer_1(output)
        output = self.layer_2(output)
        output = self.layer_3(output)
        output = self.layer_4(output)
        output = self.layer_5(output)
        output = self.avgpool(output)
        # out = torch.flatten(out, 1)
        output = output.view(output.shape[0], -1)
        output = self.fc_layer(output)

        return output