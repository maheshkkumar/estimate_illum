from torch import nn
from torchvision import models
from .resnet import resnet18

class VGG16(nn.Module):
    def __init__(self, pretrained: bool = True, chromesz: int = 64):
        super(VGG16, self).__init__()
        vgg16 = models.vgg16(pretrained=pretrained)
        features = list(vgg16.features.children())
        self.features = nn.Sequential(*features)
        layers = [nn.AdaptiveAvgPool2d((1, 1)), 
                    nn.Conv2d(512, (chromesz * chromesz * 3), 1, padding=0, bias=True)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.layers(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, pretrained: bool = True, chromesz: int = 64):
        super(ResNet18, self).__init__()
        resnet = resnet18(pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        layers = [nn.AdaptiveAvgPool2d((1, 1)), 
                    nn.Conv2d(512, (chromesz * chromesz * 3), 1, padding=0, bias=True)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layers(x)
        return x
