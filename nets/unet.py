import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
from .resnet import resnet18

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class ResNetUNetEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        
        # self.base_model = models.resnet18(pretrained=True)
        self.base_model = resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())                
        
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)        
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)        
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)        
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.conv512_1024 = nn.Conv2d(512, 1024, 3, 1, 1)
        
        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        
    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        
        layer0 = self.layer0(input)            
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)        
        layer4 = self.layer4(layer3)
        layer4 = self.conv512_1024(layer4)

        return layer4, [layer0, layer1, layer2, layer3, x_original]

class ResNetUNetDecoder(nn.Module):

    def __init__(self):
        super().__init__()
    
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)       
        self.layer2_1x1 = convrelu(128, 128, 1, 0)  
        self.layer3_1x1 = convrelu(256, 256, 1, 0)  
        self.layer4_1x1 = convrelu(512, 512, 1, 0)
        
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        self.conv_last = nn.Conv2d(64, 3, 1)
        self.conv1024_512 = nn.Conv2d(1024, 512, 3, 1, 1)
        
    def forward(self, inter_features, skip_connections):

        layer4 = inter_features 
        (layer0, layer1, layer2, layer3, x_original) = skip_connections

        # convert layer4 from 1024 to 512 standard channel dimension
        layer4 = self.conv1024_512(layer4)
        
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
 
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)        
        
        out = self.conv_last(x)        
        
        return out
    
class Feat2Env(nn.Module):
    """Class to conver the source scene features to environment map
    """
    def __init__(self, chromesz: int = 64):
        super(Feat2Env, self).__init__()
        self.chromesz = chromesz
        # self.conv = nn.Conv2d(512, (chromesz * chromesz * 3), 1, padding=0, bias=True)
        # self.ada_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.relu = nn.ReLU()
        self.layers = nn.Sequential(*[nn.Conv2d(512, (chromesz * chromesz * 3), 1, padding=0, bias=True), nn.AdaptiveAvgPool2d((1, 1))])
        
    def forward(self, x, env2feat=None):
        size = 512
        
        lighting = x[:, 0:size, :, :].clone()
        # lighting = self.conv(lighting)  
        # lighting = self.relu(lighting)
        # lighting = self.ada_pool(lighting)
        lighting = self.layers(lighting)
        
        # env2feat is not None, then we replace the source light features with target features
        if env2feat is not None:
            x[:, 0:size, :, :] = env2feat
            
        return x, x[:, size:, :, :], lighting
        

class Env2Feat(nn.Module):
    """Class to convert a target environment map to feature representation to replace source lighting features for the relighting the scene
    """
    def __init__(self):
        super(Env2Feat, self).__init__()
       
        model = [
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x)
        output = F.interpolate(output.clone(), size=(30, 46))
        return output

class UnetEnvMap(nn.Module):
    """Network only for learning to estimate illumination using resnet-based encoder
    """

    def __init__(self, chromesz: int):
        super(UnetEnvMap, self).__init__()
        self.encoder = ResNetUNetEncoder()
        self.feat_to_env = Feat2Env(chromesz=chromesz)

    def forward(self, x):
        x, _ = self.encoder(x)
        _, _, x = self.feat_to_env(x)
        return x

class RelightModel(nn.Module):
    """Network to train scene re-lighting model
    """
    def __init__(self, chromesz: int):
        super(RelightModel, self).__init__()
        self.encoder = ResNetUNetEncoder()
        self.decoder = ResNetUNetDecoder()
        self.feat_to_env = Feat2Env(chromesz=chromesz)
        self.env_to_feat = Env2Feat()

    def forward(self, source_img, target_env):

        # output of resnet encoder is intermediate features (light and non-light), and skip connections to decoder
        inter_features, skip_connections = self.encoder(source_img)

        # env_to_feat converts the target light probe to intermediate features
        if target_env is not None:
            target_inter_light_feat = self.env_to_feat(target_env)
        else:
            target_inter_light_feat = None

        # feat_to_env converts a part of the intermediate features to source light prediction and replaces the source light features
        # with target features from the above step
        inter_features, non_light_features, source_light = self.feat_to_env(inter_features, target_inter_light_feat)

        # output of decoder is target relight image
        output = self.decoder(inter_features, skip_connections)

        return output, source_light, non_light_features