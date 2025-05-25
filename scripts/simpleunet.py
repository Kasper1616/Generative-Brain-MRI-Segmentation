import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True))


class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
                
        self.conv_down1 = double_conv(3, 16)
        self.conv_down2 = double_conv(16, 32)
        self.conv_down3 = double_conv(32, 64)
        self.conv_down4 = double_conv(64, 128)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.conv_up3 = double_conv(64 + 128, 64)
        self.conv_up2 = double_conv(32 + 64, 32)
        self.conv_up1 = double_conv(32 + 16, 16)
        
        self.last_conv = nn.Conv2d(16, 1, kernel_size=1)
        
        
    def forward(self, x):
        conv1 = self.conv_down1(x)  
        x = self.maxpool(conv1)     
        conv2 = self.conv_down2(x)  
        x = self.maxpool(conv2)     
        conv3 = self.conv_down3(x) 
        x = self.maxpool(conv3)     
        x = self.conv_down4(x)      
        x = self.upsample(x)        
        
        
        x = torch.cat([x, conv3], dim=1) 
        
        x = self.conv_up3(x) 
        x = self.upsample(x)   
        x = torch.cat([x, conv2], dim=1) 

        x = self.conv_up2(x) 
        x = self.upsample(x)      
        x = torch.cat([x, conv1], dim=1) 
        
        x = self.conv_up1(x)
        
        out = self.last_conv(x) 
        out = torch.sigmoid(out)
        
        return out, x # Return the output and the last feature map for later use