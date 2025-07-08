import os, sys    
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings 
warnings.filterwarnings('ignore') 
from calflops import calculate_flops
 
import torch  
import torch.nn as nn
from torch.nn import init   
 
from engine.extre_module.ultralytics_nn.conv import Conv    

class SEAttention(nn.Module):
    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential(     
            nn.Linear(channel, channel // reduction, bias=False),    
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()  
        )     

    def init_weights(self):
        for m in self.modules():     
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)     
            elif isinstance(m, nn.BatchNorm2d):  
                init.constant_(m.weight, 1)    
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  
                init.normal_(m.weight, std=0.001) 
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)  
        return x * y.expand_as(x)
  
class ContextGuideFusionModule(nn.Module):
    def __init__(self, inc, ouc) -> None:   
        super().__init__()
  
        self.adjust_conv = nn.Identity()  
        if inc[0] != inc[1]:     
            self.adjust_conv = Conv(inc[0], inc[1], k=1)     
        
        self.se = SEAttention(inc[1] * 2)
     
        if (inc[1] * 2) != ouc:
            self.conv1x1 = Conv(inc[1] * 2, ouc)   
        else:
            self.conv1x1 = nn.Identity()
  
    def forward(self, x): 
        x0, x1 = x   
        x0 = self.adjust_conv(x0)    
    
        x_concat = torch.cat([x0, x1], dim=1) # n c h w
        x_concat = self.se(x_concat) 
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)     
        x0_weight = x0 * x0_weight    
        x1_weight = x1 * x1_weight     
        return self.conv1x1(torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1))
 
if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32
    ouc_channel = 32   
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)  
    
    module = ContextGuideFusionModule([channel_1, channel_2], ouc_channel).to(device)   

    outputs = module([inputs_1, inputs_2])    
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)
 
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     args=[[inputs_1, inputs_2]],
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)  
    print(RESET)
