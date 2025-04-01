import torch
import torch.nn as nn

class wide_res_block(nn.Module):
    """wide residual block"""
    def __init__(self, out_channels, block_num, layer_num):
        super(wide_res_block, self).__init__()
        
        # 1番目のブロック以外はチャンネル数がinputとoutputで変わる(output=4×input)
        if (layer_num==0):
            if (block_num==1):
                input_channels = 16
            else:
                input_channels = out_channels//2
        else:
            input_channels = out_channels

        # shortcutとstrideの設定
        if (layer_num == 0):
            self._is_change = True
            # 最初のresblockは(W､ H)は変更しないのでstrideは1にする
            if (block_num==1):
                stride = 1
            else:
                stride = 2
            
            self.conv_sc = nn.Conv2d(input_channels, out_channels, kernel_size=1, stride=stride)
            self.bn_sc = nn.BatchNorm2d(out_channels)
        else:
            self._is_change = False
            stride = 1

        # 1層目 3×3 畳み込み処理を行います
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.drop1 = nn.Dropout(p=0.4)
        self.conv1 = nn.Conv2d(input_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        
        
        # 2層目 3×3 畳み込み処理を行います
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop2 = nn.Dropout(p=0.4)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        shortcut = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.drop1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.drop2(out)
        out = self.conv2(out)
        
        # Projection shortcutの場合
        if self._is_change:
            shortcut = self.conv_sc(shortcut)
            shortcut = self.bn_sc(shortcut)
        
        out += shortcut
        return out

class WRN(nn.Module):
    def __init__(self, depth, k, num_classes=10):
        super(WRN, self).__init__()
        
        # 各ネットワークでのブロックの数
        N = (depth - 4) // 6
        
        conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(*[conv1])
        
        self.conv2_x = nn.Sequential(*[wide_res_block(out_channels=16*k, block_num=1, layer_num=i) for i in range(N)])
        self.conv3_x = nn.Sequential(*[wide_res_block(out_channels=32*k, block_num=2, layer_num=i) for i in range(N)])
        self.conv4_x = nn.Sequential(*[wide_res_block(out_channels=64*k, block_num=3, layer_num=i) for i in range(N)])
        
        bn = nn.BatchNorm2d(64*k)
        relu = nn.ReLU(inplace=True)
        pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(*[bn, relu, pool])
        
        self.linear = nn.Linear(in_features=64*k, out_features=num_classes)
    
    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        out = self.conv1(x)
        #print(f"After conv1: {out.shape}")
        out = self.conv2_x(out)
        #print(f"After block1: {out.shape}")
        out = self.conv3_x(out)
        #print(f"After block2: {out.shape}")
        out = self.conv4_x(out)
        #print(f"After block3: {out.shape}")
        out = self.fc(out)
        #print(f"After avg_pool2d: {out.shape}")
        out = out.view(out.shape[0], -1)
        #print(f"After flattening: {out.shape}")
        out = self.linear(out)
        
        return out
