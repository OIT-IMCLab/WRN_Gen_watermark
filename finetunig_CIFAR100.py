import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import os
import numpy as np
import random
from tqdm import tqdm                              # プログレスバーを出すために使用


# シード値の設定
def set_seed(seed=200):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(0)


# ハイパーパラメータ########################################################
CLASS = 100
LEARNING_RATE = 0.1
EPOCH = 1
BATCH_SIZE = 128

WMARK_BIT = 128
WMARK_SEED = 0
KEY_SEED = 0

USE_CH = 16
START_CH=1
LAMBDA = 0.01
##conv2conv1 :in_ch16,conv2conv2 :in_ch 160
LAYER = "conv2_x.0.conv1.weight"


SAVE_PATH = "./EMM_model/test.pth"
BASE_MODEL_PASS = "./EMM_model/WRN.pth"
##########################################################################
try:

    ###############################################################################################
    #データ処理
    from Dload import dataload_MNIST,dataload_CIFAR10,dataload_CIFAR100,dataload_CIFAR100_64
        
    ###############################################################################################
    # PyTorchモデルの定義




    from resnet import WRN



    import sys

    if __name__ == '__main__':
        trainloader,validloader,testloader = dataload_CIFAR100_64(BATCH_SIZE)
        #Gpuを使用
        device = torch.device("cuda:0")
        


        # モデルインスタンスの生成
        model = WRN(depth=28, k=10, num_classes=CLASS)

        #指定したdeviceを使用するようにする
        model = model.to(device)


        model.load_state_dict(torch.load(BASE_MODEL_PASS,weights_only=False))




        # 損失関数と最適化アルゴリズムの設定
        loss_fn = nn.CrossEntropyLoss()

        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,momentum =0.9)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
       
        """
        for name, param in model.named_parameters():
            print(name)
            param.requires_grad = True


        for name,param in model.named_parameters():# 埋め込み層の手前まで凍結
            if name == LAYER:
                break
            param.requires_grad = False
        for name, param in model.named_parameters():
            print(f"Layer: {name} | Requires Grad: {param.requires_grad}")
        """

        
        from utils import train,test
        from torch.amp import GradScaler, autocast
        scaler = GradScaler()
        train(model=model,trainloader=trainloader,validloader=validloader,
                                  device=device,optimizer= optimizer,
                                  loss_fn=loss_fn,epochs=EPOCH,scheduler=scheduler,scaler=scaler,
                                  wmark_bit = WMARK_BIT,wmark_seed=WMARK_SEED,key_seed=KEY_SEED,
                                  use_ch=USE_CH,start_ch=START_CH,layer=LAYER,reg_lambda=LAMBDA)
        
        test(model=model,testloader=testloader,device=device,loss_fn=loss_fn,
                 layer=LAYER,wmark_bit=WMARK_BIT,wmark_seed=WMARK_SEED,key_seed=KEY_SEED,use_ch=USE_CH,start_ch=START_CH)
            
        torch.save(model.state_dict(), SAVE_PATH)    

        

            
except KeyboardInterrupt:
    print("\nプログラムが中断されました。リソースを解放して終了します...")
    # ここで必要な後処理を行う（例: ファイルを保存、ログ出力など）
    torch.cuda.empty_cache()  # GPUメモリの解放など
    exit(0)  # プログラムを正常に終了"