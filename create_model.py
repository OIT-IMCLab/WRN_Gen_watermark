import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random



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
CLASS = 100                         #分類数
LEARNING_RATE = 0.1                 #学習率
EPOCH = 1                           #epoch
BATCH_SIZE = 128                    #バッチサイズ

WMARK_BIT = 128                     #透かしビット数
WMARK_SEED = 0                      #透かしseed値
KEY_SEED = 0                        #埋め込み秘密鍵の生成seed値

#埋め込みに使用する入力チャンネル数
USE_CH = 16                
#埋め込みに使用する入力チャンネルのスタート位置      
START_CH=1   
#透かし埋め込み強度                       
LAMBDA = 0.01
##conv2conv1 :in_ch16,conv2conv2 :in_ch 160
#埋め込みを行うレイヤの名前
LAYER = "conv2_x.0.conv1.weight"
#保存ファイル名
SAVE_PATH = "./EMM_model/WRN.pth"

##########################################################################




import torchvision.transforms as transforms


try:
    ###############################################################################################
    #データ処理
    from Dload import dataload_tiny_Imagenet
    ###############################################################################################
    





    # PyTorchモデルの定義

    from resnet import WRN
    import sys

    if __name__ == '__main__':
        
        trainloader,validloader,testloader = dataload_tiny_Imagenet(BATCH_SIZE)
        #Gpuを使用
        device = torch.device("cuda:0")

        # モデルインスタンスの生成
        model = WRN(depth=28, num_classes=CLASS, k=10)
        

        #指定したdeviceを使用するようにする
        model = model.to(device)

        from torchinfo import summary
        summary(model)





        # 損失関数と最適化アルゴリズムの設定
        loss_fn = nn.CrossEntropyLoss().to(device)

        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,momentum =0.9)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        #勾配スケーリング用
        from torch.amp import GradScaler, autocast
        scaler = GradScaler()
        

        from utils import train,test,train_no_emb
        #train関数
        train(model=model,trainloader=trainloader,validloader=validloader,
                                  device=device,optimizer= optimizer,
                                  loss_fn=loss_fn,epochs=EPOCH,scheduler=scheduler,scaler=scaler,
                                  wmark_bit = WMARK_BIT,wmark_seed=WMARK_SEED,key_seed=KEY_SEED,
                                  use_ch=USE_CH,start_ch=START_CH,layer=LAYER,reg_lambda=LAMBDA)
        
        #埋め込みを行わないtrain
        """
        train_no_emb (model,trainloader,validloader,device,optimizer,loss_fn,EPOCH,scheduler,scaler)
        """
        #test関数
        test(model=model,testloader=testloader,device=device,loss_fn=loss_fn,
                 layer=LAYER,wmark_bit=WMARK_BIT,wmark_seed=WMARK_SEED,key_seed=KEY_SEED,use_ch=USE_CH,start_ch=START_CH)

        #model保存 
        torch.save(model.state_dict(), SAVE_PATH)    

        

            
except KeyboardInterrupt:
    print("\nプログラムが中断されました。リソースを解放して終了します...")
    # ここで必要な後処理を行う（例: ファイルを保存、ログ出力など）
    torch.cuda.empty_cache()  # GPUメモリの解放など
    exit(0)  # プログラムを正常に終了"()