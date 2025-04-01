import torch
import torch.nn as nn
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
CLASS = 100
BATCH_SIZE = 128
WMARK_BIT = 128
WMARK_SEED = 0
KEY_SEED  = 1 
USE =16
START_CH =1
SAVE_PATH = "./model/WRN.pth"
LAYER = "conv2_x.0.conv1.weight"
##########################################################################


import torchvision
import torchvision.transforms as transforms


try:
    ###############################################################################################
    #データ処理
    from Dload import dataload_MNIST,dataload_CIFAR10,dataload_CIFAR100
    from Dload import dataload_tiny_Imagenet,dataload_CIFAR100_64, dataload_caltech101

    ###############################################################################################
    # PyTorchモデルの定義
    from resnet import WRN

    import sys

    if __name__ == '__main__':
        #モデルのタスクによって選択してください。
        #trainloader,validloader,testloader = dataload_CIFAR100_64(BATCH_SIZE)
        trainloader,validloader,testloader = dataload_tiny_Imagenet(BATCH_SIZE)
        #trainloader,validloader,testloader = dataload_caltech101(BATCH_SIZE)


        #Gpuを使用
        device = torch.device("cuda:0")


        # モデルインスタンスの生成
        model = WRN(depth=28, k=10, num_classes=CLASS)

        #指定したdeviceを使用するようにする
        model = model.to(device)

        
        #モデルロード
        model.load_state_dict(torch.load(SAVE_PATH,weights_only=False))


        #モデルサマリーを表示
        from torchinfo import summary
        #summary(model)







        # 損失関数と最適化アルゴリズムの設定
        loss_fn = nn.CrossEntropyLoss()
        
        from utils import test,GM_key_generater,wmark_rand01,key_generate
        #test(model=model,testloader=testloader,device=device,loss_fn=loss_fn) 

        #print(model.conv2.weight)
        reg_loss = 0.0
        for name, param in model.named_parameters():
            param = param.to(device)

            #print(name)

            if param.requires_grad and LAYER in name:  # 特定のレイヤー名に一致する場合のみ
                shape = param.shape
                print(shape)
                
                #重みパラメータから重み行列へ変形
                size = param.numel()
                w = param.view(1,size)
                
                #秘密鍵生成
                #x  = key_generater(size,WMARK_BIT,KEY_SEED).to(device)
                x = GM_key_generater(param=param,wmark_bit=WMARK_BIT,use=USE,start_ch=START_CH,seed=KEY_SEED).to(device)
                
                print(x)
                print("パラメータ一覧")
                print(w.shape)
                
                #重みベクトルと秘密鍵の行列積
                x_weights = torch.matmul(w,x)  

                """
                print("鍵全体の分散")
                print(torch.var(x))
                print("鍵の2ビット目")
                print(torch.var(x[:,1]))
                print("19bit目")
                print(torch.var(x[:,19]))
                """
                #sigmoid
                x_weights = torch.sigmoid(x_weights) #
                
               #################################################### 
              
                #埋め込んだ透かし
                wmark = wmark_rand01(WMARK_BIT,WMARK_SEED).to(device)
               
                #weights = x_weights.flatten()
                
                #透かし検出
                print("検出した透かし")
                
                #要素すべてをPrint
                torch.set_printoptions(profile="full")
                torch.set_printoptions(sci_mode = False)
                
                #透かし検出結果
                print(x_weights)

                ## 透かしの判定と閾値の設定
                y = (x_weights >= 0.5).int().tolist()
                #y = (x_weights >= 0.01).int().tolist()


                print("透かし判定後の結果")
                y = np.reshape(y, (1, WMARK_BIT))


                # 透かし最小値と最大値
                print("透かし最小値：", torch.min(x_weights).item())
                print("透かし最大値：", torch.max(x_weights).item())

                # 埋め込まれた透かし
                
                print("埋め込み透かし")
                print(wmark)
                wmark = wmark.to('cpu').detach().numpy().copy()

                # 透かしの成功判定
                result = np.equal(y, wmark)
                print("Trueならば透かし埋め込みが成功")
                np.set_printoptions(threshold=np.inf)
                print(y)
                print(result)

                #print("パラメータ一覧")
                #print(w[:,:100])
                
                #torch.varで不偏分散
                var_score= torch.var(w)
                print("重みパラメータの不偏分散は",var_score,"です")
                print(w.shape)
                print(len(w))





        

            
except KeyboardInterrupt:
    print("\nプログラムが中断されました。リソースを解放して終了します...")
    # ここで必要な後処理を行う（例: ファイルを保存、ログ出力など）
    torch.cuda.empty_cache()  # GPUメモリの解放など
    exit(0)  # プログラムを正常に終了