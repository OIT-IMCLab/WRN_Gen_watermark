


from tqdm import tqdm  
import torch
import numpy as np


#カスタム正則化関数
def custom_reg(model, layer, wmark_bit, wmark_seed, key_seed, device, use_ch, start_ch):
    total_reg_loss = 0.0
    
    #埋め込み透かしの定義
    wmark = wmark_rand01(wmark_bit, wmark_seed).to(torch.float32).to(device)

    for name, param in model.named_parameters():
        
        #埋め込み層の特定
        if param.requires_grad and layer in name:
            
            #paramのサイズを取得
            size = param.numel()


            #paramをベクトルに変換
            w = param.view(1, size)

            #秘密鍵生成
            x = GM_key_generater(param=param,wmark_bit=wmark_bit,use = use_ch,start_ch=start_ch
                            ,seed=key_seed).to(device)
            
            # 鍵とparamベクトルの計算
            x_weights = torch.matmul(w, x)

            #標準化
    
            mean = torch.mean(x_weights)
            std = torch.std(x_weights)
            standardized_arr = (x_weights - mean) / std
            x_weights = standardized_arr



            #x_weights = torch.sigmoid(torch.matmul(w, x))
            #reg = torch.nn.functional.binary_cross_entropy(input=x_weights, target=wmark, reduction="none")
            
            #sigmoidを行ってBCEをとる関数
            reg = torch.nn.functional.binary_cross_entropy_with_logits(input=x_weights,target=wmark,reduction="none")
            total_reg_loss += torch.mean(reg)
                
    return total_reg_loss

#前のバージョン
def train_0 (model,trainloader,validloader,device,optimizer,loss_fn,epochs,scheduler):
        # モデルのトレーニングループ
        for epoch in range(epochs):
            print(f"epoch: {epoch+1}")
            running_train_loss = 0
            model.train()
            
            for (inputs, labels) in tqdm( trainloader):
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                
                #埋め込み
                
                #reg_loss = custom_reg(model,LAYER,WMARK_BIT,WMARK_SEED,KEY_SEED,device)
                #total_loss =   loss + LAMBDA *reg_loss
                total_loss = loss

                total_loss.backward()
                optimizer.step()
                running_train_loss += total_loss.item()
                

            ave_train_loss = running_train_loss/len(trainloader)
            print(f"train_loss:{ave_train_loss:f}",end=" ")



            model.eval()
            val_loss=0
            correct_predictions = 0
            total_samples = 0
            with torch.no_grad():
                for (inputs,labels) in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    
                    #埋め込み
                    #reg_loss = custom_reg(model,LAYER,WMARK_BIT,WMARK_SEED,KEY_SEED,device)
                    #total_loss =   loss + LAMBDA *reg_loss
                    #loss += LAMBDA
                    total_loss = loss
                
                    val_loss += total_loss.item()

                    # 精度計算
                    _, predicted = torch.max(outputs, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
            # 平均損失と精度を計算
            ave_val_loss = val_loss / len(validloader)
            accuracy = correct_predictions / total_samples * 100
            print(f"val_loss:{ave_val_loss},accuracy:{accuracy}",end="\n\n")
            scheduler.step()
#train関数　
def train (model,trainloader,validloader,device,optimizer,loss_fn,epochs,scheduler,scaler,wmark_bit,wmark_seed,key_seed,use_ch,start_ch,layer,reg_lambda):
        # モデルのトレーニングループ
        from torch.amp import GradScaler, autocast

        for epoch in range(epochs):
            print(f"epoch: {epoch+1}")
            running_train_loss = 0
            #train時に実行が必要
            model.train()
            
            #各trainデータを取りだす。
            for (inputs, labels) in tqdm( trainloader):
                
                #GPUで演算を行う処理
                inputs, labels = inputs.to(device), labels.to(device)
                
                #勾配リセット
                optimizer.zero_grad()
                
                #autocastで学習高速化
                with autocast("cuda"):  # Mixed Precision Training

                    outputs = model(inputs)
                    #損失関数
                    loss = loss_fn(outputs, labels)
                    #透かし埋め込み正則化関数
                    reg_loss = custom_reg(model,layer,wmark_bit,wmark_seed,key_seed,device,use_ch,start_ch)
                    
                    #上記二種の足し合わせ
                    total_loss =   loss + reg_lambda *reg_loss
                

                # 勾配計算と更新
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_train_loss += total_loss.item()
                        
            #lossの計算
            ave_train_loss = running_train_loss/len(trainloader)
            print(f"train_loss:{ave_train_loss:f}",end=" ")


            #推論時に実行
            model.eval()

            val_loss=0
            correct_predictions = 0
            total_samples = 0
            with torch.no_grad():
                for (inputs,labels) in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    
                    #埋め込み
                    reg_loss = custom_reg(model,layer,wmark_bit,wmark_seed,key_seed,device,use_ch,start_ch)
                    total_loss =   loss + reg_lambda *reg_loss
                    #loss += LAMBDA
                    #total_loss = loss
                
                    val_loss += total_loss.item()

                    # 精度計算
                    _, predicted = torch.max(outputs, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
            # 平均損失と精度を計算
            ave_val_loss = val_loss / len(validloader)
            accuracy = correct_predictions / total_samples * 100
            print(f"val_loss:{ave_val_loss},accuracy:{accuracy}",end="\n\n")
            scheduler.step()



#透かし埋め込みを行わないtrain関数

def train_no_emb (model,trainloader,validloader,device,optimizer,loss_fn,epochs,scheduler,scaler):
        # モデルのトレーニングループ
        from torch.amp import GradScaler, autocast
        for epoch in range(epochs):
            print(f"epoch: {epoch+1}")
            running_train_loss = 0
            model.train()
            
            for (inputs, labels) in tqdm( trainloader):
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                with autocast("cuda"):  # Mixed Precision Training
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    total_loss = loss

                
                #埋め込み
                

                # 勾配計算と更新
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_train_loss += total_loss.item()
                        

            ave_train_loss = running_train_loss/len(trainloader)
            print(f"train_loss:{ave_train_loss:f}",end=" ")



            model.eval()
            val_loss=0
            correct_predictions = 0
            total_samples = 0
            with torch.no_grad():
                for (inputs,labels) in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    
                    #埋め込み
                    total_loss = loss
                
                    val_loss += total_loss.item()

                    # 精度計算
                    _, predicted = torch.max(outputs, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
            # 平均損失と精度を計算
            ave_val_loss = val_loss / len(validloader)
            accuracy = correct_predictions / total_samples * 100
            print(f"val_loss:{ave_val_loss},accuracy:{accuracy}",end="\n\n")
            scheduler.step()

#test用関数
def test(model,testloader,device,loss_fn,layer,wmark_bit,wmark_seed,key_seed,use_ch,start_ch):
        print("test_result")

        model.eval()
        test_loss=0
        correct_predictions = 0
        total_samples = 0
        emb_loss =0
        total_loss = 0
        with torch.no_grad():
            for (inputs,labels) in tqdm(testloader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()

                emb_loss = custom_reg(model, layer, wmark_bit, wmark_seed, key_seed, device, use_ch, start_ch)
                total_loss = test_loss + emb_loss
                # 精度計算
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        # 平均損失と精度を計算
        ave_test_loss = test_loss / len(testloader)
        ave_emb_loss = emb_loss / len(testloader)
        ave_total_loss = total_loss / len(testloader)
        accuracy = correct_predictions / total_samples * 100
        print(f"val_loss:{ave_test_loss},emb_loss{ave_emb_loss},total_loss{ave_total_loss},accuracy:{accuracy}")




#透かし生成
def wmark_rand01(wmark_b,seed):
    np.random.seed(seed)
    wmark= np.random.randint(0, 2, (1,wmark_b))
    wmark = torch.from_numpy((wmark.astype(np.float32)))
    
    return wmark


#秘密鍵生成　これを改良して下の関数を作成
def key_generate(size,wmark_bit,key_seed):
    np.random.seed(key_seed)
    x = np.random.rand(size,wmark_bit)
    x = torch.from_numpy((x.astype(np.float32)))
    
    return x

#秘密鍵生成　チャンネルごとのグルーピング可
"""
ex) 埋め込み層paramが[160,16,3,3]の場合
pytorchでは[out_ch,in_ch,kernel_h,kernel_w]である。
本手法では使用するparamをinput_chで区別するため
16chすべて使いたい場合は USE = 16 start=1で設定
半分の8chの場合、　　USE = 8 であり、前半を使用したい場合はstart =1
後半を使用したい場合はstart = 9
とすることで可能である。


"""

def GM_key_generater(param,wmark_bit,use,start_ch,seed):

    #numpyの乱数seedの初期設定
    np.random.seed(seed)
    #引数から使用するパラメータの数を取得
    size = param.numel()
    #size = param.size
    #パラメータ情報からカーネルサイズを取得
    kernel_size = param.shape[2]*param.shape[3]
    

    block_size =  kernel_size*param.shape[1]
    #埋め込み時に使用するパラメータのスタート位置
    start = (start_ch-1)*kernel_size
    #埋め込みに使用するパラメータ数
    emb_size = use * kernel_size

    #print(size,kernel_size,block_size,start,emb_size)
    #x = np.random.rand(size,wmark_bit)
    
    #seed設定
    rng_1 = np.random.default_rng(seed)

    #一様乱数
    #x = rng_1.uniform(-1,1,(size,wmark_bit))
    #正規分布N(0,1)平均0,分散1
    x = rng_1.normal(loc=0, scale=1, size=(size, wmark_bit))

    #使用しない重みパラメータに対応する秘密鍵の部分を0に変換
    for i in range(0,size,block_size):
        x[i:i+start,:] = 0
        x[i+start+emb_size:i+block_size,:] =0
    x = torch.from_numpy((x.astype(np.float32)))
    return x