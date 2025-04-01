import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

def to_rgb(image):
        return image.convert('RGB')  # PIL.ImageオブジェクトをRGBに変換
def dataload_MNIST(batchsize=32):

        #transformsの前処理
        trans = transforms.Compose(
            [   transforms.Lambda(lambda x: to_rgb(x)),  # グレースケール画像をRGBに変換
                transforms.ToTensor(),
                transforms.Resize((32,32)),
                transforms.Normalize((0.5,), (0.5,))
                ])
       
        


        #Datesets の前処理、およびダウンロード(なければdownload=Trueでダウンロードする)
        trainset = datasets.MNIST(root = './MNIST', train = True, download = False, 
                                            transform = trans)
        
        n_samples = len(trainset) 
        train_size = int(len(trainset) * 0.8) 
        val_size = n_samples - train_size 

        # shuffleしてから分割してくれる.
        train_dataset, val_dataset = torch.utils.data.random_split(trainset, 
                                                                [train_size, val_size])
        """
        print(len(train_dataset)) # 48000
        print(len(val_dataset)) # 12000
        """
        


        #Dateloaderによるdatesetsの使用
        trainloader = DataLoader(train_dataset, batch_size = batchsize, 
                                                shuffle = True, num_workers = 0)
        
        validloader = DataLoader(val_dataset, batch_size = batchsize,
                                                shuffle = True, num_workers = 0)


        testset =datasets.MNIST(root = './MNIST', train = False,
                                            download = False, transform = trans)


        testloader = torch.utils.data.DataLoader(testset, batch_size = batchsize, 
                                                shuffle = False, num_workers = 0)
        
        # 最初のバッチを取得
        images, labels = next(iter(trainloader))

        # 画像の形状を確認 (バッチサイズ, チャンネル数, 高さ, 幅)
        print(images.shape)
        
        return trainloader,validloader,testloader

def dataload_CIFAR10(batchsize=32):

        #transformsの前処理
        trans = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (0.5,))])

        #Datesets の前処理、およびダウンロード(なければdownload=Trueでダウンロードする))
        trainset = datasets.CIFAR10(root = './CIFAR10', train = True, download = False, 
                                            transform = trans)
        
        n_samples = len(trainset) 
        train_size = int(len(trainset) * 0.8) 
        val_size = n_samples - train_size 

        # shuffleしてから分割してくれる.
        train_dataset, val_dataset = torch.utils.data.random_split(trainset, 
                                                                [train_size, val_size])
        """
        print(len(train_dataset)) # 48000
        print(len(val_dataset)) # 12000
        """
        


        #Dateloaderによるdatesetsの使用
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batchsize, 
                                                shuffle = True, num_workers = 1)
        
        validloader = torch.utils.data.DataLoader(val_dataset, batch_size = batchsize,
                                                shuffle = True, num_workers = 1)


        testset = datasets.CIFAR10(root = './CIFAR10', train = False,
                                            download = False, transform = trans)


        testloader = torch.utils.data.DataLoader(testset, batch_size = batchsize, 
                                                shuffle = False, num_workers = 1)
        # 最初のバッチを取得
        images, labels = next(iter(trainloader))

        # 画像の形状を確認 (バッチサイズ, チャンネル数, 高さ, 幅)
        print(images.shape)
        return trainloader,validloader,testloader
      
def dataload_CIFAR100(batchsize=32):

        #transformsの前処理
        trans = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (0.5,))])

        #Datesets の前処理、およびダウンロード(なければダウンロードする)
        trainset = datasets.CIFAR100(root = './CIFAR100', train = True,
                                                  download = False, 
                                                  transform = trans)
        
        n_samples = len(trainset) # n_samples is 60000
        train_size = int(len(trainset) * 0.8) # train_size is 48000
        val_size = n_samples - train_size # val_size is 48000

        # shuffleしてから分割してくれる.
        train_dataset, val_dataset = torch.utils.data.random_split(trainset, 
                                                                [train_size, val_size])
        """
        print(len(train_dataset)) # 48000
        print(len(val_dataset)) # 12000
        """
        


        #Dateloaderによるdatesetsの使用
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batchsize, 
                                                shuffle = True, num_workers = 4,pin_memory=True)
        
        validloader = torch.utils.data.DataLoader(val_dataset, batch_size = batchsize,
                                                shuffle = True, num_workers = 4,pin_memory=True)


        testset = datasets.CIFAR100(root = './CIFAR100', train = False,
                                            download = False, transform = trans)


        testloader = torch.utils.data.DataLoader(testset, batch_size = batchsize, 
                                                shuffle = False, num_workers = 4,pin_memory=True)
        # 最初のバッチを取得
        images, labels = next(iter(trainloader))

        # 画像の形状を確認 (バッチサイズ, チャンネル数, 高さ, 幅)
        print(images.shape)
        return trainloader,validloader,testloader
def dataload_CIFAR100_64(batchsize=32):

        #transformsの前処理
        trans = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (0.5,)),
                                                transforms.Resize((64, 64))
                                                ])

        #Datesets の前処理、およびダウンロード(なければダウンロードする)
        trainset = datasets.CIFAR100(root = './CIFAR100', train = True,
                                                  download = False, 
                                                  transform = trans)
        
        #trainの80%を学習で使用、valで20%使用
        n_samples = len(trainset) 
        train_size = int(len(trainset) * 0.8) 
        val_size = n_samples - train_size 

        # shuffleしてから分割してくれる.
        train_dataset, val_dataset = torch.utils.data.random_split(trainset, 
                                                                [train_size, val_size])
        """
        print(len(train_dataset)) # 48000
        print(len(val_dataset)) # 12000
        """
        


        #Dateloaderによるdatesetsの使用
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batchsize, 
                                                shuffle = True, num_workers = 4,pin_memory=True)
        
        validloader = torch.utils.data.DataLoader(val_dataset, batch_size = batchsize,
                                                shuffle = True, num_workers = 4,pin_memory=True)


        testset = datasets.CIFAR100(root = './CIFAR100', train = False,
                                            download = False, transform = trans)


        testloader = torch.utils.data.DataLoader(testset, batch_size = batchsize, 
                                                shuffle = False, num_workers = 4,pin_memory=True)
        # 最初のバッチを取得
        images, labels = next(iter(trainloader))

        # 画像の形状を確認 (バッチサイズ, チャンネル数, 高さ, 幅)
        print(images.shape)
        return trainloader,validloader,testloader
def dataload_tiny_Imagenet(batchsize=32):


    #transformsの前処理
    trans = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((64, 64)),
                                            transforms.Normalize
                                            (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  ])

    #Datesets の前処理
    train_set = datasets.ImageFolder(root = 'tiny-imagenet-100/train', 
                                        transform = trans)
   
    
    val_dataset = datasets.ImageFolder(root = './tiny-imagenet-100/val_processed', 
                                        transform = trans)
    
    #trainデータを分割し、テストデータを用意する
    n_samples = len(train_set) 
    train_size = int(len(train_set) * 0.9) 
    test_size = n_samples - train_size 

    #print(len(train_dataset))

    # shuffleしてから分割してくれる.
    train_dataset, test_dataset = torch.utils.data.random_split(train_set, [train_size, test_size])
    """
    print(len(train_dataset)) # 48000
    print(len(val_dataset)) # 12000
    """
    


    #Dateloaderによるdatesetsの使用
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batchsize, 
                                            shuffle = True, num_workers = 4,pin_memory=True)
    
    validloader = torch.utils.data.DataLoader(val_dataset, batch_size = batchsize,
                                            shuffle = True, num_workers = 4,pin_memory=True)




    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = batchsize, 
                                            shuffle = False, num_workers = 4,pin_memory=True)
    """
    # 最初のバッチを取得
    images, labels = next(iter(trainloader))

    # 画像の形状を確認 (バッチサイズ, チャンネル数, 高さ, 幅)
    print(images.shape)
    """
    
    return trainloader,validloader,testloader


def dataload_caltech101(batchsize=32):
        
        
        from torch.utils.data import DataLoader, Subset
        from sklearn.model_selection import train_test_split

        # データ変換
        transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # データセット
        dataset = datasets.ImageFolder(root="./caltech-101/101_ObjectCategories", transform=transform)

        # インデックスとラベルの取得
        indices = list(range(len(dataset)))
        targets = dataset.targets  # クラスラベル

        # Train/Test分割 (70% Train, 30% Test+Validation)
        train_indices, test_val_indices = train_test_split(
        indices, test_size=0.3, stratify=targets, random_state=42
        )

        # Test/Validation分割 (50% Test, 50% Validation from Test+Validation)
        test_indices, val_indices = train_test_split(
        test_val_indices, test_size=0.5, stratify=[targets[i] for i in test_val_indices], random_state=42
        )

        # 各データセットの作成
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        # DataLoader作成
        trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
        validloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)
        testloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

        # 結果確認
        print(f"Trainデータ数: {len(train_dataset)}")
        print(f"Validationデータ数: {len(val_dataset)}")
        print(f"Testデータ数: {len(test_dataset)}")

        
        
        
        
        return trainloader,validloader,testloader
       