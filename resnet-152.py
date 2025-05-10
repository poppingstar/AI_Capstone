import torch, torchvision
import Trainer
from pathlib import Path
from torchvision.models import resnet152, ResNet152_Weights
import torch.utils.data as data, torch.nn as nn

def main():
    dir_path = Path(r"C:\Users\user\Desktop\deepfake and real images")
    save_dir = dir_path/'weights'/'paper_competition'/'ResNet_152'

    hyper=Trainer.HyperParameter()

    train_set = Trainer.DirDataset(dir_path/'train', transforms=hyper.transforms['train'])
    valid_set = Trainer.DirDataset(dir_path/'valid', transforms=hyper.transforms['valid'])
    test_set = Trainer.DirDataset(dir_path/'test', transforms=hyper.transforms['test'])

    train_loader = data.DataLoader(train_set,hyper.batch,True,num_workers=hyper.workers,pin_memory=True,drop_last=False)
    valid_loader = data.DataLoader(valid_set,hyper.batch,True,num_workers=hyper.workers,pin_memory=True,drop_last=False)
    test_loader = data.DataLoader(test_set,hyper.batch,True,num_workers=hyper.workers,pin_memory=True,drop_last=False)
    classes_num = len(train_set.classes)  #데이터 셋의 클래스 개수 가져오기

    model = resnet152(weights=ResNet152_Weights.DEFAULT)
    model.fc=nn.Linear(2048, classes_num)

    optimizer = torch.optim.Adam(model.parameters(), hyper.lr)
    hyper.set_optimizer(optimizer)
    
    save_dir = Trainer.no_overwrite(save_dir); save_dir.mkdir(parents=True)
    hyper.save_log(save_dir/'log.txt')

    Trainer.train_test(model, train_loader, valid_loader, 
                            test_loader, hyper, save_dir)
    
if __name__ == '__main__':
    main()