import util.trainer as trainer
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from pathlib import Path

def main():
  dir_path=Path(r"C:\Users\user\Desktop\deepfake and real images")
  save_dir=dir_path/'weights'/'paper_competition'/'MoblieNet_small'

  hyper = trainer.TrainConfig()

  #데이터 셋 및 데이터 로더 생성
  train_set=trainer.DirDataset(dir_path/'train',transforms=hyper.transforms['train'])
  valid_set=trainer.DirDataset(dir_path/'valid',transforms=hyper.transforms['valid'])
  test_set=trainer.DirDataset(dir_path/'test',transforms=hyper.transforms['test'])

  train_loader=data.DataLoader(train_set,hyper.batch_size,True,num_workers=hyper.workers,pin_memory=True,drop_last=False)
  valid_loader=data.DataLoader(valid_set,hyper.batch_size,True,num_workers=hyper.workers,pin_memory=True,drop_last=False)
  test_loader=data.DataLoader(test_set,hyper.batch_size,shuffle=True,num_workers=hyper.workers,pin_memory=True,drop_last=False)
  classes_num=len(train_set.classes)  #데이터 셋의 클래스 개수 가져오기
  
  model=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
  model.classifier[3]=nn.Linear(1024, classes_num)
  
  optimizer = torch.optim.Adam(model.parameters(), hyper.lr)
  hyper.set_optimizer(optimizer)
  
  save_dir = trainer.no_overwrite(save_dir); save_dir.mkdir(parents=True)
  hyper.save_log(save_dir/'log.txt')

  trainer.train_test(model, train_loader, valid_loader, test_loader,
                      hyper, save_dir)
  print('\a')

if __name__=='__main__':
  main()