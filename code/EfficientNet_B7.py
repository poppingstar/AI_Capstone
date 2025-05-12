import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
import util.trainer as trainer
from pathlib import Path
import yaml

def main():
    with open(r'D:\A New Start\AI\capstone\cfg\hyper_params.yaml', 'r') as f:
        hyper_params = yaml.safe_load(f)
    hyper = trainer.HyperParameter(batch_size=48)

    dataset_path = Path(hyper_params['dataset_path'])
    save_dir = dataset_path/'weights'/'EfficientNet_B7'
    save_dir = trainer.no_overwrite(save_dir)

    transforms = hyper.transforms
    train_dataset = trainer.DirDataset(dataset_path/'train', transforms['train'])
    validation_dataset = trainer.DirDataset(dataset_path/'valid',transforms['valid'])
    test_dataset = trainer.DirDataset(dataset_path/'test', transforms['test'])

    train_loader = DataLoader(train_dataset,hyper.batch_size,True,num_workers=hyper.workers,pin_memory=True,drop_last=False)
    validation_loader = DataLoader(validation_dataset, hyper.batch_size,True,num_workers=hyper.workers,pin_memory=True,drop_last=False)
    test_loader = DataLoader(test_dataset, hyper.batch_size,True,num_workers=hyper.workers,pin_memory=True,drop_last=False)
    class_num = len(train_dataset.classes)

    model = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
    trainer.gradinet_freeze(model)
    model.classifier[1] = nn.Linear(2560, out_features=class_num)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), hyper_params['lr'])
    
    hyper.set_optimizer(optimizer)
    hyper.save_log(save_dir/'log.txt')

    trainer.train_test(model, train_loader, validation_loader, 
                            test_loader, hyper, save_dir)


if __name__ == '__main__':
    main()