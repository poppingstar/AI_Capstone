import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
# from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import util.trainer as trainer
from pathlib import Path
import yaml

def main():
    hyper = trainer.TrainConfig(batch_size=64, patience=5, save_point=5)

    dataset_path = Path(r"C:\Users\user\Downloads\deep_fake_augmented")
    save_dir = dataset_path/'weights'/'EfficientNet_B7'
    save_dir = trainer.no_overwrite(save_dir)

    transformer = {  #케이스 별 transform 정의
                'train':transforms.Compose([transforms.RandomAdjustSharpness(4),transforms.RandomVerticalFlip(),
                                                        transforms.ColorJitter(0.5,0.5,0.5,0.1),transforms.RandomRotation(90)]),
                'valid':transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
                'test':transforms.Compose([transforms.Resize(224),  transforms.ToTensor()])
                }

    train_dataset = trainer.DirDataset(dataset_path/'train', transformer['train'])
    validation_dataset = trainer.DirDataset(dataset_path/'valid',transformer['valid'])
    test_dataset = trainer.DirDataset(dataset_path/'test', transformer['test'])

    train_loader = DataLoader(train_dataset,hyper.batch_size,True,num_workers=hyper.workers,pin_memory=True,drop_last=False, shuffle=True)
    validation_loader = DataLoader(validation_dataset, hyper.batch_size,True,num_workers=hyper.workers,pin_memory=True,drop_last=False)
    test_loader = DataLoader(test_dataset, hyper.batch_size,True,num_workers=hyper.workers,pin_memory=True,drop_last=False)
    class_num = len(train_dataset.classes)

    model = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    trainer.layer_freeze(model, 'features.6')
    for n, p in model.named_parameters():
        print(p.size())
    model.classifier[1] = nn.Linear(2560, out_features=class_num)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), hyper.lr)
    hyper.set_optimizer(optimizer)

    hyper.save_log(save_dir/'log.txt')

    trainer.train_test(model, train_loader, validation_loader, 
                            test_loader, hyper, save_dir)


if __name__ == '__main__':
    main()