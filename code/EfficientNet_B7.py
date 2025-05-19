import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
import util.trainer as trainer
from pathlib import Path


def main():
    hyper = trainer.HyperParameter(batch_size=64, patience=5, save_point=5)

    dataset_path = Path(r"E:\Datasets\deep_fake")
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
    trainer.layer_freeze(model, 'features.4')
    model.classifier[1].in_features
    model.classifier[1] = nn.Linear(2560, out_features=class_num)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), hyper.lr)
    hyper.set_optimizer(optimizer)

    for i, _ in model.named_parameters():
        print(i)

    hyper.save_log(save_dir/'log.txt')

    trainer.train_test(model, train_loader, validation_loader, 
                            test_loader, hyper, save_dir)


if __name__ == '__main__':
    main()