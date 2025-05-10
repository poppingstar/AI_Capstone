from pathlib import Path
import pandas as pd
import shutil

def labeling(dataset_path, label_file):
    dataset_path=Path(dataset_path)
    outputdir=dataset_path/'output'
    outputdir.mkdir(exist_ok=True)

    df=pd.read_csv(label_file)
    df=df[['file_name', 'label']]
    
    for row in df.itertuples():
        file=Path(row.file_name)
        original_path=dataset_path/file
        label_dir=outputdir/str(row.label)
        label_dir.mkdir(exist_ok=True)

        original_path.rename(label_dir/file.name)

def split(dataset_path:Path, val_rate=0.2, test_rate=0.1):
    train_dir=dataset_path/'train'
    for d in train_dir.iterdir():
        if not d.is_dir:
            continue

        flist=[f for f in d.iterdir()]
        num_files=len(flist)

        val_size=int(num_files*val_rate)
        test_size=int(num_files*test_rate)
        
        val_files=flist[:val_size]
        test_files=flist[val_size:val_size+test_size]

        val_dir=dataset_path/'valid'/d.name
        val_dir.mkdir(exist_ok=True, parents=True)
        for file in val_files:
            file.rename(val_dir/file.name)

        test_dir=dataset_path/'test'/d.name
        test_dir.mkdir(exist_ok=True, parents=True)
        for file in test_files:
            file.rename(test_dir/file.name)

def find_low_dirs(path:Path):
    while path.is_dir():
        path = path.iterdir()[0]
    low_dirs = path.parent.iterdir()
    return low_dirs

#멀티 스레딩 포함해서 재설계 ㄱㄱ
def main(dataset:Path, destination:Path, num_per_class:int):
    class_dirs = dataset.iterdir()

    for class_dir in class_dirs:
        sub_dirs = list(class_dir.iterdir())        
        file_num_maps={}
        for sub_dir in sub_dirs:
            low_dirs = list(sub_dir.iterdir())
            for low_dir in low_dirs:
                imgs = list(low_dir.iterdir())
                file_num_maps[low_dir]=len(imgs)

        num = num_per_class
        a = []
        while num > 0:
            m = min(file_num_maps.values())
            
            for k, v in file_num_maps.items():
                num -= m
                if num < 0:
                    m = -1*num
                    break

                for img in k.iterdir():
                    shutil.copy(img, destination / class_dir.name / img.name)

            file_num_maps = {k:v-m for k,v in file_num_maps.items() if v > 0}

if __name__ == '__main__':
    dataset_path = Path(r"C:\Users\user\Desktop\deepfake and real images")
    split(dataset_path)