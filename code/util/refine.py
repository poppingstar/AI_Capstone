import cv2 as cv
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor 
import shutil, os
import matplotlib.pyplot as plt


def get_saturation(img_path):
    bgr_img = cv.imread(img_path)
    hsv_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)
    
    saturation = hsv_img[:, :, 1]
    avg_saturation = np.mean(saturation)
    print(avg_saturation)


def get_mean(img_path):
    bgr_img = cv.imread(img_path)
    hsv_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)
    rgb_img = cv.cvtColor(hsv_img, cv.COLOR_HSV2RGB)
    print(rgb_img.mean())


def is_almost_gray(bgr_img:str)->bool:
    hsv_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)    
    saturation = hsv_img[:, :, 1]
    avg_saturation = np.mean(saturation)
    
    is_gray = True if avg_saturation <22 else False
    return is_gray


def find_img_files(img_dir:str) -> list[Path]:
    img_suffixs = ['.png', '.jpg']
    dir_path = Path(img_dir)
    img_paths = [f for f in dir_path.iterdir() if f.suffix in img_suffixs]
    return img_paths


def get_imgs(paths:list[str])->np.ndarray[np.ndarray]:
    max_threads_num = os.cpu_count()
    subset_size = len(paths)//max_threads_num

    if len(paths) <= max_threads_num:
        subsets = [paths]
    else: 
        subsets = [paths[(i)*subset_size:(i+1)*subset_size] for i in range(max_threads_num)]
        subsets[-1].extend(paths[max_threads_num*subset_size:])
        
    read_imgs = lambda path_subset: [cv.imread(path) for path in path_subset]
    with ThreadPoolExecutor(max_threads_num) as executor:
        futures = [executor.submit(read_imgs, subset) for subset in subsets]
    
    bgr_imgs = []
    for future in futures:
        bgr_imgs.extend(future.result())
    return np.array(bgr_imgs)


def gray_seperation(img_dir):
    img_dir = Path(img_dir)
    img_paths = find_img_files(img_dir)
    
    imgs = get_imgs(img_paths)
    gray_img_paths = [img_paths[i] for i, img in enumerate(imgs) if is_almost_gray(img)]
    
    seperation_dir = img_dir.parent/'gray'/img_dir.name
    seperation_dir.mkdir(exist_ok=True, parents=True)

    for gray_img in gray_img_paths:
        shutil.move(gray_img, seperation_dir/gray_img.name)


def refine_same_img(imgs:np.ndarray[np.ndarray]):
    img_shape = imgs.shape[1:]
    img_dtype = imgs.dtype

    img_bytes = [img.tobytes() for img in imgs]
    _, indices = np.unique(img_bytes, return_index=True)

    unique_imgs = []
    for i in indices:
        unique_imgs.append(imgs[i])
    
    return unique_imgs


def sperate_unique_imgs(src_dir:str, dst_dir:str):
    paths = list(Path(src_dir).iterdir())
    imgs = get_imgs(paths)
    
    unique_imgs = refine_same_img(imgs)

    for i, img in enumerate(unique_imgs):
        cv.imwrite(f'{dst_dir}/{i}.png', img)


def equalize_classes(*class_dirs):
    min_class_size = float('inf')
    all_class_files = []

    for class_dir in class_dirs:
        files = list(Path(class_dir).iterdir())
        min_class_size =  min_class_size if min_class_size < len(files) else len(files)
        all_class_files.append(files)
    
    for files in all_class_files:
        excess_files = files[min_class_size:]
        for excess_file in excess_files:
            shutil.move(excess_file, rf"E:\Datasets\deep_real\e\{excess_file.name}")
    

if __name__ == '__main__':
    equalize_classes(r"E:\Datasets\deep_real\ufake", r"E:\Datasets\deep_real\ureal")

# def temp(d1, d2):
#     d1_flist = list(Path(d1).iterdir())
#     d2_flist = list(Path(d2).iterdir())
#     total_flist = d1_flist+d2_flist

#     real_filst = [f for f in total_flist if f.name.split('_')[0] == 'real']
#     fake_filst = [f for f in total_flist if f.name.split('_')[0] == 'fake']

#     for f in total_flist:
#         if f.name.split('_')[0] == 'real':
#             shutil.move(f, fr"E:\Datasets\deep_real\r\{f.name}")
#         else:
#             shutil.move(f, fr"E:\Datasets\deep_real\f\{f.name}")