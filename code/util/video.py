import cv2 as cv
import numpy as np
from pathlib import Path
import concurrent.futures as futures
from typing import Iterable, Generator
from copy import deepcopy
from skimage.metrics import structural_similarity as ssim


def aggregate(func):
    def wrapper(paths:Iterable[Path]) -> Generator[Path]:
        for path in paths:
            frames = func(path)
            yield from frames

    return wrapper


def get_leaf_files(dir_path:Path|str) -> Generator[Path]:
    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        yield dir_path
    else:
        for child in dir_path.iterdir():
            if child.is_dir():
                yield from get_leaf_files(child)
            else:
                yield child


def read_frame_at(cap:cv.VideoCapture, frame_idx:int) -> np.ndarray:
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None


def get_parents_by_files(files:Iterable[Path], *suffixes:str)->dict[Path]:
    suffixes = set(suffixes)
    matches_suffix = (lambda x: x.suffix in suffixes) if suffixes else (lambda x: True)
    
    grouped_files = {}
    for f in files:
        if not matches_suffix(f):
            continue
        parent = f.parent
        name = f.name
        if parent not in grouped_files:
            grouped_files[parent] = [name]
        else:
            grouped_files[parent].append(name)
    
    return grouped_files


def get_frames_at_interval(video_path:str|Path, interval_sec:int) -> list[np.ndarray]:
    get_total_frames = lambda cap: cap.get(cv.CAP_PROP_FRAME_COUNT)
    get_fps = lambda cap: cap.get(cv.CAP_PROP_FPS)
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps = get_fps(cap)
    total_frame = int(get_total_frames(cap))
    step = int(fps*interval_sec)

    frames = []
    past_frame_idx = 0
    for frame_idx in range(1, total_frame, step):
        target_frame = np.random.randint(past_frame_idx, frame_idx)
        frame = read_frame_at(cap, target_frame)
        if frame is None:
            break
        frames.append(frame)
        past_frame_idx = frame_idx

    cap.release()
    return frames


def filter_similar_imgs(imgs:list[np.ndarray], threshold:float) -> list:
    non_similar = []
    imgs = imgs.copy()
    while imgs:
        temp = []
        non_similar.append(x)
        x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
        x = imgs[0]
        

        for y in imgs[1:]:
            y = cv.cvtColor(y, cv.COLOR_BGR2RGB)
            similarity = ssim(x, im2=y, data_range=255, channel_axis=-1)
            
            if similarity < threshold:
                temp.append(y)
        
        imgs = temp

    non_similar = [cv.cvtColor(img, cv.COLOR_RGB2BGR) for img in non_similar]
    return non_similar


def save_imgs(imgs:np.ndarray, output_dir:Path) -> None:
    output_dir = Path(output_dir)
    for i, img in enumerate(imgs):
        cv.imwrite(output_dir/f'{i}.png', img)


def process_videos(video_paths:Iterable[Path], output_dir:Path, interval_sec:int):
    output_dir.mkdir(exist_ok=True, parents=True)

    k = video_paths.keys()
    
    total_frames = []   #좀 문제 있음
    for video_path in video_paths.values:
        frame = get_frames_at_interval(video_path,1000)
        total_frames.extend(frame)
    
    non_similar_frames = filter_similar_imgs(total_frames, 0.9)

    output_dir_sub = output_dir/k.parent
    save_imgs(non_similar_frames, output_dir_sub)


if __name__ == '__main__':
    def main():
        root_dir = Path(fr"E:\Datasets\딥페이크 변조 영상\1.Train\dffs")
        output_dir = Path(fr"E:\Datasets\outputs")
        leafs = get_leaf_files(root_dir)
        grouped_files = get_parents_by_files(leafs, '.mp4')

        with futures.ProcessPoolExecutor(max_workers=24) as executor:
            executor.map(process_videos, grouped_files.items())

    # main()