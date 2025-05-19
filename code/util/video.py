import cv2 as cv
from pathlib import Path
import concurrent.futures as futures
from refine import find_files
import typing, time


def lis(func):
    def wrapper(l):
        rl = []
        for element in l:
            rl.append(func(element))
        return rl
    return wrapper


def get_leaf_files(dir_path:Path|str)->typing.Iterable[Path]:
    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        yield dir_path
    else:
        for child in dir_path.iterdir():
            if child.is_dir():
                yield from get_leaf_files(child)
            else:
                yield child


def capture_video(video_path:str|Path)->cv.VideoCapture|None:
    cap = cv.VideoCapture(video_path)
    return cap if cap.isOpened() else None


def read_frame(video_path):
    cap = capture_video(video_path)
    ret, frame = cap.read()





if __name__ == '__main__':
    p = Path(r"E:\Datasets\딥페이크 변조 영상\1.Train\dfl\dfl1")
    fs = find_files(p)
    vs = map(capture_video, fs)
    time.sleep(19999)