import cv2 as cv
from pathlib import Path
import typing, shutil, re


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


class VideoProcessor():
    def __init__(self, video_path):
        self.path = Path(video_path)
        self.cap = cv.VideoCapture(self.path)
    
    def video_open(self):
        if self.cap.isOpened():
            self.video_frame = self.cap.read()
        else:
            print(f'{self.path}를 열 수 없습니다')


def organize_files_by_regex(path:str|Path, expression:str|Path, file_only = True) -> None:
    """
    root_dir 내의 파일을 검색해,
    정규식 `expression`에 매칭된 문자열을 폴더명으로 사용해 이동시킵니다.

    Args:
        root_dir (str | Path): 탐색할 최상위 디렉토리 경로
        expression (str): 매칭할 정규식 패턴
    """
    d = Path(path)
    pattern = re.compile(expression)
    for child in d.iterdir():
        if file_only and child.is_dir():
            continue

        match_result = pattern.search(child.name)

        if not match_result:
            continue

        matched_string = match_result.group(0)
        group_dir = child.parent/matched_string
        group_dir.mkdir(exist_ok=True)

        dst = group_dir/child.name

        if child.resolve() == dst.resolve():
            continue

        shutil.move(child, dst)


if __name__ == '__main__':
    square_brackets = r'(\[.+\])+'
    organize_files_by_regex(r"H:\train", r'^\D+', True)