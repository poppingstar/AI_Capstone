import cv2 as cv
import numpy as np
from pathlib import Path
import concurrent.futures as futures
from typing import Iterable, Generator
import itertools
import ignite


def aggregate(func):
    def wrapper(paths: Iterable[Path]) -> Generator[Path, None, None]:
        for path in paths:
            frames = func(path)
            yield from frames
    return wrapper


def get_leaf_files(dir_path: Path | str) -> Generator[Path, None, None]:
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        yield dir_path
    else:
        for child in dir_path.iterdir():
            if child.is_dir():
                yield from get_leaf_files(child)
            else:
                yield child


def read_frame_at(cap: cv.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None


def get_parents_by_files(files: Iterable[Path], *suffixes: str) -> dict[Path, list[str]]:
    suffixes = set(suffixes)
    if suffixes:
        matches = lambda f: f.suffix in suffixes
    else:
        matches = lambda f: True

    grouped: dict[Path, list[str]] = {}
    for f in files:
        if not matches(f):
            continue
        grouped.setdefault(f.parent, []).append(f.name)
    return grouped


def get_frames_at_interval(video_path: Path | str, interval_sec: int) -> list[np.ndarray]:
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv.CAP_PROP_FPS)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps * interval_sec))

    frames = []
    prev_idx = 0
    for idx in range(step, total_frames, step):
        target = np.random.randint(prev_idx, idx)
        frame = read_frame_at(cap, target)
        if frame is None:
            break
        frames.append(frame)
        prev_idx = idx

    cap.release()
    return frames


def filter_similar_imgs(imgs: list[np.ndarray], threshold: float) -> list[np.ndarray]:
    """
    SSIM을 이용해 서로 유사도가 threshold 이상인 이미지는 걸러냅니다.
    """
    unique = []
    pool = imgs.copy()
    while pool:
        ref = pool.pop(0)
        unique.append(ref)
        new_pool = []
        for img in pool:
            ssim_metric = ignite.metrics.SSIM(data_range=255)
            # ignite SSIM expects (y_pred, y)
            ssim_metric.update((torch.from_numpy(ref).permute(2,0,1).float(),
                                torch.from_numpy(img).permute(2,0,1).float()))
            sim = ssim_metric.compute().item()
            if sim < threshold:
                new_pool.append(img)
        pool = new_pool
    return unique


def save(imgs: Iterable[np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs):
        cv.imwrite(str(output_dir / f'{i:04d}.png'), img)


def process_videos(
    video_paths: list[Path],
    output_dir: Path,
    interval_sec: int,
    threshold: float = 0.9
) -> None:
    """
    주어진 비디오 리스트를 순회하며,
    1) interval_sec 초 간격으로 랜덤 프레임 추출
    2) SSIM 기준으로 유사 프레임 필터링
    3) 각 비디오별로 output_dir/{video_stem}/ 아래에 저장
    """
    for video_path in video_paths:
        stem = video_path.stem
        out_subdir = output_dir / stem
        # 1) 프레임 추출
        frames = get_frames_at_interval(video_path, interval_sec)
        if not frames:
            continue
        # 2) 유사 프레임 필터링
        filtered = filter_similar_imgs(frames, threshold)
        # 3) 저장
        save(filtered, out_subdir)


if __name__ == '__main__':
    from pathlib import Path

    root_dir = Path(r"E:\Datasets\딥페이크 변조 영상\1.Train\dffs\dffs")
    output_base = Path(r"E:\Datasets\outputs")
    interval = 5  # 초 단위

    # 1) 모든 leaf 파일 수집
    leaves = list(get_leaf_files(root_dir))
    # 2) .mp4 파일만 부모 디렉터리별로 그룹화
    grouped = get_parents_by_files(leaves, '.mp4')

    # 3) 그룹별로 병렬 처리용 파라미터 준비
    video_groups = [
        [parent / name for name in names]
        for parent, names in grouped.items()
    ]
    out_dirs = [
        output_base / parent.name
        for parent in grouped.keys()
    ]
    intervals = itertools.repeat(interval)

    # 4) 병렬 실행
    with futures.ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(process_videos, video_groups, out_dirs, intervals)
