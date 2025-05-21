import cv2 as cv
import numpy as np
from pathlib import Path
import concurrent.futures as futures
import typing, time, ignite, itertools


def get_leaf_files(dir_path:Path|str) -> typing.Generator[Path]:
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


def get_total_frames(cap:cv.VideoCapture):
    return cap.get(cv.CAP_PROP_FRAME_COUNT)


def get_fps(cap:cv.VideoCapture):
    return cap.get(cv.CAP_PROP_FPS)


def iter_frames_at_interval(video_path:str|Path, interval_sec:int) -> list[np.ndarray]:
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


def process_videos(video_paths:typing.Iterable[Path], output_dir:Path, interval_sec:int):
    output_dir.mkdir(exist_ok=True, parents=True)
    interval_sec = itertools.repeat(interval_sec)
    with futures.ThreadPoolExecutor(max_workers=22) as executor:
        results = executor.map(iter_frames_at_interval, video_paths, interval_sec)

    print('파일 로드 완료')

    i=0
    for frames in results:
        for frame in frames:
            out = output_dir/f'{i}.png'
            cv.imwrite(out, frame)
            i+=1





if __name__ == '__main__':
    def main():
        for i in range(2,7):
            p = Path(fr"E:\Datasets\딥페이크 변조 영상\1.Train\dffs\dffs{i}")
            o = Path(fr"E:\Datasets\outputs")
            fs = get_leaf_files(p)
            process_videos(fs, o, 100)
    main()



# from ignite.engine import Engine
# from ignite.metrics import SSIM
# import torch

# # 평가 단계에서 사용할 함수 정의
# def eval_step(engine, batch):
#     return batch

# # 평가 엔진 생성
# evaluator = Engine(eval_step)

# # SSIM 메트릭 인스턴스 생성
# ssim_metric = SSIM(data_range=1.0)

# # 메트릭을 평가 엔진에 부착
# ssim_metric.attach(evaluator, 'ssim')

# # 예측값과 실제값 생성 (예시용 랜덤 텐서)
# preds = torch.rand([4, 3, 16, 16])
# target = preds * 0.75

# # 평가 실행
# state = evaluator.run([[preds, target]])

# # SSIM 결과 출력
# print(state.metrics['ssim'])