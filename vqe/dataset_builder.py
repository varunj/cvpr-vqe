from omegaconf import DictConfig
import hydra
import logging
import os
import cv2
import multiprocessing as mp
import subprocess
import pandas as pd
from pathlib import Path
import glob
from tqdm import tqdm


def execute_command(cmd, max_attempts=3):
    # try exectuting cmd up to max_attempts times
    attempts = 0
    while attempts < max_attempts:
        return_val = subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)
        if return_val == 0:
            return 0
        else:
            attempts += 1
    return -1


def init_worker(videos, fps):
    worker.videos = videos
    worker.fps = fps


def worker(idx: int):
    video = worker.videos[idx]
    path_root = f'{video[:-6]}/{video[-5:-4]}'
    os.makedirs(path_root, exist_ok=True)
    log = logging.getLogger(f'ds.build.{video}')
    log.info(f'extracting {video}')
    cmd = f'ffmpeg -i {video} -vf fps={worker.fps} {path_root}/frame_%04d.png'
    status = execute_command(cmd)
    if status == -1:
        assert False
    return True


def store(data, df, count, root_dir, subset):
    for idx in tqdm(range(len(data))):
        h, w, _ = cv2.imread(data[idx]).shape
        root = Path(data[idx]).relative_to(root_dir).parent.parent
        name = Path(data[idx]).name
        if 'unsupervised' in subset:
            target = None
        else:
            target = root / 'y' / name
        df.loc[count] = [
                root / 'x' / name,
                target,
                h, w,
                subset,
                root.name
        ]
        count += 1
    return count


@hydra.main(config_path='config', config_name='data/test')
def dataset_builder(cfg: DictConfig):
    cfg = cfg.data
    source_dir = Path(cfg.source_dir)
    fps = cfg.video_fps
    num_workers = mp.cpu_count()
    df_train = pd.DataFrame(columns=['img_source', 'img_target', 'h', 'w', 'subset', 'video_id'])
    df_test = pd.DataFrame(columns=['img_source', 'img_target', 'h', 'w', 'subset', 'video_id'])
    log = logging.getLogger('ds.build')


    # 1. extract frames from videos
    all_videos = glob.glob(f'{source_dir}/train/supervised/real/video_*.mp4')
    all_videos += glob.glob(f'{source_dir}/train/unsupervised/video_*.mp4')
    all_videos += glob.glob(f'{source_dir}/test/unsupervised/video_*.mp4')
    mp_pool = mp.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(all_videos, fps)
    )
    results = list(tqdm(
        mp_pool.imap_unordered(worker, range(len(all_videos))),
        total=len(all_videos),
        mininterval=1
    ))
    assert sum(results) == len(all_videos)


    # 2. read all frames
    # process train paired data
    train_sup_synth_x = glob.glob(f'{source_dir}/train/supervised/synthetic/video_*/x/*jpg')
    train_sup_synth_y = glob.glob(f'{source_dir}/train/supervised/synthetic/video_*/y/*jpg')
    train_sup_real_x = glob.glob(f'{source_dir}/train/supervised/real/video_*/x/*png')
    train_sup_real_y = glob.glob(f'{source_dir}/train/supervised/real/video_*/y/*png')
    assert len(train_sup_synth_x) == len(train_sup_synth_y)
    assert len(train_sup_real_x) == len(train_sup_real_y)
    # process train unpaired data
    train_unsup_x = glob.glob(f'{source_dir}/train/unsupervised/video_*/x/*png')

    # process test paired data
    test_sup_synth_x = glob.glob(f'{source_dir}/test/supervised/synthetic/video_*/x/*jpg')
    test_sup_synth_y = glob.glob(f'{source_dir}/test/supervised/synthetic/video_*/y/*jpg')
    assert len(test_sup_synth_x) == len(test_sup_synth_y)
    # process test unpaired data
    test_unsup_x = glob.glob(f'{source_dir}/test/unsupervised/video_*/x/*png')


    # 3. make dataframes
    c_train, c_test = 0, 0
    c_train = store(train_sup_synth_x, df_train, c_train, source_dir, 'train_supervised_synthetic')
    c_train = store(train_sup_real_x, df_train, c_train, source_dir, 'train_supervised_real')
    c_train = store(train_unsup_x, df_train, c_train, source_dir, 'train_unsupervised')
    c_test = store(test_sup_synth_x, df_test, c_test, source_dir, 'test_supervised_synthetic')
    c_test = store(test_unsup_x, df_test, c_test, source_dir, 'test_unsupervised')
    df_train.to_csv(source_dir / 'train.csv', index=False)
    df_test.to_csv(source_dir / 'test.csv', index=False)


if __name__ == '__main__':
    dataset_builder()
