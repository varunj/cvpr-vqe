import argparse
import os
from pathlib import Path
import glob
import subprocess
import multiprocessing as mp


def execute_command(cmd, max_attempts=3):
    # try exectuting cmd up to max_attempts times
    attempts = 0
    while attempts < max_attempts:
        return_val = subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)
        if return_val == 0:
            return 0
        else:
            attempts += 1
    print(f'failed to run command: {cmd}')
    return -1


def run_baseline(args):
    # run AutoAdjust
    cmd = f'AutoAdjustCmdTool.exe -in "{args.dir_in}" -out "{args.dir_out}"'
    status = execute_command(cmd)
    if status == -1:
        assert False


def convert_worker(file):
    path_in = Path(file)
    path_out = path_in.parent / f'{path_in.stem}.mp4'
    cmd = f'ffmpeg -i "{path_in}" -c:v libx264 -crf 19 -vf format=yuvj420p -preset veryslow "{path_out}"'
    status = execute_command(cmd)
    os.remove(path_in)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in', type=str, required=True, help='path to input videos')
    parser.add_argument('--dir_out', type=str, required=True, help='path to result')
    args, unknown = parser.parse_known_args()

    # AutoAdjustCmdTool writes .avi
    run_baseline(args)

    # convert saved .avi file to .mp4
    files = glob.glob(f'{Path(args.dir_out) / "*.avi"}')
    with mp.Pool(processes=4) as mp_pool:
        status = mp_pool.imap_unordered(convert_worker, files)
        for stat in status:
            assert stat == 0
