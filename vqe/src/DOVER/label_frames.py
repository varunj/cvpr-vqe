import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import yaml
from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition_for_images
from dover.models import DOVER


mean, std = (
    torch.FloatTensor([123.675, 116.28, 103.53]),
    torch.FloatTensor([58.395, 57.12, 57.375]),
)


def fuse_results(results: list):
    x = (results[1] - 0.1107) / 0.07355 * 0.6104 + (
        results[0] + 0.08285
    ) / 0.03774 * 0.3896
    return 1 / (1 + np.exp(-x))


def get_model(cfg, args):
    evaluator = DOVER(**cfg['model']['args']).to('cuda')
    evaluator.load_state_dict(
        torch.load(args.path_model, map_location='cuda')
    )
    # freeze model
    for param in evaluator.parameters():
        param.requires_grad = False
    return evaluator


def get_temporal_samplers(cfg):
    cfg_data = cfg['data']['val-l1080p']['args']
    temporal_samplers = {}
    for branch_type, cfg_branch in cfg_data['sample_types'].items():
        if 't_frag' not in cfg_branch:
            # resized temporal sampling for TQE in DOVER
            temporal_samplers[branch_type] = UnifiedFrameSampler(
                cfg_branch['clip_len'], cfg_branch['num_clips'], cfg_branch['frame_interval']
            )
        else:
            # temporal sampling for AQE in DOVER
            temporal_samplers[branch_type] = UnifiedFrameSampler(
                cfg_branch['clip_len'] // cfg_branch['t_frag'],
                cfg_branch['t_frag'],
                cfg_branch['frame_interval'],
                cfg_branch['num_clips'],
            )
    return temporal_samplers


def label(cfg, model, temporal_samplers, path_imgs):
    cfg_data = cfg['data']['val-l1080p']['args']
    # view decomposition
    views, views_idx = spatial_temporal_view_decomposition_for_images(
        path_imgs, cfg_data['sample_types'], temporal_samplers, num=32
    )

    for k, v in views.items():
        num_clips = cfg_data['sample_types'][k].get('num_clips', 1)
        views[k] = (
            ((v.permute(1, 2, 3, 0) - mean) / std)
            .permute(3, 0, 1, 2)
            .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
            .transpose(0, 1)
            .to('cuda')
        )

    results = [r.mean().item() for r in model(views)]
    # normalized fused overall score in range [0,1]
    return fuse_results(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='dover.yml')
    parser.add_argument('--path_csv_in', type=str, default='')
    parser.add_argument('--path_csv_out', type=str, default='')
    parser.add_argument('--path_model', type=str, default='')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(args.path_csv_in)
    model = get_model(cfg, args)
    temporal_samplers = get_temporal_samplers(cfg)

    df['score_dover_L'] = pd.Series(dtype='float')
    df['score_dover_R'] = pd.Series(dtype='float')
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        path_imgs_L = str(Path(args.path_csv_in).parent / row['file_name_L'])
        path_imgs_R = str(Path(args.path_csv_in).parent / row['file_name_R'])
        score_L = label(cfg, model, temporal_samplers, path_imgs_L)
        score_R = label(cfg, model, temporal_samplers, path_imgs_R)
        df.at[row.name, 'score_dover_L'] = score_L
        df.at[row.name, 'score_dover_R'] = score_R

    df.to_csv(args.path_csv_out, index=False)
