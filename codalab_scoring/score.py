import argparse
from pathlib import Path
import glob
import pandas as pd


def read_data(path_submission):
    path_submission = Path(path_submission)
    result_paths = glob.glob(f'{path_submission / "*.csv"}')
    result_names = [Path(x).name for x in result_paths]
    assert len(result_names) == 2
    assert 'metrics_supervised.csv' in result_names
    assert 'metrics_unsupervised.csv' in result_names
    df_real = pd.read_csv(f'{path_submission / "metrics_unsupervised.csv"}')
    df_synth = pd.read_csv(f'{path_submission / "metrics_supervised.csv"}')
    assert len(df_real) == 3000
    assert len(df_synth) == 500
    assert set(df_real.columns) == set(['video_i', 
        'm_1','m_2','m_3','m_4','m_5','m_6','m_7','m_8','m_9','m_10','m_11','m_12'
    ])
    assert set(df_synth.columns) == set(['video_i', 'rmse'])
    return df_real, df_synth


def compute_score(df_real, df_synth):
    # handle missing data
    missing_error = 9999
    missing_score = -missing_error
    n_nans_real = df_real.isna().sum().sum()
    n_nans_synth = df_synth.isna().sum().sum()
    df_real.fillna(missing_score, inplace=True)
    df_synth.fillna(missing_error, inplace=True)
    print(f'found {n_nans_real} in metrics_unsupervised, {n_nans_synth} in metrics_supervised')

    # compute score
    means_real = df_real.iloc[:, 1:].mean()
    means_synth = df_synth.iloc[:, 1:].mean()
    metric_1 = (
                    means_real['m_1']
                    + means_real['m_2']
                    + means_real['m_3']
                    + means_real['m_4']
                    + 1 - means_real['m_5']
                    + means_real['m_6']
                    + means_real['m_7']
                    + means_real['m_8']
                    + means_real['m_9']
                    + means_real['m_10']
                    + means_real['m_11']
                    + means_real['m_12']
                ) / 12
    metric_2 = means_synth['rmse']
    final_score = metric_1 / metric_2

    print(f'means real  = {means_real}')
    print(f'mean synth  = {means_synth}')
    print(f'final score = {final_score:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default=None, help='')
    args = parser.parse_args()

    df_real, df_synth = read_data(args.results)
    compute_score(df_real, df_synth)
