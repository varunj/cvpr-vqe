# Challenge on Video Quality Enhancement for Video Conferencing
Official repository for the first Challenge on Video Quality Enhancement for Video Conferencing held at the NTIRE Workshop at CVPR 2025.



---
## Setup
### Data
`mkdir data; cd data`
`wget https://ic3mi.z22.web.core.windows.net/vqe/train.tar.gz`
`wget https://ic3mi.z22.web.core.windows.net/vqe/test.tar.gz`
`mkdir train; tar -zxf train.tar.gz -C train/`
`mkdir test; tar -zxf test.tar.gz -C test/`
Post extraction
- train should be 139 GB (149,828,544,192 B), 460,600 files
- test should be 42.7 GB (45,944,510,680 B), 153,000 files

To extract frames from videos and make CSVs, run the following command:
`cd vqe`
`python dataset_builder.py --config-name data/all_15fps data.source_dir=<path to data/>`

### Pretrained Models
`cd data`
`wget https://ic3mi.z22.web.core.windows.net/vqe/VQA.ckpt`
`wget https://ic3mi.z22.web.core.windows.net/vqe/DOVER.pth`

### Environment
`conda install -y mamba=1.3.1 -n base -c conda-forge`
`conda create -n vqe python=3.8`
`conda activate vqe`
`mamba env update -f conda/env.gpu.yml`

---
## Baseline
We provide a baseline solution so that participants can reproduce the AutoAdjust feature as currently shipped in Microsoft Teams.
`cd baseline`
`wget https://ic3mi.z22.web.core.windows.net/vqe/autoadjust_bin_win-x64.tar.gz`
`tar -zxf autoadjust_bin_win-x64.tar.gz -C .`
`python baseline.py --dir_in <path to directory with input mp4 videos> --dir_out <path to result directory>`


---
## Training


---
## Evaluation
### Subjective
Remember that final rankings will be based on P.910 scores. It is important to join the Slack workspace where organizers will provide instructions on how to submit the 3000 enhanced videos. The joining link can be found in the CodaLab forum (https://codalab.lisn.upsaclay.fr/forums/21235/).

### Objective
However, for continuous & independent evaluation, teams can track their objective metrics by submitting `metrics_supervised.csv` and `metrics_unsupervised.csv` to CodaLab in a zip.
The metric m used to rank is a combination of
1. For 3000 unsupervised videos: VQA score v & 11 auxiliary scores a1 ... a11
3. For 500 supervised videos: RMSE on synthetic data r

`m = [(v+a1+a2+a3+(1-a4)+a5+a6+a7+a8+a9+a10+a11)/12]/r`

You can check your score offline by running:
```
python codalab_scoring/score.py --results codalab_scoring/sample_submission
```
