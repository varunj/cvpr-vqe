# Challenge on Video Quality Enhancement for Video Conferencing
Official repository for the first Challenge on Video Quality Enhancement for Video Conferencing held at the NTIRE Workshop at CVPR 2025.


## Setup
### Data
`wget https://ic3mi.z22.web.core.windows.net/vqe/train.tar.gz`
`wget https://ic3mi.z22.web.core.windows.net/vqe/test.tar.gz`

### Pretrained Models
`wget https://ic3mi.z22.web.core.windows.net/vqe/VQA.ckpt`
`wget https://ic3mi.z22.web.core.windows.net/vqe/DOVER.pth`

### Environment
`conda install -y mamba=1.3.1 -n base -c conda-forge`
`conda create -n vqe python=3.8`
`conda activate vqe`
`mamba env update -f conda/env.gpu.yml`


## Training


## Scoring
Remember that final ranking will be based on P.910 scores. However, for continuous & independent evaluation, teams can track their objective metrics by submitting `metrics_supervised.csv` and `metrics_unsupervised.csv` to CodaLab in a zip.

The metric m used to rank is a combination of
1. For 3000 unsupervised videos: VQA score v & 11 auxiliary scores a1 ... a11
3. For 500 supervised videos: RMSE on synthetic data r

`m = [(v+a1+a2+a3+(1-a4)+a5+a6+a7+a8+a9+a10+a11)/12]/r`

You can check your score offline by running:
``` 
python codalab_scoring/score.py --results codalab_scoring/sample_submission
```
