from abc import ABC
import torch
from omegaconf import DictConfig, ListConfig
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from importlib.metadata import version
from pytorch_lightning.utilities import rank_zero_only
import pandas as pd
import onnx
from src.checkpoint import ONNXExportCallback, get_resume_chkpt
from src.dataset import ImageVQEDataset, DDPGroupedBatchSampler
from src.DOVER import label_dover
import os
import logging
import thop
import functools
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader
import torch.distributed as td
from src.utils import suppress_error_stream
from typing import Optional, Union, List


class VQEModel(LightningModule, ABC):
    def __init__(self, cfg):
        super(VQEModel, self).__init__()
        cfg.model_vqe.ideal_width = cfg.dataset.width
        cfg.model_vqe.ideal_height = cfg.dataset.height
        cfg.model_vqa.ideal_width = cfg.dataset.width
        cfg.model_vqa.ideal_height = cfg.dataset.height
        self._model_vqe = instantiate(cfg.model_vqe)
        self._csv_sup = cfg.result_csv_supervised
        self._csv_unsup = cfg.result_csv_unsupervised
        self._cfg = cfg
        self._loss = instantiate(cfg.loss)

        # load vqa model
        assert cfg.model_vqa.weights is not None
        self._model_vqa = instantiate(cfg.model_vqa)

        # load dover model
        assert cfg.model_dover.weights is not None
        self._dover_cfg = instantiate(cfg.model_dover)
        self._model_dover, self._sampler_dover = label_dover.setup(self._dover_cfg)

    def on_pretrain_routine_start(self) -> None:
        self.prepare_csv_data()

    def on_pretrain_routine_end(self):
        if self.trainer.logger is not None:
            self.trainer.logger.log_hyperparams(self._cfg)

    @rank_zero_only
    @suppress_error_stream
    def save_onnx(self, file_path, *extra_args):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        if callable(getattr(self._model_vqe, 'onnx_export', None)):
            onnx.save(self._model_vqe.onnx_export(), file_path)

    @staticmethod
    def prepare_loader(
            ds,
            root_folder,
            aug_policy=None,
            drop_last=False,
            shuffle=False,
            height=0,
            width=0,
            batch_size=1,
            pin_memory=False,
            multiprocessing_context='fork',
            num_workers=0,
            seed=0
    ):
        groupby = ['h', 'w', 'subset', 'video_id']
        ds['img_target_present'] = pd.notna(ds['img_target'])
        if ds['img_target'].isna().any():
            groupby.append('img_target_present')

        dataset = ImageVQEDataset(ds, root_folder, augmentation_policy=aug_policy, h=height, w=width)
        world_size = 1
        rank = 0
        if td.is_initialized():
            world_size = td.get_world_size()
            rank = td.get_rank()

        return DataLoader(
            dataset,
            batch_sampler=DDPGroupedBatchSampler(
                ds,
                group_by=groupby,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
                batch_size=batch_size,
                drop_last=drop_last,
                shuffle=shuffle,
                dropna=False
            ),
            pin_memory=pin_memory,
            num_workers=num_workers,
            multiprocessing_context=multiprocessing_context
        )

    @staticmethod
    def filter_csv(csv, query):
        df = pd.read_csv(csv, low_memory=False)
        bn = os.path.basename(csv)
        log = logging.getLogger('lightning')
        if query is None or query == 'none' or query == '':
            log.info('Loaded %d rows from %s.' % (len(df), bn))
        else:
            full_len = len(df)
            df = df.query(query, engine='python')
            log.info('Loaded %d/%d rows from %s using <%s>.' % (len(df), full_len, bn, query))
        return df

    def prepare_csv_data(self):
        log = logging.getLogger('lightning')
        self._train_set = self.filter_csv(
            os.path.join(self._cfg.dataset.path, self._cfg.dataset.train),
            self._cfg.dataset.train_query
        )
        self._test_set = self.filter_csv(
            os.path.join(self._cfg.dataset.path, self._cfg.dataset.holdout),
            self._cfg.dataset.holdout_query
        )
        self._valid_set = self.filter_csv(
            os.path.join(self._cfg.dataset.path, self._cfg.dataset.valid),
            self._cfg.dataset.valid_query
        )
        log.info('preparing train loader')
        self._train_loader = VQEModel.prepare_loader(
            self._train_set, self._cfg.dataset.path,
            aug_policy=self._cfg.augmentation,
            drop_last=True,
            shuffle=True,
            height=self._cfg.dataset.height,
            width=self._cfg.dataset.width,
            **self._cfg.dataloader
        )
        log.info('preparing test loader')
        self._test_loader = VQEModel.prepare_loader(
            self._test_set, self._cfg.dataset.path,
            drop_last=False,
            shuffle=False,
            height=self._cfg.dataset.height,
            width=self._cfg.dataset.width,
            **self._cfg.dataloader
        )
        log.info('preparing valid loader')
        self._valid_loader = VQEModel.prepare_loader(
            self._valid_set, self._cfg.dataset.path,
            drop_last=False,
            shuffle=False,
            height=self._cfg.dataset.height,
            width=self._cfg.dataset.width,
            **self._cfg.dataloader
        )

    def on_epoch_start(self) -> None:
        self._train_loader.batch_sampler.set_epoch(self.current_epoch)

    def forward(self, x):
        y_hat = self._model_vqe(x)
        return y_hat

    def batch_eval(self, batch):
        # [b, 3, h, w]
        img_x = batch['img_x']
        # [b, 3, h, w] or [b, 0] if unsupervised sample
        img_y = batch['img_y']
        is_unsupervised = img_y.shape[1] == 0
        # list of len b
        subset, video_id = batch['subset'], batch['video_id']

        if self.precision == 16:
            img_x = img_x.half()
            img_y = img_y.half()
        else:
            img_x = img_x.float()
            img_y = img_y.float()

        img_x = (img_x - 128.) / 128.
        img_y = (img_y - 128.) / 128.
        y_hat = self(img_x)

        # pass data through VQA model
        b = img_x.shape[0]
        scores_dover_x = torch.zeros(b, 1).to(img_x)
        for idx in range(b):
            scores_dover_x[idx] = label_dover.label_dover(
                self._dover_cfg,
                self._model_dover,
                self._sampler_dover,
                img_x[idx]
            )
        scores_dover_y_hat = torch.zeros(b, 1).to(y_hat)
        for idx in range(b):
            scores_dover_y_hat[idx] = label_dover.label_dover(
                self._dover_cfg,
                self._model_dover,
                self._sampler_dover,
                y_hat[idx]
            )
        p_vqa, aux_vqa_y_hat, aux_vqa_x = self._model_vqa(
            y_hat,
            img_x,
            scores_dover_y_hat,
            scores_dover_x
        )

        return img_x, img_y, y_hat, p_vqa, aux_vqa_y_hat, is_unsupervised, subset, video_id

    def compute_loss(self, y_hat, y, p, aux, is_unsupervised):
        # vqa loss
        # you may clamp aux vals to [0,1]
        aux[:,3] = 1 - aux[:,3]
        scores = torch.cat((p, aux), dim=1)
        # MSE loss wrt [1]
        loss = self._loss(scores, torch.ones_like(scores))
        # supervised loss
        if not is_unsupervised:
            loss += self._loss(y_hat, y)
        return loss

    def training_step(self, batch, batch_nb):
        x, y, y_hat, p, aux, is_unsup, subset, video_id = self.batch_eval(batch)
        loss = self.compute_loss(y_hat, y, p, aux, is_unsup)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def compute_metrics(self, y_hat, y, is_unsupervised, prefix=''):
        metrics = dict()
        if not is_unsupervised:
            metrics[prefix + 'mse'] = torch.nn.functional.mse_loss(y_hat, y, reduction='mean')
        return {**metrics}

    @staticmethod
    def aggregate_metrics(outputs):
        metrics = dict()
        all_keys = set()
        for d in outputs:
           all_keys.update(d.keys())
        all_keys.discard('supervised')
        all_keys.discard('unsupervised')
        all_keys = sorted(all_keys)
        for key in all_keys:
            metrics[key] = torch.stack([x[key] for x in outputs if key in x]).mean()
        return metrics

    @staticmethod
    def save_metrics_csv(outputs, file_df_unsup, file_df_sup):
        metrics_unsup = dict()
        metrics_sup = dict()
        # aggregate minibatches
        for out in outputs:
            if 'unsupervised' in out:
                for v in out['unsupervised']:
                    if v in metrics_unsup:
                        metrics_unsup[v].extend(out['unsupervised'][v])
                    else:
                        metrics_unsup[v] = out['unsupervised'][v]
            if 'supervised' in out:
                for v in out['supervised']:
                    if v in metrics_sup:
                        metrics_sup[v].extend(out['supervised'][v])
                    else:
                        metrics_sup[v] = out['supervised'][v]
        # aggregate per video
        for v in metrics_unsup:
            metrics_unsup[v] = torch.stack(metrics_unsup[v]).mean(0).cpu().numpy()
        for v in metrics_sup:
            metrics_sup[v] = torch.stack(metrics_sup[v]).mean(0).unsqueeze(0).cpu().numpy()
        df_unsup = pd.DataFrame(metrics_unsup).transpose().reset_index()
        df_sup = pd.DataFrame(metrics_sup).transpose().reset_index()
        df_unsup.set_axis(['video_i','m_1','m_2','m_3','m_4','m_5','m_6','m_7','m_8','m_9','m_10','m_11','m_12'], axis=1, inplace=True)
        df_sup.set_axis(['video_i','rmse'], axis=1, inplace=True)
        df_unsup.to_csv(file_df_unsup, index=False)
        df_sup.to_csv(file_df_sup, index=False)

    def validation_step(self, batch, batch_nb):
        x, y, y_hat, p, aux, is_unsup, subset, video_id = self.batch_eval(batch)
        loss = self.compute_loss(y_hat, y, p, aux, is_unsup)
        metrics = dict()
        metrics.update(self.compute_metrics(y_hat, y, is_unsup, prefix='val_'))
        return {'val_loss': loss, **metrics}

    def get_metrics_csv_data(self, y_hat, y, p, aux, is_unsupervised, video_id):
        if not is_unsupervised:
            tag = 'supervised'
            data = torch.mean(
                torch.nn.functional.mse_loss(y_hat, y, reduction='none'), dim=(1,2,3)
            ) ** 0.5
        else:
            tag = 'unsupervised'
            data = torch.cat((p, aux), dim=1)
        metrics = dict()
        metrics[tag] = {}
        b = y_hat.shape[0]
        for idx in range(b):
            video_name = video_id[idx]
            if video_name in metrics[tag]:
                metrics[tag][video_name].append(data[idx])
            else:
                metrics[tag][video_name] = [data[idx]]
        return {**metrics}

    def test_step(self, batch, batch_nb):
        x, y, y_hat, p, aux, is_unsup, subset, video_id = self.batch_eval(batch)
        csv_data = self.get_metrics_csv_data(y_hat, y, p, aux, is_unsup, video_id)
        metrics = self.compute_metrics(y_hat, y, is_unsup, prefix='test_')
        return {**metrics, **csv_data}

    def validation_epoch_end(self, outputs):
        metrics = self.aggregate_metrics(outputs)
        # note that logging here is done by x=epoch and not x=step
        # so it would not be x-axis aligned with train metrics
        self.log_dict({**metrics}, prog_bar=True)

    def test_epoch_end(self, outputs):
        metrics = self.aggregate_metrics(outputs)
        self.save_metrics_csv(outputs, self._csv_unsup, self._csv_sup)
        self.log_dict({**metrics})

    def configure_optimizers(self):
        optimizer = instantiate(self._cfg.optimizer, self._model_vqe.parameters())
        if 'scheduler' in self._cfg and self._cfg.scheduler is not None:
            scheduler = instantiate(self._cfg.scheduler, optimizer)
            return [optimizer], [scheduler]
        return [optimizer], []

    def train_dataloader(self):
        return self._train_loader

    def test_dataloader(self):
        return self._test_loader

    def val_dataloader(self):
        return self._valid_loader

    @functools.cached_property
    def macs_params(self):
        x_ = torch.randn(1, 3, 720, 1280)
        macs, params = thop.profile(instantiate(self._cfg.model_vqe), inputs=(x_), verbose=False)
        if macs > 20e9:
            logging.getLogger('lightning').warning('macs exceed the limit')
        return macs, params

    @rank_zero_only
    def performance_metrics(self):
        logging.getLogger('lightning').info('macs: %d' % self.macs_params[0])
        logging.getLogger('lightning').info('params: %d' % self.macs_params[1])


@hydra.main(config_path='config')
def train(cfg: DictConfig) -> Optional[Union[float, List[float]]]:
    log = logging.getLogger('lightning')
    log.setLevel(logging.INFO)
    log.info('PL version %s' % version('pytorch-lightning'))
    log.info('cwd %s' % os.getcwd())
    vqe_model = VQEModel(cfg)
    loggers = list()

    if cfg.tb_logger:
        loggers.append(TensorBoardLogger(os.getcwd(), name='', version=''))

    monitor_value = 'val_mse'

    callbacks = []

    checkpoint = ModelCheckpoint(
        dirpath='%s/checkpoints' % os.getcwd(),
        filename='{epoch:04d}_{%s:.4f}' % monitor_value,
        save_weights_only=True,
        monitor=monitor_value,
        every_n_epochs=1,
        save_top_k=1,
        mode='min'
    )

    callbacks.append(checkpoint)

    callbacks.append(ModelCheckpoint(
        dirpath='%s/checkpoints' % os.getcwd(),
        filename=cfg.resume_from,
        save_weights_only=False,
        every_n_epochs=1)
    )

    if 'early_stopping' in cfg and cfg.early_stopping:
        callbacks.append(instantiate(cfg.early_stopping))

    if cfg.onnx_export:
        callbacks.append(ONNXExportCallback(
            dirpath='%s/onnx' % os.getcwd(),
            filename='{epoch:04d}_{%s:.4f}' % monitor_value,
            monitor=monitor_value,
            every_n_epochs=1,
            save_top_k=1,
            mode='min')
        )

    if 'scheduler' in cfg and cfg.scheduler:
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    else:
        log.info('no scheduler is provided, constant learning rate %f is used' % cfg.lr)

    if cfg.progress_bar_refresh_rate > 0:
        callbacks.append(TQDMProgressBar(refresh_rate=cfg.progress_bar_refresh_rate))

    trainer = Trainer(
        auto_lr_find=cfg.auto_lr_find,
        gpus=cfg.gpus,
        num_nodes=cfg.nodes if hasattr(cfg, 'nodes') and cfg.nodes else 1,
        logger=loggers,
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        callbacks=callbacks,
        benchmark=cfg.cudnn_benchmark,
        val_check_interval=cfg.val_check_interval,
        amp_backend=cfg.amp_backend,
        strategy=DDPPlugin(find_unused_parameters=False),
        gradient_clip_val=cfg.gradient_clip_val,
        gradient_clip_algorithm=cfg.gradient_clip_algorithm,
        enable_progress_bar=cfg.progress_bar_refresh_rate > 0,
        replace_sampler_ddp=False,
        num_sanity_val_steps=cfg.num_sanity_val_steps
    )

    # do only validation
    if cfg.only_validation:
        vqe_model.on_pretrain_routine_start()
        trainer.validate(vqe_model)
        return

    # train and time training
    trainer.fit(vqe_model, ckpt_path=get_resume_chkpt(cfg.resume_from))

    # test model
    test_results = trainer.test(vqe_model)[0]
    vqe_model.performance_metrics()

    # flush loggers
    if cfg.tb_logger:
        for each_logger in loggers:
            if isinstance(each_logger, TensorBoardLogger):
                each_logger.experiment.flush()

    ret = None
    if 'sweeper_object' in cfg:
        if isinstance(cfg.sweeper_object, ListConfig):
            ret = [test_results[p] for p in cfg.sweeper_object]
        else:
            ret = test_results[cfg.sweeper_object]

    return ret


if __name__ == '__main__':
    train()
