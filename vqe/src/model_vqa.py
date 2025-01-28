import onnx
import torch
import torch.nn as nn
from io import BytesIO
from collections import OrderedDict
from src.operations import make_downsample, make_block_list


class UnetLite(nn.Module):
    def __init__(self, **kwargs):
        super(UnetLite, self).__init__()
        self._cfg = kwargs
        assert 'skip_mix' not in self._cfg['level'][-1] or self._cfg['level'][-1]['skip_mix'] is None, \
            'lowest resolution level should have skip_mix'
        assert 'downsample' not in self._cfg['level'][-1] or self._cfg['level'][-1]['downsample'] is None, \
            'lowest resolution level should not have downsampler'

        self._encoder = nn.ModuleList()
        self._num_levels = len(self._cfg['level'])
        self._fc_dropout = self._cfg['fc_dropout']

        self._nos_ch_aux = 11
        self._nos_ch_dover = 1

        # create encoders
        channels_in = self._cfg['input_channels']
        for level, level_config in enumerate(self._cfg['level']):
            self._encoder.append(
                make_block_list(
                    channels_in,
                    level_config['encoder']['m1_blocks'],
                    level_config['encoder']['activation'],
                    self._cfg['over_parametrize'],
                    se_block=level_config['encoder']['se_block'] if 'se_block' in level_config['encoder'] else None
                ))
            if len(level_config['encoder']['m1_blocks']) > 0:
                channels_in = level_config['encoder']['m1_blocks'][-1]

        # create downsamplers
        self._downsampler = nn.ModuleList()
        for level_config in self._cfg['level'][:-1]:
            self._downsampler.append(make_downsample(level_config['downsample']))

        self.activation = nn.ReLU(True)

        # use Conv1d instead because of consistent initialisation
        self.main_head = nn.Sequential(
            nn.Dropout(p=self._fc_dropout),
            nn.Linear(in_features=(512 + self._nos_ch_dover)*2, out_features=512, bias=True),
            self.activation,
            nn.Dropout(p=self._fc_dropout),
            nn.Linear(in_features=512, out_features=256, bias=True),
            self.activation,
            nn.Dropout(p=self._fc_dropout),
            nn.Linear(in_features=256, out_features=128, bias=True),
            self.activation,
            nn.Dropout(p=self._fc_dropout),
            nn.Linear(in_features=128, out_features=64, bias=True),
            self.activation,
            nn.Dropout(p=self._fc_dropout),
            nn.Linear(in_features=64, out_features=1, bias=True),
        )

        # auxiliary task: predict aux scores
        self.aux_head = nn.Sequential(
            nn.Linear(in_features=512 + self._nos_ch_dover, out_features=128, bias=True),
            self.activation,
            nn.Linear(in_features=128, out_features=self._nos_ch_aux, bias=True),
        )

        if 'weights' in kwargs and kwargs['weights'] is not None:
            self._load_from_checkpoint(kwargs['weights'])

    def _load_from_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in state_dict:
            new_state_dict = OrderedDict()
            for key in state_dict['state_dict']:
                if key.startswith('_model.'):
                    new_state_dict[key[7:]] = state_dict['state_dict'][key]
                else:
                    new_state_dict[key] = state_dict['state_dict'][key]
            state_dict = new_state_dict
        self.load_state_dict(state_dict, strict=True)
        for param in self.parameters():
            param.requires_grad = False
        print(f'load model weights from checkpoint {checkpoint_path}')

    def model_backbone(self, x):
        encoder_out = list()
        for idx in range(0, self._num_levels):
            if idx != 0:
                x = self._downsampler[idx - 1](encoder_out[idx - 1])
            encoder_out.append(self._encoder[idx](x))
        feat = encoder_out[-1]

        feat = torch.mean(feat, dim=(2, 3))
        return feat

    def forward(self, x1, x2, d1, d2):
        feat_x1 = self.model_backbone(x1)
        feat_x2 = self.model_backbone(x2)

        # concat dover feat
        feat_x1 = torch.cat([feat_x1, d1], dim=1)
        feat_x2 = torch.cat([feat_x2, d2], dim=1)

        # predict aux scores
        y1_aux = self.aux_head(feat_x1).clamp(-1.0, 1.0)
        y2_aux = self.aux_head(feat_x2).clamp(-1.0, 1.0)

        # predict main score
        y = torch.cat([feat_x1, feat_x2], dim=1)
        y = self.main_head(y)
        y = torch.clamp(y, 0, 1)
        return y, y1_aux, y2_aux
