import onnx
import torch
import torch.nn as nn
from io import BytesIO
import copy
from collections import OrderedDict
from src.operations import make_downsample, make_block_list


class Sample(nn.Module):
    def __init__(self, **kwargs):
        super(Sample, self).__init__()
        self._cfg = kwargs
        c = self._cfg['input_channels']
        self.layers = nn.Conv2d(c, c, 1, padding=0, bias=False)

    def forward(self, x):
        y = self.layers(x)
        return y

    def onnx_export(self, optimize=False):
        model_state = self.state_dict()
        bio = BytesIO()
        torch.save(model_state, bio)
        export_model = self.__class__(**self._cfg)
        bio.seek(0)
        export_model.load_state_dict(torch.load(bio, map_location=torch.device('cpu')), strict=False)
        export_model.eval()
        c = self._cfg['input_channels']
        h, w = self._cfg['ideal_height'], self._cfg['ideal_width']
        x = torch.randn(1, c, h, w)
        onnx_buffer = BytesIO()
        torch.onnx.export(
            export_model,
            x,
            onnx_buffer,
            opset_version=11,
            training=torch.onnx.TrainingMode.EVAL if optimize else torch.onnx.TrainingMode.PRESERVE,
            keep_initializers_as_inputs=False,
            do_constant_folding=optimize,
            input_names=['img_input'],
            output_names=['img_enhanced']   
        )
        onnx_buffer.seek(0)
        onnx_model = onnx.load(onnx_buffer)
        onnx.checker.check_model(onnx_model, full_check=True)
        return onnx_model
