from turtle import forward
import numpy as np
import torch
import torch.nn as nn

# import detr_modules.transformer as transformer
from .detr_modules.transformer import TransformerEncoder, TransformerEncoderLayer

class BaseTransBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.input_channels = input_channels

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]

        c_in = sum(num_upsample_filters)
        self.num_bev_features = c_in

        enc_nhead = self.model_cfg.ENC_NHEAD
        ffn_dim = self.model_cfg.FFN_DIM

        # self.encoder_layer:TransformerEncoderLayer = None
        # self.add_module(
        #     'encoder_layer',
        #     TransformerEncoderLayer(
        #         d_model=input_channels,
        #         nhead=enc_nhead,
        #         dim_feedforward=ffn_dim,
        #         # dropout=args.enc_dropout,
        #         # activation=args.enc_activation,
        #     )
        # )
        enc_args = {'d_model': input_channels, 'nhead': enc_nhead, 'dim_feedforward': ffn_dim}

        # self.encoder:TransformerEncoder = None
        self.add_module(
            'encoder',
            TransformerEncoder(
                encoder_args=enc_args, num_layers=self.model_cfg.N_LAYERS
            )
        )
        # nn.functional.pixel_shuffle(out, 4)
        self.add_module(
            'shuffle',
            nn.PixelShuffle(4)
        )

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features_2d']
        ups = []
        ret_dict = {}
        x = spatial_features # bs, c, h, w [2, 64, 468, 468]

        xyz, x, xyz_inds = self.encoder(x, transpose_swap=True)
        x = self.shuffle(x)
        data_dict['spatial_features_2d'] = x
        
        return data_dict