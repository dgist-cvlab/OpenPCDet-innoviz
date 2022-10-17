from .detector3d_template import Detector3DTemplate
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from ...utils.ema_pytorch import EMA
import torch.nn as nn
import copy

class PVRCNNPlusPlusDual(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.doEMA = True
        self.add_module(
            'model_direct',
            PVRCNNPlusPlus(model_cfg, num_class, dataset)
        )
        self.add_module(
            'model_shadow',
            PVRCNNPlusPlus(model_cfg, num_class, dataset)
        )
        self.model_shadow.requires_grad_(False)
        self.add_module(
            'ema',
            EMA(
                self.model_direct,
                ema_model = self.model_shadow,
                beta = 0.9999,              # exponential moving average factor
                update_after_step = 100,    # only after this number of .update() calls will it start updating
                update_every = 10,          # how often to actually update, to save on compute (updates every 10th .update() call)
            )
        )
        self.add_module(
            'mseloss',
            nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        )
        # self.module_list = self.build_networks()

    def update_ema(self):
        self.ema.update()

    def forward_model(self, batch_dict, model):
        ## forward direct model with augmented input
        batch_dict = model.vfe(batch_dict)
        batch_dict = model.backbone_3d(batch_dict)
        batch_dict = model.map_to_bev_module(batch_dict)
        batch_dict = model.backbone_2d(batch_dict)
        batch_dict = model.dense_head(batch_dict)

        batch_dict = model.roi_head.proposal_layer(
            batch_dict, nms_config=model.roi_head.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = model.roi_head.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_targets_dict'] = targets_dict
            num_rois_per_scene = targets_dict['rois'].shape[1]
            if 'roi_valid_num' in batch_dict:
                batch_dict['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_dict['batch_size'])]

        batch_dict = model.pfe(batch_dict)
        batch_dict = model.point_head(batch_dict)
        batch_dict = model.roi_head(batch_dict)
        ## forward direct model DONE
        return batch_dict

    def forward(self, batch_dict):
        # batch_dict: original pipeline
        # batch_dict_aug2: more augmented data pipeline
        if self.training:
            batch_dict_aug2 = copy.deepcopy(batch_dict)
            batch_dict_aug2['points'] = batch_dict_aug2.pop('points_aug2')
            batch_dict_aug2['gt_boxes'] = batch_dict_aug2.pop('gt_boxes_aug2')

            batch_dict_aug2 = self.forward_model(batch_dict_aug2, self.model_direct)
            batch_dict = self.forward_model(batch_dict, self.model_shadow)

            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict, batch_dict_aug2)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            batch_dict = self.forward_model(batch_dict, self.model_direct)

            pred_dicts, recall_dicts = self.model_direct.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict, batch_dict_aug2):
        disp_dict = {}
        mask = batch_dict['correct_gt']
        if mask.sum() != 0:
            loss_rpn, tb_dict = self.model_direct.dense_head.get_loss_with_mask(mask=mask)
            if self.model_direct.point_head is not None:
                loss_point, tb_dict = self.model_direct.point_head.get_loss_with_mask(mask=mask, tb_dict=tb_dict)
            else:
                loss_point = 0
            loss_rcnn, tb_dict = self.model_direct.roi_head.get_loss_with_mask(mask=mask, tb_dict=tb_dict)

            loss_consist = self.mseloss(batch_dict['spatial_features_2d'], batch_dict_aug2['spatial_features_2d'])

            # for k in tb_dict:
            #     tb_dict[k] *= batch_dict['correct_gt']

            tb_dict = {
                **tb_dict,
                'loss_consist': loss_consist.item(),
            }
            loss = loss_rpn + loss_point + loss_rcnn + loss_consist

        else:
            loss_consist = self.mseloss(batch_dict['spatial_features_2d'], batch_dict_aug2['spatial_features_2d'])
            loss = loss_consist

            tb_dict = {
                'loss_consist': loss_consist.item(),
            }

        return loss, tb_dict, disp_dict
