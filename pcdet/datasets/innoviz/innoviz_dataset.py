
import json
import glob

import numpy as np
import copy
import pickle
import os
from tqdm import tqdm
import torch

from pcdet.datasets import DatasetTemplate
from ...utils import box_utils, common_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils

class InnovizDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, split=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        self.kfold_ratio = dataset_cfg.get('KFOLD', None)
        if self.kfold_ratio is not None and training:
            aug_cfg_list = dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST
            sample_paths = dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].DB_INFO_PATH
            for idx, sample_pkl in enumerate(sample_paths):
                if sample_pkl == "innoviz_processed_data_v0_1_0_innoviz_dbinfos_train_sampled_1.pkl":
                    kfold_idx_str = str(int(self.kfold_ratio[0] * 5 + 0.5))
                    new_pkl_name = "innoviz_processed_data_v0_1_0_" + f"f{kfold_idx_str}" + "_innoviz_dbinfos_train_sampled_1.pkl"
                    dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].DB_INFO_PATH[idx] = new_pkl_name
        
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        # TODO get folder names from configurations
        self._lidar_dir = 'itwo'
        self._lidar_ext = '.bin'

        self._gt_dir = 'gt_boxes'
        self._gt_boxes_ext = '.bin'
        self._gt_map_fname = 'classmap.json'

        self.split = split if split is not None else self.dataset_cfg.DATA_SPLIT[self.mode]
        self.lidar_path = self.root_path / self.split / self._lidar_dir
        self.gt_path = self.root_path / self.split / self._gt_dir
        print("### LIDAR PATH: " ,self.lidar_path)

        self.use_sub = self.dataset_cfg.get('USE_SUB1300', False) and self.training
        if self.use_sub:
            self._lidar_sub_dir = 'itwo_sub'
            self.lidar_sub_path = self.root_path / self.split / self._lidar_sub_dir
            data_sub_file_list = glob.glob(str(self.lidar_sub_path / f'*{self._lidar_ext}'))
            data_sub_file_list.sort()
            print("### LIDAR SUB PATH: " ,self.lidar_sub_path)
        else:
            data_sub_file_list = []

        self.only2d = self.dataset_cfg.get('ONLY2D', False)
        self.zoff = self.dataset_cfg.get('ZOFF', None)
        
        data_file_list = glob.glob(str(self.lidar_path / f'*{self._lidar_ext}'))
        data_file_list.sort()
        # apply kfold ratio
        self.kfold_ratio = self.dataset_cfg.get('KFOLD', None)
        if self.kfold_ratio is not None and self.split == 'testing':
            start_idx = int(len(data_file_list) * self.kfold_ratio[0])
            end_idx = int(len(data_file_list) * self.kfold_ratio[1])
            if self.mode == "train":
                data_file_list = data_file_list[:start_idx] + data_file_list[end_idx:]
            else:
                data_file_list = data_file_list[start_idx:end_idx]
            print(f"KFOLD applied: {self.kfold_ratio}")
        
        basic_class_remap = dict({
            "Car": "Car",
            "Vehicle": "Vehicle",
            "Pedestrian": "Pedestrian",
            "Cyclist": "Cyclist",
            "Motorcycle": "Motorcycle",
            "Truck": "Truck",
            "Unknown": "Unknown",
        })
        self.class_remap = self.dataset_cfg.get('CLASS_REMAP', basic_class_remap)

        self.ids_list = [f.split('/')[-1][:-len(self._lidar_ext)] for f in data_file_list]
        self.ids_list.sort()

        self.ids_sub_list = [f.split('/')[-1][:-len(self._lidar_ext)] for f in data_sub_file_list]
        self.ids_sub_list.sort()
        
        # get gt data
        self._gt_map = None
        if self.gt_path.exists():
            with open(self.gt_path / self._gt_map_fname) as f:
                self._gt_map = json.load(f)
                # inverse dict to int: str
                self._gt_map = {v: k for k, v in self._gt_map.items()}

        self.innoviz_infos = []
        self.include_innoviz_data(self.mode)

    def __len__(self):
        if self.use_sub:
            # return len(self.ids_list) + len(self.ids_sub_list)
            return len(self.ids_list) * 2
        else:
            return len(self.ids_list)
    
    def include_innoviz_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Innoviz dataset')
        innoviz_infos = []
        if self.split in ["submit", "training"]:
            return

        for idx, file_name in enumerate(self.ids_list):
            gt_path = self.gt_path / f'{file_name}{self._gt_boxes_ext}'
            gt_boxes = np.fromfile(gt_path, dtype=np.float32).reshape(-1, 8)

            ori_names = [self._gt_map[int(class_val + 0.5)] for class_val in gt_boxes[:, 7]]
            annos = {
                'frame_id': idx,
                'gt_boxes_lidar': gt_boxes[:, :7],
                # 'name': np.array([self._gt_map[int(class_val + 0.5)] for class_val in gt_boxes[:, 7]])
                # 'name': np.array(  self.class_remap[[self._gt_map[int(class_val + 0.5)] for class_val in gt_boxes[:, 7]]]  ),
                'name': np.array(  list(map(lambda x: self.class_remap[x], ori_names))  ),
                'difficulty': np.ones_like(gt_boxes[:, 7]),
                'num_points_in_gt': np.ones_like(gt_boxes[:, 7]) * 10,
            }
            info = {
                'annos': annos,
            }
            innoviz_infos.append(info)

        self.innoviz_infos.extend(innoviz_infos)

        if self.logger is not None:
            self.logger.info('Total samples for Innoviz dataset: %d' % (len(innoviz_infos)))

    def __getitem__(self, index):
        if self.use_sub and len(self.ids_list) <= index:
            new_index = np.random.randint(0, len(self.ids_sub_list))
            file_name = self.ids_sub_list[new_index]
            points = np.fromfile(self.lidar_sub_path / f'{file_name}{self._lidar_ext}', dtype=np.float32).reshape(-1, 4)
            correct_gt = 0.0
            file_name = self.ids_list[0] # mismatch gt as intended
        else:
            file_name = self.ids_list[index]
            points = np.fromfile(self.lidar_path / f'{file_name}{self._lidar_ext}', dtype=np.float32).reshape(-1, 4)
            correct_gt = 1.0

        if self.zoff is not None:
            points[:,2] += self.zoff

        input_dict = {
            'points': points,
            'frame_id': index,
            'correct_gt': correct_gt,
        }

        gt_path = self.gt_path / f'{file_name}{self._gt_boxes_ext}' 
        
        if gt_path.exists():
            gt_boxes = np.fromfile(gt_path, dtype=np.float32).reshape(-1, 8)
            input_dict['gt_boxes'] = gt_boxes[:, :7]
            # input_dict['gt_names'] = np.array([self._gt_map[int(class_val + 0.5)] for class_val in gt_boxes[:, 7]])
            # input_dict['gt_names'] = np.array(  self.class_remap[[self._gt_map[int(class_val + 0.5)] for class_val in gt_boxes[:, 7]]]  ),
            ori_names = [self._gt_map[int(class_val + 0.5)] for class_val in gt_boxes[:, 7]]
            input_dict['gt_names'] = np.array(  list(map(lambda x: self.class_remap[x], ori_names))  )

        data_dict = self.prepare_data(data_dict=input_dict)
        
        if self.only2d:
            data_dict['gt_boxes_yesz'] = copy.deepcopy(data_dict['gt_boxes'])
            data_dict['gt_boxes'][:, 2] = 1.0
            data_dict['gt_boxes'][:, 5] = 5.9
        return data_dict

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 
                'boxes': np.zeros([num_samples, 7]),
                'score': np.zeros(num_samples), 
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['boxes'] = pred_boxes
            pred_dict['score'] = pred_scores

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)
            
        return annos       

    def get_infos(self, raw_data_path, save_path, has_label=True, sampled_interval=1, update_info_only=False, do_semantic_label=False):
        from . import innoviz_utils
        from functools import partial

        single_sequence_infos = []
        for idx, file_name in enumerate(self.ids_list):
            gt_path = self.gt_path / f'{file_name}{self._gt_boxes_ext}'
            gt_boxes = np.fromfile(gt_path, dtype=np.float32).reshape(-1, 8)
            ori_names = [self._gt_map[int(class_val + 0.5)] for class_val in gt_boxes[:, 7]]
            annos = {
                'frame_id': idx,
                'gt_boxes_lidar': gt_boxes[:, :7],
                # 'name': np.array([self._gt_map[int(class_val + 0.5)] for class_val in gt_boxes[:, 7]])
                # 'name': np.array(  self.class_remap[[self._gt_map[int(class_val + 0.5)] for class_val in gt_boxes[:, 7]]]  ),
                'name': np.array(  list(map(lambda x: self.class_remap[x], ori_names))  ),
                'difficulty': np.ones_like(gt_boxes[:, 7]),
                'num_points_in_gt': np.ones_like(gt_boxes[:, 7]) * 10,
            }
            info = {}
            info['annos'] = annos

            pc_info = {'num_features': 4, 'lidar_sequence': self.split, 'sample_idx': idx}
            info['point_cloud'] = pc_info
            info['frame_id'] = self.split + ('_%03d' % idx)

            points = np.fromfile(self.lidar_path / f'{file_name}{self._lidar_ext}', dtype=np.float32).reshape(-1, 4)
            
            num_points_of_each_lidar = [points.shape[0]]
            info['num_points_of_each_lidar'] = num_points_of_each_lidar

            single_sequence_infos.append(info)
        sequcnes_infos = [single_sequence_infos]
        all_sequences_infos = [item for infos in sequcnes_infos for item in infos]

        return all_sequences_infos

    def create_groundtruth_database(self, info_path, save_path, used_classes=None, split='train', sampled_interval=10,
                                    processed_data_tag=None):
        # from scipy.spatial import ConvexHull, Delaunay
        
        # use_sequence_data = self.dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.dataset_cfg.SEQUENCE_CONFIG.ENABLED

        # if use_sequence_data:
        #     st_frame, ed_frame = self.dataset_cfg.SEQUENCE_CONFIG.SAMPLE_OFFSET[0], self.dataset_cfg.SEQUENCE_CONFIG.SAMPLE_OFFSET[1]
        #     self.dataset_cfg.SEQUENCE_CONFIG.SAMPLE_OFFSET[0] = min(-4, st_frame)  # at least we use 5 frames for generating gt database to support various sequence configs (<= 5 frames)
        #     st_frame = self.dataset_cfg.SEQUENCE_CONFIG.SAMPLE_OFFSET[0]
        #     database_save_path = save_path / ('%s_gt_database_%s_sampled_%d_multiframe_%s_to_%s' % (processed_data_tag, split, sampled_interval, st_frame, ed_frame))
        #     db_info_save_path = save_path / ('%s_waymo_dbinfos_%s_sampled_%d_multiframe_%s_to_%s.pkl' % (processed_data_tag, split, sampled_interval, st_frame, ed_frame))
        #     db_data_save_path = save_path / ('%s_gt_database_%s_sampled_%d_multiframe_%s_to_%s_global.npy' % (processed_data_tag, split, sampled_interval, st_frame, ed_frame))
        database_save_path = save_path / ('%s_gt_database_%s_sampled_%d' % (processed_data_tag, split, sampled_interval))
        db_info_save_path = save_path / ('%s_innoviz_dbinfos_%s_sampled_%d.pkl' % (processed_data_tag, split, sampled_interval))
        db_data_save_path = save_path / ('%s_gt_database_%s_sampled_%d_global.npy' % (processed_data_tag, split, sampled_interval))

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}
        for used_class in used_classes: # To prevent empty class
            all_db_infos[used_class] = []

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        point_offset_cnt = 0
        stacked_gt_points = []
        for k in tqdm(range(0, len(infos), sampled_interval)):
            # print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]

            # ratio = [0.8, 1.0] # TO1DO: Delete this tmp code
            # if int(len(infos) * ratio[0]) <= k < int(len(infos) * ratio[1]):
            #     continue

            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']
            # points = self.get_lidar(sequence_name, sample_idx)
            file_name = info['frame_id'].split('_')[1].zfill(10)
            points = np.fromfile(self.lidar_path / f'{file_name}{self._lidar_ext}', dtype=np.float32).reshape(-1, 4)

            # if use_sequence_data:
            #     points, num_points_all, sample_idx_pre_list, _, _, _, _ = self.get_sequence_data(
            #         info, points, sequence_name, sample_idx, self.dataset_cfg.SEQUENCE_CONFIG
            #     )

            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            gt_boxes = annos['gt_boxes_lidar']

            # comment out the following line to ignore skipping
            # if k % 4 != 0 and len(names) > 0:
            #     mask = (names == 'Vehicle')
            #     names = names[~mask]
            #     difficulty = difficulty[~mask]
            #     gt_boxes = gt_boxes[~mask]

            # if k % 2 != 0 and len(names) > 0:
            #     mask = (names == 'Pedestrian')
            #     names = names[~mask]
            #     difficulty = difficulty[~mask]
            #     gt_boxes = gt_boxes[~mask]

            num_obj = gt_boxes.shape[0]
            if num_obj == 0:
                continue

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(num_obj):
                filename = '%s_%04d_%s_%d.bin' % (sequence_name, sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]
                gt_points[:, :3] -= gt_boxes[i, :3]

                if (used_classes is None) or names[i] in used_classes:
                    gt_points = gt_points.astype(np.float32)
                    assert gt_points.dtype == np.float32
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'sequence_name': sequence_name,
                               'sample_idx': sample_idx, 'gt_idx': i, 'box3d_lidar': gt_boxes[i],
                               'num_points_in_gt': gt_points.shape[0], 'difficulty': difficulty[i]}

                    # it will be used if you choose to use shared memory for gt sampling
                    stacked_gt_points.append(gt_points)
                    db_info['global_data_offset'] = [point_offset_cnt, point_offset_cnt + gt_points.shape[0]]
                    point_offset_cnt += gt_points.shape[0]

                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

        # it will be used if you choose to use shared memory for gt sampling
        stacked_gt_points = np.concatenate(stacked_gt_points, axis=0)
        np.save(db_data_save_path, stacked_gt_points)

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.innoviz_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                'Vehicle': 'Car',
                'Pedestrian': 'Pedestrian',
                'Cyclist': 'Cyclist',
                'Sign': 'Sign',
                'Car': 'Car'
            }
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from ..waymo.waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict
        
        def innoviz_eval(eval_det_annos, eval_gt_annos):
            from ..innoviz.innoviz_utils import eval_xyiou, eval_xyiou_multiprocessing
            ap_result_str = '\n'
            ap_dict = eval_xyiou_multiprocessing(list(zip(eval_det_annos, eval_gt_annos)))
            for key in ap_dict:
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.innoviz_infos]

        # eval_det_annos['boxes_lidar'] = eval_det_annos['boxes']
        for eval_det_anno in eval_det_annos:
            eval_det_anno['boxes_lidar'] = eval_det_anno['boxes']

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
        elif kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        elif kwargs['eval_metric'] == 'innoviz':
            ap_result_str, ap_dict = innoviz_eval(eval_det_annos, eval_gt_annos)
        elif kwargs['eval_metric'] == 'waymo+innoviz':
            ap_result_str1, ap_dict1 = waymo_eval(eval_det_annos, eval_gt_annos)
            ap_result_str2, ap_dict2 = innoviz_eval(eval_det_annos, eval_gt_annos)
            ap_result_str = ap_result_str1 + ap_result_str2
            ap_dict = {**ap_dict1, **ap_dict2}
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

def create_innoviz_infos(dataset_cfg, class_names, data_path, save_path,
                       raw_data_tag='raw_data', processed_data_tag='waymo_processed_data',
                       workers=1, update_info_only=False):
    dataset = InnovizDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=True, logger=common_utils.create_logger()  # traing = True is OK? ... not sure...
    )
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, train_split))
    # val_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, val_split))

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('---------------Start to generate data infos---------------')

    # dataset.set_split(val_split)
    # waymo_infos_val = dataset.get_infos(
    #     raw_data_path=data_path / raw_data_tag,
    #     save_path=save_path / processed_data_tag, has_label=True,
    #     sampled_interval=1, update_info_only=update_info_only,
    # )
    # with open(val_filename, 'wb') as f:
    #     pickle.dump(waymo_infos_val, f)
    # print('----------------Waymo info val file is saved to %s----------------' % val_filename)

    # dataset.set_split(train_split)
    waymo_infos_train = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, has_label=True,
        sampled_interval=1, update_info_only=update_info_only,
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(waymo_infos_train, f)
    print('----------------Innoviz info train file is saved to %s----------------' % train_filename)    

    if update_info_only:
        return

    print('---------------Start create groundtruth database for data augmentation---------------')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # dataset.set_split(train_split)
    dataset.create_groundtruth_database(
        info_path=train_filename, save_path=save_path, split='train', sampled_interval=1,
        used_classes=class_names, processed_data_tag=processed_data_tag
    )
    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import argparse
    import yaml
    from easydict import EasyDict
    from pathlib import Path

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_waymo_infos', help='')
    parser.add_argument('--processed_data_tag', type=str, default='innoviz_processed_data_v0_1_0', help='')
    parser.add_argument('--update_info_only', action='store_true', default=False, help='')
    parser.add_argument('--use_parallel', action='store_true', default=False, help='')
    parser.add_argument('--wo_crop_gt_with_tail', action='store_true', default=False, help='')
    # parser.add_argument('--do_semantic_label', action='store_true', default=False, help='save semantic labels per point')

    args = parser.parse_args()

    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()

    if args.func == 'create_innoviz_infos':
        try:
            yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
        except:
            yaml_config = yaml.safe_load(open(args.cfg_file))
        dataset_cfg = EasyDict(yaml_config)
        dataset_cfg.PROCESSED_DATA_TAG = args.processed_data_tag
        create_innoviz_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist', 'Truck'],
            data_path=ROOT_DIR / 'data' / 'innoviz',
            save_path=ROOT_DIR / 'data' / 'innoviz',
            raw_data_tag='raw_data',
            processed_data_tag=args.processed_data_tag,
            update_info_only=args.update_info_only,
        )
    # elif args.func == 'create_waymo_gt_database':
    #     try:
    #         yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
    #     except:
    #         yaml_config = yaml.safe_load(open(args.cfg_file))
    #     dataset_cfg = EasyDict(yaml_config)
    #     dataset_cfg.PROCESSED_DATA_TAG = args.processed_data_tag
    #     create_waymo_gt_database(
    #         dataset_cfg=dataset_cfg,
    #         class_names=['Vehicle', 'Pedestrian', 'Cyclist', 'Truck'],
    #         data_path=ROOT_DIR / 'data' / 'waymo',
    #         save_path=ROOT_DIR / 'data' / 'waymo',
    #         processed_data_tag=args.processed_data_tag,
    #         use_parallel=args.use_parallel, 
    #         crop_gt_with_tail=not args.wo_crop_gt_with_tail
    #     )
    else:
        raise NotImplementedError
