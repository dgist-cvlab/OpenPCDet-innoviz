import os
import glob
import pickle
from pathlib import Path
import subprocess
import copy
import numpy as np
from easydict import EasyDict
from pcdet.models.model_utils.model_nms_utils import class_agnostic_nms
import torch

USE_NMS = True
USE_ZFILL = True
PHASE = "submit"
PHASE_NUMBER_DICT = {
    "training": 1219,
    "submit": 97,
}
PHASE_NUMBER = PHASE_NUMBER_DICT[PHASE]

class_names_dict = {
    "Vehicle" : 1,
    "Car" : 1,
    "Pedestrian" : 2,
    "Cyclist" : 3,
    "Motorcyclist" : 3,
    "Truck" : 4,
}

PTH_LIST = [

]

def check_and_run(base_path, sweep_name_list, cfg_file_list, nms_config):
    # --cfg_file ./cfgs/innoviz_models/pv_rcnn_plusplus_noint_finetune4_truck_submit.yaml --ckpt ../output/cfgs/innoviz_models/pv_rcnn_plusplus_noint_finetune4_truck/sweep_restful-sweep-36-2_1ksn3olx/ckpt/checkpoint_epoch_36.pth
    # ../output/cfgs/innoviz_models/pv_rcnn_plusplus_noint_finetune4_truck/sweep_restful-sweep-36-2_1ksn3olx/ckpt/checkpoint_epoch_36.pth
    base_path = Path(os.path.join(base_path))
    all_sweep_list = list(base_path.glob("*"))
    sweep_list = []
    for sweep in all_sweep_list:
        # print(sweep.name)
        for sweep_name in sweep_name_list:
            if sweep_name in sweep.name:
                sweep_list.append(sweep)
    # sweep_list.sort()
    pth_list = []
    for sweep in sweep_list:
        pth = list(sweep.glob("**/*.pth"))
        pth.sort()
        pth_list.append(pth[-1])
    print(pth_list)

    names_all = []
    boxes_all = []
    scores_all = []
    annos_all = []
    annos_scene = []
    for frame_idx in range(PHASE_NUMBER):
        annos_scene.append([])

    print(pth_list)

    for i in range(len(pth_list)):
        sweep_name = sweep_name_list[i]
        cfg_file = cfg_file_list[i]
        cmd = f'CUDA_VISIBLE_DEVICES=0 python test.py --cfg_file {cfg_file} --batch_size 1 --ckpt {pth_list[i]}'
        print(f"run cmd: {cmd}")

        epoch_idx = pth_list[i].name.split("_")[-1].split(".")[0]
        dst = base_path.parent / str(Path(cfg_file).stem) / "default" / "eval" / f"epoch_{epoch_idx}" / PHASE / "default" / f"result_{sweep_name}.pkl"

        # if dst.exists():
        if False:
            pass
        else:
            ret_str = subprocess.run(cmd, shell=True)   
            sweep_folder_name = pth_list[i].parent.parent.name
            print(sweep_folder_name, "  Runnig Done")

            path_parts = list(base_path.parts)
            path_parts.remove('cfgs')
            base_path2 = Path('/'.join(path_parts)[1:])

            result_file = base_path.parent / str(Path(cfg_file).stem) / "default" / "eval" / f"epoch_{epoch_idx}" / PHASE / "default" / "result.pkl"

            import shutil
            dst = base_path.parent / str(Path(cfg_file).stem) / "default" / "eval" / f"epoch_{epoch_idx}" / PHASE / "default" / f"result_{sweep_name}.pkl"
            shutil.copyfile(result_file, dst)

        if dst.exists():
            result_file = dst
        with open(result_file, 'rb') as dat:
            eval_data = pickle.load(dat)

        for frame_idx, det_anno in enumerate(eval_data):
            names = copy.deepcopy(det_anno['name'])
            boxes = copy.deepcopy(det_anno['boxes'])
            scores = copy.deepcopy(det_anno['score'])
            name_idxes = np.asarray(list(map(lambda x: class_names_dict[x], names))).astype(np.float32)
            frame_annos = np.concatenate((boxes, name_idxes[..., np.newaxis], scores[..., np.newaxis]), axis=1) # 7 1 1
            annos_scene[frame_idx].append(frame_annos)
    
    print(f'****************Running done on *****************')

    save_dir = base_path.parent / str(Path(cfg_file).stem) / "default" / PHASE
    save_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, anno in enumerate(annos_scene):
        con_anno = np.concatenate(anno)
        con_anno_cuda = torch.from_numpy(con_anno).to(torch.float32).to('cuda')
        # def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
        selected, scores = class_agnostic_nms(con_anno_cuda[:,8], con_anno_cuda[:, :8], nms_config, score_thresh=0.3)
        if USE_NMS:
            result = con_anno[np.asarray(selected.cpu())][:,:8]
        else: 
            result = con_anno[:,:8]
        if USE_ZFILL and result.shape[0] == 0:
            result = np.asarray([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.0, 1.0]], dtype=np.float32)
        result.tofile(save_dir / f"{str(frame_idx).zfill(10)}.bin")

    print(f'****************Submission create done on *****************')

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    nms_config = EasyDict({
        'NMS_PRE_MAXSIZE': 4096,
        'NMS_POST_MAXSIZE': 1000,
        'NMS_THRESH': 0.6,
        # 'NMS_CLASS_AGNOSTIC': True,
        'NMS_TYPE': 'nms_gpu'
    })
    # check_and_run(base_path, sweep_name_list, cfg_file_list, nms_config)
    result_dst_list = [
        # paths to result.pkl
    ]
    offset = [
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
    ]
    annos_scene = []
    for frame_idx in range(PHASE_NUMBER):
        annos_scene.append([])
    for idx, dst in enumerate(result_dst_list):
        with open(dst, 'rb') as dat:
            eval_data = pickle.load(dat)
        for frame_idx, det_anno in enumerate(eval_data):
            names = copy.deepcopy(det_anno['name'])
            boxes = copy.deepcopy(det_anno['boxes'])
            boxes[:, 2] -= offset[idx]
            scores = copy.deepcopy(det_anno['score'])
            name_idxes = np.asarray(list(map(lambda x: class_names_dict[x], names))).astype(np.float32)
            frame_annos = np.concatenate((boxes, name_idxes[..., np.newaxis], scores[..., np.newaxis]), axis=1).astype(np.float32) # 7 1 1
            annos_scene[frame_idx].append(frame_annos)
    print("READ DONE")
    save_dir = Path("merged_out") / PHASE
    save_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, anno in enumerate(annos_scene):
        con_anno = np.concatenate(anno).astype(np.float32)
        con_anno_cuda = torch.from_numpy(con_anno).to(torch.float32).to('cuda')
        # def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
        selected, scores = class_agnostic_nms(con_anno_cuda[:,8], con_anno_cuda[:, :8], nms_config, score_thresh=0.3)
        if USE_NMS:
            result = con_anno[np.asarray(selected.cpu())][:,:8]
        else: 
            result = con_anno[:,:8]
        if USE_ZFILL and result.shape[0] == 0:
            result = np.asarray([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.0, 1.0]], dtype=np.float32)
        result.tofile(save_dir / f"{str(frame_idx).zfill(10)}.bin")
    print("DONE")

if __name__ == "__main__":
    main()
