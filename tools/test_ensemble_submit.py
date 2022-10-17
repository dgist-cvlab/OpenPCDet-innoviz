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

USE_NMS = False
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
    dst_list = []
    for i in range(len(pth_list)):
        sweep_name = sweep_name_list[i]
        cfg_file = cfg_file_list[i]
        cmd = f'CUDA_VISIBLE_DEVICES=0 python test.py --cfg_file {cfg_file} --batch_size 1 --ckpt {pth_list[i]}'
        print(f"run cmd: {cmd}")

        epoch_idx = pth_list[i].name.split("_")[-1].split(".")[0]
        dst = base_path.parent / str(Path(cfg_file).stem) / "default" / "eval" / f"epoch_{epoch_idx}" / PHASE / "default" / f"result_{str(base_path.stem)}_{sweep_name}.pkl"
        print("#### DESTINATION: ", dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
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
            dst = base_path.parent / str(Path(cfg_file).stem) / "default" / "eval" / f"epoch_{epoch_idx}" / PHASE / "default" / f"result_{str(base_path.stem)}_{sweep_name}.pkl"
            shutil.copyfile(result_file, dst)
        dst_list.append(dst)
        if dst.exists():
            result_file = dst
        with open(result_file, 'rb') as dat:
            eval_data = pickle.load(dat)

        for frame_idx, det_anno in enumerate(eval_data):
            names = copy.deepcopy(det_anno['name'])
            boxes = copy.deepcopy(det_anno['boxes']).astype(np.float32)
            scores = copy.deepcopy(det_anno['score']).astype(np.float32)
            name_idxes = np.asarray(list(map(lambda x: class_names_dict[x], names))).astype(np.float32)
            frame_annos = np.concatenate((boxes, name_idxes[..., np.newaxis], scores[..., np.newaxis]), axis=1) # 7 1 1
            annos_scene[frame_idx].append(frame_annos.astype(np.float32))
    
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
    for dst_path in dst_list:
        print(str(dst_path))
    print(f'****************Submission create done on *****************')

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # base_path = "/data0/AL/source/OpenPCDet/output/cfgs/innoviz_models/pv_rcnn_plusplus_noint_finetune72_truck"
    # "../output/cfgs/innoviz_models/pv_rcnn_plusplus_noint_finetune72_truck/sweep_j9renndp-kignufwo-4/ckpt/checkpoint_epoch_16.pth"
    base_path = "/data0/AL/source/OpenPCDet/output/cfgs/innoviz_models/pv_rcnn_plusplus_final_kf0"

    # sweep_name_list = [
    #     "sweep_j9renndp-kignufwo-4",
    #     "sweep_j9renndp-kignufwo-4",
    #     "sweep_j9renndp-kignufwo-4",
    #     "sweep_j9renndp-kignufwo-4",
    #     "sweep_j9renndp-kignufwo-4",
    #     "sweep_j9renndp-kignufwo-4",

    #     "sweep_j9renndp-eju1hjht-3",
    #     "sweep_j9renndp-eju1hjht-3",
    #     "sweep_j9renndp-eju1hjht-3",
    #     "sweep_j9renndp-eju1hjht-3",
    #     "sweep_j9renndp-eju1hjht-3",
    #     "sweep_j9renndp-eju1hjht-3",

    #     "sweep_j9renndp-nwiukxs3-2",
    #     "sweep_j9renndp-nwiukxs3-2",
    #     "sweep_j9renndp-nwiukxs3-2",
    #     "sweep_j9renndp-nwiukxs3-2",
    #     "sweep_j9renndp-nwiukxs3-2",
    #     "sweep_j9renndp-nwiukxs3-2",

    #     "sweep_j9renndp-vb8p5hp6-1",
    #     "sweep_j9renndp-vb8p5hp6-1",
    #     "sweep_j9renndp-vb8p5hp6-1",
    #     "sweep_j9renndp-vb8p5hp6-1",
    #     "sweep_j9renndp-vb8p5hp6-1",
    #     "sweep_j9renndp-vb8p5hp6-1",

    #     "sweep_j9renndp-eo90k9h4-0",
    #     "sweep_j9renndp-eo90k9h4-0",
    #     "sweep_j9renndp-eo90k9h4-0",
    #     "sweep_j9renndp-eo90k9h4-0",
    #     "sweep_j9renndp-eo90k9h4-0",
    #     "sweep_j9renndp-eo90k9h4-0",
    # ]
    # cfg_file_list = [
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-30.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-25.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-20.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-15.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-10.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-05.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-00.yaml",

    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-30.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-25.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-20.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-15.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-10.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-05.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-00.yaml",

    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-30.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-25.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-20.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-15.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-10.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-05.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-00.yaml",

    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-30.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-25.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-20.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-15.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-10.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-05.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-00.yaml",

    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-30.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-25.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-20.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-15.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-10.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-05.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_final16_z-00.yaml",
    # ]
    sweep_name_list = [
        "default",
        "default",
        "default",
        "default",
        "default",
        "default",
    ]
    cfg_file_list = [
        "./cfgs/innoviz_models/pv_rcnn_plusplus_final_debug_z-15.yaml",
        "./cfgs/innoviz_models/pv_rcnn_plusplus_final_debug_z-10.yaml",
        "./cfgs/innoviz_models/pv_rcnn_plusplus_final_debug_z-05.yaml",
        "./cfgs/innoviz_models/pv_rcnn_plusplus_final_debug_z-00.yaml",
        "./cfgs/innoviz_models/pv_rcnn_plusplus_final_debug_z+05.yaml",
        "./cfgs/innoviz_models/pv_rcnn_plusplus_final_debug_z+10.yaml",
    ]
    # cfg_file_list = [
    #     # "./cfgs/innoviz_models/pv_rcnn_plusplus_noint_finetune4_truck_submit.yaml",
    #     # "./cfgs/innoviz_models/pv_rcnn_plusplus_noint_finetune4_truck_submit.yaml",
    #     # "./cfgs/innoviz_models/pv_rcnn_plusplus_noint_finetune4_truck_submit.yaml",
    #     # "./cfgs/innoviz_models/pv_rcnn_plusplus_noint_finetune4_truck_submit.yaml",
    #     # "./cfgs/innoviz_models/pv_rcnn_plusplus_noint_finetune4_truck_submit.yaml",

    #     # "./cfgs/innoviz_models/pv_rcnn_plusplus_noint_finetune7_truck_submit.yaml",
    #     # "./cfgs/innoviz_models/pv_rcnn_plusplus_noint_finetune7_truck_submit.yaml",
    #     # "./cfgs/innoviz_models/pv_rcnn_plusplus_noint_finetune7_truck_submit.yaml",
    #     # "./cfgs/innoviz_models/pv_rcnn_plusplus_noint_finetune7_truck_submit.yaml",
    #     # "./cfgs/innoviz_models/pv_rcnn_plusplus_noint_finetune7_truck_submit.yaml",

    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_noint_finetune72_truck_submit.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_noint_finetune72_truck_submit.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_noint_finetune72_truck_submit.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_noint_finetune72_truck_submit.yaml",
    #     "./cfgs/innoviz_models/pv_rcnn_plusplus_noint_finetune72_truck_submit.yaml",
    # ]
    nms_config = EasyDict({
        'NMS_PRE_MAXSIZE': 4096,
        'NMS_POST_MAXSIZE': 1000,
        'NMS_THRESH': 0.01,
        # 'NMS_CLASS_AGNOSTIC': True,
        'NMS_TYPE': 'nms_gpu'
    })
    check_and_run(base_path, sweep_name_list, cfg_file_list, nms_config)


if __name__ == "__main__":
    main()
