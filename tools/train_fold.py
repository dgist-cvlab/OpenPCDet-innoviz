import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt_fold

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import wandb

from pcdet.config import cfg
from pcdet.config import cfg_from_list, cfg_from_yaml_file, log_config_to_file, merge_new_config
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
# from .train import main

import multiprocessing
import collections
import random
from easydict import EasyDict
import dpath.util
import copy

Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("num", "sweep_id", "sweep_run_name", "cfg", "args")
)
WorkerDoneData = collections.namedtuple("WorkerDoneData", ("val_dict"))

class Tb_fool():
    def __init__(self):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=5, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--wandb', type=str, default='openpcdet_sweep_truck', help='wandb project name')
    parser.add_argument('--sweep_id', type=str, default='unknown', help='wandb project name')
    parser.add_argument('--sweep_name', type=str, default='unknown', help='wandb project name')
    
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--kfold', type=int, default=None, help='kfold number k for test, 5 means 1/5')
    parser.add_argument('--kfold_idx', type=int, default=None, help='kfold number k ')
    

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    print("Unknown args: ", unknown)

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    cfg_append = copy.deepcopy(cfg)
    # overwrite unknown cfgs (from wandb)
    unknown_dict = dict(map(lambda x: (x[2:].split('=')), unknown))
    for key, value in unknown_dict.items():
        list_key = key.split(".")
        if len(list_key) == 1:
            pass
        else:
            leaf_key = list_key[-1]
            if leaf_key in ["LR", "NMS_THRESH", "SAMPLE_RADIUS_WITH_ROI", "SCORE_THRESH"]:
                value = float(value)
            elif leaf_key in ["NUM_EPOCHS", "WARMUP_EPOCH"]:
                value = int(value)
            
            dpath.util.new(cfg_append, "/".join(list_key), value)
    
    merge_new_config(cfg, cfg_append)
    
    return args, cfg

def main_with_args_cfg_wrapper(sweep_q, worker_q):
    reset_wandb_env()
    worker_data = worker_q.get()
    run_name = "{}-{}".format(worker_data.sweep_run_name, worker_data.num)

    from pcdet.config import cfg
    merge_new_config(cfg, worker_data.cfg)
    # cfg = worker_data.cfg
    args = worker_data.args
    # args, cfg = parse_config()
    kfold_num = worker_data.num
    kfold_ratio = 1 / args.kfold
    cfg.DATA_CONFIG.KFOLD = [kfold_ratio*kfold_num, kfold_ratio*(kfold_num+1)]

    run_wandb = wandb.init(
        group=worker_data.sweep_id,
        job_type=worker_data.sweep_run_name,
        name=run_name,
        config=cfg,
    )
    # cfg = EasyDict(run_wandb.config)
    
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    print(cfg)
    # main fucntion goes here
    # val_accuracy = random.random()
    args.extra_tag = f'sweep_{run_name}_{run_wandb.id}'
    val_dict = main_with_args_cfg(args, cfg)
    # val_accuracy = val_dict["val/LEVEL2_mAP"]
    # val_accuracy = val_dict

    run_wandb.log(val_dict)
    wandb.join() # wait run_wandb to finish
    sweep_q.put(WorkerDoneData(val_dict=val_dict))
    
def main_with_args_cfg(args, cfg):
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    # tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None
    tb_log = Tb_fool()

    if tb_log is not None and args.wandb is not None:
        wandb_name = f"{cfg.EXP_GROUP_PATH}/{cfg.TAG}/{args.extra_tag}"
        wandb.init(project=args.wandb, config=cfg, dir=str(output_dir), name=wandb_name)
        wandb.save(f'{args.cfg_file}', policy='now')
        wandb.save(f'{log_file}', policy='live')

    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )
    print("BUILD DATALOADER DONE")

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))
              
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            while len(ckpt_list) > 0:
                try:
                    it, start_epoch = model.load_params_with_optimizer(
                        ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
                    )
                    last_epoch = start_epoch + 1
                    break
                except:
                    ckpt_list = ckpt_list[:-1]

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch, 
        logger=logger, 
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record, 
        show_gpu_stat=not args.wo_gpu_stat
    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    # args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)  # Only evaluate the last args.num_epochs_to_eval epochs
    args.start_epoch = 0

    tb_dict = repeat_eval_ckpt_fold(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    return tb_dict

def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]

def kfold_main_wandb(args, _, num_folds=5):
    # Spin up workers before calling wandb.init()
    # Workers will be blocked on a queue waiting to start
    multiprocessing.set_start_method('spawn')

    sweep_q = multiprocessing.Queue()
    workers = []
    for num in range(num_folds):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=main_with_args_cfg_wrapper, kwargs=dict(sweep_q=sweep_q, worker_q=q)
        )
        p.start()
        workers.append(Worker(queue=q, process=p))

    # sweep_run = wandb.init()
    wandb_name = f"{cfg.EXP_GROUP_PATH}/{cfg.TAG}/{args.extra_tag}"
    cfg.DATA_CONFIG.KFOLD = None # to log
    sweep_run = wandb.init(project=args.wandb, config=cfg)

    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown"

    metrics = []
    for num in range(num_folds):
        worker = workers[num]
        # start worker
        worker.queue.put(
            WorkerInitData(
                sweep_id=sweep_id,
                num=num,
                sweep_run_name=sweep_run_name,
                cfg=EasyDict(sweep_run.config),
                args=args,
            )
        )
        # get metric from worker
        result = sweep_q.get()
        # wait for worker to finish
        worker.process.join()
        # log metric to sweep_run
        metrics.append(result.val_dict)

    # rearrange metrics (sum)
    metric_reduced = {}
    for k,v in metrics[0].items():
        metric_reduced[k] = 0.0
    for metric in metrics:
        for k,v in metric.items():
            metric_reduced[k] += v
    for k,v in metric_reduced.items():
        metric_reduced[k] /= len(metrics)
    # rearrange metrics ends
    
    sweep_run.log(metric_reduced)
    wandb.join() # wait for sweep_run to finish

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)

def main_warp():
    args, _ = parse_config()
    print(args, cfg)
    if args.kfold is None:
        # main()
        raise NotImplementedError
    else:
        kfold_main_wandb(args, cfg, num_folds=args.kfold)

if __name__ == '__main__':
    main_warp()
