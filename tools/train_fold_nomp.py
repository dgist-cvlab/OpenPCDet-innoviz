#!/usr/bin/env python

import wandb
import os
import multiprocessing
import collections
import random
import subprocess
import argparse
import json
from easydict import EasyDict
from pathlib import Path
import copy
import yaml
import dpath.util

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    print("Unknown args: ", unknown)
    
    return unknown

def main(unknown):
    num_folds = 5
    # Spin up workers before calling wandb.init()
    # Workers will be blocked on a queue waiting to start

    sweep_run = wandb.init()
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown"

    metrics = []
    extra_args = ' '.join(unknown)
    print("*" * 40)
    print(extra_args)
    print("*" * 40)
    for num in range(num_folds):
        ret_str = subprocess.run(f'python train_nomp.py --kfold={num_folds} --kfold_idx={num} --sweep_id={sweep_id} --sweep_name={sweep_run_name} --wandb={"openpcdet3_sweep"} {extra_args}', shell=True)
        print(ret_str)
        run_name = "{}-{}".format(sweep_id, num)
        with open(f'val_out_tmp/val_{run_name}.json', 'r') as f:
            val_text = f.read()
        ret_dict = json.loads(val_text)
        print(ret_dict)
        metrics.append(ret_dict)

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
    wandb.join()

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)

if __name__ == "__main__":
    unknown = parse_config()
    main(unknown)
