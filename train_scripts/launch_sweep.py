import os
import sys

sys.path.append("./")

import matplotlib.pyplot as plt

import toml
import wandb
import pprint
import argparse
from importlib import import_module

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('-s', '--sweep_id')
    
    args = parser.parse_args()

    config_path = args.config_path
    existing_sweep_id = args.sweep_id

    if not os.path.exists(config_path):
        raise FileExistsError('config.toml doesn\'t exist')

    base_config = toml.load(config_path)

    const_dict = base_config["const"]
    var_dict = base_config["var"]
    
    const_dict["config_path"] = config_path
    
    param_dict = {k: {"value": v} for k, v in const_dict.items()} | {k: {"values": v} for k, v in var_dict.items()}

    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "episode_reward"},
        "parameters": param_dict,
    }

    pprint.pprint(param_dict)

    project_name = base_config["wandb_project_name"]

    train_python_file = base_config["train_python_file"]

    train_func = getattr(import_module(train_python_file), "train")

    def train():
        run = wandb.init(
            project=project_name,
            config=wandb.config,
        )

        config = wandb.config

        history = train_func(config)

        for reward in history["episode_reward"]:
            wandb.log({
                "episode_reward": reward,
            })

    sweep_count = 1
    for v in var_dict.values():
        sweep_count *= len(v)

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
    
    if existing_sweep_id is not None:
        sweep_id = existing_sweep_id
        print("The sweep ID we are going to use:", sweep_id)
        
    wandb.agent(sweep_id, function=train, count=sweep_count)

    wandb.finish()