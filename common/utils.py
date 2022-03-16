import collections
import copy
import json
import os
import time
import torch
import yaml
import itertools
import numpy as np

def save_checkpoint(state, checkpoint_dir="checkpoints/", name=None):
    if name:
        filename = os.path.join(checkpoint_dir, "checkpoint_%s.pt" %name)
    else:
        filename = os.path.join(checkpoint_dir, "checkpoint.pt")
    torch.save(state, filename)

# https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression
def update_config(original, update, override=True):
    """
    Recursively update a dict.
    Subdict's won't be overwritten but also updated.
    """
    for key, value in update.items():
        if key not in original or (not isinstance(value, dict) and override):
            original[key] = value
        else:
            update_config(original[key], value)
    return original

def save_experiment_log(args, jobs, configs):
    log_file = args.logdir / "exp" / time.strftime("%Y-%m-%d-%I-%M-%S%p.log")
    log_file.parent.mkdir(exist_ok=True, parents=True)
    with open(log_file, "w") as f:
        for job, config in zip(jobs, configs):
            print(
                json.dumps(
                    {
                        "config": config,
                        "slurm_id": job.job_id,
                        "timestamp": time.strftime("%I:%M:%S%p %Z %b %d, %Y"),
                    }
                ),
                file=f,
            )
    return log_file

def build_user_config(config_yml, config_override=None):
    config = yaml.safe_load(open(config_yml, "r"))

    # Load config from included files.
    includes = config.get("includes", [])
    if not isinstance(includes, list):
        raise AttributeError(
            "Includes must be a list, {} provided".format(type(includes))
        )

    for include in includes:
        include_config = yaml.safe_load(open(include, "r"))
        update_config(config, include_config, override=False)

    if includes != []:
        config.pop("includes")

    if config_override is not None:
        config_override = json.loads(config_override)
        config = update_config(config, config_override)
    return config

def build_config(args):
    config = build_user_config(args.config_yml, args.config_override)
    cmd_config = {}
    # Some other flags.
    cmd_config["mode"] = args.mode
    cmd_config["identifier"] = args.identifier
    cmd_config["seed"] = args.seed
    cmd_config["is_debug"] = args.debug
    cmd_config["run_dir"] = args.run_dir
    cmd_config["is_vis"] = args.vis
    cmd_config["print_every"] = args.print_every
    cmd_config["amp"] = args.amp
    cmd_config["checkpoint"] = args.checkpoint
    cmd_config["cpu"] = args.cpu
    # Submit
    cmd_config["submit"] = args.submit
    # Distributed
    cmd_config["local_rank"] = args.local_rank
    cmd_config["distributed_port"] = args.distributed_port
    cmd_config["world_size"] = args.num_nodes * args.num_gpus
    cmd_config["distributed_backend"] = args.distributed_backend
    cmd_config["continue_saved"] = args.continue_saved
    cmd_config["only_save_checkpoints"] = args.only_save_checkpoints

    config["cmd"] = cmd_config
    return config

def create_grid(base_config, sweep_file):
    def _flatten_sweeps(sweeps, root_key="", sep="."):
        flat_sweeps = []
        for key, value in sweeps.items():
            new_key = root_key + sep + key if root_key else key
            if isinstance(value, collections.MutableMapping):
                flat_sweeps.extend(_flatten_sweeps(value, new_key).items())
            else:
                if isinstance(value, list):
                    flat_sweeps.append((new_key, value))
                else:
                    flat_sweeps.append((new_key, [value]))
        return collections.OrderedDict(flat_sweeps)

    def _update_config(config, keys, override_vals, sep="."):
        for key, value in zip(keys, override_vals):
            key_path = key.split(sep)
            child_config = config
            for name in key_path[:-1]:
                child_config = child_config[name]
            child_config[key_path[-1]] = value
        return config

    sweeps = yaml.safe_load(open(sweep_file, "r"))
    flat_sweeps = _flatten_sweeps(sweeps)
    keys = list(flat_sweeps.keys())
    values = list(itertools.product(*flat_sweeps.values()))

    configs = []
    for i, override_vals in enumerate(values):
        config = copy.deepcopy(base_config)
        config = _update_config(config, keys, override_vals)
        configs.append(config)
    return configs

class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.is_best = False
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, loss, **kwargs):
        self.is_best = False
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.is_best = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.is_best = True
            self.best_score = score
            self.counter = 0

    def state_dict(self):
        return dict(
            patience=self.patience,
            verbose=self.verbose,
            counter=self.counter,
            best_score=self.best_score,
            early_stop=self.early_stop,
            is_best=self.is_best,
            delta=self.delta
        )

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            setattr(self, key, value)

def run_func_in_batches(func, x, max_batch_size, out_dim):
    k = int(np.ceil(x.shape[0] / max_batch_size))
    if out_dim is None:
        out = torch.empty(x.shape[0])
    else:
        out = torch.empty(x.shape[0], out_dim)
    for i in range(k):
        x_i = x[i * max_batch_size: (i+1) * max_batch_size]
        out[i * max_batch_size: (i+1) * max_batch_size] = func(x_i).detach().view(x_i.shape[0], out_dim)
    torch.cuda.empty_cache()
    return out