import os
import torch

import numpy as np
import yaml
import datetime
from socket import gethostname
from pathlib import Path
from common.flags import flags
from common.utils import (
    build_config,
    create_grid,
    save_experiment_log,
)
from base_runner import MoserRunner, FFJORDRunner
from sweep import create_sweep_jobs, VanillaJobRunner

import waic
import submitit

CONFIGURATION_FILENAME = "configuration.yaml"
CHECKPOINT_NAME = "checkpoint.pt"

RUNNER_CLASSES = {
    "moser": MoserRunner,
    "ffjord": FFJORDRunner
}

def get_run_dir(config):
    run_dir = config["cmd"]["run_dir"]
    if run_dir is not None:
        return Path(run_dir)
    identifier = config["cmd"].get("identifier")
    run_dir = os.path.join(os.getcwd(), "_experiments", config["dataset"]["type"])
    if not config["cmd"]["continue_saved"]:
        timestamp = torch.tensor(datetime.datetime.now().timestamp()).to(
            dtype=int
        )
        timestamp = datetime.datetime.fromtimestamp(timestamp).strftime(
                "%Y-%m-%d-%H-%M-%S"
        )
        if identifier is not None:
            dir_name = "%s_%s" %(timestamp, identifier)
        else:
            dir_name = timestamp
        run_dir = os.path.join(run_dir, dir_name)

    return Path(run_dir)

def get_last_checkpoint(checkpoint_dir):
    checkpoints = os.listdir(checkpoint_dir)
    if len(checkpoints) == 1:
        return os.path.join(checkpoint_dir, checkpoints[0])
    epochs = [float(re.match("checkpoint_(\d+\.?\d*).pt", checkpoint).groups()[0]) for checkpoint in checkpoints]
    max_epoch_index = np.argmax(epochs)
    return os.path.join(checkpoint_dir, checkpoints[max_epoch_index])

def run(config):
    run_dir = get_run_dir(config)
    model_name = config["model"].get("name", "moser")
    runner_class = RUNNER_CLASSES[model_name]
    runner = runner_class(config, run_dir)
    run_dir = os.path.dirname(runner.config["cmd"]["checkpoint_dir"])
    with open(os.path.join(run_dir, CONFIGURATION_FILENAME), "w") as f:
        yaml.safe_dump(config, f)

    # Load model
    if config["cmd"]["continue_saved"]:
        checkpoint_path = get_last_checkpoint(runner.config["cmd"]["checkpoint_dir"])
        print("continuing from checkpoint %s" %checkpoint_path)
        runner.load_pretrained(checkpoint_path)

    if config["cmd"]["checkpoint"] is not None:
        runner.load_pretrained(config["cmd"]["checkpoint"])

    # Train model
    runner.start()
    if config["cmd"]["mode"] == "train":
        try:
            runner.train()
        except KeyboardInterrupt:
            runner.finalize()

    # Test model
    if config["cmd"]["mode"] == "validate":
        runner.train_loader.dataset.initial_plots(runner.config["cmd"]["results_dir"], model=runner.model)
        runner.validate(split="test")

    return runner

class Runner(submitit.helpers.Checkpointable):
    def __init__(self):
        self.config = None
        self.chkpt_path = None

    def __call__(self, config):
        run(config)       

def main():
    parser = flags.get_parser()
    args = parser.parse_args()
    if not args.config_yml and not args.checkpoint and not args.continue_saved:
        raise ValueError("either config-yml or checkpoint needs to be given")
    
    if args.checkpoint:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(os.path.dirname(checkpoint_dir), CONFIGURATION_FILENAME)
        args.config_yml = config_path
        args.run_dir = os.path.join(os.path.dirname(checkpoint_dir) + "_continued", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    if args.continue_saved:
        config_path = os.path.join(args.run_dir, CONFIGURATION_FILENAME)
        args.config_yml = config_path

    config = build_config(args)

    if args.submit == 'vanilla':  # Run on cluster
        run_dir = os.path.join(os.getcwd(), "_experiments", "sweep_runs")
        jobs = create_sweep_jobs(args, run_dir)
        runner = VanillaJobRunner()
        device = args.local_rank if args.local_rank is not None else torch.cuda.device_count() - len(jobs)
        for i, job in enumerate(jobs):
            job.params["local_rank"] = device + i
        for job in jobs:
            runner.run_job(job)
    elif args.submit == 'submitit':
        if args.sweep_yml:  # Run grid search
            configs = create_grid(config, args.sweep_yml)
        else:
            configs = [config]

        print(f"Submitting {len(configs)} jobs")
        executor = submitit.AutoExecutor(folder=os.path.join(args.run_dir, "slurm", "%j"))
        executor.update_parameters(
            name=args.identifier,
            mem_gb=args.slurm_mem,
            timeout_min=args.slurm_timeout * 60,
            slurm_partition=args.slurm_partition,
            gpus_per_node=args.num_gpus,
            cpus_per_task=(args.num_workers + 1),
            tasks_per_node=(args.num_gpus if args.distributed else 1),
            nodes=args.num_nodes,
        )
        jobs = executor.map_array(Runner(), configs)
        print("Submitted jobs:", ", ".join([job.job_id for job in jobs]))
        log_file = save_experiment_log(args, jobs, configs)
        print(f"Experiment log saved to: {log_file}")
    else:  # Run locally
        Runner()(config)

if __name__ == "__main__":
    main()