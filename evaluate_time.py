import os
import re
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from common.utils import build_user_config
from base_runner import MoserRunner, FFJORDRunner
from main import Path

FLOAT_RE = "\d+\.?\d*"
MAX_TIME = 5000

def get_train_metrics(run_dir):
    log_path = os.path.join(run_dir, "logs_dir", "training.log")
    with open(log_path) as f:
        log_lines = f.read().split("\n")
    fields = ["nll", "epoch", "total_time", "cummulative_training_time", "training_step_time"]
    out = defaultdict(list)
    for log_line in log_lines[:-2]:
        if "validation" in log_line:
            continue
        for field in fields:
            values = re.findall(f"{field}: ({FLOAT_RE})", log_line)
            if not values:
                continue
            value = values[0]
            out[field].append(float(value))
    for field in fields:
        out[field] = np.array(out[field])
    return dict(out)

def get_val_metrics(run_dir):
    log_path = os.path.join(run_dir, "logs_dir", "training.log")
    with open(log_path) as f:
        log_lines = f.read().split("\n")
    fields = ["nll", "step", "total_time"]
    out = defaultdict(list)
    for log_line in log_lines[:-2]:
        if "validation" not in log_line:
            continue
        for field in fields:
            values = re.findall(f"{field}: ({FLOAT_RE})", log_line)
            if not values:
                continue
            value = values[0]
            out[field].append(float(value))

    for field in fields:
        out[field] = np.array(out[field])
    out["epoch"] = out["step"] / 100
    return dict(out)

def evaluate_checkpoints(checkpoint_dir, runner, nll_field):
    split = "train"
    checkpoint_re = re.compile(f"checkpoint_({FLOAT_RE})_{split}.pt")
    checkpoints = [cp for cp in os.listdir(checkpoint_dir) if checkpoint_re.match(cp) is not None]
    epochs = []
    for checkpoint in checkpoints:
        epoch_re_match = checkpoint_re.match(checkpoint)
        epochs.append(float(epoch_re_match.groups()[0]))
    idx = np.argsort(epochs)
    epochs = np.array(epochs)[idx]
    checkpoints = np.array(checkpoints)[idx]
    total_times = []
    train_nlls = []
    iteration_times = []
    for i, checkpoint in enumerate(checkpoints):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        data = torch.load(checkpoint_path, map_location=runner.device)
        
        train_metrics = data["metrics"]
        
        total_times.append(train_metrics["total_time"])
        iteration_times.append(train_metrics["training_step_time"])
        train_nlls.append(train_metrics[nll_field])
        
    return checkpoints, epochs, np.array(total_times), np.array(iteration_times), np.array(train_nlls)
    
def plot_model(runner, checkpoint, eval_dir):
    data = torch.load(checkpoint, map_location=runner.device)
    runner.model.load_state_dict(data["state_dict"])
    eval_dir = eval_dir.format(**data["metrics"])
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    runner.train_loader.dataset.evaluate_model(runner.model, epoch=data["epoch"], eval_dir=eval_dir)
    runner.train_loader.dataset.test_model(runner.model, test_dir=eval_dir)

def plot_mid_checkpoints(runner, label, checkpoints, times, n_midpoints, out_dir):
    mid_checkpoints = []
    time_points = np.linspace(np.min(times), MAX_TIME, n_midpoints + 1)[1:]
    for time_point in time_points:
        ind = np.nonzero(times <= time_point)[0][-1]
        mid_checkpoints.append((times[ind], checkpoints[ind]))
    
    runner.train_loader.dataset.initial_plots(out_dir)
    for mid_time, mid_checkpoint in mid_checkpoints:        
        eval_dir = os.path.join(out_dir, label, "time_%s" %mid_time)
        print(f"evaluating for checkpoint in time {mid_time}")
        plot_model(runner, mid_checkpoint, eval_dir)


FONT_SIZE = 24

class Plotter:
    def __init__(self, title, max_time=MAX_TIME):
        self.title = title
        self.fig, self.ax = plt.subplots()
        self.max_time = max_time

    def set_labels(self, xlabel, ylabel, font_size=FONT_SIZE):
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.xaxis.label.set_size(font_size)
        self.ax.yaxis.label.set_size(font_size)

    def plot(self, times, nlls, label):
        idx = times < self.max_time
        times = times[idx]
        nlls = nlls[idx]
        self.ax.plot(times, nlls, label=label) 

    def save(self, out_dir):
        self.ax.legend()
        self.fig.savefig(os.path.join(out_dir, "%s.png" %self.title.replace(" ", "_")), bbox_inches="tight")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moser", type=str, required=True, help="path to Moser Flow experiment results")
    parser.add_argument("--ffjord", type=str, required=True, help="path to FFJORD experiment results")
    parser.add_argument("--out_dir", default="./time_evals", help="path for directory to save comparison results")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    runs = [
        (args.moser, MoserRunner, "MF", "nll"),
        (args.ffjord, FFJORDRunner, "FFJORD", "loss"),
    ]

    iteration_times_plt = Plotter("iteration times")
    iteration_times_plt.ax.set_yscale("log")
    iteration_times_plt.set_labels("running time", "iteration time")

    train_total_time_plt = Plotter("train nll vs total time")
    val_total_time_plt = Plotter("val nll vs total time")
    for plotter in [train_total_time_plt, val_total_time_plt]:
        plotter.set_labels("running time", "negative log-likelihood")

    for run_dir, runner_class, label, nll_field in runs:
        print(f"started evaluating {label}")
        config_path = os.path.join(run_dir, "configuration.yaml")
        config = build_user_config(config_path)
        config["cmd"]["local_rank"] = device
        runner = runner_class(config, run_dir=Path(os.path.join(args.out_dir, label)))
        runner.start()
        print("loading saved metrics")
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        checkpoints, epochs, total_times, iteration_times, train_nlls = evaluate_checkpoints(checkpoint_dir, runner, nll_field=nll_field)
        
        ref_times = total_times

        idx = ref_times < MAX_TIME
        total_times = total_times[idx]
        iteration_times = iteration_times[idx]
        train_nlls = train_nlls[idx]

        iteration_times_plt.plot(total_times, iteration_times, label)
        train_total_time_plt.plot(total_times, train_nlls, label)
        
        if not args.debug:
            plot_mid_checkpoints(runner, label, [os.path.join(checkpoint_dir, cp) for cp in checkpoints], ref_times, 5, args.out_dir)

    for plotter in [iteration_times_plt, train_total_time_plt]:
        plotter.save(args.out_dir)

if __name__ == "__main__":
    main()